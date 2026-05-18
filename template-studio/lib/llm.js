import http from "node:http";
import https from "node:https";

import OpenAI from "openai";

import { createBlankTemplate, normalizeTemplate, normalizeText, slugifyTemplateId } from "./utils.js";

const DEFAULT_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1";
const DEFAULT_MODEL = "qwen3-coder-plus";
const TOKEN_RE = /[a-z0-9_.-]+|[\u4e00-\u9fff]+/g;
const THINK_TAG_RE = /<thi(?:nk|ngk)\b[^>]*>[\s\S]*?<\/thi(?:nk|ngk)>/gi;
const THINKING_TAG_RE = /<thinking\b[^>]*>[\s\S]*?<\/thinking>/gi;
const CODE_BLOCK_RE = /```(?:json)?\s*([\s\S]*?)```/gi;

export const DEFAULT_LLM_SETTINGS = {
  baseUrl: DEFAULT_BASE_URL,
  model: DEFAULT_MODEL,
  insecureSSL: false
};

export async function generateTemplateFromSentence({
  text,
  answer = "",
  currentMatch = null,
  currentTemplate = null,
  missingSlots = [],
  optimizationMode = "mixed",
  currentConfig,
  apiKey,
  baseUrl = DEFAULT_BASE_URL,
  model = DEFAULT_MODEL,
  insecureSSL = false,
  client = null
}) {
  if (!apiKey) {
    throw new Error("Missing API key.");
  }
  const httpAgent = buildHttpAgent(baseUrl, insecureSSL);
  const sdkClient =
    client ||
    new OpenAI({
      apiKey,
      baseURL: baseUrl,
      ...(httpAgent ? { httpAgent } : {})
    });
  const response = await sdkClient.chat.completions.create({
    model,
    temperature: 0.05,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "You are a senior template designer for a metric-query matcher. " +
          "You must return strict JSON only. Do not wrap the answer in markdown. " +
          "The system only supports metric_query templates and supports keyword_value, regex, time_range, and metric_conditions extractors."
      },
      {
        role: "user",
        content: buildGenerationPrompt({
          text,
          answer,
          currentMatch,
          currentTemplate,
          missingSlots,
          optimizationMode,
          currentConfig
        })
      }
    ]
  });
  const rawContent = extractMessageContent(response);
  const payload = parseModelJsonObject(rawContent);
  if (!payload) {
    const cleaned = cleanupModelText(rawContent);
    throw new Error(`Model did not return valid JSON. Raw output: ${truncate(cleaned || String(rawContent || ""), 320)}`);
  }
  const blank = createBlankTemplate();
  const generatedTemplate = normalizeTemplate({
    ...blank,
    ...(payload.template || {}),
    template_id:
      payload.template?.template_id ||
      inferTemplateId(payload.analysis?.entity, payload.analysis?.metric, payload.analysis?.query_operator, text)
  });
  const analysis = normalizeGenerationAnalysis(payload.analysis, generatedTemplate);
  return {
    analysis,
    template: generatedTemplate
  };
}

export function buildGenerationPrompt({
  text,
  answer = "",
  currentMatch = null,
  currentTemplate = null,
  missingSlots = [],
  optimizationMode = "mixed",
  currentConfig
}) {
  const sampleTemplates = selectRelevantTemplateExamples(text, currentConfig, 6);
  const blockedTerms = currentConfig?.matcher?.blocked_terms || [];
  return `
你要把一句中文问数 query 抽象成一个“更容易命中、也更容易维护”的模板 JSON。

当前引擎的关键约束：
- query_mode 固定写 metric_query
- 真正区分查询形态用 query_operator，例如 count / topn / list
- 模板必须是单意图，不要把 list / count / topn / trend / aggregate 混在一个模板里
- 先按“查询形态”拆模板，再把实体、指标、定位方式做成槽位
- required_slots 只保留“缺了就无法执行查询”的参数
- 如果出现 ip / mac / 名称 这种三选一定位方式，不要生成多个并行必填槽位；统一设计成 selector_type + selector_value
- 模板必须是单意图
- 尽量生成可执行的 slot_extractors
- 支持 keyword_value / regex / time_range / metric_conditions 四类 extractor
- 有包含关系的 keyword_value（如 端口 / 以太网端口）要设置 "match_policy": "longest"
- 通用时间优先用根级或模板级 {"type": "time_range"}，不要穷举所有时间词
- 单/多指标场景优先用 metric_conditions + slot_validations 控制 exact_items / min_items
- 如果是阈值、TopN、数值，优先给 regex
- 如果是时间、区域、算子、实体、指标、聚合类型，优先给 keyword_value
- 如果是设备标识、IP、MAC、名称这类自由值，优先给带前缀锚点的 regex
- 如果后端枚举值依赖资源类型，使用 derived_slots 从语义槽位派生，不要直接把“正常/故障”写死成单一枚举
- optional_slots 参与匹配但不保证执行参数完整；执行所需默认值写 slot_defaults
- negative_terms 默认包含 原因 / 根因 / 报告 / 总结 / 预测
- llm_slot_extraction 要尽量收窄，只补少量高价值槽位
- 除非确实需要补中文数字或少量自由值，否则 llm_slot_extraction 默认关闭
- 生成的 utterances 至少 5 条，要覆盖语序变化、口语化、提参顺序变化、常见 typo
- must_terms 要按“语义组”拆开，不要把多个概念混在一个 group 里
- slot_constraints 要尽量钉死 entity_type / metric / query_operator 这类容易冲突的槽位

当前全局 blocked_terms：
${JSON.stringify(blockedTerms, null, 2)}

当前最相近的模板样例：
${JSON.stringify(sampleTemplates, null, 2)}

目标句子：
${text}

参考答案或人工说明（如果为空就忽略）：
${answer || ""}

当前系统对这句 query 的已有结果（如果为空就忽略）：
${currentMatch ? JSON.stringify(currentMatch, null, 2) : ""}

当前优先修复的模板（如果为空就忽略）：
${currentTemplate ? JSON.stringify(currentTemplate, null, 2) : ""}

当前缺失槽位（如果为空就忽略）：
${JSON.stringify(Array.isArray(missingSlots) ? missingSlots : [], null, 2)}

当前优化模式：
${optimizationMode}

输出 JSON，结构固定为：
{
  "analysis": {
    "intent": "...",
    "entity": "...",
    "metric": "...",
    "query_operator": "...",
    "template_strategy": "...",
    "required_slots": ["..."],
    "optional_slots": ["..."],
    "notes": ["..."],
    "design_rationale": [
      "解释为什么这样拆模板",
      "解释为什么这些槽位是 required 或 optional",
      "解释为什么 extractor 要这样设计"
    ]
  },
  "template": {
    "template_id": "...",
    "query_mode": "metric_query",
    "description": "...",
    "utterances": ["..."],
    "required_slots": ["..."],
    "optional_slots": ["..."],
    "must_terms": [["..."]],
    "negative_terms": ["..."],
    "slot_constraints": {},
    "slot_extractors": {},
    "slot_defaults": {},
    "derived_slots": [],
    "slot_validations": {},
    "llm_slot_extraction": {
      "enabled": true,
      "slots": ["..."],
      "instructions": "..."
    },
    "metadata": {}
  }
}

请特别注意：
- design_rationale 必须可读、具体，方便人工二次调整
- template_id 要尽量复用当前仓库已有的命名风格
- 如果目标句子更像“已有模式的变体”，优先沿用已有 slot 命名，不要发明新名字
- 如果提供了 answer，要把它当成意图解释和字段边界的辅助信息，但不要把 answer 文本原样塞进模板
- 如果提供了当前已有结果，要优先思考“修已有模板更合理，还是新增模板更合理”
- 如果优化模式是 slot_completion，优先保留当前模板的语义边界、template_id、query_operator 和大部分槽位命名；优先补 utterances、must_terms、slot_extractors、required_slots / optional_slots，目标是把 partial 推成 matched
- 如果优化模式是 new_template，不要硬修一个明显不适配的旧模板；更倾向于新增一个边界更清晰的模板
- 如果提供了 currentTemplate，除非明确需要拆出新模板，否则不要随意改掉它的模板语义
`.trim();
}

export function extractMessageContent(response) {
  const message = response?.choices?.[0]?.message || {};
  const content = message?.content;
  if (typeof content === "string") {
    return content.trim();
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") {
          return item.trim();
        }
        if (item && typeof item === "object") {
          if (typeof item.text === "string") {
            return item.text.trim();
          }
          if (typeof item.content === "string") {
            return item.content.trim();
          }
        }
        return "";
      })
      .filter(Boolean)
      .join("");
  }
  return typeof content === "string" ? content.trim() : String(content || "").trim();
}

export function parseModelJsonObject(rawContent) {
  const cleaned = cleanupModelText(rawContent);
  if (!cleaned) {
    return null;
  }
  const direct = tryParseJsonObject(cleaned);
  if (direct) {
    return direct;
  }

  for (const block of extractCodeBlocks(cleaned)) {
    const parsed = tryParseJsonObject(block);
    if (parsed) {
      return parsed;
    }
  }

  const balanced = extractBalancedJsonObject(cleaned);
  if (!balanced) {
    return null;
  }
  return tryParseJsonObject(balanced);
}

export function cleanupModelText(rawContent) {
  const text = String(rawContent || "")
    .replace(/^\uFEFF/, "")
    .trim();
  if (!text) {
    return "";
  }
  return text
    .replace(THINK_TAG_RE, "")
    .replace(THINKING_TAG_RE, "")
    .trim();
}

function inferTemplateId(entity, metric, queryOperator, text) {
  const raw = [entity, metric, queryOperator, text].filter(Boolean).join(".");
  return slugifyTemplateId(raw, "template.generated");
}

function buildHttpAgent(baseUrl, insecureSSL) {
  if (!insecureSSL) {
    return null;
  }
  const protocol = safeProtocol(baseUrl);
  if (protocol === "https:") {
    return new https.Agent({
      keepAlive: true,
      rejectUnauthorized: false
    });
  }
  if (protocol === "http:") {
    return new http.Agent({
      keepAlive: true
    });
  }
  return null;
}

function safeProtocol(baseUrl) {
  try {
    return new URL(baseUrl).protocol;
  } catch {
    return "";
  }
}

function normalizeGenerationAnalysis(rawAnalysis, generatedTemplate) {
  const analysis = rawAnalysis && typeof rawAnalysis === "object" && !Array.isArray(rawAnalysis) ? rawAnalysis : {};
  const designRationale = normalizeStringArray(analysis.design_rationale);
  return {
    intent: String(analysis.intent || generatedTemplate.description || ""),
    entity: String(analysis.entity || generatedTemplate.slot_constraints?.entity_type?.[0] || ""),
    metric: String(analysis.metric || generatedTemplate.slot_constraints?.metric?.[0] || ""),
    query_operator: String(analysis.query_operator || generatedTemplate.slot_constraints?.query_operator?.[0] || ""),
    template_strategy: String(analysis.template_strategy || ""),
    required_slots: normalizeStringArray(analysis.required_slots, generatedTemplate.required_slots),
    optional_slots: normalizeStringArray(analysis.optional_slots, generatedTemplate.optional_slots),
    notes: normalizeStringArray(analysis.notes),
    design_rationale: designRationale.length ? designRationale : buildFallbackDesignRationale(generatedTemplate)
  };
}

function buildFallbackDesignRationale(template) {
  const rationale = [];
  rationale.push(`模板按单一查询形态收敛，description 聚焦在“${template.description || template.template_id}”。`);
  if (template.required_slots.length) {
    rationale.push(`required_slots 只保留执行查询最关键的参数：${template.required_slots.join(", ")}。`);
  }
  if (Object.keys(template.slot_constraints || {}).length) {
    rationale.push("slot_constraints 用来钉死容易冲突的 entity_type / metric / query_operator 等槽位。");
  }
  if (Object.keys(template.slot_extractors || {}).length) {
    rationale.push("slot_extractors 尽量按槽位最小可用来设计，枚举值优先 keyword_value，数值或自由值优先 regex。");
  }
  return rationale;
}

function selectRelevantTemplateExamples(text, currentConfig, limit) {
  const templates = Array.isArray(currentConfig?.templates) ? currentConfig.templates : [];
  const queryTerms = new Set(mixedPromptTerms(text));
  return templates
    .map((template) => ({
      template,
      score: scoreTemplateExample(queryTerms, template)
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.max(1, limit))
    .map(({ template }) => ({
      template_id: template.template_id,
      description: template.description,
      required_slots: template.required_slots,
      optional_slots: template.optional_slots,
      slot_constraints: template.slot_constraints,
      slot_extractors: template.slot_extractors,
      utterances: (template.utterances || []).slice(0, 4)
    }));
}

function scoreTemplateExample(queryTerms, template) {
  const docTerms = new Set(
    mixedPromptTerms(
      [
        template.template_id,
        template.description,
        ...(template.utterances || []),
        ...(template.required_slots || []),
        ...(template.optional_slots || [])
      ]
        .filter(Boolean)
        .join(" ")
    )
  );
  if (!queryTerms.size || !docTerms.size) {
    return 0;
  }
  let overlap = 0;
  for (const term of queryTerms) {
    if (docTerms.has(term)) {
      overlap += 1;
    }
  }
  return overlap / Math.max(queryTerms.size, 1);
}

function mixedPromptTerms(text) {
  const normalized = normalizeText(text);
  const chunks = normalized.match(TOKEN_RE) || [];
  const terms = [];
  for (const chunk of chunks) {
    terms.push(chunk);
    if (/[\u4e00-\u9fff]/.test(chunk)) {
      terms.push(...charNgrams(chunk, 2));
      if (chunk.length >= 3) {
        terms.push(...charNgrams(chunk, 3));
      }
      continue;
    }
    if (chunk.length >= 4) {
      terms.push(...charNgrams(chunk, 3));
    }
  }
  return Array.from(new Set(terms.filter(Boolean)));
}

function charNgrams(text, size) {
  if (text.length < size) {
    return [text];
  }
  const grams = [];
  for (let index = 0; index <= text.length - size; index += 1) {
    grams.push(text.slice(index, index + size));
  }
  return grams;
}

function normalizeStringArray(value, fallback = []) {
  const source = Array.isArray(value) ? value : fallback;
  return source.map((item) => String(item)).filter(Boolean);
}

function tryParseJsonObject(text) {
  if (!text) {
    return null;
  }
  try {
    const payload = JSON.parse(text);
    return payload && typeof payload === "object" && !Array.isArray(payload) ? payload : null;
  } catch {
    return null;
  }
}

function extractCodeBlocks(text) {
  const blocks = [];
  for (const match of text.matchAll(CODE_BLOCK_RE)) {
    blocks.push(String(match[1] || "").trim());
  }
  return blocks;
}

function extractBalancedJsonObject(text) {
  let start = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (start < 0) {
      if (char === "{") {
        start = index;
        depth = 1;
      }
      continue;
    }
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (char === "\"") {
      inString = !inString;
      continue;
    }
    if (inString) {
      continue;
    }
    if (char === "{") {
      depth += 1;
      continue;
    }
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return text.slice(start, index + 1);
      }
    }
  }
  return "";
}

function truncate(text, maxLength) {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength)}...`;
}
