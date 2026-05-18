import { exec } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import express from "express";

import { batchGenerateTemplates, evaluateBatchMatch, parseBatchSource, summarizeBatchMatches } from "./lib/batch.js";
import { generateTemplateFromSentence, DEFAULT_LLM_SETTINGS } from "./lib/llm.js";
import { TemplateMatcher } from "./lib/matcher.js";
import { optimizeConfigIteratively } from "./lib/optimizer.js";
import { normalizeConfig } from "./lib/utils.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.join(__dirname, "public");
const WORKSPACE_FILE = path.join(__dirname, "workspace", "current-config.json");
const DEFAULT_CONFIG_FILE = path.join(ROOT, "configs", "templates.json");
const DEFAULT_BATCH_CASES_FILE = path.join(ROOT, "tests", "fixtures", "evaluation_cases.json");
const PORT = Number(process.env.TEMPLATE_STUDIO_PORT || 3847);
const DEFAULT_INSECURE_SSL = ["1", "true", "yes", "on"].includes(
  String(process.env.TEMPLATE_STUDIO_INSECURE_SSL || process.env.OPENAI_INSECURE_SSL || "").toLowerCase()
);

const app = express();
app.use(express.json({ limit: "5mb" }));
app.use(express.static(PUBLIC_DIR));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true });
});

app.get("/api/workspace", async (_req, res) => {
  const { config, source } = await loadWorkspaceConfig();
  res.json({
    config,
    source,
    defaultConfigPath: DEFAULT_CONFIG_FILE,
    workspacePath: WORKSPACE_FILE,
    llmDefaults: {
      ...DEFAULT_LLM_SETTINGS,
      apiKey: process.env.DASHSCOPE_API_KEY || "",
      insecureSSL: DEFAULT_INSECURE_SSL
    }
  });
});

app.post("/api/workspace/save", async (req, res) => {
  const config = normalizeConfig(req.body?.config || {});
  await fs.mkdir(path.dirname(WORKSPACE_FILE), { recursive: true });
  await fs.writeFile(WORKSPACE_FILE, JSON.stringify(config, null, 2) + "\n", "utf8");
  res.json({ ok: true, workspacePath: WORKSPACE_FILE, config });
});

app.post("/api/workspace/reset", async (_req, res) => {
  const config = normalizeConfig(JSON.parse(await fs.readFile(DEFAULT_CONFIG_FILE, "utf8")));
  await fs.mkdir(path.dirname(WORKSPACE_FILE), { recursive: true });
  await fs.writeFile(WORKSPACE_FILE, JSON.stringify(config, null, 2) + "\n", "utf8");
  res.json({ ok: true, config, source: "default" });
});

app.post("/api/match", async (req, res) => {
  const text = String(req.body?.text || "");
  const config = normalizeConfig(req.body?.config || (await loadWorkspaceConfig()).config);
  const templateIds = Array.isArray(req.body?.templateIds) ? req.body.templateIds.map(String) : null;
  const matcher = new TemplateMatcher(config, { baseDir: ROOT });
  const result = matcher.match(text, { templateIds });
  res.json({ ok: true, result });
});

app.post("/api/route-for-sql", async (req, res) => {
  const text = String(req.body?.text || "");
  const config = normalizeConfig(req.body?.config || (await loadWorkspaceConfig()).config);
  const templateIds = Array.isArray(req.body?.templateIds) ? req.body.templateIds.map(String) : null;
  const matcher = new TemplateMatcher(config, { baseDir: ROOT });
  const decision = matcher.routeForSql(text, { templateIds });
  res.json({ ok: true, decision });
});

app.post("/api/batch/parse", async (req, res) => {
  try {
    const content = String(req.body?.content || "");
    const filename = String(req.body?.filename || "");
    const parsed = parseBatchSource(content, filename);
    res.json({ ok: true, ...parsed });
  } catch (error) {
    res.status(400).json({
      ok: false,
      error: error instanceof Error ? error.message : "Batch parsing failed."
    });
  }
});

app.get("/api/examples/evaluation-cases", async (_req, res) => {
  try {
    const content = await fs.readFile(DEFAULT_BATCH_CASES_FILE, "utf8");
    const parsed = parseBatchSource(content, path.basename(DEFAULT_BATCH_CASES_FILE));
    res.json({
      ok: true,
      source_name: DEFAULT_BATCH_CASES_FILE,
      ...parsed
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : "Failed to load evaluation cases."
    });
  }
});

app.post("/api/match/batch", async (req, res) => {
  try {
    const items = Array.isArray(req.body?.items) ? req.body.items : [];
    if (!items.length) {
      res.status(400).json({ ok: false, error: "Missing batch items." });
      return;
    }
    const config = normalizeConfig(req.body?.config || (await loadWorkspaceConfig()).config);
    const templateIds = Array.isArray(req.body?.templateIds) ? req.body.templateIds.map(String) : null;
    const matcher = new TemplateMatcher(config, { baseDir: ROOT });
    const results = items.map((item) => {
      const normalizedItem = {
        id: String(item?.id || ""),
        text: String(item?.text || "").trim(),
        expected_template_id: String(item?.expected_template_id || ""),
        expected_status: String(item?.expected_status || ""),
        expected_not_full_match: Boolean(item?.expected_not_full_match),
        answer: String(item?.answer || ""),
        metadata: item?.metadata && typeof item.metadata === "object" ? item.metadata : {}
      };
      const result = matcher.match(normalizedItem.text, { templateIds });
      return {
        item: normalizedItem,
        result,
        evaluation: evaluateBatchMatch(normalizedItem, result)
      };
    });
    res.json({
      ok: true,
      results,
      summary: summarizeBatchMatches(results)
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : "Batch match failed."
    });
  }
});

app.post("/api/llm/generate-template", async (req, res) => {
  try {
    const text = String(req.body?.text || "").trim();
    if (!text) {
      res.status(400).json({ ok: false, error: "Missing source text." });
      return;
    }
    const settings = req.body?.settings || {};
    const output = await generateTemplateFromSentence({
      text,
      currentConfig: normalizeConfig(req.body?.config || {}),
      apiKey: settings.apiKey || process.env.DASHSCOPE_API_KEY || "",
      baseUrl: settings.baseUrl || process.env.DASHSCOPE_BASE_URL || DEFAULT_LLM_SETTINGS.baseUrl,
      model: settings.model || process.env.DASHSCOPE_MODEL || DEFAULT_LLM_SETTINGS.model,
      insecureSSL: Boolean(settings.insecureSSL ?? DEFAULT_INSECURE_SSL)
    });
    res.json({ ok: true, ...output });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : "Unknown LLM generation error."
    });
  }
});

app.post("/api/llm/generate-template/batch", async (req, res) => {
  try {
    const items = Array.isArray(req.body?.items) ? req.body.items : [];
    if (!items.length) {
      res.status(400).json({ ok: false, error: "Missing batch items." });
      return;
    }
    const settings = req.body?.settings || {};
    const config = normalizeConfig(req.body?.config || {});
    const output = await batchGenerateTemplates(items, async (item) =>
      generateTemplateFromSentence({
        text: String(item?.text || "").trim(),
        answer: String(item?.answer || "").trim(),
        currentConfig: config,
        apiKey: settings.apiKey || process.env.DASHSCOPE_API_KEY || "",
        baseUrl: settings.baseUrl || process.env.DASHSCOPE_BASE_URL || DEFAULT_LLM_SETTINGS.baseUrl,
        model: settings.model || process.env.DASHSCOPE_MODEL || DEFAULT_LLM_SETTINGS.model,
        insecureSSL: Boolean(settings.insecureSSL ?? DEFAULT_INSECURE_SSL)
      })
    );
    res.json({ ok: true, ...output });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : "Unknown batch LLM generation error."
    });
  }
});

app.post("/api/optimize/run", async (req, res) => {
  try {
    const items = Array.isArray(req.body?.items) ? req.body.items : [];
    if (!items.length) {
      res.status(400).json({ ok: false, error: "Missing optimization items." });
      return;
    }
    const settings = req.body?.settings || {};
    const config = normalizeConfig(req.body?.config || (await loadWorkspaceConfig()).config);
    const targetMetric = String(req.body?.targetMetric || "pass_rate");
    const targetValue = Number(req.body?.targetValue ?? 0.9);
    const maxRounds = Math.max(1, Number(req.body?.maxRounds ?? 3));
    const maxCandidatesPerRound = Math.max(1, Number(req.body?.maxCandidatesPerRound ?? 5));
    const optimizationStrategy = String(req.body?.optimizationStrategy || "balanced");

    const output = await optimizeConfigIteratively({
      items,
      config,
      baseDir: ROOT,
      targetMetric,
      targetValue,
      maxRounds,
      maxCandidatesPerRound,
      optimizationStrategy,
      generateCandidate: async ({
        item,
        currentConfig,
        currentResult,
        currentEvaluation,
        currentTemplate,
        generationMode,
        missingSlots
      }) =>
        generateTemplateFromSentence({
          text: String(item?.text || "").trim(),
          answer: String(item?.answer || "").trim(),
          currentMatch: {
            result: currentResult,
            evaluation: currentEvaluation
          },
          currentTemplate,
          missingSlots,
          optimizationMode: generationMode,
          currentConfig,
          apiKey: settings.apiKey || process.env.DASHSCOPE_API_KEY || "",
          baseUrl: settings.baseUrl || process.env.DASHSCOPE_BASE_URL || DEFAULT_LLM_SETTINGS.baseUrl,
          model: settings.model || process.env.DASHSCOPE_MODEL || DEFAULT_LLM_SETTINGS.model,
          insecureSSL: Boolean(settings.insecureSSL ?? DEFAULT_INSECURE_SSL)
        })
    });
    res.json({ ok: true, ...output });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error instanceof Error ? error.message : "Optimization run failed."
    });
  }
});

app.use((_req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, "index.html"));
});

export const server = app.listen(PORT, () => {
  const url = `http://127.0.0.1:${PORT}`;
  console.log(`Template Studio running at ${url}`);
  if (process.env.TEMPLATE_STUDIO_OPEN_BROWSER !== "0") {
    openBrowser(url);
  }
});

async function loadWorkspaceConfig() {
  try {
    const payload = JSON.parse(await fs.readFile(WORKSPACE_FILE, "utf8"));
    return {
      config: normalizeConfig(payload),
      source: "workspace"
    };
  } catch {
    const payload = JSON.parse(await fs.readFile(DEFAULT_CONFIG_FILE, "utf8"));
    return {
      config: normalizeConfig(payload),
      source: "default"
    };
  }
}

function openBrowser(url) {
  if (process.platform === "win32") {
    exec(`start "" "${url}"`);
    return;
  }
  if (process.platform === "darwin") {
    exec(`open "${url}"`);
    return;
  }
  exec(`xdg-open "${url}"`);
}
