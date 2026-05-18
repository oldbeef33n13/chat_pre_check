import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { TemplateMatcher } from "../lib/matcher.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..", "..");

async function loadConfig() {
  const payload = await fs.readFile(path.join(ROOT, "configs", "templates.json"), "utf8");
  return JSON.parse(payload);
}

test("node matcher matches a known topn template", async () => {
  const matcher = new TemplateMatcher(await loadConfig());
  const result = matcher.match("近24小时接口错误包告警Top10");

  assert.equal(result.template_id, "alarm.interface.error.topn");
  assert.equal(result.status, "matched");
  assert.equal(result.slots.topn, 10);
});

test("node matcher can isolate current template scope", async () => {
  const matcher = new TemplateMatcher(await loadConfig());
  const result = matcher.match("查询最近cpu大于80且内存大于70的设备列表", {
    templateIds: ["device.cpu.memory.over.list"]
  });

  assert.equal(result.template_id, "device.cpu.memory.over.list");
  assert.equal(result.status, "matched");
});

test("node matcher rejects blocked intents", async () => {
  const matcher = new TemplateMatcher(await loadConfig());
  const result = matcher.match("帮我分析最近cpu异常原因");

  assert.equal(result.template_id, -1);
  assert.equal(result.status, "unmatched");
});

test("node matcher applies query rewrite before matching", () => {
  const matcher = new TemplateMatcher({
    matcher: {
      match_threshold: 0.4,
      ambiguity_margin: 0.03,
      recall_top_k: 10,
      weights: {
        lexical: 0.6,
        sample: 0.1,
        vector: 0.1,
        fusion: 0.1,
        slot_fit: 0.05,
        constraint: 0.05,
        structure: 0
      }
    },
    query_rewrite: {
      enabled: true,
      max_passes: 1,
      rules: [
        {
          rule_id: "school.short",
          source: "北二小",
          target: "北京第二小学"
        }
      ]
    },
    templates: [
      {
        template_id: "school.alarm.count",
        query_mode: "metric_query",
        description: "查询北京第二小学告警数量",
        utterances: ["查询北京第二小学的告警数量"],
        required_slots: [],
        optional_slots: [],
        must_terms: [["北京第二小学"], ["告警"]],
        negative_terms: [],
        slot_constraints: {},
        slot_extractors: {},
        llm_slot_extraction: {
          enabled: false,
          slots: [],
          instructions: ""
        },
        metadata: {}
      }
    ]
  });

  const result = matcher.match("查询北二小的告警");

  assert.equal(result.template_id, "school.alarm.count");
  assert.equal(result.trace.rewrite_trace.changed, true);
  assert.equal(result.trace.rewrite_trace.rewritten_text, "查询北京第二小学的告警");
});

test("node matcher supports longest keyword, derived slots, defaults, and route decisions", () => {
  const matcher = new TemplateMatcher({
    matcher: {
      match_threshold: 0.2,
      ambiguity_margin: 0.03,
      recall_top_k: 10,
      weights: {
        lexical: 0.1,
        sample: 0,
        vector: 0,
        fusion: 0,
        slot_fit: 0.45,
        constraint: 0.2,
        structure: 0.25
      },
      template_route: {
        min_score: 0.2,
        min_structure_score: 0,
        top_k: 1
      }
    },
    slot_extractors: {
      resource_type: {
        extractors: [
          {
            type: "keyword_value",
            match_policy: "longest",
            cases: [
              { terms: ["端口"], value: "generic_port" },
              { terms: ["以太网端口"], value: "ethernet_port" },
              { terms: ["硬盘"], value: "disk" }
            ]
          }
        ]
      },
      health_status_label: {
        extractors: [
          {
            type: "keyword_value",
            cases: [{ terms: ["正常"], value: "normal" }]
          }
        ]
      },
      query_operator: {
        extractors: [
          {
            type: "keyword_value",
            cases: [
              { terms: ["列表"], value: "list" },
              { terms: ["信息"], value: "info" }
            ]
          }
        ]
      }
    },
    templates: [
      {
        template_id: "resource.health.list",
        query_mode: "metric_query",
        description: "查询资源健康状态列表",
        utterances: ["查询硬盘健康状态为正常的列表"],
        required_slots: ["resource_type", "health_status_label", "healthStatus", "query_operator"],
        optional_slots: ["page_no", "page_size"],
        must_terms: [["健康状态"], ["列表"]],
        negative_terms: [],
        slot_constraints: { query_operator: ["list"] },
        slot_defaults: { page_no: 1, page_size: 100 },
        derived_slots: [
          {
            slot_name: "healthStatus",
            source_slots: ["resource_type", "health_status_label"],
            mapping: [{ when: { resource_type: "disk", health_status_label: "normal" }, value: 2 }]
          }
        ],
        slot_validations: {},
        slot_extractors: {},
        llm_slot_extraction: { enabled: false, slots: [], instructions: "" },
        metadata: {}
      }
    ]
  });

  const result = matcher.match("查询硬盘健康状态为正常的列表");
  const route = matcher.routeForSql("查询硬盘健康状态为正常的列表");

  assert.equal(matcher.extractSlots("resource.health.list", "查询以太网端口信息").resource_type, "ethernet_port");
  assert.equal(result.status, "matched");
  assert.equal(result.slots.healthStatus, 2);
  assert.equal(result.slots.page_size, 100);
  assert.equal(route.route_to, "template");
});

test("node matcher separates single and multi metric condition templates", () => {
  const config = {
    matcher: {
      match_threshold: 0.2,
      ambiguity_margin: 0.03,
      recall_top_k: 10,
      weights: {
        lexical: 0.1,
        sample: 0,
        vector: 0,
        fusion: 0,
        slot_fit: 0.55,
        constraint: 0.15,
        structure: 0.2
      }
    },
    slot_extractors: {
      metric_conditions: {
        extractors: [
          {
            type: "metric_conditions",
            metrics: [
              { terms: ["内存利用率", "内存"], value: "memory_usage" },
              { terms: ["cpu利用率", "cpu"], value: "cpu_usage" }
            ]
          }
        ]
      },
      query_operator: {
        extractors: [
          {
            type: "keyword_value",
            cases: [{ terms: ["列表"], value: "list" }]
          }
        ]
      }
    },
    templates: [
      {
        template_id: "storage.single_metric.list",
        query_mode: "metric_query",
        description: "查询单一指标超过阈值的分布式存储列表",
        utterances: ["查询内存利用率大于50的分布式存储列表"],
        required_slots: ["query_operator", "metric_conditions"],
        optional_slots: [],
        must_terms: [["分布式存储"], ["列表"]],
        negative_terms: [],
        slot_constraints: { query_operator: ["list"] },
        slot_validations: { metric_conditions: { exact_items: 1 } },
        slot_extractors: {},
        llm_slot_extraction: { enabled: false, slots: [], instructions: "" },
        metadata: {}
      },
      {
        template_id: "storage.multi_metric.list",
        query_mode: "metric_query",
        description: "查询多个指标超过阈值的分布式存储列表",
        utterances: ["查询内存利用率大于50 cpu利用率大于90的分布式存储列表"],
        required_slots: ["query_operator", "metric_conditions"],
        optional_slots: [],
        must_terms: [["分布式存储"], ["列表"]],
        negative_terms: [],
        slot_constraints: { query_operator: ["list"] },
        slot_validations: { metric_conditions: { min_items: 2 } },
        slot_extractors: {},
        llm_slot_extraction: { enabled: false, slots: [], instructions: "" },
        metadata: {}
      }
    ]
  };

  const matcher = new TemplateMatcher(config);

  assert.equal(
    matcher.match("查询今天内存利用率大于50%的分布式存储列表").template_id,
    "storage.single_metric.list"
  );
  assert.equal(
    matcher.match("查询今天内存利用率大于50% cpu利用率大于90%的分布式存储列表").template_id,
    "storage.multi_metric.list"
  );
});
