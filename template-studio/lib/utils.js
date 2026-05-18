export const DEFAULT_SCORE_WEIGHTS = {
  lexical: 0.2,
  sample: 0.1,
  vector: 0.15,
  fusion: 0.1,
  slot_fit: 0.15,
  constraint: 0.1,
  structure: 0.2
};

export const DEFAULT_LEXICAL_FIELD_WEIGHTS = {
  description: 0.6,
  utterances: 1.0,
  must_terms: 1.6
};

export const DEFAULT_MATCHER = {
  match_threshold: 0.58,
  ambiguity_margin: 0.03,
  recall_top_k: 30,
  weights: DEFAULT_SCORE_WEIGHTS,
  fusion_rrf_k: 60,
  lexical_field_weights: DEFAULT_LEXICAL_FIELD_WEIGHTS,
  blocked_terms: [
    "分析",
    "原因",
    "根因",
    "报告",
    "预测",
    "总结",
    "建议",
    "解决方案",
    "为什么",
    "走势",
    "优化"
  ],
  llm_fallback: {
    enabled: false,
    max_candidates: 3,
    score_margin: 0.08,
    max_missing_slots: 2
  },
  llm_slot_fallback: {
    enabled: false,
    max_missing_slots: 2,
    min_score: 0.58,
    allow_on_matched: false
  },
  template_route: {
    min_score: 0.58,
    min_structure_score: 0,
    top_k: 1
  },
  vector: {
    provider: "local_tfidf",
    dimension: 512
  }
};

export const DEFAULT_QUERY_REWRITE = {
  enabled: false,
  max_passes: 1,
  dictionary_path: "",
  reload_on_change: true,
  rules: []
};

export function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function normalizeText(text = "") {
  return String(text)
    .normalize("NFKC")
    .trim()
    .toLowerCase()
    .replace(/[，。？！!?,:：;；、()\[\]{}<>《》"'`]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function ensureArray(value) {
  return Array.isArray(value) ? value : [];
}

export function ensureObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

export function clampScore(score) {
  if (Number.isNaN(score)) {
    return 0;
  }
  return Math.max(0, Math.min(1, score));
}

export function slugifyTemplateId(input, fallback = "template.generated") {
  const slug = String(input || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ".")
    .replace(/\.+/g, ".")
    .replace(/^\.|\.$/g, "");
  return slug || fallback;
}

export function normalizeConfig(payload) {
  const raw = ensureObject(payload);
  const matcher = ensureObject(raw.matcher);
  const vector = ensureObject(matcher.vector);
  const normalizedMatcher = {
    match_threshold: Number(matcher.match_threshold ?? DEFAULT_MATCHER.match_threshold),
    ambiguity_margin: Number(matcher.ambiguity_margin ?? DEFAULT_MATCHER.ambiguity_margin),
    recall_top_k: Number(matcher.recall_top_k ?? DEFAULT_MATCHER.recall_top_k),
    weights: {
      ...DEFAULT_SCORE_WEIGHTS,
      ...coerceNumberMap(matcher.weights)
    },
    fusion_rrf_k: Number(matcher.fusion_rrf_k ?? DEFAULT_MATCHER.fusion_rrf_k),
    lexical_field_weights: {
      ...DEFAULT_LEXICAL_FIELD_WEIGHTS,
      ...coerceNumberMap(matcher.lexical_field_weights)
    },
    blocked_terms: ensureArray(matcher.blocked_terms).map((item) => String(item)),
    llm_fallback: {
      ...DEFAULT_MATCHER.llm_fallback,
      ...ensureObject(matcher.llm_fallback)
    },
    llm_slot_fallback: {
      ...DEFAULT_MATCHER.llm_slot_fallback,
      ...ensureObject(matcher.llm_slot_fallback)
    },
    template_route: {
      ...DEFAULT_MATCHER.template_route,
      ...ensureObject(matcher.template_route)
    },
    vector: {
      provider: String(vector.provider ?? DEFAULT_MATCHER.vector.provider),
      dimension: Number(vector.dimension ?? DEFAULT_MATCHER.vector.dimension)
    }
  };

  const slotExtractors = normalizeSlotExtractorMap(raw.slot_extractors);
  const queryRewrite = normalizeQueryRewrite(raw.query_rewrite);
  const templates = ensureArray(raw.templates)
    .filter((item) => item && typeof item === "object")
    .map((item, index) => normalizeTemplate(item, index));

  return {
    matcher: normalizedMatcher,
    query_rewrite: queryRewrite,
    slot_extractors: slotExtractors,
    templates
  };
}

export function normalizeTemplate(template, index = 0) {
  const item = ensureObject(template);
  const llm = ensureObject(item.llm_slot_extraction);
  return {
    template_id: String(item.template_id || `template.generated.${index + 1}`),
    query_mode: String(item.query_mode || "metric_query"),
    description: String(item.description || ""),
    utterances: ensureArray(item.utterances).map((value) => String(value)).filter(Boolean),
    required_slots: ensureArray(item.required_slots).map((value) => String(value)).filter(Boolean),
    optional_slots: ensureArray(item.optional_slots).map((value) => String(value)).filter(Boolean),
    must_terms: ensureArray(item.must_terms)
      .map((group) => ensureArray(group).map((value) => String(value)).filter(Boolean))
      .filter((group) => group.length > 0),
    negative_terms: ensureArray(item.negative_terms).map((value) => String(value)).filter(Boolean),
    slot_constraints: normalizeConstraintMap(item.slot_constraints),
    slot_extractors: normalizeSlotExtractorMap(item.slot_extractors),
    slot_defaults: ensureObject(item.slot_defaults),
    derived_slots: normalizeDerivedSlots(item.derived_slots),
    slot_validations: normalizeValidationMap(item.slot_validations),
    llm_slot_extraction: {
      enabled: Boolean(llm.enabled),
      slots: ensureArray(llm.slots).map((value) => String(value)).filter(Boolean),
      instructions: String(llm.instructions || ""),
      allow_on_matched: Boolean(llm.allow_on_matched)
    },
    metadata: ensureObject(item.metadata)
  };
}

export function createBlankTemplate() {
  return {
    template_id: `template.generated.${Date.now()}`,
    query_mode: "metric_query",
    description: "",
    utterances: [],
    required_slots: ["query_operator"],
    optional_slots: [],
    must_terms: [],
    negative_terms: ["原因", "根因", "报告", "总结", "预测"],
    slot_constraints: {},
    slot_extractors: {},
    slot_defaults: {},
    derived_slots: [],
    slot_validations: {},
    llm_slot_extraction: {
      enabled: false,
      slots: [],
      instructions: "",
      allow_on_matched: false
    },
    metadata: {}
  };
}

function normalizeSlotExtractorMap(value) {
  const input = ensureObject(value);
  return Object.fromEntries(
    Object.entries(input).map(([slotName, definition]) => {
      const slotObject = ensureObject(definition);
      return [
        String(slotName),
        {
          slot_name: String(slotObject.slot_name || slotName),
          extractors: ensureArray(slotObject.extractors)
            .filter((item) => item && typeof item === "object")
            .map((item) => normalizeExtractor(item))
        }
      ];
    })
  );
}

function normalizeQueryRewrite(value) {
  const input = ensureObject(value);
  return {
    ...DEFAULT_QUERY_REWRITE,
    enabled: Boolean(input.enabled),
    max_passes: Math.max(1, Number(input.max_passes ?? DEFAULT_QUERY_REWRITE.max_passes)),
    dictionary_path: String(input.dictionary_path || ""),
    reload_on_change: Boolean(input.reload_on_change ?? DEFAULT_QUERY_REWRITE.reload_on_change),
    rules: ensureArray(input.rules)
      .filter((item) => item && typeof item === "object")
      .map((item, index) => ({
        rule_id: String(item.rule_id || `rewrite_rule_${index + 1}`),
        source: String(item.source || ""),
        target: String(item.target || ""),
        match_mode: String(item.match_mode || "substring")
      }))
      .filter((item) => item.source.trim() && item.target.trim())
  };
}

function normalizeExtractor(extractor) {
  const item = ensureObject(extractor);
  const type = String(item.type || "").toLowerCase();
  if (type === "regex") {
    return {
      type: "regex",
      patterns: ensureArray(item.patterns)
        .filter((pattern) => pattern && typeof pattern === "object")
        .map((pattern) => ({
          pattern: String(pattern.pattern || ""),
          group: Number(pattern.group ?? 1),
          value_type: String(pattern.value_type || "string"),
          min: pattern.min ?? null,
          max: pattern.max ?? null,
          value: pattern.value ?? null
        }))
    };
  }
  if (type === "time_range") {
    return {
      type: "time_range",
      include_absolute: Boolean(item.include_absolute ?? true)
    };
  }
  if (type === "metric_conditions") {
    return {
      type: "metric_conditions",
      metrics: ensureArray(item.metrics)
        .filter((entry) => entry && typeof entry === "object")
        .map((entry) => ({
          terms: ensureArray(entry.terms).map((term) => String(term)).filter(Boolean),
          value: entry.value ?? entry.metric ?? null
        })),
      operators: ensureArray(item.operators)
        .filter((entry) => entry && typeof entry === "object")
        .map((entry) => ({
          terms: ensureArray(entry.terms).map((term) => String(term)).filter(Boolean),
          value: String(entry.value || "")
        })),
      max_gap_chars: Number(item.max_gap_chars ?? 24),
      value_type: String(item.value_type || "float"),
      value_pattern: String(item.value_pattern || "(-?\\d+(?:\\.\\d+)?)\\s*%?")
    };
  }
  return {
    type: "keyword_value",
    match_policy: String(item.match_policy || "first"),
    cases: ensureArray(item.cases)
      .filter((entry) => entry && typeof entry === "object")
      .map((entry) => ({
        terms: ensureArray(entry.terms).map((term) => String(term)).filter(Boolean),
        value: entry.value ?? null
      }))
  };
}

function normalizeDerivedSlots(value) {
  if (Array.isArray(value)) {
    return value.filter((entry) => entry && typeof entry === "object").map((entry) => normalizeDerivedSlot(entry));
  }
  const input = ensureObject(value);
  return Object.entries(input).map(([slotName, definition]) =>
    normalizeDerivedSlot({
      ...ensureObject(definition),
      slot_name: ensureObject(definition).slot_name || slotName
    })
  );
}

function normalizeDerivedSlot(value) {
  const input = ensureObject(value);
  return {
    slot_name: String(input.slot_name || ""),
    source_slots: ensureArray(input.source_slots).map((slot) => String(slot)).filter(Boolean),
    mapping: Array.isArray(input.mapping) || (input.mapping && typeof input.mapping === "object") ? input.mapping : [],
    default: input.default ?? null,
    key_separator: String(input.key_separator || "."),
    overwrite: Boolean(input.overwrite),
    require_all_sources: Boolean(input.require_all_sources ?? true)
  };
}

function normalizeValidationMap(value) {
  const input = ensureObject(value);
  return Object.fromEntries(
    Object.entries(input)
      .filter(([, rule]) => rule && typeof rule === "object" && !Array.isArray(rule))
      .map(([slotName, rule]) => [String(slotName), { ...rule }])
  );
}

function normalizeConstraintMap(value) {
  const input = ensureObject(value);
  return Object.fromEntries(
    Object.entries(input).map(([slotName, allowed]) => [
      String(slotName),
      ensureArray(allowed).length > 0 ? ensureArray(allowed) : [allowed]
    ])
  );
}

function coerceNumberMap(value) {
  const input = ensureObject(value);
  return Object.fromEntries(
    Object.entries(input).map(([key, item]) => [String(key), Number(item)])
  );
}
