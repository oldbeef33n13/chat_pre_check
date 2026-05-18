import { buildSlotRegistry } from "./extractors.js";
import {
  adaptiveScoreWeights,
  BM25FieldIndex,
  buildUtteranceTermSets,
  constraintScore,
  hasNegativeTerm,
  mixedTerms,
  missingRequiredSlots,
  normalizeCandidateScores,
  reciprocalRankFusion,
  sampleSimilarityFromTerms,
  slotFitScore,
  structuralAlignmentScore,
  weightedScore
} from "./scoring.js";
import { QueryRewriteStateMachine } from "./rewrite.js";
import { InMemoryVectorIndex, LocalHashVectorProvider, LocalTfidfVectorProvider } from "./vector.js";
import { deepClone, normalizeConfig, normalizeText } from "./utils.js";

export class TemplateMatcher {
  constructor(rawConfig, options = {}) {
    this.config = normalizeConfig(rawConfig);
    this.settings = this.config.matcher;
    this.queryRewriter = new QueryRewriteStateMachine(this.config.query_rewrite, {
      baseDir: options.baseDir
    });
    this.templates = Object.fromEntries(this.config.templates.map((template) => [template.template_id, template]));
    this.slotRegistry = buildSlotRegistry(this.config.slot_extractors);
    this.templateSlotRegistries = Object.fromEntries(
      this.config.templates.map((template) => [template.template_id, buildSlotRegistry(this.buildTemplateSlotDefinitions(template))])
    );
    this.templateDocuments = Object.fromEntries(
      this.config.templates.map((template) => [template.template_id, this.buildTemplateDocument(template)])
    );
    this.templateLexicalFields = Object.fromEntries(
      this.config.templates.map((template) => [template.template_id, this.buildTemplateFields(template)])
    );
    this.templateSampleTerms = Object.fromEntries(
      this.config.templates.map((template) => [template.template_id, buildUtteranceTermSets(template.utterances)])
    );
    this.lexicalIndex = new BM25FieldIndex(this.templateLexicalFields, this.settings.lexical_field_weights);
    this.vectorBackend = new InMemoryVectorIndex(buildVectorProvider(this.settings.vector));
    this.vectorBackend.build(this.templateDocuments);
  }

  match(inputText, options = {}) {
    const originalNormText = normalizeText(inputText);
    const rewriteResult = this.queryRewriter.rewriteNormalized(originalNormText);
    const normText = rewriteResult.rewritten_text;
    const sharedSlots = this.slotRegistry.extract(normText);
    const blockedTerm = this.matchBlockedTerm(normText);
    if (blockedTerm) {
      return {
        template_id: -1,
        status: "unmatched",
        score: 0,
        query_mode: null,
        slots: sharedSlots,
        missing_slots: [],
        metadata: {},
        trace: {
          norm_text: normText,
          original_norm_text: originalNormText,
          rewrite_trace: rewriteResult,
          blocked_term: blockedTerm,
          reason: "blocked_intent"
        }
      };
    }

    const ranked = this.rankTemplates(normText, options.templateIds || null);
    const top = ranked[0];
    const second = ranked[1];
    if (!top || top.score < this.settings.match_threshold || this.isAmbiguous(top, second)) {
      return {
        template_id: -1,
        status: "unmatched",
        score: top?.score || 0,
        query_mode: null,
        slots: sharedSlots,
        missing_slots: [],
        metadata: {},
        trace: {
          norm_text: normText,
          original_norm_text: originalNormText,
          rewrite_trace: rewriteResult,
          top_candidates: ranked.slice(0, 5),
          threshold: this.settings.match_threshold,
          ambiguity_margin: this.settings.ambiguity_margin
        }
      };
    }

    const template = this.templates[top.template_id];
    const finalized = this.rebuildCandidate(top, normText, { includeDefaults: true });
    const missingSlots = missingRequiredSlots(template, finalized.slots);
    const status = missingSlots.length ? "partial" : "matched";
    return {
      template_id: template.template_id,
      status,
      score: finalized.score,
      query_mode: template.query_mode,
      slots: finalized.slots,
      missing_slots: missingSlots,
      metadata: template.metadata,
      trace: {
        norm_text: normText,
        original_norm_text: originalNormText,
        rewrite_trace: rewriteResult,
        selected_template: finalized,
        top_candidates: [finalized, ...ranked.filter((item) => item.template_id !== finalized.template_id)].slice(0, 5)
      }
    };
  }

  routeForSql(inputText, options = {}) {
    const result = this.match(inputText, options);
    if (result.status === "matched" && result.score >= Number(this.settings.template_route?.min_score ?? this.settings.match_threshold)) {
      return {
        route_to: "template",
        reason: "template_satisfied",
        result
      };
    }
    return {
      route_to: "nl2sql",
      reason: result.trace?.reason || "template_requirements_not_satisfied",
      result: {
        ...result,
        template_id: -1,
        status: "unmatched",
        query_mode: null,
        missing_slots: []
      }
    };
  }

  rankTemplates(normText, templateIds = null) {
    const queryTermSet = new Set(mixedTerms(normText));
    const scopeIds = templateIds ? new Set(templateIds) : null;

    const lexicalRecall = filterRanking(this.lexicalIndex.search(normText, this.settings.recall_top_k), scopeIds);
    const vectorRecall = filterRanking(this.vectorBackend.search(normText, this.settings.recall_top_k), scopeIds);
    const sampleRecall = filterRanking(this.sampleRecall(queryTermSet), scopeIds);

    const lexicalScores = normalizeCandidateScores(lexicalRecall);
    const vectorScores = normalizeCandidateScores(vectorRecall);
    const sampleScores = normalizeCandidateScores(sampleRecall);
    const fusionScores = reciprocalRankFusion([lexicalRecall, vectorRecall, sampleRecall], this.settings.fusion_rrf_k);

    const candidateIds = new Set([
      ...Object.keys(lexicalScores),
      ...Object.keys(vectorScores),
      ...Object.keys(sampleScores),
      ...Object.keys(fusionScores)
    ]);
    if (scopeIds) {
      for (const templateId of scopeIds) {
        candidateIds.add(templateId);
      }
    }
    const ranked = [];
    for (const templateId of candidateIds) {
      const template = this.templates[templateId];
      if (!template) {
        continue;
      }
      const slots = this.extractSlots(templateId, normText);
      const weights = adaptiveScoreWeights(this.settings.weights, slots);
      ranked.push(
        this.buildCandidate({
          templateId,
          normText,
          lexicalScore: lexicalScores[templateId] || 0,
          sampleScore: sampleScores[templateId] || 0,
          vectorScore: vectorScores[templateId] || 0,
          fusionScore: fusionScores[templateId] || 0,
          slots,
          weights
        })
      );
    }
    ranked.sort((left, right) => right.score - left.score);
    return ranked;
  }

  sampleRecall(queryTermSet) {
    return Object.entries(this.templateSampleTerms)
      .map(([templateId, utteranceTermSets]) => [
        templateId,
        sampleSimilarityFromTerms(queryTermSet, utteranceTermSets)
      ])
      .sort((left, right) => right[1] - left[1])
      .slice(0, this.settings.recall_top_k);
  }

  extractSlots(templateId, text) {
    return this.templateSlotRegistries[templateId].extract(text);
  }

  buildTemplateSlotDefinitions(template) {
    const relevantSlots = new Set([
      ...(template.required_slots || []),
      ...(template.optional_slots || []),
      ...Object.keys(template.slot_constraints || {}),
      ...Object.keys(template.slot_extractors || {}),
      ...Object.keys(template.slot_defaults || {}),
      ...Object.keys(template.slot_validations || {})
    ]);
    for (const rule of template.derived_slots || []) {
      relevantSlots.add(rule.slot_name);
      for (const sourceSlot of rule.source_slots || []) {
        relevantSlots.add(sourceSlot);
      }
    }
    const merged = {};
    for (const slotName of relevantSlots) {
      const localDefinition = template.slot_extractors?.[slotName];
      if (localDefinition) {
        merged[slotName] = deepClone(localDefinition);
        continue;
      }
      const sharedDefinition = this.config.slot_extractors?.[slotName];
      if (sharedDefinition) {
        merged[slotName] = deepClone(sharedDefinition);
      }
    }
    return merged;
  }

  buildTemplateDocument(template) {
    const parts = [template.description, ...(template.utterances || [])];
    parts.push(...(template.must_terms || []).map((group) => group.join(" ")));
    return normalizeText(parts.filter(Boolean).join(" "));
  }

  buildTemplateFields(template) {
    const mustTermsText = (template.must_terms || []).map((group) => group.join(" ")).join(" ");
    return {
      description: mixedTerms(normalizeText(template.description)),
      utterances: mixedTerms(normalizeText((template.utterances || []).join(" "))),
      must_terms: mixedTerms(normalizeText(mustTermsText))
    };
  }

  buildCandidate({ templateId, normText, lexicalScore, sampleScore, vectorScore, fusionScore, slots, weights }) {
    const template = this.templates[templateId];
    const slotResolution = this.resolveSlots(template, slots, { includeDefaults: false });
    slots = slotResolution.slots;
    const slotScore = slotFitScore(template, slots);
    const constraint = constraintScore(template, normText, slots);
    const structure = structuralAlignmentScore(template, slots);
    const total = hasNegativeTerm(template, normText)
      ? 0
      : weightedScore(
          {
            lexical: lexicalScore,
            sample: sampleScore,
            vector: vectorScore,
            fusion: fusionScore,
            slot_fit: slotScore,
            constraint,
            structure
          },
          weights
        );

    return {
      template_id: template.template_id,
      query_mode: template.query_mode,
      score: total,
      lexical_score: lexicalScore,
      sample_score: sampleScore,
      vector_score: vectorScore,
      fusion_score: fusionScore,
      slot_fit_score: slotScore,
      constraint_score: constraint,
      structure_score: structure,
      slots,
      missing_slots: missingRequiredSlots(template, slots),
      metadata: template.metadata,
      trace: {
        slot_resolution: slotResolution.trace
      }
    };
  }

  rebuildCandidate(candidate, normText, { includeDefaults = false } = {}) {
    const weights = adaptiveScoreWeights(this.settings.weights, candidate.slots);
    const rebuilt = this.buildCandidate({
      templateId: candidate.template_id,
      normText,
      lexicalScore: candidate.lexical_score,
      sampleScore: candidate.sample_score,
      vectorScore: candidate.vector_score,
      fusionScore: candidate.fusion_score,
      slots: candidate.slots,
      weights
    });
    if (!includeDefaults) {
      return rebuilt;
    }
    const template = this.templates[candidate.template_id];
    const slotResolution = this.resolveSlots(template, rebuilt.slots, { includeDefaults: true });
    return this.buildCandidate({
      templateId: candidate.template_id,
      normText,
      lexicalScore: candidate.lexical_score,
      sampleScore: candidate.sample_score,
      vectorScore: candidate.vector_score,
      fusionScore: candidate.fusion_score,
      slots: slotResolution.slots,
      weights
    });
  }

  resolveSlots(template, slots, { includeDefaults = false } = {}) {
    const resolved = deepClone(slots || {});
    const trace = {
      defaults_applied: {},
      derived_applied: {}
    };
    if (includeDefaults) {
      for (const [slotName, value] of Object.entries(template.slot_defaults || {})) {
        if (slotValuePresent(resolved[slotName])) {
          continue;
        }
        resolved[slotName] = deepClone(value);
        trace.defaults_applied[slotName] = deepClone(value);
      }
    }
    for (let pass = 0; pass < 3; pass += 1) {
      let changed = false;
      for (const rule of template.derived_slots || []) {
        if (!rule.overwrite && slotValuePresent(resolved[rule.slot_name])) {
          continue;
        }
        const value = deriveSlotValue(rule, resolved);
        if (!slotValuePresent(value)) {
          continue;
        }
        resolved[rule.slot_name] = deepClone(value);
        trace.derived_applied[rule.slot_name] = deepClone(value);
        changed = true;
      }
      if (!changed) {
        break;
      }
    }
    return { slots: resolved, trace };
  }

  isAmbiguous(top, second) {
    if (!top || !second) {
      return false;
    }
    if (second.score < this.settings.match_threshold) {
      return false;
    }
    return top.score - second.score < this.settings.ambiguity_margin;
  }

  matchBlockedTerm(normText) {
    for (const term of this.settings.blocked_terms || []) {
      if (term && normText.includes(String(term).toLowerCase())) {
        return term;
      }
    }
    return null;
  }
}

function buildVectorProvider(vectorSettings = {}) {
  const provider = String(vectorSettings.provider || "local_tfidf");
  const dimension = Number(vectorSettings.dimension || 512);
  if (provider === "hashing" || provider === "local_hash") {
    return new LocalHashVectorProvider(dimension);
  }
  return new LocalTfidfVectorProvider(dimension);
}

function filterRanking(ranking, scopeIds) {
  if (!scopeIds) {
    return ranking;
  }
  return ranking.filter(([templateId]) => scopeIds.has(templateId));
}

function deriveSlotValue(rule, slots) {
  const sourceValues = [];
  for (const slotName of rule.source_slots || []) {
    const value = slots[slotName];
    if (!slotValuePresent(value)) {
      if (rule.require_all_sources !== false) {
        return null;
      }
      sourceValues.push("");
      continue;
    }
    sourceValues.push(value);
  }
  const mapping = rule.mapping;
  if (mapping && !Array.isArray(mapping) && typeof mapping === "object") {
    const key = sourceValues.map(slotValueKey).join(rule.key_separator || ".");
    return mapping[key] ?? rule.default ?? null;
  }
  if (Array.isArray(mapping)) {
    for (const item of mapping) {
      const conditions = Array.isArray(item.source_values)
        ? Object.fromEntries((rule.source_slots || []).map((slotName, index) => [slotName, item.source_values[index]]))
        : item.when;
      if (!conditions || typeof conditions !== "object") {
        continue;
      }
      if (Object.entries(conditions).every(([slotName, expected]) => slotValueMatches(slots[slotName], expected))) {
        return item.value ?? null;
      }
    }
  }
  return rule.default ?? null;
}

function slotValueMatches(actual, expected) {
  if (!slotValuePresent(actual)) {
    return false;
  }
  if (Array.isArray(expected)) {
    return expected.some((item) => slotValueMatches(actual, item));
  }
  const actualCandidates = slotValueCandidates(actual);
  const expectedCandidates = slotValueCandidates(expected);
  return [...actualCandidates].some((item) => expectedCandidates.has(item));
}

function slotValueCandidates(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const candidates = new Set();
    for (const key of ["id", "value", "code", "preset"]) {
      if (value[key] !== undefined) {
        candidates.add(String(value[key]).toLowerCase());
      }
    }
    if (!candidates.size) {
      candidates.add(JSON.stringify(value).toLowerCase());
    }
    return candidates;
  }
  return new Set([String(value).toLowerCase()]);
}

function slotValueKey(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const key of ["id", "value", "code", "preset"]) {
      if (value[key] !== undefined) {
        return String(value[key]);
      }
    }
  }
  return String(value);
}

function slotValuePresent(value) {
  if (value === null || value === undefined || value === "") {
    return false;
  }
  if (Array.isArray(value) && !value.length) {
    return false;
  }
  if (value && typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length) {
    return false;
  }
  return true;
}
