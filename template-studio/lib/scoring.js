import { clampScore } from "./utils.js";

const TOKEN_RE = /[a-z0-9_.-]+|[\u4e00-\u9fff]+/g;
const LOW_SIGNAL_SLOTS = new Set(["time_range", "region_id"]);
const FILTER_SLOTS = new Set(["topn", "severity", "device_id", "protocol", "query_operator", "metric_conditions"]);

export function mixedTerms(text) {
  const chunks = String(text || "").toLowerCase().match(TOKEN_RE) || [];
  const terms = [];
  for (const chunk of chunks) {
    if (!chunk) {
      continue;
    }
    terms.push(chunk);
    if (containsCjk(chunk)) {
      terms.push(...charNgrams(chunk, 2));
      if (chunk.length >= 3) {
        terms.push(...charNgrams(chunk, 3));
      }
    } else if (chunk.length >= 4) {
      terms.push(...charNgrams(chunk, 3));
    }
  }
  return terms.length ? terms : [String(text || "").toLowerCase()];
}

export function buildUtteranceTermSets(utterances = []) {
  return utterances.filter(Boolean).map((utterance) => new Set(mixedTerms(utterance)));
}

export function sampleSimilarityFromTerms(queryTerms, utteranceTermSets) {
  if (!queryTerms || !queryTerms.size) {
    return 0;
  }
  let best = 0;
  for (const utteranceTerms of utteranceTermSets) {
    const overlap = intersectionSize(queryTerms, utteranceTerms);
    const total = queryTerms.size + utteranceTerms.size;
    if (!total) {
      continue;
    }
    best = Math.max(best, (2 * overlap) / total);
  }
  return best;
}

export class BM25FieldIndex {
  constructor(documents, fieldWeights, k1 = 1.5, b = 0.75) {
    this.documents = documents;
    this.fieldWeights = fieldWeights;
    this.k1 = k1;
    this.b = b;
    this.totalDocs = Math.max(1, Object.keys(documents).length);
    this.docTermFreqs = {};
    this.docLengths = {};
    this.avgFieldLengths = {};
    this.fieldDocFreqs = {};

    for (const [docId, fields] of Object.entries(documents)) {
      this.docTermFreqs[docId] = {};
      this.docLengths[docId] = {};
      for (const fieldName of Object.keys(fieldWeights)) {
        const terms = fields[fieldName] || [];
        const termFreq = countTerms(terms);
        this.docTermFreqs[docId][fieldName] = termFreq;
        this.docLengths[docId][fieldName] = sumObject(termFreq);
      }
    }

    for (const fieldName of Object.keys(fieldWeights)) {
      let totalLength = 0;
      const docFreqs = {};
      for (const docId of Object.keys(documents)) {
        totalLength += this.docLengths[docId][fieldName] || 0;
        for (const term of Object.keys(this.docTermFreqs[docId][fieldName] || {})) {
          docFreqs[term] = (docFreqs[term] || 0) + 1;
        }
      }
      this.avgFieldLengths[fieldName] = totalLength / this.totalDocs;
      this.fieldDocFreqs[fieldName] = docFreqs;
    }
  }

  search(queryText, topK) {
    const queryTerms = mixedTerms(queryText);
    return Object.keys(this.documents)
      .map((docId) => [docId, this.scoreDoc(queryTerms, docId)])
      .sort((left, right) => right[1] - left[1])
      .slice(0, topK);
  }

  scoreDoc(queryTerms, docId) {
    let score = 0;
    for (const [fieldName, fieldWeight] of Object.entries(this.fieldWeights)) {
      score += fieldWeight * this.scoreField(queryTerms, docId, fieldName);
    }
    return score;
  }

  scoreField(queryTerms, docId, fieldName) {
    const termFreq = this.docTermFreqs[docId]?.[fieldName] || {};
    const docLength = this.docLengths[docId]?.[fieldName] || 0;
    const avgLength = this.avgFieldLengths[fieldName] || 0;
    const docFreqs = this.fieldDocFreqs[fieldName] || {};
    let score = 0;
    for (const term of queryTerms) {
      const freq = termFreq[term];
      if (!freq) {
        continue;
      }
      const docFreq = docFreqs[term] || 0;
      const idf = Math.log(1 + (this.totalDocs - docFreq + 0.5) / (docFreq + 0.5));
      const numerator = freq * (this.k1 + 1);
      const denominator = freq + this.k1 * (1 - this.b + this.b * docLength / Math.max(1, avgLength));
      score += idf * numerator / denominator;
    }
    return score;
  }
}

export function normalizeCandidateScores(items) {
  if (!items.length) {
    return {};
  }
  const topScore = Math.max(...items.map(([, score]) => score));
  if (topScore <= 0) {
    return Object.fromEntries(items.map(([docId]) => [docId, 0]));
  }
  return Object.fromEntries(items.map(([docId, score]) => [docId, clampScore(score / topScore)]));
}

export function reciprocalRankFusion(rankings, rrfK = 60) {
  const fused = {};
  for (const ranking of rankings) {
    ranking.forEach(([docId], index) => {
      fused[docId] = (fused[docId] || 0) + 1 / (rrfK + index + 1);
    });
  }
  const values = Object.values(fused);
  if (!values.length) {
    return {};
  }
  const topScore = Math.max(...values);
  return Object.fromEntries(
    Object.entries(fused).map(([docId, score]) => [docId, topScore > 0 ? clampScore(score / topScore) : 0])
  );
}

export function slotFitScore(template, slots) {
  const required = template.required_slots || [];
  const optional = template.optional_slots || [];
  if (!required.length && !optional.length) {
    const validationScores = slotValidationScores(template, slots);
    return validationScores.length ? average(validationScores) : 1;
  }
  const requiredHit = coverageScore(required, slots);
  const validationScores = slotValidationScores(template, slots);
  const coreHit = validationScores.length ? average([requiredHit, average(validationScores)]) : requiredHit;
  if (!optional.length) {
    return coreHit;
  }
  const optionalHit = coverageScore(optional, slots);
  return clampScore(coreHit * 0.8 + optionalHit * 0.2);
}

export function structuralAlignmentScore(template, slots) {
  const extractedSlots = new Set(
    Object.entries(slots)
      .filter(([, value]) => slotPresentValue(value))
      .map(([slotName]) => slotName)
  );
  if (!extractedSlots.size) {
    return 0;
  }
  const supportedSlots = new Set([
    ...(template.required_slots || []),
    ...(template.optional_slots || []),
    ...Object.keys(template.slot_constraints || {})
  ]);
  if (!supportedSlots.size) {
    return 0;
  }

  const totalWeight = sumSet(extractedSlots, slotSignalWeight);
  const unexpectedSlots = [...extractedSlots].filter((slotName) => !supportedSlots.has(slotName));
  const unexpectedWeight = unexpectedSlots.reduce((sum, slotName) => sum + slotSignalWeight(slotName), 0);
  const coverage = 1 - unexpectedWeight / Math.max(1, totalWeight);

  const extractedFilterSlots = [...extractedSlots].filter(isFilterSlot);
  if (!extractedFilterSlots.length) {
    return clampScore(coverage);
  }
  const supportedFilterSlots = new Set([...supportedSlots].filter(isFilterSlot));
  const matchedFilterWeight = extractedFilterSlots
    .filter((slotName) => supportedFilterSlots.has(slotName))
    .reduce((sum, slotName) => sum + slotSignalWeight(slotName), 0);
  const filterTotalWeight = extractedFilterSlots.reduce((sum, slotName) => sum + slotSignalWeight(slotName), 0);
  const filterScore = matchedFilterWeight / Math.max(1, filterTotalWeight);
  const requiredFilterSlots = (template.required_slots || []).filter(isFilterSlot);
  if (!requiredFilterSlots.length) {
    return clampScore(0.55 * coverage + 0.45 * filterScore);
  }
  const requiredFilterWeight = requiredFilterSlots.reduce((sum, slotName) => sum + slotSignalWeight(slotName), 0);
  const matchedRequiredWeight = requiredFilterSlots
    .filter((slotName) => extractedFilterSlots.includes(slotName))
    .reduce((sum, slotName) => sum + slotSignalWeight(slotName), 0);
  const completeness = matchedRequiredWeight / Math.max(1, requiredFilterWeight);
  return clampScore(0.45 * coverage + 0.3 * filterScore + 0.25 * completeness);
}

export function constraintScore(template, text, slots) {
  const lowered = String(text || "").toLowerCase();
  if (hasNegativeTerm(template, lowered)) {
    return 0;
  }
  const mustTerms = template.must_terms || [];
  let mustScore = 1;
  if (mustTerms.length) {
    let matchedGroups = 0;
    for (const group of mustTerms) {
      if (group.some((term) => lowered.includes(String(term).toLowerCase()))) {
        matchedGroups += 1;
      }
    }
    mustScore = matchedGroups / mustTerms.length;
  }
  const slotConstraintScores = [];
  for (const [slotName, allowedValues] of Object.entries(template.slot_constraints || {})) {
    const extracted = slots[slotName];
    if (extracted === null || extracted === undefined || extracted === "") {
      if ((template.required_slots || []).includes(slotName)) {
        slotConstraintScores.push(0);
      }
      continue;
    }
    slotConstraintScores.push(matchesAllowedValues(extracted, allowedValues) ? 1 : 0);
  }
  if (!slotConstraintScores.length) {
    return clampScore(mustScore);
  }
  return clampScore(0.6 * mustScore + 0.4 * average(slotConstraintScores));
}

export function hasNegativeTerm(template, text) {
  return (template.negative_terms || []).some((term) => String(text).includes(String(term).toLowerCase()));
}

export function weightedScore(parts, weights) {
  let total = 0;
  for (const [name, weight] of Object.entries(weights || {})) {
    total += clampScore(parts[name] || 0) * weight;
  }
  return clampScore(total);
}

export function adaptiveScoreWeights(baseWeights, slots) {
  const weights = { ...baseWeights };
  const filterSlotCount = Object.entries(slots).filter(
    ([slotName, value]) => slotPresentValue(value) && isFilterSlot(slotName)
  ).length;
  const complexity = Math.min(1, filterSlotCount / 3);
  if (complexity <= 0) {
    return normalizeWeights(weights);
  }
  for (const key of ["lexical", "sample", "vector"]) {
    if (weights[key] !== undefined) {
      weights[key] *= 1 - (0.15 + (key === "vector" ? 0.05 : 0)) * complexity;
    }
  }
  if (weights.slot_fit !== undefined) {
    weights.slot_fit *= 1 + 0.2 * complexity;
  }
  if (weights.structure !== undefined) {
    weights.structure *= 1 + 0.45 * complexity;
  }
  if (weights.constraint !== undefined) {
    weights.constraint *= 1 + 0.1 * complexity;
  }
  if (weights.fusion !== undefined) {
    weights.fusion *= 1 + 0.1 * complexity;
  }
  return normalizeWeights(weights);
}

export function missingRequiredSlots(template, slots) {
  const missing = (template.required_slots || []).filter((slotName) => !slotPresent(slots, slotName));
  for (const [slotName, rule] of Object.entries(template.slot_validations || {})) {
    const count = slotItemCount(slots[slotName]);
    if (rule.exact_items !== undefined && rule.exact_items !== null && count !== Number(rule.exact_items)) {
      if (!missing.includes(slotName)) {
        missing.push(slotName);
      }
      continue;
    }
    if (rule.min_items !== undefined && rule.min_items !== null && count < Number(rule.min_items)) {
      if (!missing.includes(slotName)) {
        missing.push(slotName);
      }
    }
    if (rule.max_items !== undefined && rule.max_items !== null && count > Number(rule.max_items)) {
      if (!missing.includes(slotName)) {
        missing.push(slotName);
      }
    }
  }
  return missing;
}

function coverageScore(slotNames, slots) {
  if (!slotNames.length) {
    return 1;
  }
  const hit = slotNames.filter((slotName) => {
    return slotPresent(slots, slotName);
  }).length;
  return hit / slotNames.length;
}

function slotValidationScores(template, slots) {
  return Object.entries(template.slot_validations || {}).map(([slotName, rule]) => slotValidationScore(slotName, rule, slots));
}

function slotValidationScore(slotName, rule, slots) {
  const count = slotItemCount(slots[slotName]);
  if (rule.exact_items !== undefined && rule.exact_items !== null) {
    return count === Number(rule.exact_items) ? 1 : 0;
  }
  if (rule.min_items !== undefined && rule.min_items !== null && count < Number(rule.min_items)) {
    return count / Math.max(1, Number(rule.min_items));
  }
  if (rule.max_items !== undefined && rule.max_items !== null && count > Number(rule.max_items)) {
    return Math.max(0, Number(rule.max_items) / Math.max(1, count));
  }
  return 1;
}

function containsCjk(text) {
  return [...text].some((char) => char >= "\u4e00" && char <= "\u9fff");
}

function charNgrams(text, size) {
  if (text.length < size) {
    return [text];
  }
  const output = [];
  for (let index = 0; index <= text.length - size; index += 1) {
    output.push(text.slice(index, index + size));
  }
  return output;
}

function countTerms(terms) {
  return terms.reduce((acc, term) => {
    acc[term] = (acc[term] || 0) + 1;
    return acc;
  }, {});
}

function sumObject(value) {
  return Object.values(value).reduce((sum, item) => sum + item, 0);
}

function intersectionSize(left, right) {
  let total = 0;
  for (const value of left) {
    if (right.has(value)) {
      total += 1;
    }
  }
  return total;
}

function flattenValue(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const candidates = new Set();
    for (const key of ["preset", "id", "value", "code"]) {
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

function matchesAllowedValues(extracted, allowedValues) {
  const extractedCandidates = flattenValue(extracted);
  return (allowedValues || []).some((allowed) => {
    const allowedCandidates = flattenValue(allowed);
    for (const candidate of extractedCandidates) {
      if (allowedCandidates.has(candidate)) {
        return true;
      }
    }
    return false;
  });
}

function slotSignalWeight(slotName) {
  if (slotName === "metric_conditions") {
    return 2;
  }
  if (LOW_SIGNAL_SLOTS.has(slotName)) {
    return 0.5;
  }
  if (slotName.endsWith("_threshold")) {
    return 1.5;
  }
  if (FILTER_SLOTS.has(slotName)) {
    return 1.25;
  }
  return 1;
}

function slotPresent(slots, slotName) {
  return slotPresentValue(slots[slotName]);
}

function slotPresentValue(value) {
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

function slotItemCount(value) {
  if (!slotPresentValue(value)) {
    return 0;
  }
  if (Array.isArray(value)) {
    return value.length;
  }
  return 1;
}

function isFilterSlot(slotName) {
  return slotName.endsWith("_threshold") || FILTER_SLOTS.has(slotName);
}

function normalizeWeights(weights) {
  const total = Object.values(weights).reduce((sum, value) => sum + Math.max(0, value), 0);
  if (total <= 0) {
    return weights;
  }
  return Object.fromEntries(Object.entries(weights).map(([key, value]) => [key, Math.max(0, value) / total]));
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function sumSet(values, iteratee) {
  let total = 0;
  for (const value of values) {
    total += iteratee(value);
  }
  return total;
}
