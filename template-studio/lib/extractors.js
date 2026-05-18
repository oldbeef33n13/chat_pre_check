import { deepClone, normalizeText } from "./utils.js";

export function buildSlotRegistry(definitions = {}) {
  const registry = {};
  for (const [slotName, definition] of Object.entries(definitions || {})) {
    const slotExtractors = [];
    for (const extractor of definition.extractors || []) {
      const type = String(extractor.type || "").toLowerCase();
      if (type === "keyword_value") {
        slotExtractors.push(new KeywordValueExtractor(extractor.cases || [], extractor.match_policy || "first"));
        continue;
      }
      if (type === "regex") {
        slotExtractors.push(new RegexValueExtractor(extractor.patterns || []));
        continue;
      }
      if (type === "time_range") {
        slotExtractors.push(new TimeRangeExtractor(extractor));
        continue;
      }
      if (type === "metric_conditions") {
        slotExtractors.push(new MetricConditionsExtractor(extractor));
      }
    }
    registry[slotName] = slotExtractors;
  }
  return new SlotExtractorRegistry(registry);
}

class KeywordValueExtractor {
  constructor(cases, matchPolicy = "first") {
    this.cases = cases;
    this.matchPolicy = String(matchPolicy || "first").toLowerCase();
  }

  extract(text) {
    if (this.matchPolicy === "longest") {
      return this.extractLongest(text);
    }
    for (const entry of this.cases) {
      const terms = (entry.terms || []).map((term) => normalizeText(String(term))).filter(Boolean);
      if (terms.some((term) => text.includes(term))) {
        return deepClone(entry.value);
      }
    }
    return null;
  }

  extractLongest(text) {
    let best = null;
    this.cases.forEach((entry, caseIndex) => {
      const terms = (entry.terms || []).map((term) => normalizeText(String(term))).filter(Boolean);
      for (const term of terms) {
        const start = text.indexOf(term);
        if (start < 0) {
          continue;
        }
        const candidate = {
          length: term.length,
          start,
          caseIndex,
          value: deepClone(entry.value)
        };
        if (
          !best ||
          candidate.length > best.length ||
          (candidate.length === best.length && candidate.start < best.start) ||
          (candidate.length === best.length && candidate.start === best.start && candidate.caseIndex < best.caseIndex)
        ) {
          best = candidate;
        }
      }
    });
    return best ? best.value : null;
  }
}

class RegexValueExtractor {
  constructor(patterns) {
    this.patterns = patterns
      .filter((entry) => entry && entry.pattern)
      .map((entry) => ({
        regex: new RegExp(String(entry.pattern), "i"),
        group: Number(entry.group ?? 1),
        valueType: String(entry.value_type || "string"),
        min: entry.min ?? null,
        max: entry.max ?? null,
        value: entry.value ?? null
      }));
  }

  extract(text) {
    for (const pattern of this.patterns) {
      const match = pattern.regex.exec(text);
      if (!match) {
        continue;
      }
      if (pattern.value !== null && pattern.value !== undefined) {
        return deepClone(pattern.value);
      }
      const rawValue = match[pattern.group];
      const value = castValue(rawValue, pattern.valueType);
      if (value === null || value === undefined || value === "") {
        continue;
      }
      if (typeof value === "number") {
        if (pattern.min !== null && Number(value) < Number(pattern.min)) {
          continue;
        }
        if (pattern.max !== null && Number(value) > Number(pattern.max)) {
          continue;
        }
      }
      return value;
    }
    return null;
  }
}

class TimeRangeExtractor {
  constructor(options = {}) {
    this.includeAbsolute = Boolean(options.include_absolute ?? true);
    this.relativeRe = /(?:最近|近|过去)?\s*(\d+)\s*(小时|天|日|周|星期|个月|月)\s*(?:内)?/i;
    this.dateRe = /(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})(?:日)?/;
    this.dateRangeRe = new RegExp(`${this.dateRe.source}\\s*(?:到|至|-|~)\\s*${this.dateRe.source}`);
  }

  extract(text) {
    const presets = [
      ["今天", "today"],
      ["今日", "today"],
      ["昨天", "yesterday"],
      ["昨日", "yesterday"],
      ["前天", "day_before_yesterday"],
      ["本周", "this_week"],
      ["本月", "this_month"]
    ];
    for (const [term, preset] of presets) {
      if (text.includes(term)) {
        return { mode: "relative", preset };
      }
    }
    const relative = this.relativeRe.exec(text);
    if (relative) {
      const amount = Number.parseInt(relative[1], 10);
      const unit = normalizeTimeUnit(relative[2]);
      return {
        mode: "relative",
        value: amount,
        unit,
        preset: relativeTimePreset(amount, unit)
      };
    }
    if (!this.includeAbsolute) {
      return null;
    }
    const range = this.dateRangeRe.exec(text);
    if (range) {
      return {
        mode: "absolute_range",
        start: formatDate(range[1], range[2], range[3]),
        end: formatDate(range[4], range[5], range[6])
      };
    }
    const date = this.dateRe.exec(text);
    if (date) {
      return {
        mode: "absolute_date",
        date: formatDate(date[1], date[2], date[3])
      };
    }
    return null;
  }
}

class MetricConditionsExtractor {
  constructor(options = {}) {
    this.maxGapChars = Number(options.max_gap_chars ?? 24);
    this.valueType = String(options.value_type || "float");
    this.valueRe = new RegExp(String(options.value_pattern || "(-?\\d+(?:\\.\\d+)?)\\s*%?"), "i");
    this.metrics = (options.metrics || [])
      .flatMap((metric) =>
        (metric.terms || [])
          .map((term) => [normalizeText(String(term)), metric.value ?? metric.metric])
          .filter(([term]) => term)
      )
      .sort((left, right) => right[0].length - left[0].length);
    const operators = options.operators?.length
      ? options.operators
      : [
          { terms: ["大于等于", "不小于", ">="], value: ">=" },
          { terms: ["小于等于", "不大于", "<="], value: "<=" },
          { terms: ["不等于", "!="], value: "!=" },
          { terms: ["大于", "超过", "高于", ">"], value: ">" },
          { terms: ["小于", "低于", "少于", "<"], value: "<" },
          { terms: ["等于", "为", "="], value: "=" }
        ];
    this.operators = operators
      .flatMap((operator) =>
        (operator.terms || [])
          .map((term) => [normalizeText(String(term)), String(operator.value || "")])
          .filter(([term, value]) => term && value)
      )
      .sort((left, right) => right[0].length - left[0].length);
  }

  extract(text) {
    const candidates = [];
    for (const [term, metricValue] of this.metrics) {
      for (const match of text.matchAll(new RegExp(escapeRegExp(term), "g"))) {
        const start = match.index || 0;
        const suffix = text.slice(start + term.length, start + term.length + this.maxGapChars);
        const operator = this.findOperator(suffix);
        if (!operator) {
          continue;
        }
        const [, operatorEnd, operatorValue] = operator;
        const valueMatch = this.valueRe.exec(suffix.slice(operatorEnd));
        if (!valueMatch) {
          continue;
        }
        const value = castValue(valueMatch[1], this.valueType);
        if (value === null || value === undefined) {
          continue;
        }
        candidates.push({
          start,
          termLength: term.length,
          condition: {
            metric: deepClone(metricValue),
            operator: operatorValue,
            value,
            raw: text.slice(start, start + term.length + operatorEnd + valueMatch.index + valueMatch[0].length)
          }
        });
      }
    }
    candidates.sort((left, right) => left.start - right.start || right.termLength - left.termLength);
    const seen = new Set();
    const output = [];
    for (const candidate of candidates) {
      const key = `${candidate.condition.metric}|${candidate.condition.operator}|${candidate.condition.value}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      output.push(candidate.condition);
    }
    return output.length ? output : null;
  }

  findOperator(text) {
    let best = null;
    for (const [term, value] of this.operators) {
      const start = text.indexOf(term);
      if (start < 0) {
        continue;
      }
      const candidate = [start, start + term.length, value];
      if (!best || candidate[0] < best[0] || (candidate[0] === best[0] && candidate[1] - candidate[0] > best[1] - best[0])) {
        best = candidate;
      }
    }
    return best;
  }
}

class SlotExtractorRegistry {
  constructor(extractors) {
    this.extractors = extractors;
  }

  extract(text) {
    const slots = {};
    for (const [slotName, extractors] of Object.entries(this.extractors)) {
      for (const extractor of extractors) {
        const value = extractor.extract(text);
        if (value !== null && value !== undefined && value !== "") {
          slots[slotName] = value;
          break;
        }
      }
    }
    return slots;
  }
}

function castValue(rawValue, valueType) {
  try {
    if (valueType === "int") {
      return Number.parseInt(rawValue, 10);
    }
    if (valueType === "float") {
      return Number.parseFloat(rawValue);
    }
    return String(rawValue);
  } catch {
    return null;
  }
}

function normalizeTimeUnit(rawUnit) {
  if (rawUnit === "小时") {
    return "hour";
  }
  if (rawUnit === "周" || rawUnit === "星期") {
    return "week";
  }
  if (rawUnit === "个月" || rawUnit === "月") {
    return "month";
  }
  return "day";
}

function relativeTimePreset(amount, unit) {
  const suffix = {
    hour: "h",
    day: "d",
    week: "w",
    month: "m"
  }[unit] || unit;
  return `last_${amount}${suffix}`;
}

function formatDate(year, month, day) {
  return `${String(Number(year)).padStart(4, "0")}-${String(Number(month)).padStart(2, "0")}-${String(Number(day)).padStart(2, "0")}`;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
