from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol

from template_capability.models import SlotExtractorDefinition


PUNCTUATION_RE = re.compile(r"[，。？！!?,:：;；、()\[\]{}<>《》\"'`]+")


def normalize_text(text: str) -> str:
    """只做轻量标准化，不做业务改写。"""
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = PUNCTUATION_RE.sub(" ", normalized)
    return " ".join(normalized.split())


class SlotValueExtractor(Protocol):
    """所有 extractor 的统一接口。"""
    def extract(self, text: str) -> Any | None:
        ...


@dataclass(slots=True)
class KeywordValueExtractor:
    cases: list[dict[str, Any]]
    match_policy: str = "first"

    def extract(self, text: str) -> Any | None:
        """命中任一关键词组就返回预定义值。"""
        if self.match_policy == "longest":
            return self._extract_longest(text)
        for case in self.cases:
            terms = [normalize_text(str(term)) for term in case.get("terms", [])]
            if not terms:
                continue
            # 任意一个近义词命中就算该 case 成立；case 的先后顺序就是优先级。
            if any(term and term in text for term in terms):
                return copy.deepcopy(case.get("value"))
        return None

    def _extract_longest(self, text: str) -> Any | None:
        best: tuple[int, int, int, Any] | None = None
        for case_index, case in enumerate(self.cases):
            for term in [normalize_text(str(term)) for term in case.get("terms", [])]:
                if not term:
                    continue
                start = text.find(term)
                if start < 0:
                    continue
                # term 越长越精确；同长度时优先出现位置更靠前、配置顺序更靠前的 case。
                current = (len(term), -start, -case_index, copy.deepcopy(case.get("value")))
                if best is None or current[:3] > best[:3]:
                    best = current
        if best is None:
            return None
        return best[3]


@dataclass(slots=True)
class RegexValueExtractor:
    patterns: list[dict[str, Any]]
    _compiled_patterns: list[tuple[re.Pattern[str], int, str, Any, Any, Any]] = field(
        init=False,
        default_factory=list,
    )

    def __post_init__(self) -> None:
        # 提前编译 regex，避免每次提参时重复编译模式。
        self._compiled_patterns = [
            (
                re.compile(str(spec["pattern"]), re.IGNORECASE),
                int(spec.get("group", 1)),
                str(spec.get("value_type", "string")),
                spec.get("min"),
                spec.get("max"),
                spec.get("value"),
            )
            for spec in self.patterns
            if "pattern" in spec
        ]

    def extract(self, text: str) -> Any | None:
        """按顺序尝试 regex，首个成功命中的规则直接返回。"""
        for pattern, group, value_type, min_value, max_value, fixed_value in self._compiled_patterns:
            match = pattern.search(text)
            if not match:
                continue
            if fixed_value is not None:
                # 某些 regex 只负责识别一个表达，不需要读取分组值，直接返回固定配置值。
                return copy.deepcopy(fixed_value)
            raw_value = match.group(group)
            value = _cast_value(raw_value, value_type)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                # min/max 允许在配置层提前拦掉明显异常的阈值。
                if min_value is not None and value < min_value:
                    continue
                if max_value is not None and value > max_value:
                    continue
            return value
        return None


@dataclass(slots=True)
class TimeRangeExtractor:
    """通用中文时间表达抽取器。

    它负责覆盖常见相对时间，不再要求每个模板穷举“近 N 天 / 过去 N 小时”等 case。
    """

    include_absolute: bool = True
    _relative_re: re.Pattern[str] = field(init=False)
    _date_range_re: re.Pattern[str] = field(init=False)
    _date_re: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        self._relative_re = re.compile(
            r"(?:最近|近|过去)?\s*(\d+)\s*(小时|天|日|周|星期|个月|月)\s*(?:内)?",
            re.IGNORECASE,
        )
        date = r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})(?:日)?"
        self._date_range_re = re.compile(date + r"\s*(?:到|至|-|~)\s*" + date)
        self._date_re = re.compile(date)

    def extract(self, text: str) -> Any | None:
        for term, preset in (
            ("今天", "today"),
            ("今日", "today"),
            ("昨天", "yesterday"),
            ("昨日", "yesterday"),
            ("前天", "day_before_yesterday"),
            ("本周", "this_week"),
            ("本月", "this_month"),
        ):
            if term in text:
                return {"mode": "relative", "preset": preset}

        match = self._relative_re.search(text)
        if match:
            amount = int(match.group(1))
            unit = _normalize_time_unit(match.group(2))
            return {
                "mode": "relative",
                "value": amount,
                "unit": unit,
                "preset": _relative_time_preset(amount, unit),
            }

        if not self.include_absolute:
            return None
        range_match = self._date_range_re.search(text)
        if range_match:
            start = _format_date(range_match.group(1), range_match.group(2), range_match.group(3))
            end = _format_date(range_match.group(4), range_match.group(5), range_match.group(6))
            return {"mode": "absolute_range", "start": start, "end": end}

        date_match = self._date_re.search(text)
        if date_match:
            return {
                "mode": "absolute_date",
                "date": _format_date(date_match.group(1), date_match.group(2), date_match.group(3)),
            }
        return None


@dataclass(slots=True)
class MetricConditionsExtractor:
    """抽取可重复的指标比较条件。

    输出形如：
    [{"metric": "memory_usage", "operator": ">", "value": 50.0}]
    """

    metrics: list[dict[str, Any]]
    operators: list[dict[str, Any]] = field(default_factory=list)
    max_gap_chars: int = 24
    value_type: str = "float"
    value_pattern: str = r"(-?\d+(?:\.\d+)?)\s*%?"
    _metric_terms: list[tuple[str, Any]] = field(init=False, default_factory=list)
    _operator_terms: list[tuple[str, str]] = field(init=False, default_factory=list)
    _value_re: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        metric_terms: list[tuple[str, Any]] = []
        for metric in self.metrics:
            value = metric.get("value", metric.get("metric"))
            for term in metric.get("terms", []):
                normalized = normalize_text(str(term))
                if normalized:
                    metric_terms.append((normalized, value))
        self._metric_terms = sorted(metric_terms, key=lambda item: len(item[0]), reverse=True)

        raw_operators = self.operators or [
            {"terms": ["大于等于", "不小于", ">="], "value": ">="},
            {"terms": ["小于等于", "不大于", "<="], "value": "<="},
            {"terms": ["不等于", "!="], "value": "!="},
            {"terms": ["大于", "超过", "高于", ">"], "value": ">"},
            {"terms": ["小于", "低于", "少于", "<"], "value": "<"},
            {"terms": ["等于", "为", "="], "value": "="},
        ]
        operator_terms: list[tuple[str, str]] = []
        for operator in raw_operators:
            value = str(operator.get("value", "")).strip()
            for term in operator.get("terms", []):
                normalized = normalize_text(str(term))
                if normalized and value:
                    operator_terms.append((normalized, value))
        self._operator_terms = sorted(operator_terms, key=lambda item: len(item[0]), reverse=True)
        self._value_re = re.compile(self.value_pattern, re.IGNORECASE)

    def extract(self, text: str) -> Any | None:
        if not self._metric_terms or not self._operator_terms:
            return None
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for term, metric_value in self._metric_terms:
            for match in re.finditer(re.escape(term), text):
                operator = self._find_operator(text[match.end() : match.end() + self.max_gap_chars])
                if operator is None:
                    continue
                operator_start, operator_end, operator_value = operator
                value_match = self._value_re.search(text[match.end() + operator_end : match.end() + self.max_gap_chars])
                if value_match is None:
                    continue
                raw_value = value_match.group(1)
                value = _cast_value(raw_value, self.value_type)
                if value is None:
                    continue
                candidates.append(
                    (
                        match.start(),
                        -len(term),
                        {
                            "metric": copy.deepcopy(metric_value),
                            "operator": operator_value,
                            "value": value,
                            "raw": text[match.start() : match.end() + operator_end + value_match.end()],
                        },
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        output: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for _, _, condition in candidates:
            fingerprint = (
                str(condition["metric"]),
                str(condition["operator"]),
                str(condition["value"]),
            )
            if fingerprint in seen:
                continue
            output.append(condition)
            seen.add(fingerprint)
        return output or None

    def _find_operator(self, text: str) -> tuple[int, int, str] | None:
        best: tuple[int, int, str] | None = None
        for term, value in self._operator_terms:
            start = text.find(term)
            if start < 0:
                continue
            current = (start, start + len(term), value)
            if best is None or (current[0], -len(term)) < (best[0], -(best[1] - best[0])):
                best = current
        return best


def build_slot_registry(
    definitions: dict[str, SlotExtractorDefinition],
) -> "SlotExtractorRegistry":
    """把配置字典实例化成真正可执行的 extractor 注册表。"""
    extractors: dict[str, list[SlotValueExtractor]] = {}
    for slot_name, definition in definitions.items():
        slot_extractors: list[SlotValueExtractor] = []
        for extractor in definition.extractors:
            extractor_type = str(extractor.get("type", "")).lower()
            if extractor_type == "keyword_value":
                slot_extractors.append(
                    KeywordValueExtractor(
                        cases=[dict(case) for case in extractor.get("cases", [])],
                        match_policy=str(extractor.get("match_policy", "first") or "first").strip().lower(),
                    )
                )
                continue
            if extractor_type == "regex":
                slot_extractors.append(
                    RegexValueExtractor(
                        patterns=[dict(pattern) for pattern in extractor.get("patterns", [])]
                    )
                )
                continue
            if extractor_type == "time_range":
                slot_extractors.append(
                    TimeRangeExtractor(
                        include_absolute=bool(extractor.get("include_absolute", True)),
                    )
                )
                continue
            if extractor_type == "metric_conditions":
                slot_extractors.append(
                    MetricConditionsExtractor(
                        metrics=[dict(metric) for metric in extractor.get("metrics", []) if isinstance(metric, dict)],
                        operators=[dict(operator) for operator in extractor.get("operators", []) if isinstance(operator, dict)],
                        max_gap_chars=int(extractor.get("max_gap_chars", 24)),
                        value_type=str(extractor.get("value_type", "float") or "float"),
                        value_pattern=str(extractor.get("value_pattern", r"(-?\d+(?:\.\d+)?)\s*%?")),
                    )
                )
        # 配置里某个 slot 即使没有合法 extractor，也保留空列表，方便后面统一遍历。
        extractors[slot_name] = slot_extractors
    return SlotExtractorRegistry(extractors)


@dataclass(slots=True)
class SlotExtractorRegistry:
    extractors: dict[str, list[SlotValueExtractor]]

    def extract(self, text: str) -> dict[str, Any]:
        """对每个槽位只保留第一个成功提取的值。"""
        slots: dict[str, Any] = {}
        for slot_name, slot_extractors in self.extractors.items():
            for extractor in slot_extractors:
                value = extractor.extract(text)
                if value not in (None, ""):
                    # 同一槽位一旦命中，就不再继续尝试后续 extractor，
                    # 这样模板作者可以通过配置顺序表达优先级。
                    slots[slot_name] = value
                    break
        return slots


def _cast_value(raw_value: str, value_type: str) -> Any | None:
    """把 regex 命中的文本转成配置声明的值类型。"""
    try:
        if value_type == "int":
            return int(raw_value)
        if value_type == "float":
            return float(raw_value)
        return str(raw_value)
    except ValueError:
        return None


def _normalize_time_unit(raw_unit: str) -> str:
    if raw_unit in {"小时"}:
        return "hour"
    if raw_unit in {"周", "星期"}:
        return "week"
    if raw_unit in {"个月", "月"}:
        return "month"
    return "day"


def _relative_time_preset(amount: int, unit: str) -> str:
    suffix = {
        "hour": "h",
        "day": "d",
        "week": "w",
        "month": "m",
    }.get(unit, unit)
    return f"last_{amount}{suffix}"


def _format_date(year: str, month: str, day: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
