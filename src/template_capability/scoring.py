from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from template_capability.extractors import normalize_text
from template_capability.models import TemplateDefinition, template_declared_slot_names
from template_capability.text_matching import match_text_rule


TOKEN_RE = re.compile(r"[a-z0-9_.-]+|[\u4e00-\u9fff]+")
LOW_SIGNAL_SLOTS = {"time_range", "region_id"}
FILTER_SLOTS = {
    "query_operator",
    "topn",
    "severity",
    "device_id",
    "protocol",
    "selector_type",
    "selector_value",
    "metric_conditions",
}


@dataclass(slots=True)
class RequirementIssue:
    """单条槽位规则诊断。"""

    kind: str
    slots: list[str]
    message: str
    trigger_slots: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "slots": list(self.slots),
            "message": self.message,
            "trigger_slots": list(self.trigger_slots),
            "description": self.description,
        }


@dataclass(slots=True)
class SlotRequirementReport:
    """模板槽位约束的综合评估结果。"""

    missing_slots: list[str] = field(default_factory=list)
    issues: list[RequirementIssue] = field(default_factory=list)

    @property
    def is_satisfied(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_slots": list(self.missing_slots),
            "issues": [issue.to_dict() for issue in self.issues],
            "blocking_issue_count": len(self.issues),
        }


@dataclass(slots=True)
class StructureAlignmentReport:
    """结构分诊断结果。"""

    score: float
    support_coverage_score: float
    capture_score: float
    filter_capture_score: float
    requirement_completeness_score: float
    key_requirement_penalty: float
    soft_requirement_penalty: float
    conflict_penalty: float
    unsupported_query_slots: list[str] = field(default_factory=list)
    uncaptured_supported_slots: list[str] = field(default_factory=list)
    missing_requirement_slots: list[str] = field(default_factory=list)
    conflict_slots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "support_coverage_score": self.support_coverage_score,
            "capture_score": self.capture_score,
            "filter_capture_score": self.filter_capture_score,
            "requirement_completeness_score": self.requirement_completeness_score,
            "key_requirement_penalty": self.key_requirement_penalty,
            "soft_requirement_penalty": self.soft_requirement_penalty,
            "conflict_penalty": self.conflict_penalty,
            "unsupported_query_slots": list(self.unsupported_query_slots),
            "uncaptured_supported_slots": list(self.uncaptured_supported_slots),
            "missing_requirement_slots": list(self.missing_requirement_slots),
            "conflict_slots": list(self.conflict_slots),
        }


def mixed_terms(text: str) -> list[str]:
    """混合 token + ，兼顾中文短句和提参变化。"""
    chunks = TOKEN_RE.findall(text.lower())
    terms: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        # 先保留完整 token，保证精确命中的领域词有足够权重。
        terms.append(chunk)
        if _contains_cjk(chunk):
            # 中文缺少天然空格分词，用 2/3-gram 吸收语序变化、口语缩写和轻微错字。
            terms.extend(_char_ngrams(chunk, 2))
            if len(chunk) >= 3:
                terms.extend(_char_ngrams(chunk, 3))
        elif len(chunk) >= 4:
            # 英文或数字串不过度切碎，只补 3-gram 兼顾 typo 容错。
            terms.extend(_char_ngrams(chunk, 3))
    return terms or [text.lower()]


def sample_similarity_score(text: str, utterances: list[str]) -> float:
    """兼容旧调用方的便捷入口。"""
    if not utterances:
        return 0.0
    query_terms = set(mixed_terms(text))
    return sample_similarity_from_terms(query_terms, build_utterance_term_sets(utterances))


def build_utterance_term_sets(utterances: list[str]) -> list[set[str]]:
    """提前把模板示例问法切词，避免每次匹配都重复处理。"""
    return [set(mixed_terms(utterance)) for utterance in utterances if utterance]


def sample_similarity_from_terms(query_terms: set[str], utterance_term_sets: list[set[str]]) -> float:
    """用 Dice 风格的重叠率衡量 query 和示例问法的接近程度。"""
    if not query_terms:
        return 0.0
    best = 0.0
    for utterance_terms in utterance_term_sets:
        overlap = len(query_terms & utterance_terms)
        total = len(query_terms) + len(utterance_terms)
        if total == 0:
            continue
        score = (2.0 * overlap) / total
        best = max(best, score)
    return best


@dataclass(slots=True)
class BM25Index:
    """轻量 BM25 实现，用于模板候选召回。"""

    documents: dict[str, list[str]]
    k1: float = 1.5
    b: float = 0.75
    doc_term_freqs: dict[str, Counter[str]] = field(init=False, default_factory=dict)
    doc_lengths: dict[str, int] = field(init=False, default_factory=dict)
    avg_doc_length: float = field(init=False, default=0.0)
    doc_freqs: Counter[str] = field(init=False, default_factory=Counter)
    total_docs: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.doc_term_freqs = {
            doc_id: Counter(terms)
            for doc_id, terms in self.documents.items()
        }
        self.doc_lengths = {
            doc_id: sum(term_freq.values())
            for doc_id, term_freq in self.doc_term_freqs.items()
        }
        doc_count = max(1, len(self.documents))
        self.avg_doc_length = sum(self.doc_lengths.values()) / doc_count
        self.doc_freqs: Counter[str] = Counter()
        for term_freq in self.doc_term_freqs.values():
            for term in term_freq:
                self.doc_freqs[term] += 1
        self.total_docs = doc_count

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        # query 侧和模板侧使用同一套 mixed_terms，保证 lexical 召回的口径一致。
        query_terms = mixed_terms(query_text)
        results = [
            (doc_id, self._score_doc(query_terms, doc_id))
            for doc_id in self.documents
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def _score_doc(self, query_terms: list[str], doc_id: str) -> float:
        term_freq = self.doc_term_freqs.get(doc_id, Counter())
        doc_length = self.doc_lengths.get(doc_id, 0)
        score = 0.0
        for term in query_terms:
            # query 里没出现、或模板文档里没命中的词，直接跳过，不做平滑补偿。
            freq = term_freq.get(term)
            if not freq:
                continue
            doc_freq = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_length / max(1.0, self.avg_doc_length)
            )
            score += idf * numerator / denominator
        return score


@dataclass(slots=True)
class BM25FieldIndex:
    """简化版 BM25F，对模板多字段分别建模后加权求和。"""

    documents: dict[str, dict[str, list[str]]]
    field_weights: dict[str, float]
    k1: float = 1.5
    b: float = 0.75
    doc_term_freqs: dict[str, dict[str, Counter[str]]] = field(init=False, default_factory=dict)
    doc_lengths: dict[str, dict[str, int]] = field(init=False, default_factory=dict)
    avg_field_lengths: dict[str, float] = field(init=False, default_factory=dict)
    field_doc_freqs: dict[str, Counter[str]] = field(init=False, default_factory=dict)
    total_docs: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.total_docs = max(1, len(self.documents))
        self.doc_term_freqs = {}
        self.doc_lengths = {}
        for doc_id, fields in self.documents.items():
            self.doc_term_freqs[doc_id] = {}
            self.doc_lengths[doc_id] = {}
            for field_name in self.field_weights:
                terms = fields.get(field_name, [])
                term_freq = Counter(terms)
                self.doc_term_freqs[doc_id][field_name] = term_freq
                self.doc_lengths[doc_id][field_name] = sum(term_freq.values())

        self.avg_field_lengths = {}
        self.field_doc_freqs = {}
        for field_name in self.field_weights:
            # BM25F 需要维护每个字段自己的平均长度和文档频次，
            # 否则 utterances / must_terms / description 的长度差异会互相污染。
            total_length = sum(
                self.doc_lengths[doc_id].get(field_name, 0)
                for doc_id in self.documents
            )
            self.avg_field_lengths[field_name] = total_length / self.total_docs
            doc_freqs: Counter[str] = Counter()
            for doc_id in self.documents:
                for term in self.doc_term_freqs[doc_id][field_name]:
                    doc_freqs[term] += 1
            self.field_doc_freqs[field_name] = doc_freqs

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        # 每个字段先独立打分，最后再按字段权重汇总。
        query_terms = mixed_terms(query_text)
        results = [
            (doc_id, self._score_doc(query_terms, doc_id))
            for doc_id in self.documents
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def _score_doc(self, query_terms: list[str], doc_id: str) -> float:
        score = 0.0
        for field_name, field_weight in self.field_weights.items():
            score += field_weight * self._score_field(query_terms, doc_id, field_name)
        return score

    def _score_field(self, query_terms: list[str], doc_id: str, field_name: str) -> float:
        term_freq = self.doc_term_freqs.get(doc_id, {}).get(field_name, Counter())
        doc_length = self.doc_lengths.get(doc_id, {}).get(field_name, 0)
        avg_length = self.avg_field_lengths.get(field_name, 0.0)
        doc_freqs = self.field_doc_freqs.get(field_name, Counter())
        score = 0.0
        for term in query_terms:
            # 字段级 BM25 只在当前字段局部统计 tf/idf，
            # 这样 must_terms 的短字段不会被 description 的长文本掩盖。
            freq = term_freq.get(term)
            if not freq:
                continue
            doc_freq = doc_freqs.get(term, 0)
            idf = math.log(1 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_length / max(1.0, avg_length)
            )
            score += idf * numerator / denominator
        return score


def normalize_candidate_scores(items: list[tuple[str, float]]) -> dict[str, float]:
    """把不同召回路的原始分压到 0-1，方便后续融合。"""
    if not items:
        return {}
    top_score = max(score for _, score in items)
    if top_score <= 0:
        return {doc_id: 0.0 for doc_id, _ in items}
    return {doc_id: clamp_score(score / top_score) for doc_id, score in items}


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    rrf_k: int = 60,
) -> dict[str, float]:
    """RRF 只看 rank，不依赖各路原始分是否同尺度。"""
    fused: dict[str, float] = {}
    for ranking in rankings:
        # rank 越靠前，贡献越大；哪怕原始分量纲不同，也能在同一空间里融合。
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    if not fused:
        return {}
    top_score = max(fused.values())
    if top_score <= 0:
        return {doc_id: 0.0 for doc_id in fused}
    return {doc_id: clamp_score(score / top_score) for doc_id, score in fused.items()}


def slot_fit_score(template: TemplateDefinition, slots: dict[str, Any]) -> float:
    """衡量当前模板需要的槽位被填得有多完整。"""
    requirement_scores: list[float] = []
    required = template.required_slots
    optional = template.optional_slots
    if required:
        requirement_scores.append(_coverage_score(required, slots))
    if template.required_one_of:
        requirement_scores.append(
            sum(_one_of_group_score(group.slots, slots) for group in template.required_one_of)
            / len(template.required_one_of)
        )
    active_conditional_scores = _conditional_requirement_scores(template, slots)
    if active_conditional_scores:
        requirement_scores.append(sum(active_conditional_scores) / len(active_conditional_scores))
    if template.mutually_exclusive_slots:
        requirement_scores.append(
            sum(_mutual_exclusion_score(group.slots, slots) for group in template.mutually_exclusive_slots)
            / len(template.mutually_exclusive_slots)
        )
    validation_scores = _slot_validation_scores(template, slots)
    if validation_scores:
        requirement_scores.append(sum(validation_scores) / len(validation_scores))
    required_hit = sum(requirement_scores) / len(requirement_scores) if requirement_scores else 1.0
    if not optional:
        return required_hit
    optional_hit = _coverage_score(optional, slots)
    return clamp_score(0.8 * required_hit + 0.2 * optional_hit)


def structural_alignment_score(
    template: TemplateDefinition,
    *,
    query_slots: dict[str, Any],
    candidate_slots: dict[str, Any],
    requirement_report: SlotRequirementReport | None = None,
) -> float:
    """兼容旧调用方，只返回结构分数值。"""
    return evaluate_structural_alignment(
        template,
        query_slots=query_slots,
        candidate_slots=candidate_slots,
        requirement_report=requirement_report,
    ).score


def evaluate_structural_alignment(
    template: TemplateDefinition,
    *,
    query_slots: dict[str, Any],
    candidate_slots: dict[str, Any],
    requirement_report: SlotRequirementReport | None = None,
) -> StructureAlignmentReport:
    """衡量“query 条件集合”和“模板实际接住的条件集合”是否同构。"""
    requirement_report = requirement_report or evaluate_slot_requirements(template, candidate_slots)
    extracted_query_slots = {
        slot_name
        for slot_name, value in query_slots.items()
        if _value_present(value)
    }
    extracted_candidate_slots = {
        slot_name
        for slot_name, value in candidate_slots.items()
        if _value_present(value)
    }
    if not extracted_query_slots:
        return StructureAlignmentReport(
            score=0.0,
            support_coverage_score=0.0,
            capture_score=0.0,
            filter_capture_score=0.0,
            requirement_completeness_score=_requirement_completeness_score(template, candidate_slots),
            key_requirement_penalty=1.0,
            soft_requirement_penalty=1.0,
            conflict_penalty=1.0,
            missing_requirement_slots=list(requirement_report.missing_slots),
            conflict_slots=_conflict_slots_from_report(requirement_report),
        )
    supported_slots = template_declared_slot_names(template)
    if not supported_slots:
        return StructureAlignmentReport(
            score=0.0,
            support_coverage_score=0.0,
            capture_score=0.0,
            filter_capture_score=0.0,
            requirement_completeness_score=0.0,
            key_requirement_penalty=1.0,
            soft_requirement_penalty=1.0,
            conflict_penalty=1.0,
            unsupported_query_slots=sorted(extracted_query_slots),
            uncaptured_supported_slots=[],
            missing_requirement_slots=list(requirement_report.missing_slots),
            conflict_slots=_conflict_slots_from_report(requirement_report),
        )

    total_weight = sum(_slot_signal_weight(slot_name) for slot_name in extracted_query_slots)
    unexpected_slots = extracted_query_slots - supported_slots
    # query 里出现但模板不支持的条件，会直接拉低结构分。
    unexpected_weight = sum(_slot_signal_weight(slot_name) for slot_name in unexpected_slots)
    coverage_score = 1.0 - unexpected_weight / max(1.0, total_weight)

    supported_query_slots = extracted_query_slots & supported_slots
    if not supported_query_slots:
        return StructureAlignmentReport(
            score=clamp_score(coverage_score),
            support_coverage_score=clamp_score(coverage_score),
            capture_score=0.0,
            filter_capture_score=0.0,
            requirement_completeness_score=_requirement_completeness_score(template, candidate_slots),
            key_requirement_penalty=1.0,
            soft_requirement_penalty=1.0,
            conflict_penalty=1.0,
            unsupported_query_slots=sorted(unexpected_slots),
            uncaptured_supported_slots=[],
            missing_requirement_slots=list(requirement_report.missing_slots),
            conflict_slots=_conflict_slots_from_report(requirement_report),
        )

    supported_query_weight = sum(_slot_signal_weight(slot_name) for slot_name in supported_query_slots)
    captured_slots = supported_query_slots & extracted_candidate_slots
    captured_weight = sum(_slot_signal_weight(slot_name) for slot_name in captured_slots)
    captured_score = captured_weight / max(1.0, supported_query_weight)
    uncaptured_supported_slots = sorted(supported_query_slots - extracted_candidate_slots)

    # 没有过滤条件时，结构分主要看 query 条件是否被模板真正接住。
    extracted_filter_slots = {slot_name for slot_name in supported_query_slots if _is_filter_slot(slot_name)}
    captured_filter_slots = extracted_filter_slots & extracted_candidate_slots
    captured_filter_weight = sum(
        _slot_signal_weight(slot_name)
        for slot_name in captured_filter_slots
    )
    filter_total_weight = sum(_slot_signal_weight(slot_name) for slot_name in extracted_filter_slots)
    filter_score = (
        captured_filter_weight / max(1.0, filter_total_weight)
        if extracted_filter_slots
        else captured_score
    )
    requirement_completeness_score = _requirement_completeness_score(template, candidate_slots)
    key_requirement_penalty, soft_requirement_penalty, conflict_penalty = _requirement_penalties(
        template,
        requirement_report,
        query_slots=query_slots,
    )
    base_score = clamp_score(
        0.3 * coverage_score
        + 0.25 * captured_score
        + 0.25 * filter_score
        + 0.2 * requirement_completeness_score
    )
    score = clamp_score(
        base_score
        * key_requirement_penalty
        * soft_requirement_penalty
        * conflict_penalty
    )
    return StructureAlignmentReport(
        score=score,
        support_coverage_score=clamp_score(coverage_score),
        capture_score=clamp_score(captured_score),
        filter_capture_score=clamp_score(filter_score),
        requirement_completeness_score=clamp_score(requirement_completeness_score),
        key_requirement_penalty=clamp_score(key_requirement_penalty),
        soft_requirement_penalty=clamp_score(soft_requirement_penalty),
        conflict_penalty=clamp_score(conflict_penalty),
        unsupported_query_slots=sorted(unexpected_slots),
        uncaptured_supported_slots=uncaptured_supported_slots,
        missing_requirement_slots=list(requirement_report.missing_slots),
        conflict_slots=_conflict_slots_from_report(requirement_report),
    )


def constraint_score(
    template: TemplateDefinition,
    text: str,
    slots: dict[str, Any],
) -> float:
    """计算模板自身显式约束是否满足。"""
    normalized_text = normalize_text(text)
    if has_negative_term(template, normalized_text):
        return 0.0

    must_score = 1.0
    if template.must_terms:
        matched_groups = 0
        for group in template.must_terms:
            # must_terms 按组计算，只要求每组里任意一个近义词命中即可。
            if any(match_text_rule(normalized_text, rule) for rule in group):
                matched_groups += 1
        must_score = matched_groups / len(template.must_terms)

    slot_constraint_scores: list[float] = []
    for slot_name, allowed_values in template.slot_constraints.items():
        extracted = slots.get(slot_name)
        if extracted in (None, ""):
            if slot_name in template.required_slots:
                # 约束字段本身又是必填时，缺失要明确计 0 分。
                slot_constraint_scores.append(0.0)
            continue
        slot_constraint_scores.append(1.0 if _matches_allowed_values(extracted, allowed_values) else 0.0)

    if not slot_constraint_scores:
        return clamp_score(must_score)
    return clamp_score(0.6 * must_score + 0.4 * (sum(slot_constraint_scores) / len(slot_constraint_scores)))


def has_negative_term(template: TemplateDefinition, text: str) -> bool:
    """只要命中模板负向词，就视为该模板不该匹配。"""
    normalized_text = normalize_text(text)
    return any(match_text_rule(normalized_text, rule) for rule in template.negative_terms)


def weighted_score(parts: dict[str, float], weights: dict[str, float]) -> float:
    """把各子分按权重线性融合。"""
    total = 0.0
    for name, weight in weights.items():
        total += clamp_score(parts.get(name, 0.0)) * weight
    return clamp_score(total)


def adaptive_score_weights(
    base_weights: dict[str, float],
    slots: dict[str, Any],
) -> dict[str, float]:
    """根据 query 复杂度动态调权。

    过滤条件越多，越要提高结构分和槽位覆盖度的重要性，
    否则多条件 query 很容易被单条件模板抢走。
    """
    weights = {name: float(value) for name, value in base_weights.items()}
    filter_slot_count = sum(1 for slot_name, value in slots.items() if _value_present(value) and _is_filter_slot(slot_name))
    complexity = min(1.0, filter_slot_count / 3.0)
    if complexity <= 0:
        return _normalize_weights(weights)

    # query 越复杂，纯文本相似度越容易误导，因此适度下调 lexical/vector，
    # 把更多权重让给结构和槽位完整度。
    for key in ("lexical", "sample", "vector"):
        if key in weights:
            weights[key] *= 1.0 - (0.15 + (0.05 if key == "vector" else 0.0)) * complexity
    if "slot_fit" in weights:
        weights["slot_fit"] *= 1.0 + 0.2 * complexity
    if "structure" in weights:
        weights["structure"] *= 1.0 + 0.45 * complexity
    if "constraint" in weights:
        weights["constraint"] *= 1.0 + 0.1 * complexity
    if "fusion" in weights:
        weights["fusion"] *= 1.0 + 0.1 * complexity
    return _normalize_weights(weights)


def clamp_score(score: float) -> float:
    """统一把分数限制在 0-1。"""
    if math.isnan(score):
        return 0.0
    return max(0.0, min(1.0, score))


def missing_required_slots(template: TemplateDefinition, slots: dict[str, Any]) -> list[str]:
    """兼容旧调用方，返回所有阻塞 MATCHED 的缺失槽位。"""
    return evaluate_slot_requirements(template, slots).missing_slots


def evaluate_slot_requirements(
    template: TemplateDefinition,
    slots: dict[str, Any],
) -> SlotRequirementReport:
    """统一评估模板的必填、条件必填和互斥规则。

    这层是向后兼容的关键：
    - 老模板只配 `required_slots` 时，行为和以前一致
    - 新模板可以叠加 `required_one_of / conditional_required / mutually_exclusive_slots`
    """

    missing_slots: list[str] = []
    issues: list[RequirementIssue] = []

    missing_required = [
        slot_name
        for slot_name in template.required_slots
        if not _slot_present(slots, slot_name)
    ]
    if missing_required:
        _extend_unique(missing_slots, missing_required)
        for slot_name in missing_required:
            issues.append(
                RequirementIssue(
                    kind="required_slot",
                    slots=[slot_name],
                    message=f"required slot '{slot_name}' is missing.",
                )
            )

    for group in template.required_one_of:
        filled_slots = [slot_name for slot_name in group.slots if _slot_present(slots, slot_name)]
        if filled_slots:
            continue
        _extend_unique(missing_slots, group.slots)
        issues.append(
            RequirementIssue(
                kind="required_one_of",
                slots=list(group.slots),
                message=(
                    group.description
                    or f"at least one of [{', '.join(group.slots)}] must be provided."
                ),
                description=group.description,
            )
        )

    for rule in template.conditional_required:
        trigger_slots = _triggered_slots(rule.when_any, rule.when_all, slots)
        if not trigger_slots:
            continue
        missing_required_by_rule = [
            slot_name
            for slot_name in rule.require
            if not _slot_present(slots, slot_name)
        ]
        if not missing_required_by_rule:
            continue
        _extend_unique(missing_slots, missing_required_by_rule)
        issues.append(
            RequirementIssue(
                kind="conditional_required",
                slots=missing_required_by_rule,
                trigger_slots=trigger_slots,
                message=(
                    rule.description
                    or f"when [{', '.join(trigger_slots)}] is present, [{', '.join(missing_required_by_rule)}] is also required."
                ),
                description=rule.description,
            )
        )

    for group in template.mutually_exclusive_slots:
        filled_slots = [slot_name for slot_name in group.slots if _slot_present(slots, slot_name)]
        if len(filled_slots) <= 1:
            continue
        issues.append(
            RequirementIssue(
                kind="mutually_exclusive_slots",
                slots=filled_slots,
                message=(
                    group.description
                    or f"only one of [{', '.join(group.slots)}] can be provided at the same time."
                ),
                description=group.description,
            )
        )

    for slot_name, rule in template.slot_validations.items():
        count = _slot_item_count(slots.get(slot_name))
        exact_items = rule.get("exact_items")
        min_items = rule.get("min_items")
        max_items = rule.get("max_items")
        if exact_items not in (None, ""):
            expected = int(exact_items)
            if count != expected:
                _extend_unique(missing_slots, [slot_name])
                issues.append(
                    RequirementIssue(
                        kind="slot_exact_items",
                        slots=[slot_name],
                        message=f"slot '{slot_name}' must contain exactly {expected} item(s).",
                    )
                )
                continue
        if min_items not in (None, "") and count < int(min_items):
            _extend_unique(missing_slots, [slot_name])
            issues.append(
                RequirementIssue(
                    kind="slot_min_items",
                    slots=[slot_name],
                    message=f"slot '{slot_name}' must contain at least {int(min_items)} item(s).",
                )
            )
        if max_items not in (None, "") and count > int(max_items):
            issues.append(
                RequirementIssue(
                    kind="slot_max_items",
                    slots=[slot_name],
                    message=f"slot '{slot_name}' must contain at most {int(max_items)} item(s).",
                )
            )

    return SlotRequirementReport(
        missing_slots=missing_slots,
        issues=issues,
    )


def _coverage_score(slot_names: list[str], slots: dict[str, Any]) -> float:
    """简单覆盖率，用于 required/optional 的命中统计。"""
    if not slot_names:
        return 1.0
    hit = sum(1 for slot_name in slot_names if _slot_present(slots, slot_name))
    return hit / len(slot_names)


def _one_of_group_score(slot_names: list[str], slots: dict[str, Any]) -> float:
    if not slot_names:
        return 1.0
    return 1.0 if any(_slot_present(slots, slot_name) for slot_name in slot_names) else 0.0


def _conditional_requirement_scores(template: TemplateDefinition, slots: dict[str, Any]) -> list[float]:
    scores: list[float] = []
    for rule in template.conditional_required:
        if not _triggered_slots(rule.when_any, rule.when_all, slots):
            continue
        scores.append(_coverage_score(rule.require, slots))
    return scores


def _mutual_exclusion_score(slot_names: list[str], slots: dict[str, Any]) -> float:
    filled_count = sum(1 for slot_name in slot_names if _slot_present(slots, slot_name))
    return 1.0 if filled_count <= 1 else 0.0


def _slot_validation_scores(template: TemplateDefinition, slots: dict[str, Any]) -> list[float]:
    return [
        _slot_validation_score(slot_name, rule, slots)
        for slot_name, rule in template.slot_validations.items()
    ]


def _slot_validation_score(slot_name: str, rule: dict[str, Any], slots: dict[str, Any]) -> float:
    count = _slot_item_count(slots.get(slot_name))
    exact_items = rule.get("exact_items")
    if exact_items not in (None, ""):
        return 1.0 if count == int(exact_items) else 0.0
    min_items = rule.get("min_items")
    max_items = rule.get("max_items")
    if min_items not in (None, "") and count < int(min_items):
        return count / max(1, int(min_items))
    if max_items not in (None, "") and count > int(max_items):
        return max(0.0, int(max_items) / max(1, count))
    return 1.0


def _requirement_completeness_score(template: TemplateDefinition, slots: dict[str, Any]) -> float:
    """把模板自己的必填规则聚合成一个完整度分数。"""
    weighted_components: list[tuple[float, float]] = []
    for slot_name in template.required_slots:
        weighted_components.append(
            (_slot_signal_weight(slot_name), 1.0 if _slot_present(slots, slot_name) else 0.0)
        )
    for group in template.required_one_of:
        group_weight = max((_slot_signal_weight(slot_name) for slot_name in group.slots), default=1.0)
        weighted_components.append(
            (group_weight, 1.0 if any(_slot_present(slots, slot_name) for slot_name in group.slots) else 0.0)
        )
    for rule in template.conditional_required:
        if not _triggered_slots(rule.when_any, rule.when_all, slots):
            continue
        for slot_name in rule.require:
            weighted_components.append(
                (_slot_signal_weight(slot_name), 1.0 if _slot_present(slots, slot_name) else 0.0)
            )
    for slot_name, rule in template.slot_validations.items():
        weighted_components.append((_slot_signal_weight(slot_name), _slot_validation_score(slot_name, rule, slots)))
    if not weighted_components:
        return 1.0
    total_weight = sum(weight for weight, _ in weighted_components)
    satisfied_weight = sum(weight * value for weight, value in weighted_components)
    return satisfied_weight / max(1.0, total_weight)


def _requirement_penalties(
    template: TemplateDefinition,
    requirement_report: SlotRequirementReport,
    *,
    query_slots: dict[str, Any],
) -> tuple[float, float, float]:
    """把缺槽位和冲突规则转成结构分惩罚项。"""
    query_weight = sum(
        _slot_signal_weight(slot_name)
        for slot_name, value in query_slots.items()
        if _value_present(value)
    )
    required_weight = _template_requirement_weight(template)
    normalizer = max(1.0, query_weight, required_weight)

    key_issue_weight = 0.0
    soft_issue_weight = 0.0
    conflict_issue_weight = 0.0
    for issue in requirement_report.issues:
        issue_weight = _requirement_issue_weight(issue)
        if issue.kind == "mutually_exclusive_slots":
            conflict_issue_weight += issue_weight
            continue
        if any(_is_key_signal_slot(slot_name) for slot_name in issue.slots):
            key_issue_weight += issue_weight
            continue
        soft_issue_weight += issue_weight

    key_penalty = 1.0 - 0.55 * min(1.0, key_issue_weight / normalizer)
    soft_penalty = 1.0 - 0.25 * min(1.0, soft_issue_weight / normalizer)
    conflict_penalty = 1.0 - 0.45 * min(1.0, conflict_issue_weight / normalizer)
    return (
        clamp_score(key_penalty),
        clamp_score(soft_penalty),
        clamp_score(conflict_penalty),
    )


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _char_ngrams(text: str, size: int) -> list[str]:
    if len(text) < size:
        return [text]
    return [text[idx : idx + size] for idx in range(len(text) - size + 1)]


def _matches_allowed_values(extracted: Any, allowed_values: list[Any]) -> bool:
    """支持标量和字典槽位的宽松值比较。"""
    extracted_candidates = _flatten_value(extracted)
    for allowed in allowed_values:
        allowed_candidates = _flatten_value(allowed)
        if extracted_candidates & allowed_candidates:
            return True
    return False


def _flatten_value(value: Any) -> set[str]:
    if isinstance(value, dict):
        candidates: set[str] = set()
        for key in ("preset", "id", "value", "code"):
            if key in value:
                candidates.add(str(value[key]).lower())
        if not candidates:
            candidates.add(str(value).lower())
        return candidates
    return {str(value).lower()}


def _slot_signal_weight(slot_name: str) -> float:
    """不同槽位对结构判断的价值不同。"""
    if slot_name == "metric_conditions":
        return 2.0
    if slot_name in LOW_SIGNAL_SLOTS:
        # 时间、区域对问数模板通常只是修饰信息，不应该像过滤条件那样强影响结构分。
        return 0.5
    if slot_name.endswith("_threshold"):
        # 阈值条件往往决定模板颗粒度，权重给得更高。
        return 1.5
    if slot_name in FILTER_SLOTS:
        return 1.25
    return 1.0


def _template_requirement_weight(template: TemplateDefinition) -> float:
    total = sum(_slot_signal_weight(slot_name) for slot_name in template.required_slots)
    for group in template.required_one_of:
        total += max((_slot_signal_weight(slot_name) for slot_name in group.slots), default=1.0)
    for rule in template.conditional_required:
        total += sum(_slot_signal_weight(slot_name) for slot_name in rule.require)
    for group in template.mutually_exclusive_slots:
        total += sum(_slot_signal_weight(slot_name) for slot_name in group.slots) / max(1, len(group.slots))
    for slot_name in template.slot_validations:
        total += _slot_signal_weight(slot_name)
    return total or 1.0


def _requirement_issue_weight(issue: RequirementIssue) -> float:
    if issue.kind == "required_one_of":
        return max((_slot_signal_weight(slot_name) for slot_name in issue.slots), default=1.0)
    if issue.kind == "mutually_exclusive_slots":
        return sum(_slot_signal_weight(slot_name) for slot_name in issue.slots) / max(1, len(issue.slots))
    return sum(_slot_signal_weight(slot_name) for slot_name in issue.slots) or 1.0


def _conflict_slots_from_report(requirement_report: SlotRequirementReport) -> list[str]:
    conflict_slots: list[str] = []
    for issue in requirement_report.issues:
        if issue.kind != "mutually_exclusive_slots":
            continue
        _extend_unique(conflict_slots, issue.slots)
    return conflict_slots


def _is_key_signal_slot(slot_name: str) -> bool:
    return slot_name not in LOW_SIGNAL_SLOTS


def _is_filter_slot(slot_name: str) -> bool:
    return slot_name.endswith("_threshold") or slot_name in FILTER_SLOTS


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """把任意权重字典重新归一化。"""
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return weights
    return {name: max(0.0, value) / total for name, value in weights.items()}


def _slot_present(slots: dict[str, Any], slot_name: str) -> bool:
    return _value_present(slots.get(slot_name))


def _value_present(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, list) and not value:
        return False
    if isinstance(value, dict) and not value:
        return False
    return True


def _slot_item_count(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return 1 if value else 0
    return 1


def _triggered_slots(
    when_any: list[str],
    when_all: list[str],
    slots: dict[str, Any],
) -> list[str]:
    """返回实际触发条件必填规则的槽位列表。"""

    triggered: list[str] = []
    if when_all:
        missing_when_all = [slot_name for slot_name in when_all if not _slot_present(slots, slot_name)]
        if missing_when_all:
            return []
        triggered.extend(when_all)
    if when_any:
        matched_when_any = [slot_name for slot_name in when_any if _slot_present(slots, slot_name)]
        if not matched_when_any:
            return []
        triggered.extend(matched_when_any)
    if not when_any and not when_all:
        return []
    return list(dict.fromkeys(triggered))


def _extend_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value in seen:
            continue
        target.append(value)
        seen.add(value)
