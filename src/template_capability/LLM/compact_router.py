from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from template_capability.engine import TemplateCapabilityEngine, _merge_slot_maps
from template_capability.extractors import normalize_text
from template_capability.models import MatchResult, MatchStatus, TemplateCandidate, TemplateRouteDecision
from template_capability.scoring import evaluate_slot_requirements


COMPACT_SLOT_ROUTER_SYSTEM_PROMPT = """你是模板路由后的快速提参器。只做三件事：在候选模板中确认是否可用、补充槽位、返回 JSON。

固定规则：
1. 只能选择用户输入中的候选模板 id，不能创造模板。
2. 如果没有候选模板可靠满足 query，返回 r="nl2sql"。
3. 如果 required / one_of / conditional / validations 不满足，返回 r="nl2sql"。
4. 不生成 SQL，不解释业务，不回答用户问题。
5. 不编造 query 没有表达的业务条件；optional 缺失由系统默认值处理。
6. derived 槽位不要直接输出后端枚举，只输出其 source slots。例如 healthStatus 由 resource_type + health_status_label 派生。
7. metric_conditions 必须完整抽取 query 中所有指标条件，不能因为候选模板是单指标就忽略第二个指标。
8. 自由文本槽位必须保守，只有明确前缀或明确上下文才抽，例如“名称为 X”“型号为 X”。
9. 命中分析、原因、报告、预测、建议、优化等非问数意图时返回 r="nl2sql"。
10. 输出必须是严格 JSON，不要 Markdown。

字段约定：
- r: "template" 或 "nl2sql"
- id: 选中的模板 id；回退时为 -1
- s: slots，对象
- m: missing slots，数组
- cf: 置信度，0 到 1

输出格式固定：
{"r":"template|nl2sql","id":"template_id|-1","s":{},"m":[],"cf":0.0}"""


CompactLLMCall = Callable[[str, dict[str, Any]], Mapping[str, Any] | str | None]


@dataclass(slots=True)
class CompactRouterContext:
    """一次 compact LLM 路由所需的运行期上下文。"""

    input_text: str
    original_norm_text: str
    norm_text: str
    shared_slots: dict[str, object]
    global_slots: dict[str, object]
    diagnostic_slots: dict[str, object]
    ranked: list[TemplateCandidate]
    payload: dict[str, Any]
    blocked_term: str | None = None


@dataclass(slots=True)
class CompactLLMDecision:
    route_to: str
    template_id: str | int
    slots: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    confidence: float = 0.0


def mock_compact_llm_call(
    system_prompt: str,
    user_payload: dict[str, Any],
) -> Mapping[str, Any] | str | None:
    """LLM 调用占位函数。

    接真实模型时只需要替换这个函数，保持入参和返回结构不变：
    - 入参 system_prompt: 稳定系统提示词
    - 入参 user_payload: 每次请求的最小动态 JSON
    - 返回: dict / JSON string / None
    """

    _ = (system_prompt, user_payload)
    return None


def route_with_compact_llm(
    engine: TemplateCapabilityEngine,
    input_text: str,
    *,
    llm_call: CompactLLMCall = mock_compact_llm_call,
    top_k: int | None = None,
) -> TemplateRouteDecision:
    """完整串接流程：高速召回 -> compact payload -> LLM 补参 -> 确定性校验。

    这个入口适合你描述的新链路：
    1. 复用当前多路召回，快速筛出 top1/top3。
    2. 只把最小候选信息交给 LLM。
    3. LLM 只返回模板选择和槽位，不做 SQL。
    4. 系统再用 required / validations / derived / defaults 做最终判定。
    5. 满足模板才 route_to="template"，否则 route_to="nl2sql"。
    """

    context = build_compact_router_context(engine, input_text, top_k=top_k)
    if context.blocked_term is not None:
        return _nl2sql_decision(engine, context, reason="blocked_intent")

    top = context.ranked[0] if context.ranked else None
    second = context.ranked[1] if len(context.ranked) > 1 else None
    if top is None:
        return _nl2sql_decision(engine, context, reason="no_template_candidate")
    if top.score < engine.config.settings.template_route_min_score:
        return _nl2sql_decision(engine, context, reason="low_template_confidence")
    if engine._is_ambiguous(top, second):
        return _nl2sql_decision(engine, context, reason="ambiguous_template_candidates")

    raw_decision = llm_call(COMPACT_SLOT_ROUTER_SYSTEM_PROMPT, context.payload)
    decision = parse_compact_llm_decision(raw_decision)
    if decision is None:
        return _nl2sql_decision(engine, context, reason="llm_no_decision")
    if decision.route_to != "template":
        slots = _merge_slot_maps(context.diagnostic_slots, decision.slots)
        return _nl2sql_decision(engine, context, reason="llm_rejected_template", slots=slots)

    candidate = _find_candidate(context.ranked, decision.template_id)
    if candidate is None:
        return _nl2sql_decision(engine, context, reason="llm_selected_unknown_template")
    if candidate.score < engine.config.settings.template_route_min_score:
        return _nl2sql_decision(engine, context, reason="selected_candidate_below_threshold")
    if candidate.structure_score < engine.config.settings.template_route_min_structure_score:
        return _nl2sql_decision(engine, context, reason="selected_candidate_structure_too_low")

    merged_slots = _merge_missing_slots_only(candidate.slots, decision.slots)
    enriched = TemplateCandidate(
        template_id=candidate.template_id,
        query_mode=candidate.query_mode,
        score=candidate.score,
        lexical_score=candidate.lexical_score,
        sample_score=candidate.sample_score,
        vector_score=candidate.vector_score,
        fusion_score=candidate.fusion_score,
        rerank_score=candidate.rerank_score,
        slot_fit_score=candidate.slot_fit_score,
        constraint_score=candidate.constraint_score,
        structure_score=candidate.structure_score,
        slots=merged_slots,
        missing_slots=list(candidate.missing_slots),
        trace={**candidate.trace, "compact_llm_used": True, "compact_llm_confidence": decision.confidence},
        metadata=candidate.metadata,
    )
    finalized = engine._rebuild_candidate(
        enriched,
        norm_text=context.norm_text,
        global_slots=context.global_slots,
        include_defaults=True,
    )
    template = engine.templates[finalized.template_id]
    requirement_report = evaluate_slot_requirements(template, finalized.slots)
    if not requirement_report.is_satisfied:
        return _nl2sql_decision(
            engine,
            context,
            reason="llm_template_requirements_not_satisfied",
            slots=_merge_slot_maps(context.diagnostic_slots, finalized.slots),
            ranked=[finalized, *[item for item in context.ranked if item.template_id != finalized.template_id]],
        )

    result = MatchResult(
        template_id=template.template_id,
        status=MatchStatus.MATCHED,
        score=finalized.score,
        query_mode=template.query_mode,
        slots=finalized.slots,
        missing_slots=[],
        metadata=template.metadata,
        trace={
            "norm_text": context.norm_text,
            "original_norm_text": context.original_norm_text,
            "shared_slots": context.shared_slots,
            "global_slots": context.global_slots,
            "route_to": "template",
            "reason": "compact_llm_template_satisfied",
            "compact_payload": context.payload,
            "selected_template": finalized.to_dict(),
            "top_candidates": [candidate.to_dict() for candidate in context.ranked[:5]],
        },
    )
    return TemplateRouteDecision(
        route_to="template",
        result=result,
        reason="compact_llm_template_satisfied",
        candidates=[finalized, *[item for item in context.ranked if item.template_id != finalized.template_id]][:5],
    )


def build_compact_router_payload(
    engine: TemplateCapabilityEngine,
    input_text: str,
    *,
    top_k: int | None = None,
) -> dict[str, Any]:
    """只构造发给 LLM 的最小动态 payload。"""

    return build_compact_router_context(engine, input_text, top_k=top_k).payload


def build_compact_router_context(
    engine: TemplateCapabilityEngine,
    input_text: str,
    *,
    top_k: int | None = None,
) -> CompactRouterContext:
    original_norm_text = normalize_text(input_text)
    rewrite_result = engine.query_rewriter.rewrite_normalized(original_norm_text)
    norm_text = rewrite_result.rewritten_text
    shared_slots = engine.slot_registry.extract(norm_text)
    global_slots = engine.global_slot_registry.extract(norm_text)
    diagnostic_slots = _merge_slot_maps(shared_slots, global_slots)
    blocked_term = engine._match_blocked_term(norm_text)
    ranked = [] if blocked_term is not None else engine._rank_templates(norm_text, global_slots)
    candidate_limit = max(1, int(top_k or engine.config.settings.template_route_top_k))
    candidates = ranked[:candidate_limit]
    payload: dict[str, Any] = {
        "q": input_text,
        "c": [_compact_candidate_payload(engine, candidate) for candidate in candidates],
    }
    if diagnostic_slots:
        payload["g"] = diagnostic_slots
    if rewrite_result.changed:
        payload["n"] = norm_text
    return CompactRouterContext(
        input_text=input_text,
        original_norm_text=original_norm_text,
        norm_text=norm_text,
        shared_slots=shared_slots,
        global_slots=global_slots,
        diagnostic_slots=diagnostic_slots,
        ranked=ranked,
        payload=payload,
        blocked_term=blocked_term,
    )


def parse_compact_llm_decision(raw: Mapping[str, Any] | str | None) -> CompactLLMDecision | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        return None

    route_to = str(payload.get("r", payload.get("route_to", ""))).strip().lower()
    if route_to not in {"template", "nl2sql"}:
        return None
    template_id_raw = payload.get("id", payload.get("template_id", -1))
    template_id: str | int = -1 if str(template_id_raw) == "-1" else str(template_id_raw)
    slots = payload.get("s", payload.get("slots", {}))
    missing_slots = payload.get("m", payload.get("missing_slots", []))
    return CompactLLMDecision(
        route_to=route_to,
        template_id=template_id,
        slots=dict(slots) if isinstance(slots, Mapping) else {},
        missing_slots=[str(item) for item in missing_slots] if isinstance(missing_slots, list) else [],
        confidence=_clamp_float(payload.get("cf", payload.get("confidence", 0.0))),
    )


def _compact_candidate_payload(engine: TemplateCapabilityEngine, candidate: TemplateCandidate) -> dict[str, Any]:
    template = engine.templates[candidate.template_id]
    item: dict[str, Any] = {
        "id": candidate.template_id,
        "sc": round(candidate.score, 4),
        "st": round(candidate.structure_score, 4),
    }
    if template.required_slots:
        item["req"] = list(template.required_slots)
    if candidate.slots:
        item["cur"] = candidate.slots
    if candidate.missing_slots:
        item["miss"] = candidate.missing_slots
    schema = _compact_schema(template, candidate)
    if schema:
        item["schema"] = schema
    derive = {
        rule.slot_name: list(rule.source_slots)
        for rule in template.derived_slots
    }
    if derive:
        item["derive"] = derive
    if template.slot_validations:
        item["val"] = template.slot_validations
    return item


def _compact_schema(template: Any, candidate: TemplateCandidate) -> dict[str, Any]:
    target_slots = set(template.required_slots) | set(candidate.missing_slots) | set(template.slot_validations)
    for rule in template.derived_slots:
        target_slots.update(rule.source_slots)
    schema: dict[str, Any] = {}
    for slot_name in sorted(target_slots):
        if slot_name in candidate.slots and slot_name not in candidate.missing_slots:
            continue
        metric_schema = _metric_conditions_schema(template, slot_name)
        if metric_schema is not None:
            schema[slot_name] = metric_schema
            continue
        enum_values = _slot_enum_values(template, slot_name)
        if enum_values:
            schema[slot_name] = enum_values
            continue
        value_type = _slot_regex_value_type(template, slot_name)
        if value_type:
            schema[slot_name] = {"type": value_type}
    return schema


def _metric_conditions_schema(template: Any, slot_name: str) -> dict[str, Any] | None:
    definition = template.slot_extractors.get(slot_name)
    if definition is None:
        return None
    for extractor in definition.extractors:
        if str(extractor.get("type", "")).strip().lower() != "metric_conditions":
            continue
        metrics: list[Any] = []
        for item in extractor.get("metrics", []):
            if not isinstance(item, dict):
                continue
            value = item.get("value", item.get("metric"))
            if value is not None and value not in metrics:
                metrics.append(value)
        operators: list[str] = []
        for item in extractor.get("operators", []):
            if not isinstance(item, dict):
                continue
            value = str(item.get("value", "")).strip()
            if value and value not in operators:
                operators.append(value)
        return {
            "metrics": metrics,
            "ops": operators or [">", ">=", "<", "<=", "=", "!="],
        }
    return None


def _slot_enum_values(template: Any, slot_name: str) -> list[Any]:
    values: list[Any] = []
    for value in template.slot_constraints.get(slot_name, []):
        _append_unique(values, value)
    definition = template.slot_extractors.get(slot_name)
    if definition is None:
        return values
    for extractor in definition.extractors:
        if str(extractor.get("type", "")).strip().lower() != "keyword_value":
            continue
        for case in extractor.get("cases", []):
            if isinstance(case, dict) and "value" in case:
                _append_unique(values, case["value"])
    return values


def _slot_regex_value_type(template: Any, slot_name: str) -> str | None:
    definition = template.slot_extractors.get(slot_name)
    if definition is None:
        return None
    for extractor in definition.extractors:
        if str(extractor.get("type", "")).strip().lower() != "regex":
            continue
        for pattern in extractor.get("patterns", []):
            if isinstance(pattern, dict):
                return str(pattern.get("value_type", "string"))
    return None


def _find_candidate(candidates: list[TemplateCandidate], template_id: str | int) -> TemplateCandidate | None:
    token = str(template_id)
    for candidate in candidates:
        if candidate.template_id == token:
            return candidate
    return None


def _nl2sql_decision(
    engine: TemplateCapabilityEngine,
    context: CompactRouterContext,
    *,
    reason: str,
    slots: dict[str, object] | None = None,
    ranked: list[TemplateCandidate] | None = None,
) -> TemplateRouteDecision:
    ranked = context.ranked if ranked is None else ranked
    top = ranked[0] if ranked else None
    result = MatchResult(
        template_id=-1,
        status=MatchStatus.UNMATCHED,
        score=top.score if top is not None else 0.0,
        query_mode=None,
        slots=dict(slots or context.diagnostic_slots),
        missing_slots=[],
        trace={
            "norm_text": context.norm_text,
            "original_norm_text": context.original_norm_text,
            "route_to": "nl2sql",
            "reason": reason,
            "compact_payload": context.payload,
            "template_route_min_score": engine.config.settings.template_route_min_score,
            "template_route_min_structure_score": engine.config.settings.template_route_min_structure_score,
            "top_candidates": [candidate.to_dict() for candidate in ranked[:5]],
        },
    )
    return TemplateRouteDecision(route_to="nl2sql", result=result, reason=reason, candidates=ranked[:5])


def _merge_missing_slots_only(base_slots: dict[str, Any], supplemental_slots: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_slots)
    for slot_name, value in supplemental_slots.items():
        if _value_present(merged.get(slot_name)):
            continue
        if not _value_present(value):
            continue
        merged[str(slot_name)] = copy.deepcopy(value)
    return merged


def _value_present(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, list) and not value:
        return False
    if isinstance(value, dict) and not value:
        return False
    return True


def _append_unique(target: list[Any], value: Any) -> None:
    if value not in target:
        target.append(value)


def _clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
