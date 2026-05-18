from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from template_capability.models import (
    build_text_match_groups,
    build_text_match_rules,
    ConditionalSlotRequirement,
    DEFAULT_LEXICAL_FIELD_WEIGHTS,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_SCORE_WEIGHTS,
    DEFAULT_VECTOR_PROVIDER,
    DerivedSlotDefinition,
    MatcherSettings,
    QueryRewriteRule,
    QueryRewriteSettings,
    SlotGroupRequirement,
    SlotExtractorDefinition,
    TemplateDefinition,
)
from template_capability.validation import validate_template_config, validate_template_payload


@dataclass(slots=True)
class TemplateConfig:
    """模板能力完整配置。

    这是配置文件加载后的根对象，按职责拆成三块：
    - `settings`: 全局匹配策略
    - `slot_extractors`: 根级共享槽位定义
    - `templates`: 具体模板列表
    """

    settings: MatcherSettings
    query_rewrite: QueryRewriteSettings = field(default_factory=QueryRewriteSettings)
    slot_extractors: dict[str, SlotExtractorDefinition] = field(default_factory=dict)
    templates: list[TemplateDefinition] = field(default_factory=list)


def load_template_config(path: str | Path) -> TemplateConfig:
    """从 JSON 文件加载配置。"""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    # 启动期先做一次严格 lint，避免坏配置静默进入运行态。
    validate_template_payload(payload)
    matcher_payload = payload.get("matcher", {})
    rewrite_payload = payload.get("query_rewrite", {})
    vector_payload = matcher_payload.get("vector", {})
    reranker_payload = matcher_payload.get("reranker", {})
    fallback_payload = matcher_payload.get("llm_fallback", {})
    slot_fallback_payload = matcher_payload.get("llm_slot_fallback", {})
    template_route_payload = matcher_payload.get("template_route", {})
    weights = {
        str(key): float(value)
        for key, value in matcher_payload.get("weights", {}).items()
    } or dict(DEFAULT_SCORE_WEIGHTS)
    reranker_weight = reranker_payload.get("weight")
    if reranker_weight not in (None, "") and "rerank" not in weights:
        weights["rerank"] = float(reranker_weight)
    settings = MatcherSettings(
        match_threshold=float(matcher_payload.get("match_threshold", 0.58)),
        ambiguity_margin=float(matcher_payload.get("ambiguity_margin", 0.03)),
        recall_top_k=int(matcher_payload.get("recall_top_k", 30)),
        weights=weights,
        lexical_field_weights={
            str(key): float(value)
            for key, value in matcher_payload.get("lexical_field_weights", {}).items()
        }
        or dict(DEFAULT_LEXICAL_FIELD_WEIGHTS),
        fusion_rrf_k=int(matcher_payload.get("fusion_rrf_k", 60)),
        vector_dimension=int(vector_payload.get("dimension", 512)),
        vector_provider=str(vector_payload.get("provider", DEFAULT_VECTOR_PROVIDER) or DEFAULT_VECTOR_PROVIDER),
        vector_options=_build_vector_options(vector_payload),
        reranker_enabled=bool(reranker_payload.get("enabled", False)),
        reranker_provider=str(reranker_payload.get("provider", DEFAULT_RERANKER_PROVIDER) or DEFAULT_RERANKER_PROVIDER),
        reranker_top_k=max(1, int(reranker_payload.get("top_k", 15))),
        reranker_options=_build_reranker_options(reranker_payload),
        blocked_terms=build_text_match_rules(matcher_payload.get("blocked_terms", [])),
        llm_fallback_enabled=bool(fallback_payload.get("enabled", False)),
        llm_fallback_max_candidates=int(fallback_payload.get("max_candidates", 3)),
        llm_fallback_score_margin=float(fallback_payload.get("score_margin", 0.08)),
        llm_fallback_max_missing_slots=int(fallback_payload.get("max_missing_slots", 2)),
        llm_slot_fallback_enabled=bool(slot_fallback_payload.get("enabled", False)),
        llm_slot_fallback_max_missing_slots=int(slot_fallback_payload.get("max_missing_slots", 2)),
        llm_slot_fallback_min_score=float(slot_fallback_payload.get("min_score", matcher_payload.get("match_threshold", 0.58))),
        llm_slot_fallback_allow_on_matched=bool(slot_fallback_payload.get("allow_on_matched", False)),
        template_route_min_score=float(template_route_payload.get("min_score", matcher_payload.get("match_threshold", 0.58))),
        template_route_min_structure_score=float(template_route_payload.get("min_structure_score", 0.0)),
        template_route_top_k=int(template_route_payload.get("top_k", 1)),
    )

    query_rewrite = QueryRewriteSettings(
        enabled=bool(rewrite_payload.get("enabled", False)),
        max_passes=max(1, int(rewrite_payload.get("max_passes", 1))),
        dictionary_path=_resolve_dictionary_path(
            rewrite_payload.get("dictionary_path"),
            base_dir=config_path.parent,
        ),
        reload_on_change=bool(rewrite_payload.get("reload_on_change", True)),
        rules=[
            rule
            for index, item in enumerate(rewrite_payload.get("rules", []), start=1)
            if isinstance(item, dict)
            for rule in [_build_query_rewrite_rule(item, index)]
            if rule is not None
        ],
    )

    # 根级 slot_extractors 是共享定义，只在模板本地未覆写时才会生效。
    slot_extractors = {
        str(slot_name): SlotExtractorDefinition(
            slot_name=str(slot_name),
            extractors=[
                dict(extractor)
                for extractor in definition.get("extractors", [])
                if isinstance(extractor, dict)
            ],
        )
        for slot_name, definition in payload.get("slot_extractors", {}).items()
        if isinstance(definition, dict)
    }

    # templates 是实际参与召回和匹配的对象；每条模板都会被标准化成强类型 dataclass。
    templates = [
        TemplateDefinition(
            template_id=str(item["template_id"]),
            query_mode=str(item.get("query_mode", "metric_query")),
            description=str(item.get("description", "")),
            utterances=[str(text) for text in item.get("utterances", [])],
            required_slots=[str(slot) for slot in item.get("required_slots", [])],
            optional_slots=[str(slot) for slot in item.get("optional_slots", [])],
            must_terms=build_text_match_groups(item.get("must_terms", [])),
            negative_terms=build_text_match_rules(item.get("negative_terms", [])),
            slot_constraints={
                str(slot_name): _normalize_constraint_values(values)
                for slot_name, values in item.get("slot_constraints", {}).items()
            },
            required_one_of=_build_slot_group_requirements(item.get("required_one_of", [])),
            conditional_required=_build_conditional_slot_requirements(item.get("conditional_required", [])),
            mutually_exclusive_slots=_build_slot_group_requirements(item.get("mutually_exclusive_slots", [])),
            slot_extractors={
                str(slot_name): SlotExtractorDefinition(
                    slot_name=str(slot_name),
                    extractors=[
                        dict(extractor)
                        for extractor in definition.get("extractors", [])
                        if isinstance(extractor, dict)
                    ],
                )
                for slot_name, definition in item.get("slot_extractors", {}).items()
                if isinstance(definition, dict)
            },
            slot_defaults=dict(item.get("slot_defaults", {})),
            derived_slots=_build_derived_slot_definitions(item.get("derived_slots", [])),
            slot_validations={
                str(slot_name): dict(rule)
                for slot_name, rule in item.get("slot_validations", {}).items()
                if isinstance(rule, dict)
            },
            llm_slot_extraction=dict(item.get("llm_slot_extraction", {})),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload.get("templates", [])
        if isinstance(item, dict)
    ]
    config = TemplateConfig(
        settings=settings,
        query_rewrite=query_rewrite,
        slot_extractors=slot_extractors,
        templates=templates,
    )
    validate_template_config(config)
    return config


def _normalize_constraint_values(values: Any) -> list[Any]:
    # 配置层允许单值或列表写法，运行时统一转成列表，简化后续判断逻辑。
    if isinstance(values, list):
        return values
    return [values]


def _build_query_rewrite_rule(item: dict[str, Any], index: int) -> QueryRewriteRule | None:
    source = str(item.get("source", "")).strip()
    target = str(item.get("target", "")).strip()
    if not source or not target:
        return None
    return QueryRewriteRule(
        source=source,
        target=target,
        rule_id=str(item.get("rule_id", f"rewrite_rule_{index}")),
        match_mode=str(item.get("match_mode", "substring") or "substring"),
    )


def _resolve_dictionary_path(raw_path: Any, *, base_dir: Path) -> str | None:
    if raw_path in (None, ""):
        return None
    candidate = Path(str(raw_path))
    if not candidate.is_absolute():
        primary = (base_dir / candidate).resolve()
        fallback = (Path.cwd() / candidate).resolve()
        candidate = primary if primary.exists() or not fallback.exists() else fallback
    return str(candidate)


def _build_vector_options(vector_payload: dict[str, Any]) -> dict[str, Any]:
    """保留 provider 的附加配置，供 remote 模式直接透传。"""
    return {
        str(key): value
        for key, value in vector_payload.items()
        if key not in {"provider", "dimension"}
    }


def _build_reranker_options(reranker_payload: dict[str, Any]) -> dict[str, Any]:
    """保留 reranker 的附加配置，方便 provider 自己解释。"""
    return {
        str(key): value
        for key, value in reranker_payload.items()
        if key not in {"enabled", "provider", "top_k", "weight"}
    }


def _build_slot_group_requirements(value: Any) -> list[SlotGroupRequirement]:
    """兼容 `[['ip', 'mac']]` 和 `[{slots: [...], description: ...}]` 两种写法。"""
    if not isinstance(value, list):
        return []
    groups: list[SlotGroupRequirement] = []
    for item in value:
        if isinstance(item, list):
            slots = _normalize_slot_name_list(item)
            description = ""
        elif isinstance(item, dict):
            slots = _normalize_slot_name_list(item.get("slots", []))
            description = str(item.get("description", ""))
        else:
            continue
        if not slots:
            continue
        groups.append(SlotGroupRequirement(slots=slots, description=description))
    return groups


def _build_conditional_slot_requirements(value: Any) -> list[ConditionalSlotRequirement]:
    """兼容配置层的条件必填规则。"""
    if not isinstance(value, list):
        return []
    rules: list[ConditionalSlotRequirement] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        require = _normalize_slot_name_list(item.get("require", []))
        when_any = _normalize_slot_name_list(item.get("when_any", []))
        when_all = _normalize_slot_name_list(item.get("when_all", []))
        if not require:
            continue
        rules.append(
            ConditionalSlotRequirement(
                require=require,
                when_any=when_any,
                when_all=when_all,
                description=str(item.get("description", "")),
            )
        )
    return rules


def _build_derived_slot_definitions(value: Any) -> list[DerivedSlotDefinition]:
    """兼容列表写法和对象写法。

    支持：
    - [{"slot_name": "healthStatus", ...}]
    - {"healthStatus": {"source_slots": [...], "mapping": {...}}}
    """

    raw_items: list[dict[str, Any]] = []
    if isinstance(value, list):
        raw_items = [dict(item) for item in value if isinstance(item, dict)]
    elif isinstance(value, dict):
        for slot_name, definition in value.items():
            if not isinstance(definition, dict):
                continue
            item = dict(definition)
            item.setdefault("slot_name", slot_name)
            raw_items.append(item)
    definitions: list[DerivedSlotDefinition] = []
    for item in raw_items:
        slot_name = str(item.get("slot_name", "")).strip()
        source_slots = _normalize_slot_name_list(item.get("source_slots", []))
        mapping = item.get("mapping", [])
        if not slot_name or not source_slots:
            continue
        definitions.append(
            DerivedSlotDefinition(
                slot_name=slot_name,
                source_slots=source_slots,
                mapping=mapping if isinstance(mapping, (list, dict)) else [],
                default=item.get("default"),
                key_separator=str(item.get("key_separator", ".") or "."),
                overwrite=bool(item.get("overwrite", False)),
                require_all_sources=bool(item.get("require_all_sources", True)),
            )
        )
    return definitions


def _normalize_slot_name_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(slot).strip() for slot in value if str(slot).strip()]
