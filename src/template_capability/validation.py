from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from template_capability.models import (
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_VECTOR_PROVIDER,
    SUPPORTED_KEYWORD_MATCH_POLICIES,
    SUPPORTED_TEXT_MATCH_MODES,
    template_declared_slot_names,
)

if TYPE_CHECKING:
    from template_capability.config import TemplateConfig
    from template_capability.models import MatcherSettings, SlotExtractorDefinition, TemplateDefinition, TextMatchRule


SUPPORTED_EXTRACTOR_TYPES = {"keyword_value", "regex", "time_range", "metric_conditions"}
SUPPORTED_VECTOR_PROVIDERS = {"local_tfidf", "tfidf", "hashing", "local_hash", "remote"}
SUPPORTED_RERANKER_PROVIDERS = {"none", "term_overlap", "openai_compatible", "remote"}


class TemplateConfigValidationError(ValueError):
    """聚合配置校验错误，避免开发者一次只修一个问题。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def validate_template_payload(payload: dict[str, Any]) -> None:
    """在 JSON 归一化前做一轮严格校验。"""
    errors: list[str] = []
    templates = payload.get("templates", [])
    template_ids = [
        str(item.get("template_id", "")).strip()
        for item in templates
        if isinstance(item, dict)
    ]
    errors.extend(_duplicate_template_id_errors(template_ids))

    _validate_vector_payload(payload.get("matcher", {}).get("vector", {}), errors)
    _validate_reranker_payload(payload.get("matcher", {}).get("reranker", {}), errors)
    _validate_raw_text_match_rules(payload.get("matcher", {}).get("blocked_terms", []), "matcher.blocked_terms", errors)
    _validate_raw_slot_map(payload.get("slot_extractors", {}), "root slot_extractors", errors)

    for index, item in enumerate(templates, start=1):
        if not isinstance(item, dict):
            errors.append(f"templates[{index}] must be an object.")
            continue
        template_id = str(item.get("template_id", "")).strip() or f"<missing:{index}>"
        _validate_raw_template(item, template_id, errors)

    if errors:
        raise TemplateConfigValidationError(errors)


def validate_template_config(config: "TemplateConfig") -> None:
    """对运行期配置对象做校验，覆盖直接手写 dataclass 的场景。"""
    errors: list[str] = []
    errors.extend(_duplicate_template_id_errors([template.template_id for template in config.templates]))
    _validate_vector_settings(config.settings, errors)
    _validate_reranker_settings(config.settings, errors)
    _validate_text_match_rules(config.settings.blocked_terms, "matcher.blocked_terms", errors)
    _validate_slot_map(config.slot_extractors, "root slot_extractors", errors)

    for template in config.templates:
        _validate_template(template, config, errors)

    if errors:
        raise TemplateConfigValidationError(errors)


def _validate_raw_template(item: dict[str, Any], template_id: str, errors: list[str]) -> None:
    required_slots = _normalize_slot_name_list(item.get("required_slots", []))
    optional_slots = _normalize_slot_name_list(item.get("optional_slots", []))
    _validate_slot_overlap(required_slots, optional_slots, template_id, errors)
    _validate_raw_must_terms(item.get("must_terms", []), template_id, errors)
    _validate_raw_text_match_rules(item.get("negative_terms", []), f"template {template_id}.negative_terms", errors)
    _validate_raw_slot_map(item.get("slot_extractors", {}), f"template {template_id}", errors)
    _validate_raw_slot_groups(item.get("required_one_of", []), template_id, "required_one_of", errors)
    _validate_raw_conditional_rules(item.get("conditional_required", []), template_id, errors)
    _validate_raw_slot_groups(item.get("mutually_exclusive_slots", []), template_id, "mutually_exclusive_slots", errors)
    _validate_raw_derived_slots(item.get("derived_slots", []), template_id, errors)
    _validate_raw_slot_validations(item.get("slot_validations", {}), template_id, errors)


def _validate_template(template: "TemplateDefinition", config: "TemplateConfig", errors: list[str]) -> None:
    _validate_slot_overlap(template.required_slots, template.optional_slots, template.template_id, errors)
    _validate_must_terms(template.must_terms, template.template_id, errors)
    _validate_text_match_rules(template.negative_terms, f"template {template.template_id}.negative_terms", errors)
    _validate_slot_map(template.slot_extractors, f"template {template.template_id}", errors)
    _validate_slot_groups(template.required_one_of, template.template_id, "required_one_of", errors)
    _validate_slot_groups(template.mutually_exclusive_slots, template.template_id, "mutually_exclusive_slots", errors)
    _validate_conditional_rules(template, errors)
    _validate_derived_slots(template, errors)
    _validate_slot_validations(template.slot_validations, template.template_id, errors)

    available_slots = set(config.slot_extractors) | set(template.slot_extractors)
    llm_slots = set()
    if bool(template.llm_slot_extraction.get("enabled", False)):
        llm_slots = {str(slot) for slot in template.llm_slot_extraction.get("slots", [])}
    # 模板选择级 fallback 虽然不是主链路，但在开启时仍然可能补齐缺失槽位，
    # 因此不能把这类模板误判为“永远无法 matched”。
    allow_generic_llm_fill = bool(config.settings.llm_fallback_enabled)
    derived_target_slots = {rule.slot_name for rule in template.derived_slots}
    default_slots = set(template.slot_defaults)
    fillable_slots = available_slots | llm_slots | derived_target_slots | default_slots

    for slot_name in template.required_slots:
        if slot_name in fillable_slots or allow_generic_llm_fill:
            continue
        errors.append(
            f"template {template.template_id} requires slot '{slot_name}' but no extractor or llm fill path can fill it."
        )
    for slot_name in template.optional_slots:
        if slot_name in fillable_slots or allow_generic_llm_fill:
            continue
        errors.append(
            f"template {template.template_id} declares optional slot '{slot_name}' but no extractor or llm fill path can fill it."
        )
    for slot_name in template.slot_constraints:
        if slot_name in fillable_slots or allow_generic_llm_fill:
            continue
        errors.append(
            f"template {template.template_id} constrains slot '{slot_name}' but no extractor or llm fill path can fill it."
        )
    explicitly_validated_slots = set(template.required_slots) | set(template.optional_slots) | set(template.slot_constraints)
    rule_only_slots = sorted(template_declared_slot_names(template) - explicitly_validated_slots - set(template.slot_extractors))
    for slot_name in rule_only_slots:
        if slot_name in fillable_slots or allow_generic_llm_fill:
            continue
        errors.append(
            f"template {template.template_id} references slot '{slot_name}' in constraints/rules but no extractor or llm fill path can fill it."
        )


def _validate_raw_must_terms(must_terms: Any, template_id: str, errors: list[str]) -> None:
    if must_terms is None:
        return
    if not isinstance(must_terms, list):
        errors.append(f"template {template_id} must_terms must be a list.")
        return
    for index, group in enumerate(must_terms, start=1):
        if not isinstance(group, list):
            errors.append(f"template {template_id} must_terms[{index}] must be a list.")
            continue
        _validate_raw_text_match_rules(group, f"template {template_id}.must_terms[{index}]", errors)
        if not any(_raw_text_match_term(term) for term in group):
            errors.append(f"template {template_id} must_terms[{index}] cannot be empty.")


def _validate_must_terms(must_terms: list[list["TextMatchRule"]], template_id: str, errors: list[str]) -> None:
    for index, group in enumerate(must_terms, start=1):
        _validate_text_match_rules(group, f"template {template_id}.must_terms[{index}]", errors)
        if not group or not any(rule.term for rule in group):
            errors.append(f"template {template_id} must_terms[{index}] cannot be empty.")


def _validate_raw_slot_groups(value: Any, template_id: str, field_name: str, errors: list[str]) -> None:
    if value in (None, []):
        return
    if not isinstance(value, list):
        errors.append(f"template {template_id} {field_name} must be a list.")
        return
    for index, item in enumerate(value, start=1):
        if isinstance(item, list):
            slots = _normalize_slot_name_list(item)
        elif isinstance(item, dict):
            slots = _normalize_slot_name_list(item.get("slots", []))
        else:
            errors.append(f"template {template_id} {field_name}[{index}] must be a list or object.")
            continue
        if len(slots) < 2:
            errors.append(f"template {template_id} {field_name}[{index}] must contain at least 2 slots.")


def _validate_raw_conditional_rules(value: Any, template_id: str, errors: list[str]) -> None:
    if value in (None, []):
        return
    if not isinstance(value, list):
        errors.append(f"template {template_id} conditional_required must be a list.")
        return
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            errors.append(f"template {template_id} conditional_required[{index}] must be an object.")
            continue
        require = _normalize_slot_name_list(item.get("require", []))
        when_any = _normalize_slot_name_list(item.get("when_any", []))
        when_all = _normalize_slot_name_list(item.get("when_all", []))
        if not require:
            errors.append(f"template {template_id} conditional_required[{index}] must declare require.")
        if not when_any and not when_all:
            errors.append(f"template {template_id} conditional_required[{index}] must declare when_any or when_all.")


def _validate_raw_derived_slots(value: Any, template_id: str, errors: list[str]) -> None:
    if value in (None, [], {}):
        return
    if isinstance(value, dict):
        items = []
        for slot_name, definition in value.items():
            if not isinstance(definition, dict):
                errors.append(f"template {template_id} derived_slots.{slot_name} must be an object.")
                continue
            item = dict(definition)
            item.setdefault("slot_name", slot_name)
            items.append(item)
    elif isinstance(value, list):
        items = value
    else:
        errors.append(f"template {template_id} derived_slots must be a list or object.")
        return
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"template {template_id} derived_slots[{index}] must be an object.")
            continue
        slot_name = str(item.get("slot_name", "")).strip()
        source_slots = _normalize_slot_name_list(item.get("source_slots", []))
        mapping = item.get("mapping", [])
        if not slot_name:
            errors.append(f"template {template_id} derived_slots[{index}] must declare slot_name.")
        if not source_slots:
            errors.append(f"template {template_id} derived_slots[{index}] must declare source_slots.")
        if not isinstance(mapping, (list, dict)):
            errors.append(f"template {template_id} derived_slots[{index}].mapping must be a list or object.")


def _validate_raw_slot_validations(value: Any, template_id: str, errors: list[str]) -> None:
    if value in (None, {}):
        return
    if not isinstance(value, dict):
        errors.append(f"template {template_id} slot_validations must be an object.")
        return
    _validate_slot_validations(value, template_id, errors)


def _validate_raw_slot_map(slot_map: Any, context: str, errors: list[str]) -> None:
    if slot_map in (None, {}):
        return
    if not isinstance(slot_map, dict):
        errors.append(f"{context} must be an object.")
        return
    for slot_name, definition in slot_map.items():
        if not isinstance(definition, dict):
            errors.append(f"{context}.{slot_name} must be an object.")
            continue
        extractors = definition.get("extractors", [])
        if not isinstance(extractors, list):
            errors.append(f"{context}.{slot_name}.extractors must be a list.")
            continue
        _validate_extractors(extractors, f"{context}.{slot_name}", errors)


def _validate_slot_map(
    slot_map: dict[str, "SlotExtractorDefinition"],
    context: str,
    errors: list[str],
) -> None:
    for slot_name, definition in slot_map.items():
        _validate_extractors(definition.extractors, f"{context}.{slot_name}", errors)


def _validate_extractors(extractors: list[dict[str, Any]], context: str, errors: list[str]) -> None:
    for index, extractor in enumerate(extractors, start=1):
        extractor_type = str(extractor.get("type", "")).strip().lower()
        if extractor_type not in SUPPORTED_EXTRACTOR_TYPES:
            errors.append(
                f"{context}.extractors[{index}] uses unsupported extractor type '{extractor_type or '<missing>'}'."
            )
            continue
        if extractor_type == "keyword_value":
            match_policy = str(extractor.get("match_policy", "first") or "first").strip().lower()
            if match_policy not in SUPPORTED_KEYWORD_MATCH_POLICIES:
                errors.append(
                    f"{context}.extractors[{index}] uses unsupported match_policy '{match_policy}'."
                )
            continue
        if extractor_type == "metric_conditions":
            metrics = extractor.get("metrics", [])
            if not isinstance(metrics, list) or not metrics:
                errors.append(f"{context}.extractors[{index}] metric_conditions requires non-empty metrics.")
            continue
        if extractor_type != "regex":
            continue
        for pattern_index, spec in enumerate(extractor.get("patterns", []), start=1):
            pattern = str(spec.get("pattern", ""))
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as error:
                errors.append(
                    f"{context}.extractors[{index}].patterns[{pattern_index}] has invalid regex '{pattern}': {error}."
                )


def _validate_raw_text_match_rules(value: Any, context: str, errors: list[str]) -> None:
    if value in (None, []):
        return
    if not isinstance(value, list):
        errors.append(f"{context} must be a list.")
        return
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            term = item.strip()
            match_mode = "substring"
        elif isinstance(item, dict):
            term = _raw_text_match_term(item)
            match_mode = str(item.get("match_mode", "substring") or "substring").strip().lower()
        else:
            errors.append(f"{context}[{index}] must be a string or object.")
            continue
        if not term:
            errors.append(f"{context}[{index}] term cannot be empty.")
        if match_mode not in SUPPORTED_TEXT_MATCH_MODES:
            errors.append(f"{context}[{index}] uses unsupported match_mode '{match_mode}'.")


def _validate_text_match_rules(value: list["TextMatchRule"], context: str, errors: list[str]) -> None:
    for index, rule in enumerate(value, start=1):
        if not rule.term:
            errors.append(f"{context}[{index}] term cannot be empty.")
        if rule.match_mode not in SUPPORTED_TEXT_MATCH_MODES:
            errors.append(f"{context}[{index}] uses unsupported match_mode '{rule.match_mode}'.")


def _validate_slot_groups(groups: list[Any], template_id: str, field_name: str, errors: list[str]) -> None:
    for index, group in enumerate(groups, start=1):
        slots = list(dict.fromkeys(getattr(group, "slots", [])))
        if len(slots) < 2:
            errors.append(f"template {template_id} {field_name}[{index}] must contain at least 2 distinct slots.")


def _validate_conditional_rules(template: "TemplateDefinition", errors: list[str]) -> None:
    for index, rule in enumerate(template.conditional_required, start=1):
        require = list(dict.fromkeys(rule.require))
        when_any = list(dict.fromkeys(rule.when_any))
        when_all = list(dict.fromkeys(rule.when_all))
        if not require:
            errors.append(f"template {template.template_id} conditional_required[{index}] must declare require.")
        if not when_any and not when_all:
            errors.append(
                f"template {template.template_id} conditional_required[{index}] must declare when_any or when_all."
            )


def _validate_derived_slots(template: "TemplateDefinition", errors: list[str]) -> None:
    for index, rule in enumerate(template.derived_slots, start=1):
        if not rule.slot_name:
            errors.append(f"template {template.template_id} derived_slots[{index}] must declare slot_name.")
        if not rule.source_slots:
            errors.append(f"template {template.template_id} derived_slots[{index}] must declare source_slots.")
        if not isinstance(rule.mapping, (list, dict)):
            errors.append(f"template {template.template_id} derived_slots[{index}].mapping must be a list or object.")


def _validate_slot_validations(value: dict[str, Any], template_id: str, errors: list[str]) -> None:
    for slot_name, rule in value.items():
        if not isinstance(rule, dict):
            errors.append(f"template {template_id} slot_validations.{slot_name} must be an object.")
            continue
        for field_name in ("min_items", "max_items", "exact_items"):
            if field_name not in rule:
                continue
            try:
                if int(rule[field_name]) < 0:
                    errors.append(f"template {template_id} slot_validations.{slot_name}.{field_name} must be >= 0.")
            except (TypeError, ValueError):
                errors.append(f"template {template_id} slot_validations.{slot_name}.{field_name} must be an integer.")


def _validate_vector_payload(vector_payload: Any, errors: list[str]) -> None:
    payload = vector_payload if isinstance(vector_payload, dict) else {}
    provider = str(payload.get("provider", DEFAULT_VECTOR_PROVIDER) or DEFAULT_VECTOR_PROVIDER).strip().lower()
    if provider not in SUPPORTED_VECTOR_PROVIDERS:
        errors.append(f"matcher.vector.provider '{provider}' is not supported.")
        return
    if provider != "remote":
        return
    _validate_remote_vector_options(payload, errors)


def _validate_vector_settings(settings: "MatcherSettings", errors: list[str]) -> None:
    if settings.vector_provider not in SUPPORTED_VECTOR_PROVIDERS:
        errors.append(f"matcher.vector.provider '{settings.vector_provider}' is not supported.")
        return
    if settings.vector_provider != "remote":
        return
    _validate_remote_vector_options(settings.vector_options, errors)


def _validate_reranker_payload(reranker_payload: Any, errors: list[str]) -> None:
    payload = reranker_payload if isinstance(reranker_payload, dict) else {}
    enabled = bool(payload.get("enabled", False))
    provider = str(payload.get("provider", DEFAULT_RERANKER_PROVIDER) or DEFAULT_RERANKER_PROVIDER).strip().lower()
    if provider not in SUPPORTED_RERANKER_PROVIDERS:
        errors.append(f"matcher.reranker.provider '{provider}' is not supported.")
        return
    top_k = payload.get("top_k", 15)
    try:
        if int(top_k) < 1:
            errors.append("matcher.reranker.top_k must be >= 1.")
    except (TypeError, ValueError):
        errors.append("matcher.reranker.top_k must be an integer.")
    if not enabled or provider in {"none", "term_overlap"}:
        return
    _validate_remote_reranker_options(payload, errors)


def _validate_reranker_settings(settings: "MatcherSettings", errors: list[str]) -> None:
    if settings.reranker_provider not in SUPPORTED_RERANKER_PROVIDERS:
        errors.append(f"matcher.reranker.provider '{settings.reranker_provider}' is not supported.")
        return
    if settings.reranker_top_k < 1:
        errors.append("matcher.reranker.top_k must be >= 1.")
    if not settings.reranker_enabled or settings.reranker_provider in {"none", "term_overlap"}:
        return
    _validate_remote_reranker_options(settings.reranker_options, errors)


def _validate_remote_vector_options(options: Any, errors: list[str]) -> None:
    payload = options if isinstance(options, dict) else {}
    if not str(payload.get("base_url", "")).strip():
        errors.append("matcher.vector.remote requires base_url.")
    if not str(payload.get("model", "")).strip():
        errors.append("matcher.vector.remote requires model.")
    api_key = str(payload.get("api_key", "")).strip()
    api_key_env = str(payload.get("api_key_env", "")).strip()
    if not api_key and not api_key_env:
        errors.append("matcher.vector.remote requires api_key or api_key_env.")


def _validate_remote_reranker_options(options: Any, errors: list[str]) -> None:
    payload = options if isinstance(options, dict) else {}
    if not str(payload.get("base_url", "")).strip():
        errors.append("matcher.reranker.remote requires base_url.")
    if not str(payload.get("model", "")).strip():
        errors.append("matcher.reranker.remote requires model.")
    api_key = str(payload.get("api_key", "")).strip()
    api_key_env = str(payload.get("api_key_env", "")).strip()
    if not api_key and not api_key_env:
        errors.append("matcher.reranker.remote requires api_key or api_key_env.")


def _duplicate_template_id_errors(template_ids: list[str]) -> list[str]:
    duplicates = [template_id for template_id, count in Counter(template_ids).items() if template_id and count > 1]
    return [f"duplicate template_id detected: {template_id}" for template_id in sorted(duplicates)]


def _validate_slot_overlap(
    required_slots: list[str],
    optional_slots: list[str],
    template_id: str,
    errors: list[str],
) -> None:
    overlap = sorted(set(required_slots) & set(optional_slots))
    if overlap:
        errors.append(f"template {template_id} has slots listed in both required_slots and optional_slots: {', '.join(overlap)}")


def _normalize_slot_name_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(slot).strip() for slot in value if str(slot).strip()]


def _raw_text_match_term(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("term", value.get("text", ""))).strip()
    return str(value).strip()
