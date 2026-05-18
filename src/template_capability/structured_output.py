from __future__ import annotations

from typing import Any

from template_capability.models import TemplateDefinition


JSON_ANY_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "object", "additionalProperties": True},
        {"type": "array"},
        {"type": "null"},
    ]
}


def build_json_schema_response_format(*, name: str, schema: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """为 OpenAI-compatible chat.completions 构造 `json_schema` 响应格式。"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }


def build_json_object_response_format() -> dict[str, str]:
    """保守兼容模式，要求模型返回 JSON object。"""
    return {"type": "json_object"}


def build_template_selection_schema(candidate_ids: list[str]) -> dict[str, Any]:
    """约束模板选择兜底只能在候选模板里做裁决，或返回 `-1`。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["template_id", "status"],
        "properties": {
            "template_id": {
                "type": "string",
                "enum": sorted(dict.fromkeys([*candidate_ids, "-1"])),
            },
            "status": {
                "type": "string",
                "enum": ["matched", "partial", "unmatched"],
            },
            "slots": {
                "type": "object",
                "additionalProperties": True,
            },
            "missing_slots": {
                "type": "array",
                "items": {"type": "string"},
            },
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reason": {"type": "string"},
        },
    }


def build_slot_fill_schema(template: TemplateDefinition, target_slots: list[str]) -> dict[str, Any]:
    """约束模板级补参只能输出目标槽位。"""
    slot_properties = {
        slot_name: infer_slot_json_schema(template, slot_name)
        for slot_name in target_slots
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["slots"],
        "properties": {
            "slots": {
                "type": "object",
                "additionalProperties": False,
                "properties": slot_properties,
            },
            "reason": {"type": "string"},
        },
    }


def infer_slot_json_schema(template: TemplateDefinition, slot_name: str) -> dict[str, Any]:
    """根据模板约束和 extractor 线索，尽量收窄单个槽位的 JSON schema。"""
    if slot_name == "metric_conditions" or _extractor_type_exists(template, slot_name, "metric_conditions"):
        return {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "required": ["metric", "operator", "value"],
                "properties": {
                    "metric": JSON_ANY_SCHEMA,
                    "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=", "!="]},
                    "value": {"type": "number"},
                    "raw": {"type": "string"},
                },
            },
        }
    allowed_values = template.slot_constraints.get(slot_name, [])
    enum_values = _collect_enum_values(template, slot_name, allowed_values)
    if enum_values:
        return {"enum": enum_values}

    inferred_regex_type = _infer_regex_value_type(template, slot_name)
    if inferred_regex_type == "integer":
        return _numeric_schema("integer", template, slot_name)
    if inferred_regex_type == "number":
        return _numeric_schema("number", template, slot_name)
    if inferred_regex_type == "boolean":
        return {"type": "boolean"}
    if inferred_regex_type == "string":
        return {"type": "string"}
    return dict(JSON_ANY_SCHEMA)


def _collect_enum_values(
    template: TemplateDefinition,
    slot_name: str,
    allowed_values: list[Any],
) -> list[Any]:
    enum_values: list[Any] = []
    for value in allowed_values:
        _append_unique_json_value(enum_values, value)
    definition = template.slot_extractors.get(slot_name)
    if definition is None:
        return enum_values
    for extractor in definition.extractors:
        if str(extractor.get("type", "")).strip().lower() != "keyword_value":
            continue
        for case in extractor.get("cases", []):
            if not isinstance(case, dict) or "value" not in case:
                continue
            _append_unique_json_value(enum_values, case.get("value"))
    return enum_values


def _infer_regex_value_type(template: TemplateDefinition, slot_name: str) -> str | None:
    definition = template.slot_extractors.get(slot_name)
    if definition is None:
        return None
    type_priority = {
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "string": "string",
    }
    for extractor in definition.extractors:
        if str(extractor.get("type", "")).strip().lower() != "regex":
            continue
        for pattern in extractor.get("patterns", []):
            if not isinstance(pattern, dict):
                continue
            mapped = type_priority.get(str(pattern.get("value_type", "")).strip().lower())
            if mapped is not None:
                return mapped
    return None


def _extractor_type_exists(template: TemplateDefinition, slot_name: str, extractor_type: str) -> bool:
    definition = template.slot_extractors.get(slot_name)
    if definition is None:
        return False
    expected = extractor_type.strip().lower()
    return any(str(extractor.get("type", "")).strip().lower() == expected for extractor in definition.extractors)


def _numeric_schema(kind: str, template: TemplateDefinition, slot_name: str) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": kind}
    definition = template.slot_extractors.get(slot_name)
    if definition is None:
        return schema
    mins: list[float] = []
    maxs: list[float] = []
    for extractor in definition.extractors:
        if str(extractor.get("type", "")).strip().lower() != "regex":
            continue
        for pattern in extractor.get("patterns", []):
            if not isinstance(pattern, dict):
                continue
            try:
                if pattern.get("min") not in (None, ""):
                    mins.append(float(pattern["min"]))
                if pattern.get("max") not in (None, ""):
                    maxs.append(float(pattern["max"]))
            except (TypeError, ValueError):
                continue
    if mins:
        schema["minimum"] = min(mins)
    if maxs:
        schema["maximum"] = max(maxs)
    return schema


def _append_unique_json_value(target: list[Any], value: Any) -> None:
    for existing in target:
        if existing == value:
            return
    target.append(value)
