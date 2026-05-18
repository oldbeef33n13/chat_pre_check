from __future__ import annotations

from template_capability.config import TemplateConfig
from template_capability.engine import TemplateCapabilityEngine
from template_capability.models import (
    DerivedSlotDefinition,
    MatcherSettings,
    SlotExtractorDefinition,
    TemplateDefinition,
)


def base_settings(**overrides: object) -> MatcherSettings:
    payload = {
        "match_threshold": 0.25,
        "ambiguity_margin": 0.03,
        "recall_top_k": 10,
        "weights": {
            "lexical": 0.1,
            "sample": 0.0,
            "vector": 0.0,
            "fusion": 0.0,
            "slot_fit": 0.35,
            "constraint": 0.2,
            "structure": 0.35,
        },
        "template_route_min_score": 0.25,
        "template_route_min_structure_score": 0.0,
    }
    payload.update(overrides)
    return MatcherSettings(**payload)


def test_keyword_value_can_choose_longest_case_for_overlapping_terms():
    engine = TemplateCapabilityEngine(
        TemplateConfig(
            settings=base_settings(),
            slot_extractors={
                "resource_type": SlotExtractorDefinition(
                    slot_name="resource_type",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "match_policy": "longest",
                            "cases": [
                                {"terms": ["端口"], "value": "generic_port"},
                                {"terms": ["FC端口", "FCoE端口", "SAS端口"], "value": "storage_port"},
                                {"terms": ["以太网端口"], "value": "ethernet_port"},
                            ],
                        }
                    ],
                )
            },
            templates=[
                TemplateDefinition(
                    template_id="resource.port.info",
                    query_mode="metric_query",
                    description="查询端口信息",
                    utterances=["查询端口信息", "查询以太网端口信息"],
                    required_slots=["resource_type"],
                    optional_slots=[],
                    must_terms=[["端口"], ["信息"]],
                    negative_terms=[],
                    slot_constraints={},
                )
            ],
        )
    )

    slots = engine.extract_slots("resource.port.info", "查询以太网端口信息")

    assert slots["resource_type"] == "ethernet_port"


def build_health_engine() -> TemplateCapabilityEngine:
    return TemplateCapabilityEngine(
        TemplateConfig(
            settings=base_settings(),
            slot_extractors={
                "resource_type": SlotExtractorDefinition(
                    slot_name="resource_type",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "match_policy": "longest",
                            "cases": [
                                {"terms": ["存储池"], "value": "storage_pool"},
                                {"terms": ["硬盘"], "value": "disk"},
                            ],
                        }
                    ],
                ),
                "health_status_label": SlotExtractorDefinition(
                    slot_name="health_status_label",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [
                                {"terms": ["正常"], "value": "normal"},
                                {"terms": ["故障"], "value": "fault"},
                            ],
                        }
                    ],
                ),
                "query_operator": SlotExtractorDefinition(
                    slot_name="query_operator",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [{"terms": ["列表"], "value": "list"}],
                        }
                    ],
                ),
            },
            templates=[
                TemplateDefinition(
                    template_id="resource.health.list",
                    query_mode="metric_query",
                    description="按资源类型和健康状态查询列表",
                    utterances=["查询硬盘健康状态为正常的列表", "查询存储池健康状态为正常的列表"],
                    required_slots=["resource_type", "health_status_label", "healthStatus", "query_operator"],
                    optional_slots=[],
                    must_terms=[["健康状态"], ["列表"]],
                    negative_terms=[],
                    slot_constraints={"query_operator": ["list"]},
                    derived_slots=[
                        DerivedSlotDefinition(
                            slot_name="healthStatus",
                            source_slots=["resource_type", "health_status_label"],
                            mapping=[
                                {
                                    "when": {"resource_type": "storage_pool", "health_status_label": "normal"},
                                    "value": 1,
                                },
                                {
                                    "when": {"resource_type": "disk", "health_status_label": "normal"},
                                    "value": 2,
                                },
                                {
                                    "when": {"resource_type": "storage_pool", "health_status_label": "fault"},
                                    "value": 2,
                                },
                                {
                                    "when": {"resource_type": "disk", "health_status_label": "fault"},
                                    "value": 1,
                                },
                            ],
                        )
                    ],
                )
            ],
        )
    )


def test_dependent_slot_mapping_uses_resource_type_and_status_label():
    engine = build_health_engine()

    disk_payload = engine.match("查询硬盘健康状态为正常的列表").to_dict()
    pool_payload = engine.match("查询存储池健康状态为正常的列表").to_dict()

    assert disk_payload["status"] == "matched"
    assert disk_payload["slots"]["healthStatus"] == 2
    assert pool_payload["status"] == "matched"
    assert pool_payload["slots"]["healthStatus"] == 1


def test_route_for_sql_returns_nl2sql_when_template_requirements_are_not_satisfied():
    decision = build_health_engine().route_for_sql("查询硬盘健康状态为未知的列表").to_dict()

    assert decision["route_to"] == "nl2sql"
    assert decision["reason"] == "template_requirements_not_satisfied"


def test_time_range_extractor_handles_relative_numeric_time_without_template_cases():
    engine = TemplateCapabilityEngine(
        TemplateConfig(
            settings=base_settings(),
            slot_extractors={
                "time_range": SlotExtractorDefinition(
                    slot_name="time_range",
                    extractors=[{"type": "time_range"}],
                ),
                "query_operator": SlotExtractorDefinition(
                    slot_name="query_operator",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [{"terms": ["列表"], "value": "list"}],
                        }
                    ],
                ),
            },
            templates=[
                TemplateDefinition(
                    template_id="storage.list.by_time",
                    query_mode="metric_query",
                    description="按时间查询分布式存储列表",
                    utterances=["查询最近3天分布式存储列表"],
                    required_slots=["time_range", "query_operator"],
                    optional_slots=[],
                    must_terms=[["分布式存储"], ["列表"]],
                    negative_terms=[],
                    slot_constraints={"query_operator": ["list"]},
                )
            ],
        )
    )

    payload = engine.match("查询最近3天分布式存储列表").to_dict()

    assert payload["status"] == "matched"
    assert payload["slots"]["time_range"] == {
        "mode": "relative",
        "value": 3,
        "unit": "day",
        "preset": "last_3d",
    }


def build_metric_condition_engine() -> TemplateCapabilityEngine:
    metric_conditions = SlotExtractorDefinition(
        slot_name="metric_conditions",
        extractors=[
            {
                "type": "metric_conditions",
                "metrics": [
                    {"terms": ["内存利用率", "内存"], "value": "memory_usage"},
                    {"terms": ["cpu利用率", "cpu"], "value": "cpu_usage"},
                ],
            }
        ],
    )
    return TemplateCapabilityEngine(
        TemplateConfig(
            settings=base_settings(),
            slot_extractors={
                "metric_conditions": metric_conditions,
                "query_operator": SlotExtractorDefinition(
                    slot_name="query_operator",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [{"terms": ["列表"], "value": "list"}],
                        }
                    ],
                ),
            },
            templates=[
                TemplateDefinition(
                    template_id="storage.single_metric.list",
                    query_mode="metric_query",
                    description="查询单一指标超过阈值的分布式存储列表",
                    utterances=["查询内存利用率大于50的分布式存储列表"],
                    required_slots=["query_operator", "metric_conditions"],
                    optional_slots=[],
                    must_terms=[["分布式存储"], ["列表"]],
                    negative_terms=[],
                    slot_constraints={"query_operator": ["list"]},
                    slot_validations={"metric_conditions": {"exact_items": 1}},
                ),
                TemplateDefinition(
                    template_id="storage.multi_metric.list",
                    query_mode="metric_query",
                    description="查询多个指标同时超过阈值的分布式存储列表",
                    utterances=["查询内存利用率大于50 cpu利用率大于90的分布式存储列表"],
                    required_slots=["query_operator", "metric_conditions"],
                    optional_slots=[],
                    must_terms=[["分布式存储"], ["列表"]],
                    negative_terms=[],
                    slot_constraints={"query_operator": ["list"]},
                    slot_validations={"metric_conditions": {"min_items": 2}},
                ),
            ],
        )
    )


def test_single_and_multi_metric_templates_are_separated_by_condition_cardinality():
    engine = build_metric_condition_engine()

    single = engine.match("查询今天内存利用率大于50%的分布式存储列表").to_dict()
    multi = engine.match("查询今天内存利用率大于50% cpu利用率大于90%的分布式存储列表").to_dict()

    assert single["template_id"] == "storage.single_metric.list"
    assert single["status"] == "matched"
    assert len(single["slots"]["metric_conditions"]) == 1
    assert multi["template_id"] == "storage.multi_metric.list"
    assert multi["status"] == "matched"
    assert len(multi["slots"]["metric_conditions"]) == 2


def test_optional_slot_defaults_are_applied_to_final_template_slots():
    engine = TemplateCapabilityEngine(
        TemplateConfig(
            settings=base_settings(),
            slot_extractors={
                "query_operator": SlotExtractorDefinition(
                    slot_name="query_operator",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [{"terms": ["信息"], "value": "info"}],
                        }
                    ],
                ),
                "sort_by": SlotExtractorDefinition(
                    slot_name="sort_by",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [{"terms": ["按容量排序"], "value": "capacity"}],
                        }
                    ],
                ),
            },
            templates=[
                TemplateDefinition(
                    template_id="storage.pool.info",
                    query_mode="metric_query",
                    description="查询存储池信息",
                    utterances=["查询存储池信息", "查询存储池按容量排序"],
                    required_slots=["query_operator"],
                    optional_slots=["sort_by", "page_no", "page_size", "sort_direction"],
                    must_terms=[["存储池"]],
                    negative_terms=[],
                    slot_constraints={"query_operator": ["info"]},
                    slot_defaults={
                        "page_no": 1,
                        "page_size": 100,
                        "sort_direction": "desc",
                    },
                )
            ],
        )
    )

    no_sort = engine.match("查询存储池信息").to_dict()
    with_sort = engine.match("查询存储池按容量排序信息").to_dict()
    route = engine.route_for_sql("查询存储池信息").to_dict()

    assert no_sort["slots"]["page_no"] == 1
    assert no_sort["slots"]["page_size"] == 100
    assert "sort_by" not in no_sort["slots"]
    assert with_sort["slots"]["sort_by"] == "capacity"
    assert with_sort["slots"]["page_size"] == 100
    assert route["route_to"] == "template"
