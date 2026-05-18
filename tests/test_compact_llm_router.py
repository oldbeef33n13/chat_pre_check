from __future__ import annotations

from typing import Any, Mapping

from template_capability.LLM import (
    COMPACT_SLOT_ROUTER_SYSTEM_PROMPT,
    build_compact_router_payload,
    route_with_compact_llm,
)
from template_capability.config import TemplateConfig
from template_capability.engine import TemplateCapabilityEngine
from template_capability.models import DerivedSlotDefinition, MatcherSettings, SlotExtractorDefinition, TemplateDefinition


def build_demo_engine() -> TemplateCapabilityEngine:
    return TemplateCapabilityEngine(
        TemplateConfig(
            settings=MatcherSettings(
                match_threshold=0.2,
                ambiguity_margin=0.03,
                recall_top_k=10,
                weights={
                    "lexical": 0.1,
                    "sample": 0.0,
                    "vector": 0.0,
                    "fusion": 0.0,
                    "slot_fit": 0.4,
                    "constraint": 0.2,
                    "structure": 0.3,
                },
                template_route_min_score=0.2,
                template_route_top_k=1,
            ),
            slot_extractors={
                "resource_type": SlotExtractorDefinition(
                    slot_name="resource_type",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [{"terms": ["硬盘"], "value": "disk"}],
                        }
                    ],
                ),
                "health_status_label": SlotExtractorDefinition(
                    slot_name="health_status_label",
                    extractors=[
                        {
                            "type": "keyword_value",
                            "cases": [{"terms": ["正常"], "value": "normal"}],
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
                    description="按健康状态查询资源列表",
                    utterances=["查询硬盘健康状态为正常的列表"],
                    required_slots=["resource_type", "health_status_label", "query_operator", "healthStatus"],
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
                                    "when": {"resource_type": "disk", "health_status_label": "normal"},
                                    "value": 2,
                                }
                            ],
                        )
                    ],
                )
            ],
        )
    )


def test_compact_router_payload_omits_large_static_template_fields():
    payload = build_compact_router_payload(
        build_demo_engine(),
        "查询硬盘健康状态为正常的列表",
        top_k=1,
    )

    candidate = payload["c"][0]
    assert payload["q"] == "查询硬盘健康状态为正常的列表"
    assert candidate["id"] == "resource.health.list"
    assert "description" not in candidate
    assert "utterances" not in candidate
    assert "must_terms" not in candidate
    assert candidate["derive"] == {"healthStatus": ["resource_type", "health_status_label"]}


def test_route_with_compact_llm_accepts_stub_decision_and_runs_deterministic_validation():
    def stub_llm(system_prompt: str, user_payload: dict[str, Any]) -> Mapping[str, Any]:
        assert "输出格式固定" in system_prompt
        assert user_payload["c"][0]["id"] == "resource.health.list"
        return {
            "r": "template",
            "id": "resource.health.list",
            "s": {
                "resource_type": "disk",
                "health_status_label": "normal",
                "query_operator": "list",
            },
            "m": [],
            "cf": 0.92,
        }

    decision = route_with_compact_llm(
        build_demo_engine(),
        "查询硬盘健康状态为正常的列表",
        llm_call=stub_llm,
    ).to_dict()

    assert decision["route_to"] == "template"
    assert decision["result"]["template_id"] == "resource.health.list"
    assert decision["result"]["slots"]["healthStatus"] == 2
    assert decision["result"]["trace"]["reason"] == "compact_llm_template_satisfied"


def test_route_with_compact_llm_mock_call_falls_back_to_nl2sql_cleanly():
    decision = route_with_compact_llm(
        build_demo_engine(),
        "查询硬盘健康状态为正常的列表",
    ).to_dict()

    assert decision["route_to"] == "nl2sql"
    assert decision["reason"] == "llm_no_decision"
    assert COMPACT_SLOT_ROUTER_SYSTEM_PROMPT.startswith("你是模板路由后的快速提参器")
