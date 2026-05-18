"""Compact LLM routing helpers for template capability."""

from template_capability.LLM.compact_router import (
    COMPACT_SLOT_ROUTER_SYSTEM_PROMPT,
    build_compact_router_payload,
    mock_compact_llm_call,
    route_with_compact_llm,
)

__all__ = [
    "COMPACT_SLOT_ROUTER_SYSTEM_PROMPT",
    "build_compact_router_payload",
    "mock_compact_llm_call",
    "route_with_compact_llm",
]
