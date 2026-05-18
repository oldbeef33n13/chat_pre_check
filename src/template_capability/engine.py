from __future__ import annotations

import copy
import json
import os

from template_capability.config import TemplateConfig, load_template_config
from template_capability.extractors import build_slot_registry, normalize_text
from template_capability.fallback import (
    FallbackSuggestion,
    LLMFallbackResolver,
    LLMTemplateSlotResolver,
)
from template_capability.models import (
    DerivedSlotDefinition,
    MatchResult,
    MatchStatus,
    SlotExtractorDefinition,
    TemplateCandidate,
    TemplateDefinition,
    TemplateRouteDecision,
    template_declared_slot_names,
)
from template_capability.rerankers import (
    NoopTemplateReranker,
    OpenAICompatibleTemplateReranker,
    TemplateReranker,
    TermOverlapTemplateReranker,
    resolve_reranker_api_key,
)
from template_capability.rewrite import QueryRewriteStateMachine
from template_capability.scoring import (
    BM25FieldIndex,
    adaptive_score_weights,
    build_utterance_term_sets,
    constraint_score,
    evaluate_slot_requirements,
    evaluate_structural_alignment,
    has_negative_term,
    missing_required_slots,
    mixed_terms,
    normalize_candidate_scores,
    reciprocal_rank_fusion,
    sample_similarity_from_terms,
    slot_fit_score,
    weighted_score,
)
from template_capability.text_matching import match_text_rule
from template_capability.vector_index import (
    InMemoryVectorIndex,
    LocalHashVectorProvider,
    LocalTfidfVectorProvider,
    RemoteEmbeddingProvider,
    VectorSearchBackend,
)
from template_capability.validation import validate_template_config


HIGH_SIGNAL_EXACT_SLOTS = {
    "query_operator",
    "topn",
    "selector_type",
    "selector_value",
    "time_range",
    "region_id",
    "resource_type",
    "health_status_label",
    "metric_conditions",
}


class TemplateCapabilityEngine:
    """面向问数场景的通用模板匹配能力。

    主链路分两段：
    1. 先只基于 query 文本做模板召回和粗排。
    2. 对候选模板分别按“模板自己的 extractor”抽参，再做精排和可选补参。
    """

    def __init__(
        self,
        config: TemplateConfig,
        vector_backend: VectorSearchBackend | None = None,
        template_reranker: TemplateReranker | None = None,
        llm_fallback_resolver: LLMFallbackResolver | None = None,
        llm_template_slot_resolver: LLMTemplateSlotResolver | None = None,
    ) -> None:
        # 直接构造 dataclass 配置时，也要走同一套校验，避免坏配置绕过 load_template_config。
        validate_template_config(config)
        self.config = config
        self.llm_fallback_resolver = llm_fallback_resolver
        self.llm_template_slot_resolver = llm_template_slot_resolver
        self.query_rewriter = QueryRewriteStateMachine(config.query_rewrite)
        self.slot_registry = build_slot_registry(config.slot_extractors)
        # 这层 registry 只抽高信号公共槽位，专门用于结构判断，不直接决定模板命中。
        self.global_slot_registry = build_slot_registry(self._build_global_signal_slot_definitions())
        self.templates = {template.template_id: template for template in config.templates}
        # 每个模板都维护一套独立的槽位抽取器。这样模板之间可以复用槽位名，
        # 但不需要共享完全一致的提参逻辑。
        self.template_slot_registries = {
            template.template_id: build_slot_registry(self._build_template_slot_definitions(template))
            for template in config.templates
        }
        # 下面这些结构都是预计算索引，避免在线匹配时重复处理模板文本。
        self.template_documents = {
            template.template_id: self._build_template_document(template)
            for template in config.templates
        }
        self.template_lexical_fields = {
            template.template_id: self._build_template_fields(template)
            for template in config.templates
        }
        self.template_sample_terms = {
            template.template_id: build_utterance_term_sets(template.utterances)
            for template in config.templates
        }
        self.lexical_index = BM25FieldIndex(
            self.template_lexical_fields,
            config.settings.lexical_field_weights,
        )
        self.vector_backend = vector_backend or _build_configured_vector_backend(config)
        self.template_reranker = template_reranker or _build_configured_template_reranker(config)
        self.vector_backend.build(self.template_documents)

    @classmethod
    def from_file(cls, path: str) -> "TemplateCapabilityEngine":
        return cls(load_template_config(path))

    def match(self, input_text: str) -> MatchResult:
        """执行一次完整匹配。

        返回时只暴露一个最终模板；如果分数不够或歧义太高，则直接返回 `-1`。
        """
        original_norm_text = normalize_text(input_text)
        rewrite_result = self.query_rewriter.rewrite_normalized(original_norm_text)
        norm_text = rewrite_result.rewritten_text
        # 共享 extractor 只承担非常轻的公共补充作用，真正决定模板是否成立，
        # 仍然以后续“模板内提参”的结果为准。
        shared_slots = self.slot_registry.extract(norm_text)
        global_slots = self.global_slot_registry.extract(norm_text)
        diagnostic_slots = _merge_slot_maps(shared_slots, global_slots)
        # 非问数意图先全局拦截，避免后面把“报告/分析/预测”误打到任何模板上。
        blocked_term = self._match_blocked_term(norm_text)
        if blocked_term is not None:
            return MatchResult(
                template_id=-1,
                status=MatchStatus.UNMATCHED,
                score=0.0,
                query_mode=None,
                slots=diagnostic_slots,
                missing_slots=[],
                trace={
                    "norm_text": norm_text,
                    "original_norm_text": original_norm_text,
                    "shared_slots": shared_slots,
                    "global_slots": global_slots,
                    "rewrite_trace": rewrite_result.to_dict(),
                    "blocked_term": blocked_term,
                    "reason": "blocked_intent",
                },
            )
        # 只有通过全局拦截后，才进入多路召回和精排。
        ranked = self._rank_templates(norm_text, global_slots)
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None

        if (
            top is None
            or top.score < self.config.settings.match_threshold
            or self._is_ambiguous(top, second)
        ):
            # 当规则链路无法稳定选出模板时，才把控制权交给更贵的模板选择级 fallback。
            fallback_result = self._resolve_with_fallback(
                input_text=input_text,
                norm_text=norm_text,
                slots=diagnostic_slots,
                ranked=ranked,
                missing_slots=[],
                base_status=MatchStatus.UNMATCHED,
            )
            if fallback_result is not None:
                fallback_result.trace.update(
                    {
                        "original_norm_text": original_norm_text,
                        "shared_slots": shared_slots,
                        "global_slots": global_slots,
                        "rewrite_trace": rewrite_result.to_dict(),
                    }
                )
                return fallback_result
            return MatchResult(
                template_id=-1,
                status=MatchStatus.UNMATCHED,
                score=top.score if top is not None else 0.0,
                query_mode=None,
                slots=diagnostic_slots,
                missing_slots=[],
                trace={
                    "norm_text": norm_text,
                    "original_norm_text": original_norm_text,
                    "shared_slots": shared_slots,
                    "global_slots": global_slots,
                    "rewrite_trace": rewrite_result.to_dict(),
                    "top_candidates": [candidate.to_dict() for candidate in ranked[:5]],
                    "threshold": self.config.settings.match_threshold,
                    "ambiguity_margin": self.config.settings.ambiguity_margin,
                },
            )

        template = self.templates[top.template_id]
        # top 已经是当前最优模板，先基于规则抽参判断它是 matched 还是 partial。
        slots = dict(top.slots)
        requirement_report = evaluate_slot_requirements(template, slots)
        missing_slots = requirement_report.missing_slots
        status = MatchStatus.MATCHED if requirement_report.is_satisfied else MatchStatus.PARTIAL
        # 模板已经稳定命中后，才允许用更贵的 LLM 做定向补参。
        top = self._resolve_template_slots_with_fallback(
            input_text=input_text,
            norm_text=norm_text,
            global_slots=global_slots,
            candidate=top,
            status=status,
        )
        top = self._rebuild_candidate(
            top,
            norm_text=norm_text,
            global_slots=global_slots,
            include_defaults=True,
        )
        # 如果 top1 因为补参而发生变化，需要让 trace 里的 top_candidates
        # 也反映补参后的最新 top1，而不是旧版本。
        display_candidates = [top] + [
            candidate
            for candidate in ranked
            if candidate.template_id != top.template_id
        ]
        slots = dict(top.slots)
        requirement_report = evaluate_slot_requirements(template, slots)
        missing_slots = requirement_report.missing_slots
        status = MatchStatus.MATCHED if requirement_report.is_satisfied else MatchStatus.PARTIAL
        if status is MatchStatus.PARTIAL:
            # 已经知道 top1 是哪个模板，但它还缺关键槽位时，
            # 允许模板选择级 fallback 再看一次是否应该直接换模板或返回 -1。
            fallback_result = self._resolve_with_fallback(
                input_text=input_text,
                norm_text=norm_text,
                slots=_merge_slot_maps(shared_slots, global_slots, slots),
                ranked=ranked,
                missing_slots=missing_slots,
                base_status=status,
            )
            if fallback_result is not None:
                fallback_result.trace.update(
                    {
                        "original_norm_text": original_norm_text,
                        "shared_slots": shared_slots,
                        "global_slots": global_slots,
                        "rewrite_trace": rewrite_result.to_dict(),
                    }
                )
                return fallback_result
        return MatchResult(
            template_id=template.template_id,
            status=status,
            score=top.score,
            query_mode=template.query_mode,
            slots=slots,
            missing_slots=missing_slots,
            metadata=template.metadata,
            trace={
                "norm_text": norm_text,
                "original_norm_text": original_norm_text,
                "shared_slots": shared_slots,
                "global_slots": global_slots,
                "rewrite_trace": rewrite_result.to_dict(),
                "requirement_issues": requirement_report.to_dict()["issues"],
                "blocking_issue_count": len(requirement_report.issues),
                "selected_template": top.to_dict(),
                "top_candidates": [candidate.to_dict() for candidate in display_candidates[:5]],
            },
        )

    def route(self, input_text: str) -> MatchResult:
        """兼容旧入口。"""
        return self.match(input_text)

    def route_for_sql(self, input_text: str) -> TemplateRouteDecision:
        """面向“模板 SQL vs NL2SQL”的高置信路由入口。

        这个入口比 `match()` 更保守：
        - 快速召回分数不够，直接 `nl2sql`
        - 模板结构分不够，直接 `nl2sql`
        - 经过默认值、派生槽位和可选 LLM 补参后仍不满足模板，直接 `nl2sql`
        """

        original_norm_text = normalize_text(input_text)
        rewrite_result = self.query_rewriter.rewrite_normalized(original_norm_text)
        norm_text = rewrite_result.rewritten_text
        shared_slots = self.slot_registry.extract(norm_text)
        global_slots = self.global_slot_registry.extract(norm_text)
        diagnostic_slots = _merge_slot_maps(shared_slots, global_slots)
        blocked_term = self._match_blocked_term(norm_text)
        if blocked_term is not None:
            result = MatchResult(
                template_id=-1,
                status=MatchStatus.UNMATCHED,
                score=0.0,
                query_mode=None,
                slots=diagnostic_slots,
                trace={
                    "norm_text": norm_text,
                    "original_norm_text": original_norm_text,
                    "rewrite_trace": rewrite_result.to_dict(),
                    "blocked_term": blocked_term,
                    "route_to": "nl2sql",
                    "reason": "blocked_intent",
                },
            )
            return TemplateRouteDecision(route_to="nl2sql", result=result, reason="blocked_intent")

        ranked = self._rank_templates(norm_text, global_slots)
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        if top is None:
            return self._build_nl2sql_route_result(
                reason="no_template_candidate",
                norm_text=norm_text,
                original_norm_text=original_norm_text,
                rewrite_trace=rewrite_result.to_dict(),
                slots=diagnostic_slots,
                ranked=ranked,
            )
        if top.score < self.config.settings.template_route_min_score:
            return self._build_nl2sql_route_result(
                reason="low_template_confidence",
                norm_text=norm_text,
                original_norm_text=original_norm_text,
                rewrite_trace=rewrite_result.to_dict(),
                slots=diagnostic_slots,
                ranked=ranked,
            )
        if self._is_ambiguous(top, second):
            return self._build_nl2sql_route_result(
                reason="ambiguous_template_candidates",
                norm_text=norm_text,
                original_norm_text=original_norm_text,
                rewrite_trace=rewrite_result.to_dict(),
                slots=diagnostic_slots,
                ranked=ranked,
            )

        evaluated_candidates: list[TemplateCandidate] = []
        for candidate in ranked[: self.config.settings.template_route_top_k]:
            if candidate.score < self.config.settings.template_route_min_score:
                continue
            if candidate.structure_score < self.config.settings.template_route_min_structure_score:
                evaluated_candidates.append(candidate)
                continue
            template = self.templates[candidate.template_id]
            requirement_report = evaluate_slot_requirements(template, candidate.slots)
            status = MatchStatus.MATCHED if requirement_report.is_satisfied else MatchStatus.PARTIAL
            enriched = self._resolve_template_slots_with_fallback(
                input_text=input_text,
                norm_text=norm_text,
                global_slots=global_slots,
                candidate=candidate,
                status=status,
            )
            finalized = self._rebuild_candidate(
                enriched,
                norm_text=norm_text,
                global_slots=global_slots,
                include_defaults=True,
            )
            evaluated_candidates.append(finalized)
            final_report = evaluate_slot_requirements(template, finalized.slots)
            if (
                final_report.is_satisfied
                and finalized.score >= self.config.settings.template_route_min_score
                and finalized.structure_score >= self.config.settings.template_route_min_structure_score
            ):
                result = MatchResult(
                    template_id=template.template_id,
                    status=MatchStatus.MATCHED,
                    score=finalized.score,
                    query_mode=template.query_mode,
                    slots=finalized.slots,
                    missing_slots=[],
                    metadata=template.metadata,
                    trace={
                        "norm_text": norm_text,
                        "original_norm_text": original_norm_text,
                        "shared_slots": shared_slots,
                        "global_slots": global_slots,
                        "rewrite_trace": rewrite_result.to_dict(),
                        "route_to": "template",
                        "reason": "template_satisfied",
                        "requirement_issues": [],
                        "selected_template": finalized.to_dict(),
                        "top_candidates": [item.to_dict() for item in evaluated_candidates[:5]],
                    },
                )
                return TemplateRouteDecision(
                    route_to="template",
                    result=result,
                    reason="template_satisfied",
                    candidates=evaluated_candidates,
                )

        return self._build_nl2sql_route_result(
            reason="template_requirements_not_satisfied",
            norm_text=norm_text,
            original_norm_text=original_norm_text,
            rewrite_trace=rewrite_result.to_dict(),
            slots=diagnostic_slots,
            ranked=evaluated_candidates or ranked,
        )

    def _rank_templates(
        self,
        norm_text: str,
        global_slots: dict[str, object],
    ) -> list[TemplateCandidate]:
        """先召回候选模板，再按模板内抽参结果做精排。"""
        query_term_set = set(mixed_terms(norm_text))
        # 三路召回各自关注不同信号：
        # 1. lexical: 领域词、关键词命中
        # 2. vector: 粗语义接近度
        # 3. sample: 对模板示例问法的贴近程度
        lexical_recall = self.lexical_index.search(norm_text, self.config.settings.recall_top_k)
        vector_recall = self.vector_backend.search(norm_text, self.config.settings.recall_top_k)
        sample_recall = self._sample_recall(query_term_set)

        # 不同召回路的原始分数不在同一量纲上，先各自压到 0-1 再参与融合。
        lexical_scores = normalize_candidate_scores(lexical_recall)
        vector_scores = normalize_candidate_scores(vector_recall)
        sample_scores = normalize_candidate_scores(sample_recall)
        fusion_scores = reciprocal_rank_fusion(
            [lexical_recall, vector_recall, sample_recall],
            rrf_k=self.config.settings.fusion_rrf_k,
        )
        # 候选池取三路召回和融合排序的并集，避免任何单一路召回把好模板漏掉。
        candidate_ids = set(lexical_scores) | set(vector_scores) | set(sample_scores) | set(fusion_scores)
        preliminary_ranked: list[TemplateCandidate] = []
        base_weight_template = _weights_without_score_part(self.config.settings.weights, "rerank")

        for template_id in candidate_ids:
            # 进入精排后才按模板自己的规则提参。
            slots = self.extract_slots(template_id, norm_text)
            # query 条件越多，越要提高 slot/structure 的权重，
            # 否则单条件模板会因为文本更像而挤掉多条件模板。
            base_weights = adaptive_score_weights(
                base_weight_template,
                _merge_slot_maps(global_slots, slots),
            )
            preliminary_ranked.append(
                self._build_candidate(
                    template_id=template_id,
                    norm_text=norm_text,
                    global_slots=global_slots,
                    lexical_score=lexical_scores.get(template_id, 0.0),
                    sample_score=sample_scores.get(template_id, 0.0),
                    vector_score=vector_scores.get(template_id, 0.0),
                    fusion_score=fusion_scores.get(template_id, 0.0),
                    slots=slots,
                    weights=base_weights,
                )
            )

        preliminary_ranked.sort(key=lambda candidate: candidate.score, reverse=True)
        rerank_scores = self.template_reranker.rerank(
            normalized_text=norm_text,
            candidates=preliminary_ranked[: self.config.settings.reranker_top_k]
            if self.config.settings.reranker_enabled
            else [],
            templates=self.templates,
            template_documents=self.template_documents,
        )
        if not rerank_scores:
            return preliminary_ranked

        reranked: list[TemplateCandidate] = []
        for candidate in preliminary_ranked:
            rerank_result = rerank_scores.get(candidate.template_id)
            rerank_score = rerank_result.score if rerank_result is not None else 0.0
            rerank_trace = rerank_result.trace if rerank_result is not None else {}
            rerank_weights = adaptive_score_weights(
                self.config.settings.weights,
                _merge_slot_maps(global_slots, candidate.slots),
            )
            reranked.append(
                self._build_candidate(
                    template_id=candidate.template_id,
                    norm_text=norm_text,
                    global_slots=global_slots,
                    lexical_score=candidate.lexical_score,
                    sample_score=candidate.sample_score,
                    vector_score=candidate.vector_score,
                    fusion_score=candidate.fusion_score,
                    rerank_score=rerank_score,
                    slots=candidate.slots,
                    weights=rerank_weights,
                    rerank_trace=rerank_trace,
                )
            )
        reranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return reranked

    def _build_template_document(self, template: TemplateDefinition) -> str:
        """生成向量召回使用的模板文档。"""
        parts = [template.description, *template.utterances]
        parts.extend(" ".join(rule.term for rule in group) for group in template.must_terms)
        return normalize_text(" ".join(part for part in parts if part))

    def _build_template_fields(self, template: TemplateDefinition) -> dict[str, list[str]]:
        """生成 BM25F 的多字段文本。"""
        must_terms_text = " ".join(" ".join(rule.term for rule in group) for group in template.must_terms)
        return {
            "description": mixed_terms(normalize_text(template.description)),
            # utterances 保留模板最接近用户原话的表达，是 BM25F 里最重要的召回字段之一。
            "utterances": mixed_terms(normalize_text(" ".join(template.utterances))),
            # must_terms 作为硬线索单独建字段，避免被 description/utterances 的长文本稀释。
            "must_terms": mixed_terms(normalize_text(must_terms_text)),
        }

    def _sample_recall(self, query_term_set: set[str]) -> list[tuple[str, float]]:
        """用示例问法相似度做一条廉价召回路。"""
        scored = [
            (
                template_id,
                sample_similarity_from_terms(query_term_set, utterance_term_sets),
            )
            for template_id, utterance_term_sets in self.template_sample_terms.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.config.settings.recall_top_k]

    def extract_slots(self, template_id: str, text: str) -> dict[str, object]:
        """按模板自己的 extractor 抽槽位。"""
        # 每个模板都只看自己的 extractor，这样同名槽位也可以按模板语义独立解释。
        return self.template_slot_registries[template_id].extract(text)

    def _build_global_signal_slot_definitions(self) -> dict[str, SlotExtractorDefinition]:
        """合并所有模板里的高信号槽位定义，供结构判断共享使用。"""
        merged: dict[str, SlotExtractorDefinition] = {}
        for slot_name, definition in self.config.slot_extractors.items():
            if not _is_high_signal_slot_name(slot_name):
                continue
            _append_slot_definition(merged, slot_name, definition)
        for template in self.config.templates:
            for slot_name, definition in template.slot_extractors.items():
                if not _is_high_signal_slot_name(slot_name):
                    continue
                _append_slot_definition(merged, slot_name, definition)
        return merged

    def _build_template_slot_definitions(
        self,
        template: TemplateDefinition,
    ) -> dict[str, SlotExtractorDefinition]:
        """组装模板可见的槽位定义。

        优先级是：
        1. 模板本地 `slot_extractors`
        2. 根级共享 `slot_extractors`
        """
        relevant_slots = (
            template_declared_slot_names(template)
            | set(template.slot_extractors)
        )
        merged: dict[str, SlotExtractorDefinition] = {}
        for slot_name in relevant_slots:
            local_definition = template.slot_extractors.get(slot_name)
            if local_definition is not None:
                # 模板本地定义优先，允许模板针对同名槽位覆写共享逻辑。
                merged[slot_name] = SlotExtractorDefinition(
                    slot_name=local_definition.slot_name,
                    extractors=[copy.deepcopy(extractor) for extractor in local_definition.extractors],
                )
                continue
            shared_definition = self.config.slot_extractors.get(slot_name)
            if shared_definition is None:
                continue
            # 只有模板本地没声明时，才回退到根级共享定义。
            merged[slot_name] = SlotExtractorDefinition(
                slot_name=shared_definition.slot_name,
                extractors=[copy.deepcopy(extractor) for extractor in shared_definition.extractors],
            )
        return merged

    def _build_candidate(
        self,
        *,
        template_id: str,
        norm_text: str,
        global_slots: dict[str, object],
        lexical_score: float,
        sample_score: float,
        vector_score: float,
        fusion_score: float,
        rerank_score: float = 0.0,
        slots: dict[str, object],
        weights: dict[str, float],
        rerank_trace: dict[str, object] | None = None,
        include_defaults: bool = False,
    ) -> TemplateCandidate:
        """把一个模板在当前 query 下的所有子分数组装成最终候选。"""
        template = self.templates[template_id]
        slots, slot_resolution_trace = self._resolve_slots_for_template(
            template,
            slots,
            include_defaults=include_defaults,
        )
        # slot_fit 看“该有的槽位有没有抽出来”，
        # constraint 看“抽到的值是否满足模板显式约束”，
        # structure 看“query 的条件复杂度和模板结构是否对得上”。
        slot_score = slot_fit_score(template, slots)
        constraint = constraint_score(template, norm_text, slots)
        requirement_report = evaluate_slot_requirements(template, slots)
        structure_report = evaluate_structural_alignment(
            template,
            query_slots=global_slots,
            candidate_slots=slots,
            requirement_report=requirement_report,
        )
        structure_score = structure_report.score
        unexpected_slots = _unexpected_global_slots(template, global_slots)
        if has_negative_term(template, norm_text):
            # 负向词命中属于硬否决，直接把该模板总分清零。
            total = 0.0
        else:
            total = weighted_score(
                {
                    "lexical": lexical_score,
                    "sample": sample_score,
                    "vector": vector_score,
                    "fusion": fusion_score,
                    "rerank": rerank_score,
                    "slot_fit": slot_score,
                    "constraint": constraint,
                    "structure": structure_score,
                },
                weights,
            )
        return TemplateCandidate(
            template_id=template.template_id,
            query_mode=template.query_mode,
            score=total,
            lexical_score=lexical_score,
            sample_score=sample_score,
            vector_score=vector_score,
            fusion_score=fusion_score,
            rerank_score=rerank_score,
            slot_fit_score=slot_score,
            constraint_score=constraint,
            structure_score=structure_score,
            slots=dict(slots),
            missing_slots=requirement_report.missing_slots,
            trace={
                # 这里记录 query 里出现、但当前模板不支持的高信号槽位，
                # 方便排查“为什么单条件模板没选上”。
                "unexpected_global_slots": unexpected_slots,
                "structure_details": structure_report.to_dict(),
                "requirement_issues": requirement_report.to_dict()["issues"],
                "blocking_issue_count": len(requirement_report.issues),
                "slot_resolution": slot_resolution_trace,
                "rerank_trace": dict(rerank_trace or {}),
            },
            metadata=template.metadata,
        )

    def _rebuild_candidate(
        self,
        candidate: TemplateCandidate,
        *,
        norm_text: str,
        global_slots: dict[str, object],
        include_defaults: bool = False,
    ) -> TemplateCandidate:
        """用同一组召回子分重新计算槽位相关分数。"""
        weights = adaptive_score_weights(
            self.config.settings.weights,
            _merge_slot_maps(global_slots, candidate.slots),
        )
        rebuilt = self._build_candidate(
            template_id=candidate.template_id,
            norm_text=norm_text,
            global_slots=global_slots,
            lexical_score=candidate.lexical_score,
            sample_score=candidate.sample_score,
            vector_score=candidate.vector_score,
            fusion_score=candidate.fusion_score,
            rerank_score=candidate.rerank_score,
            slots=candidate.slots,
            weights=weights,
            rerank_trace=dict(candidate.trace.get("rerank_trace", {})),
            include_defaults=include_defaults,
        )
        for key in ("slot_fallback_used", "slot_fallback_trace"):
            if key in candidate.trace:
                rebuilt.trace[key] = candidate.trace[key]
        return rebuilt

    def _resolve_slots_for_template(
        self,
        template: TemplateDefinition,
        slots: dict[str, object],
        *,
        include_defaults: bool,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """补充默认值和派生槽位，返回可追踪的槽位来源。"""
        resolved: dict[str, object] = dict(slots)
        trace: dict[str, object] = {
            "defaults_applied": {},
            "derived_applied": {},
        }
        if include_defaults:
            defaults_applied: dict[str, object] = {}
            for slot_name, value in template.slot_defaults.items():
                if _slot_value_present(resolved.get(slot_name)):
                    continue
                resolved[slot_name] = copy.deepcopy(value)
                defaults_applied[slot_name] = copy.deepcopy(value)
            trace["defaults_applied"] = defaults_applied

        derived_applied: dict[str, object] = {}
        for _ in range(3):
            changed = False
            for rule in template.derived_slots:
                if not rule.overwrite and _slot_value_present(resolved.get(rule.slot_name)):
                    continue
                value = _derive_slot_value(rule, resolved)
                if not _slot_value_present(value):
                    continue
                resolved[rule.slot_name] = copy.deepcopy(value)
                derived_applied[rule.slot_name] = copy.deepcopy(value)
                changed = True
            if not changed:
                break
        trace["derived_applied"] = derived_applied
        return resolved, trace

    def _is_ambiguous(
        self,
        top: TemplateCandidate,
        second: TemplateCandidate | None,
    ) -> bool:
        """top1 和 top2 都够高，但差距不够时视为歧义。"""
        if second is None:
            return False
        if second.score < self.config.settings.match_threshold:
            return False
        return (top.score - second.score) < self.config.settings.ambiguity_margin

    def _match_blocked_term(self, norm_text: str) -> str | None:
        """全局拦截明显非问数意图。"""
        for rule in self.config.settings.blocked_terms:
            if match_text_rule(norm_text, rule):
                return rule.term
        return None

    def _resolve_template_slots_with_fallback(
        self,
        *,
        input_text: str,
        norm_text: str,
        global_slots: dict[str, object],
        candidate: TemplateCandidate,
        status: MatchStatus,
    ) -> TemplateCandidate:
        """只对已稳定命中的 top1 模板尝试模板级 LLM 补参。"""
        template = self.templates[candidate.template_id]
        missing_slots = missing_required_slots(template, candidate.slots)
        if not self._should_use_template_slot_fallback(candidate, missing_slots, status):
            return candidate
        if self.llm_template_slot_resolver is None:
            return candidate
        # 这里已经知道模板 id，不再让 LLM 选模板，只让它补少量缺参。
        suggestion = self.llm_template_slot_resolver.resolve_slots(
            input_text=input_text,
            normalized_text=norm_text,
            template=template,
            current_slots=dict(candidate.slots),
            missing_slots=missing_slots,
        )
        if suggestion is None or not suggestion.slots:
            return candidate
        # LLM 只补充缺失值，不覆盖已有槽位的业务判断结果。
        merged_slots = dict(candidate.slots)
        merged_slots.update(suggestion.slots)
        # 补参后要重新算一次分数，因为 slot_fit/structure/constraint 都可能变化。
        rerank_weights = adaptive_score_weights(
            self.config.settings.weights,
            _merge_slot_maps(global_slots, merged_slots),
        )
        enriched = self._build_candidate(
            template_id=candidate.template_id,
            norm_text=norm_text,
            global_slots=global_slots,
            lexical_score=candidate.lexical_score,
            sample_score=candidate.sample_score,
            vector_score=candidate.vector_score,
            fusion_score=candidate.fusion_score,
            rerank_score=candidate.rerank_score,
            slots=merged_slots,
            weights=rerank_weights,
            rerank_trace=dict(candidate.trace.get("rerank_trace", {})),
        )
        return TemplateCandidate(
            template_id=enriched.template_id,
            query_mode=enriched.query_mode,
            score=enriched.score,
            lexical_score=enriched.lexical_score,
            sample_score=enriched.sample_score,
            vector_score=enriched.vector_score,
            fusion_score=enriched.fusion_score,
            rerank_score=enriched.rerank_score,
            slot_fit_score=enriched.slot_fit_score,
            constraint_score=enriched.constraint_score,
            structure_score=enriched.structure_score,
            slots=enriched.slots,
            missing_slots=enriched.missing_slots,
            trace={
                **enriched.trace,
                "slot_fallback_used": True,
                "slot_fallback_trace": suggestion.trace,
            },
            metadata=enriched.metadata,
        )

    def _should_use_template_slot_fallback(
        self,
        candidate: TemplateCandidate,
        missing_slots: list[str],
        status: MatchStatus,
    ) -> bool:
        """模板级 LLM 补参的触发门槛。"""
        if not self.config.settings.llm_slot_fallback_enabled:
            return False
        template = self.templates[candidate.template_id]
        template_settings = template.llm_slot_extraction
        if not template_settings.get("enabled", False):
            return False
        # 只有规则链路已经把模板打到较高置信度时，才值得为它付一次 LLM 成本。
        if candidate.score < self.config.settings.llm_slot_fallback_min_score:
            return False
        if status is MatchStatus.MATCHED:
            return bool(
                self.config.settings.llm_slot_fallback_allow_on_matched
                or template_settings.get("allow_on_matched", False)
            )
        if status is not MatchStatus.PARTIAL:
            return False
        if not missing_slots:
            # 纯冲突型 partial 没有“可补的缺槽位”，不应该走模板级补参。
            return False
        return len(missing_slots) <= self.config.settings.llm_slot_fallback_max_missing_slots

    def _resolve_with_fallback(
        self,
        *,
        input_text: str,
        norm_text: str,
        slots: dict[str, object],
        ranked: list[TemplateCandidate],
        missing_slots: list[str],
        base_status: MatchStatus,
    ) -> MatchResult | None:
        """旧的模板选择级 fallback，作用范围比模板级补参更大。"""
        if not self._should_use_fallback(ranked, missing_slots, base_status):
            return None
        if self.llm_fallback_resolver is None:
            return None
        # 这里只把少量 top 候选给 LLM，避免退化成“全模板逐个推理”。
        suggestion = self.llm_fallback_resolver.resolve(
            input_text=input_text,
            normalized_text=norm_text,
            slots=dict(slots),
            candidates=ranked[: self.config.settings.llm_fallback_max_candidates],
            templates=self.templates,
        )
        if suggestion is None:
            return None
        return self._build_fallback_result(norm_text, ranked, suggestion)

    def _should_use_fallback(
        self,
        ranked: list[TemplateCandidate],
        missing_slots: list[str],
        base_status: MatchStatus,
    ) -> bool:
        """只有在低置信边界 case 上才放行模板选择 fallback。"""
        if not self.config.settings.llm_fallback_enabled:
            return False
        top = ranked[0] if ranked else None
        if base_status is MatchStatus.PARTIAL:
            # partial 代表模板大概率是对的，只是缺少少量必要参数。
            return len(missing_slots) <= self.config.settings.llm_fallback_max_missing_slots
        if top is None:
            return False
        # unmatched 只放行接近阈值的边界 case，避免把明显不相关的 query 也送给 LLM。
        lower_bound = self.config.settings.match_threshold - self.config.settings.llm_fallback_score_margin
        return top.score >= max(0.0, lower_bound)

    def _build_fallback_result(
        self,
        norm_text: str,
        ranked: list[TemplateCandidate],
        suggestion: FallbackSuggestion,
    ) -> MatchResult:
        """把模板选择 fallback 的结果转成统一输出。"""
        slots = dict(suggestion.slots)
        missing_slots = list(suggestion.missing_slots)
        status = suggestion.status
        slot_resolution_trace: dict[str, object] = {}
        if isinstance(suggestion.template_id, str) and suggestion.template_id in self.templates:
            template = self.templates[suggestion.template_id]
            slots, slot_resolution_trace = self._resolve_slots_for_template(
                template,
                slots,
                include_defaults=True,
            )
            requirement_report = evaluate_slot_requirements(template, slots)
            missing_slots = requirement_report.missing_slots
            status = MatchStatus.MATCHED if requirement_report.is_satisfied else MatchStatus.PARTIAL
        return MatchResult(
            template_id=suggestion.template_id,
            status=status,
            score=suggestion.score,
            query_mode=suggestion.query_mode,
            slots=slots,
            missing_slots=missing_slots,
            metadata=suggestion.metadata,
            trace={
                "norm_text": norm_text,
                "fallback_used": True,
                "fallback_trace": suggestion.trace,
                "slot_resolution": slot_resolution_trace,
                "top_candidates": [candidate.to_dict() for candidate in ranked[:5]],
            },
        )

    def _build_nl2sql_route_result(
        self,
        *,
        reason: str,
        norm_text: str,
        original_norm_text: str,
        rewrite_trace: dict[str, object],
        slots: dict[str, object],
        ranked: list[TemplateCandidate],
    ) -> TemplateRouteDecision:
        top = ranked[0] if ranked else None
        result = MatchResult(
            template_id=-1,
            status=MatchStatus.UNMATCHED,
            score=top.score if top is not None else 0.0,
            query_mode=None,
            slots=slots,
            missing_slots=[],
            trace={
                "norm_text": norm_text,
                "original_norm_text": original_norm_text,
                "rewrite_trace": rewrite_trace,
                "route_to": "nl2sql",
                "reason": reason,
                "top_candidates": [candidate.to_dict() for candidate in ranked[:5]],
                "template_route_min_score": self.config.settings.template_route_min_score,
                "template_route_min_structure_score": self.config.settings.template_route_min_structure_score,
            },
        )
        return TemplateRouteDecision(
            route_to="nl2sql",
            result=result,
            reason=reason,
            candidates=ranked[:5],
        )


def _build_configured_vector_backend(config: TemplateConfig) -> VectorSearchBackend:
    """按 matcher.vector.provider 真正实例化向量后端。"""
    provider_name = config.settings.vector_provider
    dimension = config.settings.vector_dimension
    if provider_name in {"local_tfidf", "tfidf"}:
        return InMemoryVectorIndex(provider=LocalTfidfVectorProvider(dimension))
    if provider_name in {"hashing", "local_hash"}:
        return InMemoryVectorIndex(provider=LocalHashVectorProvider(dimension))
    if provider_name == "remote":
        options = config.settings.vector_options
        api_key = str(options.get("api_key", "")).strip()
        api_key_env = str(options.get("api_key_env", "")).strip()
        if not api_key and api_key_env:
            api_key = os.environ.get(api_key_env, "")
        provider = RemoteEmbeddingProvider(
            api_key=api_key,
            base_url=str(options.get("base_url", "")).strip(),
            model=str(options.get("model", "")).strip(),
            dimension=dimension,
            timeout_seconds=float(options.get("timeout_seconds", 10.0)),
            batch_size=int(options.get("batch_size", 32)),
            extra_body=dict(options.get("extra_body", {})),
            include_dimensions=bool(options.get("include_dimensions", True)),
        )
        return InMemoryVectorIndex(provider=provider)
    raise ValueError(f"Unsupported vector provider: {provider_name}")


def _build_configured_template_reranker(config: TemplateConfig) -> TemplateReranker:
    """按 matcher.reranker.provider 实例化二阶段精排器。"""
    if not config.settings.reranker_enabled:
        return NoopTemplateReranker()
    provider_name = config.settings.reranker_provider
    if provider_name in {"none"}:
        return NoopTemplateReranker()
    if provider_name == "term_overlap":
        return TermOverlapTemplateReranker()
    if provider_name in {"openai_compatible", "remote"}:
        options = config.settings.reranker_options
        return OpenAICompatibleTemplateReranker(
            api_key=resolve_reranker_api_key(options),
            base_url=str(options.get("base_url", "")).strip(),
            model=str(options.get("model", "")).strip(),
            timeout_seconds=float(options.get("timeout_seconds", 8.0)),
        )
    raise ValueError(f"Unsupported reranker provider: {provider_name}")


def _append_slot_definition(
    target: dict[str, SlotExtractorDefinition],
    slot_name: str,
    definition: SlotExtractorDefinition,
) -> None:
    """合并槽位定义时按 extractor 内容去重，避免全局 registry 无限膨胀。"""
    existing = target.setdefault(
        slot_name,
        SlotExtractorDefinition(slot_name=definition.slot_name, extractors=[]),
    )
    seen = {
        json.dumps(extractor, ensure_ascii=False, sort_keys=True)
        for extractor in existing.extractors
    }
    for extractor in definition.extractors:
        fingerprint = json.dumps(extractor, ensure_ascii=False, sort_keys=True)
        if fingerprint in seen:
            continue
        existing.extractors.append(copy.deepcopy(extractor))
        seen.add(fingerprint)


def _is_high_signal_slot_name(slot_name: str) -> bool:
    return slot_name in HIGH_SIGNAL_EXACT_SLOTS or slot_name.endswith("_threshold")


def _merge_slot_maps(*slot_maps: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for slot_map in slot_maps:
        for slot_name, value in slot_map.items():
            if not _slot_value_present(value):
                continue
            merged[slot_name] = value
    return merged


def _unexpected_global_slots(template: TemplateDefinition, global_slots: dict[str, object]) -> list[str]:
    supported_slots = template_declared_slot_names(template)
    extracted_slots = {
        slot_name
        for slot_name, value in global_slots.items()
        if _slot_value_present(value)
    }
    return sorted(extracted_slots - supported_slots)


def _weights_without_score_part(weights: dict[str, float], field_name: str) -> dict[str, float]:
    """构造“去掉某个子分”的权重模板，供二阶段精排前做初排。"""
    stripped = {
        name: float(value)
        for name, value in weights.items()
        if name != field_name
    }
    return stripped or dict(weights)


def _derive_slot_value(rule: DerivedSlotDefinition, slots: dict[str, object]) -> object | None:
    source_values: list[object] = []
    for slot_name in rule.source_slots:
        value = slots.get(slot_name)
        if not _slot_value_present(value):
            if rule.require_all_sources:
                return None
            source_values.append("")
            continue
        source_values.append(value)

    mapping = rule.mapping
    if isinstance(mapping, dict):
        key = rule.key_separator.join(_slot_value_key(value) for value in source_values)
        if key in mapping:
            return copy.deepcopy(mapping[key])
        return copy.deepcopy(rule.default)

    if isinstance(mapping, list):
        for item in mapping:
            if not isinstance(item, dict):
                continue
            conditions = item.get("when", item.get("source_values", {}))
            if isinstance(conditions, list):
                conditions = {
                    slot_name: conditions[index]
                    for index, slot_name in enumerate(rule.source_slots)
                    if index < len(conditions)
                }
            if not isinstance(conditions, dict):
                continue
            if all(_slot_value_matches(slots.get(str(slot_name)), expected) for slot_name, expected in conditions.items()):
                return copy.deepcopy(item.get("value"))
    return copy.deepcopy(rule.default)


def _slot_value_key(value: object) -> str:
    if isinstance(value, dict):
        for key in ("id", "value", "code", "preset"):
            if key in value:
                return str(value[key])
    return str(value)


def _slot_value_matches(actual: object, expected: object) -> bool:
    if not _slot_value_present(actual):
        return False
    if isinstance(expected, list):
        return any(_slot_value_matches(actual, item) for item in expected)
    actual_candidates = _slot_value_candidates(actual)
    expected_candidates = _slot_value_candidates(expected)
    return bool(actual_candidates & expected_candidates)


def _slot_value_candidates(value: object) -> set[str]:
    if isinstance(value, dict):
        candidates: set[str] = set()
        for key in ("id", "value", "code", "preset"):
            if key in value:
                candidates.add(str(value[key]).lower())
        if not candidates:
            candidates.add(str(value).lower())
        return candidates
    return {str(value).lower()}


def _slot_value_present(value: object) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, list) and not value:
        return False
    if isinstance(value, dict) and not value:
        return False
    return True
