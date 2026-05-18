from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# 这一层默认权重只定义“没有显式配置时系统怎么工作”，
# 真正生效的值仍然优先来自 templates.json 里的 matcher 配置。
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "lexical": 0.2,
    "sample": 0.1,
    "vector": 0.15,
    "fusion": 0.1,
    "slot_fit": 0.15,
    "constraint": 0.1,
    "structure": 0.2,
}

DEFAULT_LEXICAL_FIELD_WEIGHTS: dict[str, float] = {
    "description": 0.6,
    "utterances": 1.0,
    "must_terms": 1.6,
}

DEFAULT_VECTOR_PROVIDER = "local_tfidf"
DEFAULT_RERANKER_PROVIDER = "none"
DEFAULT_RERANK_WEIGHT = 0.12
SUPPORTED_TEXT_MATCH_MODES = {"substring", "whole_word", "exact"}
SUPPORTED_KEYWORD_MATCH_POLICIES = {"first", "longest"}


@dataclass(slots=True)
class QueryRewriteRule:
    """前置 query 改写规则。

    这层规则不参与模板选择，它只负责把行业黑话、别名、内部简称
    统一改写成模板侧更稳定的表达。
    """

    source: str
    target: str
    rule_id: str = ""
    match_mode: str = "substring"


@dataclass(slots=True)
class QueryRewriteSettings:
    """前置 query 改写模块配置。"""

    enabled: bool = False
    max_passes: int = 1
    dictionary_path: str | None = None
    reload_on_change: bool = True
    rules: list[QueryRewriteRule] = field(default_factory=list)


@dataclass(slots=True)
class TextMatchRule:
    """文本匹配规则。

    默认仍然使用 `substring`，这样老模板和老配置完全不需要改。
    只有显式声明 `whole_word / exact` 时，才会启用更严格的边界匹配。
    """

    term: str
    match_mode: str = "substring"

    def __post_init__(self) -> None:
        self.term = str(self.term or "").strip()
        self.match_mode = str(self.match_mode or "substring").strip().lower()

    def to_dict(self) -> dict[str, str]:
        return {
            "term": self.term,
            "match_mode": self.match_mode,
        }


class MatchStatus(str, Enum):
    """模板匹配只区分命中、部分命中、未命中。

    这是整个能力层最外层的状态枚举：
    - MATCHED: 模板已确定，且必填槽位齐全
    - PARTIAL: 模板已基本确定，但仍缺少关键槽位
    - UNMATCHED: 没有稳定命中的模板，外部应按 -1 理解
    """

    MATCHED = "matched"
    PARTIAL = "partial"
    UNMATCHED = "unmatched"


@dataclass(slots=True)
class MatcherSettings:
    """匹配引擎运行参数。

    这类字段不描述任何具体模板，只描述“所有模板共用的匹配策略”。
    阅读顺序建议按下面三组理解：
    1. 召回与阈值: `match_threshold` / `ambiguity_margin` / `recall_top_k`
    2. 打分融合: `weights` / `lexical_field_weights` / `fusion_rrf_k`
    3. 兜底策略: `blocked_terms` / `llm_fallback_*` / `llm_slot_fallback_*`
    """

    # 最终 top1 低于该阈值时，直接视为未命中。
    match_threshold: float
    # top1 和 top2 同时够高但差距过小时，判为歧义，返回 -1 或交给 fallback。
    ambiguity_margin: float
    # 每条召回链各自保留多少候选进入后续融合。
    recall_top_k: int
    # 最终总分的线性融合权重，key 对应 scoring.py 里的子分名。
    weights: dict[str, float]
    # BM25F 的字段权重，通常 must_terms > utterances > description。
    lexical_field_weights: dict[str, float] = field(default_factory=dict)
    # Reciprocal Rank Fusion 的平滑参数，值越大，不同召回路的 rank 差异越不敏感。
    fusion_rrf_k: int = 60
    # 向量召回约定的 embedding 维度，便于外部向量服务接入时保持一致。
    vector_dimension: int = 512
    # 向量 provider 名称，允许通过配置在本地 tfidf / hashing / remote 之间切换。
    vector_provider: str = DEFAULT_VECTOR_PROVIDER
    # provider 额外参数，例如 remote 模式下的 base_url / model / api_key_env。
    vector_options: dict[str, Any] = field(default_factory=dict)
    # 二阶段 reranker 默认关闭；打开后只对 top-k 候选做更贵的精排。
    reranker_enabled: bool = False
    # 当前内置 `none / term_overlap / openai_compatible`，后续可以平滑扩展。
    reranker_provider: str = DEFAULT_RERANKER_PROVIDER
    # 只对初排前几名做 rerank，控制额外时延和成本。
    reranker_top_k: int = 15
    # reranker 的附加参数，例如 remote 模式下的 base_url / model / api_key_env。
    reranker_options: dict[str, Any] = field(default_factory=dict)
    # 全局负向意图词，命中后直接短路为 UNMATCHED。
    blocked_terms: list[TextMatchRule] = field(default_factory=list)
    # 模板选择级 LLM fallback，总开关。
    llm_fallback_enabled: bool = False
    # 模板选择级 fallback 最多看多少个 top 候选。
    llm_fallback_max_candidates: int = 3
    # top1 距离主阈值多近时，允许 unmatched case 进入 fallback。
    llm_fallback_score_margin: float = 0.08
    # partial case 最多缺多少必填槽位时，允许进入模板选择级 fallback。
    llm_fallback_max_missing_slots: int = 2
    # 模板级 LLM 补参，总开关。它只在模板已基本命中后触发。
    llm_slot_fallback_enabled: bool = False
    # 模板级补参最多允许补几个缺失槽位。
    llm_slot_fallback_max_missing_slots: int = 2
    # 候选模板至少达到该分数，才值得付一次模板级 LLM 成本。
    llm_slot_fallback_min_score: float = 0.58
    # 是否允许对已经 MATCHED 的模板也尝试再补充可选槽位。
    llm_slot_fallback_allow_on_matched: bool = False
    # 面向 SQL 路由的高置信模板准入阈值；默认跟主匹配阈值保持一致。
    template_route_min_score: float = 0.0
    # SQL 路由时要求模板结构分达到的下限，避免文本像但结构接不住的模板放行。
    template_route_min_structure_score: float = 0.0
    # SQL 路由提参最多查看几个候选；默认 top1，低延迟优先。
    template_route_top_k: int = 1

    def __post_init__(self) -> None:
        # 兼容直接手写 MatcherSettings 的场景；未传权重时补默认值。
        if not self.weights:
            self.weights = dict(DEFAULT_SCORE_WEIGHTS)
        if not self.lexical_field_weights:
            self.lexical_field_weights = dict(DEFAULT_LEXICAL_FIELD_WEIGHTS)
        self.vector_provider = str(self.vector_provider or DEFAULT_VECTOR_PROVIDER).strip().lower()
        self.reranker_provider = str(self.reranker_provider or DEFAULT_RERANKER_PROVIDER).strip().lower()
        if self.reranker_enabled and "rerank" not in self.weights:
            # 只有显式打开 reranker 时，才自动补一个保守权重，避免老配置被无意改变。
            self.weights["rerank"] = DEFAULT_RERANK_WEIGHT
        self.blocked_terms = build_text_match_rules(self.blocked_terms)
        if self.template_route_min_score <= 0:
            self.template_route_min_score = self.match_threshold
        self.template_route_top_k = max(1, int(self.template_route_top_k))


@dataclass(slots=True)
class SlotExtractorDefinition:
    """配置驱动的槽位抽取定义。

    它描述的是“某一个槽位可以怎样被抽出来”，而不是模板本身。
    同一个 `slot_name` 在不同模板里可以复用，也可以被模板本地覆写。
    """

    slot_name: str
    # extractor 按顺序尝试，先命中的规则优先级更高。
    extractors: list[dict[str, Any]]


@dataclass(slots=True)
class SlotGroupRequirement:
    """一组槽位的组合规则。

    这类结构被两个场景复用：
    1. `required_one_of`: 至少命中一个
    2. `mutually_exclusive_slots`: 最多命中一个
    """

    slots: list[str]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": list(self.slots),
            "description": self.description,
        }


@dataclass(slots=True)
class ConditionalSlotRequirement:
    """条件必填规则。

    当 `when_any` / `when_all` 触发后，`require` 里的槽位必须被填充，
    适合表达“给了 A 就必须给 B”“A+B 出现时必须补 C”这类约束。
    """

    require: list[str]
    when_any: list[str] = field(default_factory=list)
    when_all: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "require": list(self.require),
            "when_any": list(self.when_any),
            "when_all": list(self.when_all),
            "description": self.description,
        }


@dataclass(slots=True)
class DerivedSlotDefinition:
    """由多个已有槽位确定性派生出的槽位。

    典型场景是同一个语义值在不同资源类型下映射到不同后端枚举：
    `resource_type=disk + health_status_label=normal -> healthStatus=2`。
    """

    slot_name: str
    source_slots: list[str]
    mapping: list[dict[str, Any]] | dict[str, Any]
    default: Any | None = None
    key_separator: str = "."
    overwrite: bool = False
    require_all_sources: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "source_slots": list(self.source_slots),
            "mapping": self.mapping,
            "default": self.default,
            "key_separator": self.key_separator,
            "overwrite": self.overwrite,
            "require_all_sources": self.require_all_sources,
        }


@dataclass(slots=True)
class TemplateDefinition:
    """只面向问数场景的模板定义。

    这是系统里最重要的静态配置对象。一个模板同时承担三类职责：
    1. 说明“这类 query 想问什么”      -> `description` / `utterances`
    2. 说明“成立至少需要什么条件”    -> `required_slots` / `must_terms`
    3. 说明“参数应该怎么抽、怎么约束” -> `slot_extractors` / `slot_constraints`
    """

    # 模板唯一标识，也是最终返回给外部系统的主键。
    template_id: str
    # 模板所属的“能力大类”。
    # 在当前工程里它基本固定为 `metric_query`，用于告诉下游：
    # 这是一个问数模板，而不是报告、分析、动作等别的能力类型。
    # 真正区分 `count / topn / list` 的不是它，而是槽位里的 `query_operator`。
    query_mode: str
    # 供开发和召回理解使用的简短说明，适合写“这个模板到底问什么”。
    description: str
    # 模板的代表性问法样例，主要用于 lexical/sample 召回。
    utterances: list[str]
    # 没有这些槽位时，模板最多只能是 PARTIAL，不能是 MATCHED。
    required_slots: list[str]
    # 这些槽位会提升模板完整度，但缺失时不阻止命中。
    optional_slots: list[str]
    # 必须命中的词组列表；每组内是近义词关系，组与组之间是“都要满足”。
    must_terms: list[list[TextMatchRule]]
    # 负向排斥词，只要命中就明确说明“不该走这个模板”。
    negative_terms: list[TextMatchRule]
    # 对抽取值的显式限制，例如某个槽位只能是特定枚举值。
    slot_constraints: dict[str, list[Any]]
    # 至少满足一项即可的槽位组，适合表达“ip / mac / name 三选一”。
    required_one_of: list["SlotGroupRequirement"] = field(default_factory=list)
    # 条件必填规则，适合表达“如果给了 A，就必须给 B”。
    conditional_required: list["ConditionalSlotRequirement"] = field(default_factory=list)
    # 互斥槽位组，适合表达“ip / mac / name 不能同时给多个”。
    mutually_exclusive_slots: list["SlotGroupRequirement"] = field(default_factory=list)
    # 模板自己的槽位抽取器。它优先于根级共享定义。
    slot_extractors: dict[str, "SlotExtractorDefinition"] = field(default_factory=dict)
    # 可选槽位或执行层必需默认参数。默认值只在最终输出/SQL 路由前补齐，不参与早期召回。
    slot_defaults: dict[str, Any] = field(default_factory=dict)
    # 由 resource_type、状态语义等上游槽位派生出的后端字段值。
    derived_slots: list["DerivedSlotDefinition"] = field(default_factory=list)
    # 针对列表型槽位的数量约束，例如单指标模板要求 metric_conditions 最多 1 个。
    slot_validations: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 模板级 LLM 补参配置，只在模板已基本命中后才会使用。
    llm_slot_extraction: dict[str, Any] = field(default_factory=dict)
    # 业务附加信息，原样透传到匹配结果里，不参与排序。
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.must_terms = build_text_match_groups(self.must_terms)
        self.negative_terms = build_text_match_rules(self.negative_terms)


@dataclass(slots=True)
class TemplateCandidate:
    """单个模板候选的打分轨迹。

    这是纯运行期对象，只存在于一次 match 过程中。
    它回答的是：“当前 query 下，这个模板为什么排在这个位置？”
    """

    template_id: str
    query_mode: str
    # 融合后的最终分，用来和其他模板排序。
    score: float
    # 下面这些都是组成 `score` 的子分，便于调试和调参。
    lexical_score: float
    sample_score: float
    vector_score: float
    fusion_score: float
    rerank_score: float
    slot_fit_score: float
    constraint_score: float
    structure_score: float
    # 这是“按该模板自己的 extractor”抽出来的槽位，不是全局抽参结果。
    slots: dict[str, Any]
    # 当前模板如果想成为 MATCHED，还缺哪些槽位。
    # 这里兼容保留旧字段名，但内容可能来自 required_slots / one_of / conditional_required。
    missing_slots: list[str]
    # 预留给更细的诊断轨迹，例如 LLM 补参耗时、重排原因。
    trace: dict[str, Any] = field(default_factory=dict)
    # 透传模板的 metadata，方便候选调试时直接看到业务标签。
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # trace 输出统一走 dict，方便 CLI、测试和报告直接序列化。
        return {
            "template_id": self.template_id,
            "query_mode": self.query_mode,
            "score": self.score,
            "lexical_score": self.lexical_score,
            "sample_score": self.sample_score,
            "vector_score": self.vector_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "slot_fit_score": self.slot_fit_score,
            "constraint_score": self.constraint_score,
            "structure_score": self.structure_score,
            "slots": self.slots,
            "missing_slots": self.missing_slots,
            "trace": self.trace,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MatchResult:
    """模板匹配输出。

    这是能力层对外的最终契约。外部通常只需要关心三件事：
    1. `template_id` 是哪个；未命中时固定为 `-1`
    2. `status` 是 matched / partial / unmatched
    3. `slots` 和 `missing_slots` 说明参数是否齐全
    """

    # 命中时是模板 id，未命中时固定返回 -1。
    template_id: str | int
    status: MatchStatus
    # 这里是最终对外分数，已经是融合后的 top1 分。
    score: float
    # 模板声明的能力大类；当前一般是 `metric_query`，未命中时为 None。
    query_mode: str | None
    # 最终确认可用的槽位集合。
    slots: dict[str, Any] = field(default_factory=dict)
    # 对 partial 场景尤其关键，告诉外部还差哪些必要参数。
    # 这里兼容保留旧字段名，但内容可能来自更丰富的约束规则。
    missing_slots: list[str] = field(default_factory=list)
    # 透传模板 metadata，供业务侧做后续路由或展示。
    metadata: dict[str, Any] = field(default_factory=dict)
    # 调试轨迹，不应被业务逻辑强依赖。
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "status": self.status.value,
            "score": self.score,
            "query_mode": self.query_mode,
            "slots": self.slots,
            "missing_slots": self.missing_slots,
            "metadata": self.metadata,
            "trace": self.trace,
        }


@dataclass(slots=True)
class TemplateRouteDecision:
    """面向 SQL 执行入口的路由决策。

    `match()` 仍然保留 matched / partial / unmatched 的诊断语义；
    这个结构额外告诉上游本次是否应该直接走模板 SQL，还是回退 NL2SQL。
    """

    route_to: str
    result: MatchResult
    reason: str
    candidates: list[TemplateCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_to": self.route_to,
            "reason": self.reason,
            "result": self.result.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def template_declared_slot_names(template: TemplateDefinition) -> set[str]:
    """收集模板显式声明过的所有槽位名。

    这层聚合用于三类场景：
    1. 构建模板可见 extractor 集合
    2. 结构分判断“query 条件模板能不能接住”
    3. 配置校验判断规则里引用的槽位是否有填充路径
    """

    declared_slots = (
        set(template.required_slots)
        | set(template.optional_slots)
        | set(template.slot_constraints)
        | set(template.slot_extractors)
        | set(template.slot_defaults)
        | set(template.slot_validations)
    )
    for group in template.required_one_of:
        declared_slots.update(group.slots)
    for rule in template.conditional_required:
        declared_slots.update(rule.when_any)
        declared_slots.update(rule.when_all)
        declared_slots.update(rule.require)
    for group in template.mutually_exclusive_slots:
        declared_slots.update(group.slots)
    for rule in template.derived_slots:
        declared_slots.add(rule.slot_name)
        declared_slots.update(rule.source_slots)
    return declared_slots


def build_text_match_rule(value: Any) -> TextMatchRule | None:
    """兼容字符串写法和对象写法。

    支持：
    - `"cpu"`
    - `{"term": "cpu", "match_mode": "whole_word"}`
    """

    if isinstance(value, TextMatchRule):
        return TextMatchRule(term=value.term, match_mode=value.match_mode)
    if isinstance(value, dict):
        term = str(value.get("term", value.get("text", ""))).strip()
        match_mode = str(value.get("match_mode", "substring") or "substring")
    else:
        term = str(value).strip()
        match_mode = "substring"
    if not term:
        return None
    return TextMatchRule(term=term, match_mode=match_mode)


def build_text_match_rules(values: Any) -> list[TextMatchRule]:
    if not isinstance(values, list):
        return []
    rules: list[TextMatchRule] = []
    for value in values:
        rule = build_text_match_rule(value)
        if rule is None:
            continue
        rules.append(rule)
    return rules


def build_text_match_groups(values: Any) -> list[list[TextMatchRule]]:
    if not isinstance(values, list):
        return []
    groups: list[list[TextMatchRule]] = []
    for group in values:
        if not isinstance(group, list):
            continue
        rules = build_text_match_rules(group)
        if not rules:
            continue
        groups.append(rules)
    return groups
