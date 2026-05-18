# template-capability

面向“问数”场景的通用模板匹配能力。

这是一个面向问数场景的模板匹配与槽位提取系统。

它会根据已配置的模板，对输入 query 进行标准化、候选召回、参数抽取、结构校验和结果判定，最终输出模板是否命中，以及关键参数是否齐全。

返回结果只有三类：

- `matched`：命中模板且必填槽位齐全
- `partial`：命中模板但缺少必填槽位
- `unmatched`：没有可信模板，返回 `template_id = -1`

## 目标与边界

这个工程只做：

1. 配置驱动的槽位抽取
2. 模板候选召回
3. 模板精排打分
4. 命中模板 / 部分命中 / `-1`
5. 模板回归评测

这个工程不做：

1. 报告、总结、根因、预测等非问数场景
2. 下游查询执行
3. 追问文案生成
4. 业务实体写死在代码里
5. 在线大模型逐模板全量推断

## 能力规格

这项能力承诺的是“同一类问题的不同说法可以稳定识别”，不是“任何新问题都能自动理解”。

适用对象：

- 已经定义好的业务模板
- 同一业务意图下的表达变化

支持内容：

- 说法变化
- 语序变化
- 常见近义词
- 少量 typo
- 行业简称、别名、黑话
- 时间、数量、阈值、对象名称这类参数变化
- 单条件、双条件、三条件这类结构相同的问题

超出范围时：

- 不会强行猜
- 优先返回 `partial` 或 `unmatched`

举例：

- `查询最近cpu大于80的设备列表`
- `列出最近CPU超过80的设备`
- `最近有哪些设备cpu高于80`

这几句虽然说法不同，但本质上还是同一个“设备 CPU 超阈值列表”问题，所以可以归到同一个模板。

但如果用户问：

- `为什么这些设备CPU高`
- `帮我分析CPU异常原因`

这就已经变成分析类问题，不在这个模板能力的承诺范围内。

为什么能做到：

- 系统不只看关键词，也会一起看问题结构和参数
- 常见简称、黑话、别名会先统一成标准表达
- 一个模板会覆盖多种典型问法，不要求用户每次说得一模一样
- 拿不准时会保守返回，不会强行乱猜
- 能力会持续通过回归测试校验

## 近期增强

最近一轮核心能力增强主要包括：

- 前置 `query_rewrite` 改写
  支持行业黑话、简称、别名在匹配前统一改写
- 启动期配置校验和 `lint`
  坏模板、坏正则、坏配置会在启动前直接拦住
- 全局高信号槽位和更强的结构判断
  复杂 query 会优先匹配结构更完整的模板
- 组合约束升级
  支持 `required_one_of / conditional_required / mutually_exclusive_slots`
- 二阶段 `reranker`
  可以在初排候选之上继续细排
- 结构化 fallback
  LLM 兜底优先走 `json_schema`，不支持时自动回退到 `json_object`
- 高置信模板 SQL 路由
  新增 `route_for_sql()`，模板完全满足才返回 `route_to = template`，否则明确回退 `nl2sql`
- 多属性联动和执行默认值
  支持 `derived_slots` 和 `slot_defaults`，解决资源类型与枚举值联动、可选槽位缺失导致 SQL 拼接异常等问题
- 更强的抽取器
  `keyword_value` 支持最长匹配，新增通用 `time_range` 和可重复 `metric_conditions`
- 轻量 compact LLM 串接
  新增 [LLM/compact_router.py](D:/GitHub/chat_pre_check_blank/src/template_capability/LLM/compact_router.py)，固定规则放 system prompt，每次只传 top 候选的极简 payload
- 更强的诊断 trace
  可以看到 `global_slots / unexpected_global_slots / requirement_issues / structure_details / rerank_trace`

## 核心设计

整体链路：

1. `normalize_text`：文本标准化
2. 可选 `query_rewrite`：前置行业黑话、别名、内部简称改写
3. 共享 `slot_extractors` 与 `global_slots`：补充高信号公共参数
4. `BM25F` + `char ngram` + `vector search`：多路候选召回
5. `RRF` + 动态权重初排：融合多路召回，按 query 复杂度调权
6. 可选 `reranker`：对 top-k 候选继续细排
7. 命中候选后，按模板自己的 `slot_extractors` 做模板内提参
8. `slot_fit` + `constraint` + `structure_score`：模板约束和结构校验
9. 可选 LLM fallback：支持模板选择裁决和模板级补参
10. `slot_defaults / derived_slots / slot_validations`：补默认值、派生联动槽位、校验集合约束
11. 阈值和歧义判断：输出 `matched / partial / unmatched`
12. SQL 路由场景可走 `route_for_sql()` 或 compact LLM 串接，只有模板满足才放行，否则回退 NL2SQL

设计原则：

- 模板定义和槽位定义都配置化
- 参数抽取默认下沉到模板，不依赖全局统一参数字典
- 匹配支持提参、语序变化、口语化表达
- 大模型默认不进入主链路；只支持命中后的模板级窄触发补参
- 向量后端可替换，默认按 `512` 维接口设计
- 模板扩展后必须可批量评测
- 面向 SQL 执行的入口必须比普通 `match()` 更保守：不能满足模板就回退 NL2SQL

## 流程能力说明

### 1. `normalize_text`

做什么：

- 把原始 query 先整理成统一、稳定的文本形式

输入：

- 用户原始输入文本

输出：

- 标准化后的文本

典型用例：

- `CPU大于80？`
- `cpu 大于 80`
- `ＣＰＵ大于80`

这几种表达会先被整理成更一致的形式，再进入后续流程。

算法与原因：

- 这一步做轻量标准化，包括大小写统一、全半角归一、标点清洗、空白折叠
- 这样成本低、稳定性高，是后续召回和提参的共同基础

### 2. `query_rewrite`

做什么：

- 把行业黑话、简称、别名、内部说法改写成模板更稳定的标准说法

输入：

- 标准化后的文本

输出：

- 改写后的文本
- 改写命中轨迹

典型用例：

- `北二小` 改写为 `北京第二小学`
- `错包` 改写为 `错误包`

算法与原因：

- 使用 Trie 状态机做短语匹配和最长匹配
- 这种方式比简单字符串替换更稳，适合做大量固定短语改写，也方便维护和排查

### 3. 共享 `slot_extractors` 与 `global_slots`

做什么：

- 先抽取一批高价值公共参数，给后面的结构判断提供依据

输入：

- 改写后的文本

输出：

- 共享槽位结果
- 全局高信号槽位结果

典型用例：

- 先识别出 `近24小时`
- 先识别出 `cpu>80`
- 先识别出 `前10`

算法与原因：

- 这里主要使用配置化的 `keyword_value` 和 `regex`
- 因为时间、阈值、TopN、IP、MAC 这类信息本身就比较结构化，用确定性规则更稳，也更容易解释

### 4. `BM25F` + `char ngram` + `vector search`

做什么：

- 从所有模板里先召回一批可能相关的候选模板

输入：

- 改写后的文本

输出：

- 多路召回得到的候选模板集合

典型用例：

- 用户说法和模板样例不完全一致时，仍然能把相关模板召回来

算法与原因：

- `BM25F` 适合抓关键词和领域术语
- `char ngram` 适合处理语序变化和轻微口语化
- `vector search` 适合处理更弱的表达改写
- 三路组合比只靠一种方法更稳，不容易漏召回

### 5. `RRF` + 动态权重初排

做什么：

- 把多路召回结果合并成统一排序，并根据 query 复杂度自动调权

输入：

- 多路召回结果及其分数

输出：

- 初步排序后的模板候选列表

典型用例：

- 双条件、三条件 query 不会只因为文本像，就被简单模板抢走

算法与原因：

- `RRF` 适合融合不同召回器的结果，因为它不依赖不同分数的绝对量纲一致
- 动态权重会让复杂 query 更看重结构和参数覆盖，而不是只看文本相似度

### 6. 可选 `reranker`

做什么：

- 对 top-k 候选再做一轮更细的判断

输入：

- 初排后的少量候选模板

输出：

- 细排后的候选顺序
- 额外的 rerank 分

典型用例：

- 两个模板都很像时，继续拉开差距

算法与原因：

- 这是第二阶段精排
- 因为成本更高，所以只对少量候选使用
- 这样既能提升效果，又不会把整体性能拖得太重

### 7. 模板内 `slot_extractors`

做什么：

- 在候选模板已经基本确定后，按该模板自己的规则抽具体参数

输入：

- 当前 query
- 当前候选模板

输出：

- 该模板下的槽位结果

典型用例：

- 同样是 `topn`、`threshold`，不同模板可以有不同解释方式

算法与原因：

- 槽位抽取下沉到模板内部，是因为很多参数的含义依赖模板语境
- 这样比全局统一抽参更灵活，也更贴近真实业务

### 8. `slot_fit` + `constraint` + `structure_score`

做什么：

- 判断这个模板是不是“真的适合这条 query”

输入：

- 模板
- 模板内抽到的槽位
- 全局高信号条件

输出：

- 参数完整度分
- 约束一致性分
- 结构一致性分

典型用例：

- `cpu>80 且 内存>70` 这种 query，会优先匹配双条件模板，而不是单条件模板

算法与原因：

- `slot_fit` 看参数齐不齐
- `constraint` 看值合不合法
- `structure_score` 看 query 的条件结构和模板结构是否一致
- 这样能避免“文本像，但结构不对”的误匹配

### 9. 可选 LLM fallback

做什么：

- 在边界 case 上做有限兜底，不作为主链路

输入：

- 少量 top 候选
- 或已命中但缺少少量槽位的模板

输出：

- 模板裁决结果
- 或补充后的槽位结果

典型用例：

- 模板已经大概率对，但 `topn` 没抽到
- top1 和 top2 很接近时，做一次有限裁决

算法与原因：

- 只在窄范围触发，并优先要求结构化输出
- 这样既能利用 LLM 的理解能力，又不会让系统退化成不可控的全量大模型推理

### 10. 阈值与歧义判断

做什么：

- 把整个流程的结果收口成最终状态

输入：

- 最终候选分数
- 缺失槽位
- 歧义情况

输出：

- `matched`
- `partial`
- `unmatched`

典型用例：

- 模板基本对但缺少关键参数时返回 `partial`
- 两个模板太接近时返回 `unmatched`

算法与原因：

- 这是系统的安全阀
- 宁可保守一点，也不要把不确定结果硬判成命中，这样更适合真实业务落地

### 11. 高置信 SQL 路由

做什么：

- 为“模板 SQL”和“NL2SQL”之间做保守路由

输入：

- 用户 query
- 当前模板配置

输出：

- `route_to = template`：模板完整满足，可以直接走模板 SQL
- `route_to = nl2sql`：模板不满足或不够稳定，交给 NL2SQL

典型用例：

- `查询硬盘健康状态为正常的列表`
  如果资源类型、健康状态、查询动作和派生枚举都满足模板，则走模板
- `查询硬盘健康状态为未知的列表`
  如果状态语义不在模板可解释范围内，则回退 NL2SQL

算法与原因：

- 它复用当前多路召回，但门槛比普通 `match()` 更保守
- 分数低、top 候选歧义、结构分不足、模板约束不满足，都会直接回退
- 这样可以把模板能力作为“高置信快捷路径”，而不是强行覆盖所有 query

### 12. compact LLM 快速提参

做什么：

- 在高置信 top1/top3 候选上，用极小 payload 让 LLM 补少量槽位或拒绝模板

输入：

- 固定 system prompt
- 每次请求只传 `q / g / c`
  - `q`：原始 query
  - `g`：规则链路已抽到的公共槽位
  - `c`：top 候选的极简字段

输出：

- `{"r":"template","id":"...","s":{},"m":[],"cf":0.0}`
- 或 `{"r":"nl2sql","id":-1,"s":{},"m":[],"cf":0.0}`

典型用例：

- 模板已经大概率对，但某个自由文本或语义槽位规则没有抽到
- top1 是单指标模板，但 query 明确包含两个指标条件，LLM 必须返回 `nl2sql`

算法与原因：

- 固定规则放进 system prompt，减少每次请求 token
- user payload 不传 `description / utterances / must_terms` 这类大字段
- LLM 输出后仍然回到确定性校验，不满足模板就回退 NL2SQL

## 目录结构

```text
.
|-- configs/
|   |-- query_rewrite_rules.json
|   `-- templates.json
|-- src/template_capability/
|   |-- LLM/
|   |   |-- __init__.py
|   |   `-- compact_router.py
|   |-- config.py
|   |-- engine.py
|   |-- evaluation.py
|   |-- extractors.py
|   |-- fallback.py
|   |-- models.py
|   |-- openai_client.py
|   |-- rerankers.py
|   |-- rewrite.py
|   |-- scoring.py
|   |-- structured_output.py
|   |-- text_matching.py
|   |-- validation.py
|   `-- vector_index.py
|-- tests/
|   |-- fixtures/evaluation_cases.json
|   |-- test_config_validation.py
|   |-- test_engine.py
|   |-- test_engine_capabilities.py
|   |-- test_engine_core_enhancements.py
|   |-- test_evaluation_corpus.py
|   |-- test_evaluation_report.py
|   |-- test_openai_sdk_integration.py
|   |-- test_rerankers.py
|   |-- test_structured_fallback.py
|   |-- test_compact_llm_router.py
|   |-- test_sql_route_extensions.py
|   |-- test_template_constraints.py
|   `-- test_term_match_modes.py
|-- tools/
|   |-- evaluate_matcher.py
|   |-- lint_templates.py
|   `-- generate_eval_corpus.py
|-- main.py
`-- README.md
```

## 运行方式

安装：

```bash
pip install -e ".[dev]"
```

单条输入：

```bash
python main.py --input "过去24小时接口错包告警前10名"
```

代码内高置信 SQL 路由：

```python
from template_capability.engine import TemplateCapabilityEngine

engine = TemplateCapabilityEngine.from_file("configs/templates.json")
decision = engine.route_for_sql("查询硬盘健康状态为正常的列表")

payload = decision.to_dict()
if payload["route_to"] == "template":
    # 直接走模板 SQL
    slots = payload["result"]["slots"]
else:
    # 回退 NL2SQL
    pass
```

代码内 compact LLM 串接：

```python
from template_capability.engine import TemplateCapabilityEngine
from template_capability.LLM import route_with_compact_llm

engine = TemplateCapabilityEngine.from_file("configs/templates.json")

def real_llm_call(system_prompt, user_payload):
    # 这里替换成真实模型调用，返回 dict 或 JSON string
    return None

decision = route_with_compact_llm(
    engine,
    "查询硬盘健康状态为正常的列表",
    llm_call=real_llm_call,
)
```

交互模式：

```bash
python main.py --interactive
```

开启模板级 LLM 补参：

```bash
$env:DASHSCOPE_API_KEY="***"
python main.py --llm-slot-fallback --input "近24小时接口错误包告警前十"
```

运行测试：

```bash
python -m pytest -q
python tools/lint_templates.py
```

本地模板工作台：

```bash
npm install
npm run studio
npm run studio:test
```

Windows 下也可以直接双击：

```text
start-template-studio.cmd
```

## 输出契约

普通 `match()` 的输出仍然是 `MatchResult`，保持向后兼容。

示例：

```json
{
  "template_id": "alarm.interface.error.topn",
  "status": "matched",
  "score": 0.96,
  "query_mode": "metric_query",
  "slots": {
    "time_range": {
      "mode": "relative",
      "preset": "last_24h"
    },
    "topn": 10,
    "entity_type": "interface",
    "metric": "error_packet",
    "query_operator": "topn"
  },
  "missing_slots": [],
  "metadata": {
    "metric_code": "interface_error_alarm_topn"
  },
  "trace": {}
}
```

字段说明：

- `template_id`：命中的模板；未命中时为 `-1`
- `status`：`matched / partial / unmatched`
- `score`：最终融合分
- `query_mode`：模板所属能力大类，当前基本固定是 `metric_query`
  它回答的是“这是问数模板还是别的大类模板”
  它不负责区分 `count / topn / list`
- `slots`：抽出的槽位
- `missing_slots`：`partial` 时缺失的必填槽位
- `metadata`：模板透传字段
- `trace`：调试轨迹，不建议下游业务强依赖
  其中如果开启了 `query_rewrite`，会看到 `rewrite_trace`
  当前还会带 `shared_slots / global_slots`
  候选模板里还会记录 `unexpected_global_slots`
  用来解释“query 里有哪些高信号条件，但当前模板没接住”
  如果模板启用了组合槽位约束，还会看到 `requirement_issues`
  结构分细节会放在候选模板的 `trace.structure_details`
  如果开启了二阶段精排，还会看到 `rerank_score` 和 `trace.rerank_trace`

SQL 路由入口 `route_for_sql()` 返回 `TemplateRouteDecision`：

```json
{
  "route_to": "template",
  "reason": "template_satisfied",
  "result": {
    "template_id": "resource.health.list",
    "status": "matched",
    "score": 0.91,
    "query_mode": "metric_query",
    "slots": {
      "resource_type": "disk",
      "health_status_label": "normal",
      "healthStatus": 2,
      "query_operator": "list"
    },
    "missing_slots": [],
    "metadata": {},
    "trace": {}
  },
  "candidates": []
}
```

字段说明：

- `route_to`
  `template` 表示可直接走模板 SQL
  `nl2sql` 表示模板不够稳定或不满足，应该交给 NL2SQL
- `reason`
  当前路由原因，例如 `template_satisfied / low_template_confidence / template_requirements_not_satisfied`
- `result`
  与普通 `MatchResult` 结构一致
- `candidates`
  用于调试的候选模板轨迹

## 配置说明

主配置文件是 [templates.json](D:/GitHub/chat_pre_check_blank/configs/templates.json)。

如果你要自己新增模板，优先看单独的模板编写文档：

- [template_authoring_guide.md](D:/GitHub/chat_pre_check_blank/docs/template_authoring_guide.md)
- [slot_extractors_authoring_guide.md](D:/GitHub/chat_pre_check_blank/docs/slot_extractors_authoring_guide.md)
- [typical_template_examples.md](D:/GitHub/chat_pre_check_blank/docs/typical_template_examples.md)

顶层结构：

```json
{
  "matcher": {},
  "query_rewrite": {},
  "slot_extractors": {},
  "templates": []
}
```

### matcher

示例字段：

- `match_threshold`
  最终总分达到这个阈值才允许进入 `matched / partial`
  值越低，召回更激进，但误匹配会增加
- `ambiguity_margin`
  top1 和 top2 的分差小于这个值时直接返回 `-1`
  用来避免两个模板都像时硬选错
- `recall_top_k`
  每一路召回保留多少候选进入融合
  模板规模在几百到一两千时，`20-50` 通常够用
- `weights`
  最终重排时各子分数的基础权重
  当前支持：`lexical / sample / vector / fusion / slot_fit / constraint / structure`
- `lexical_field_weights`
  BM25F 的字段权重
  当前字段：`description / utterances / must_terms`
  一般 `must_terms` 应该最高，因为它决定模板语义锚点
- `fusion_rrf_k`
  RRF 的倒数排序参数
  值越大，不同召回路之间的 rank 差异被压得越平
- `vector.dimension`
  向量维度
  当前主流程按 `512` 维设计，后续替换真实向量接口时保持一致即可
- `vector.provider`
  向量后端实现
  当前支持 `local_tfidf / hashing / remote`
  默认推荐 `local_tfidf`
  `remote` 需要额外提供 `base_url / model / api_key` 或 `api_key_env`
- `reranker`
  二阶段精排器
  默认关闭，老配置零影响
  当前支持 `none / term_overlap / openai_compatible`
  常见写法是只对初排 top `10-30` 做 rerank
  如果启用但没有显式写 `weights.rerank`，系统会补一个保守默认值
- `blocked_terms`
  全局拦截词
  命中后直接返回 `unmatched`
  适合放 `报告 / 分析 / 总结 / 根因 / 预测` 这类明确非问数词
  默认写字符串即可，等价于 `substring`
  也支持对象写法：`{"term": "idc", "match_mode": "whole_word"}`
- `llm_fallback`
  模板选择阶段的 LLM 兜底
  只建议在 `partial` 或接近阈值的 `unmatched` 上窄触发
  不建议作为主链路能力
- `llm_slot_fallback`
  模板已命中后的 LLM 补参开关
  这是当前更推荐的用法
  只在 top1 模板比较稳定，但还有少量关键槽位没抽到时补参
- `template_route`
  面向 SQL 路由的保守门槛
  `route_for_sql()` 和 compact LLM 串接会使用它
  常见字段是 `min_score / min_structure_score / top_k`

`weights` 的含义：

- `lexical`
  BM25F 词法匹配分
  适合稳住领域关键词、固定词组
- `sample`
  基于模板示例问法的 `char ngram` 相似度
  对语序变化、轻微口语化更稳
- `vector`
  向量召回分
  用来处理更弱的表达改写
- `fusion`
  多路召回经 RRF 融合后的排序分
  用来减少单一路召回偏置
- `rerank`
  二阶段精排分
  只在打开 `matcher.reranker` 时生效
  推荐只作为 top-k 候选之间的细排信号，不要一开始就给太高
- `slot_fit`
  槽位覆盖度
  query 抽到的关键参数越齐，分越高
- `constraint`
  模板约束分
  包括 `must_terms` 命中和 `slot_constraints` 一致性
- `structure`
  结构一致性分
  当前会先做一层全局高信号槽位抽取，再判断“query 有的条件模板接不接得住”
  用来惩罚“query 有的条件模板接不住”或“模板要求的关键过滤条件没给全”
  对关键过滤条件缺失、组合约束未满足、互斥槽位冲突，会比低信号缺失惩罚更重

模板约束支持平滑升级：

- `required_slots`
  单槽位必填，老模板继续按这个字段工作
- `required_one_of`
  一组槽位至少满足一个，适合 `ip / mac / name 三选一`
- `conditional_required`
  条件必填，适合“给了 A 就必须给 B”
- `mutually_exclusive_slots`
  互斥槽位组，适合多种定位方式不能同时出现的场景

兼容说明：

- 老模板完全不用改
- `missing_slots` 字段继续保留
- 更细的规则缺失或冲突原因进入 `trace.requirement_issues`

建议：

- `match_threshold` 不要太低，否则误匹配会明显增加
- `blocked_terms` 主要放非问数意图词，如 `报告 / 分析 / 根因 / 预测`
- `llm_fallback` 只建议在 `partial` 或接近阈值的 `unmatched` 上窄触发
- `llm_slot_fallback` 只建议在 top1 模板已稳定命中、但缺少少量关键参数时触发

文本匹配模式支持：

- `substring`
  默认模式，完全兼容旧配置
- `whole_word`
  适合英文缩写、编码、设备名等 token 化表达
- `exact`
  适合必须整句精确匹配的极少数场景

`llm_slot_fallback` 细项：

- `enabled`
  是否启用模板级 LLM 补参
- `max_missing_slots`
  最多允许缺多少个必填槽位时触发 LLM
  建议控制在 `1-2`
- `min_score`
  top1 模板分数至少达到多少才允许触发 LLM
  这能避免把 LLM 用在本来就不稳定的命中上
- `allow_on_matched`
  即使已经 `matched`，是否还允许 LLM 二次补参
  默认建议关闭，除非你确实需要补充可选槽位

`template_route` 示例：

```json
{
  "matcher": {
    "template_route": {
      "min_score": 0.62,
      "min_structure_score": 0.3,
      "top_k": 1
    }
  }
}
```

字段说明：

- `min_score`
  候选模板至少达到这个分数，才允许进入模板 SQL 路由
- `min_structure_score`
  结构分下限，用来挡住“文本很像但条件结构接不住”的模板
- `top_k`
  compact LLM 最多查看几个候选
  默认建议 `1`
  只有 top1/top2 差距很小且你愿意多付一点时延时，再调到 `2-3`

### query_rewrite

`query_rewrite` 是一个独立的前置改写模块。

它的职责不是选模板，也不是抽槽位，而是在进入召回前先把行业黑话、内部简称、别名统一改写成模板侧更稳定的表达。

推荐放在这里的场景：

- 全局通用别名
  例如 `北二小 -> 北京第二小学`
- 领域简称或黑话
  例如 `错包 -> 错误包`
- 内部系统名或厂商映射
  例如 `思科设备 -> cisic`

不推荐放在这里的场景：

- 只在单个模板里才有意义的同义词
- 需要抽取自由值的槽位
- 依赖上下文推断才能决定改成什么的复杂语义

当前实现是基于 Trie 状态机的短语改写，特点是：

- 支持最长匹配
- 支持多轮改写
- 支持 `whole_word` 边界匹配
- 支持独立词典文件
- 支持词典文件热加载
- 命中轨迹会进入 `trace.rewrite_trace`

示例：

```json
{
  "query_rewrite": {
    "enabled": true,
    "max_passes": 2,
    "dictionary_path": "configs/query_rewrite_rules.json",
    "reload_on_change": true,
    "rules": [
      {
        "rule_id": "alias.school.short_name",
        "source": "北二小",
        "target": "北京第二小学"
      },
      {
        "rule_id": "alias.vendor.cisco",
        "source": "思科设备",
        "target": "cisic"
      },
      {
        "rule_id": "alias.cpu_typo",
        "source": "cup",
        "target": "cpu",
        "match_mode": "whole_word"
      }
    ]
  }
}
```

字段说明：

- `dictionary_path`
  可选外部词典文件路径；适合把行业黑话和别名单独维护
- `reload_on_change`
  是否在运行时检测词典文件变化并热加载
- `match_mode`
  当前支持 `substring` 和 `whole_word`

建议：

- 中文短语默认继续用 `substring`
- 英文缩写、设备编码、厂商简称这类 token 化表达，再考虑 `whole_word`

### slot_extractors

根级 `slot_extractors` 现在主要用于兼容旧配置；主路径推荐把参数抽取下沉到每个模板的 `slot_extractors`。

槽位抽取内置四类抽取器：

- `keyword_value`
- `regex`
- `time_range`
- `metric_conditions`

`keyword_value` 适合：

- 区域
- 指标
- 查询算子
- 严重级别
- 有限枚举值

`regex` 适合：

- `topn`
- 数字类参数
- 格式稳定的标识

`time_range` 适合：

- 今天、昨天、前天
- 最近 N 小时 / N 天 / N 周 / N 月
- 过去 N 天
- 日期和日期范围

`metric_conditions` 适合：

- 单指标阈值条件
- 多指标阈值条件
- 单/多指标模板区分

模板内 `keyword_value` 示例：

```json
{
  "slot_extractors": {
    "time_range": {
      "extractors": [
        {
          "type": "keyword_value",
          "cases": [
            {
              "terms": ["近24小时", "过去24小时", "24小时内"],
              "value": {
                "mode": "relative",
                "preset": "last_24h"
              }
            }
          ]
        }
      ]
    }
  }
}
```

如果 case 之间有包含关系，建议开启最长匹配：

```json
{
  "resource_type": {
    "extractors": [
      {
        "type": "keyword_value",
        "match_policy": "longest",
        "cases": [
          {"terms": ["端口"], "value": "generic_port"},
          {"terms": ["FC端口", "FCoE端口", "SAS端口"], "value": "storage_port"},
          {"terms": ["以太网端口"], "value": "ethernet_port"}
        ]
      }
    ]
  }
}
```

模板内 `regex` 示例：

```json
{
  "topn": {
    "extractors": [
      {
        "type": "regex",
        "patterns": [
          {
            "pattern": "(?:top\\s*|前)\\s*(\\d+)",
            "group": 1,
            "value_type": "int",
            "min": 1,
            "max": 1000
          }
        ]
      }
    ]
  }
}
```

通用时间抽取示例：

```json
{
  "time_range": {
    "extractors": [
      {
        "type": "time_range",
        "include_absolute": true
      }
    ]
  }
}
```

多指标条件抽取示例：

```json
{
  "metric_conditions": {
    "extractors": [
      {
        "type": "metric_conditions",
        "metrics": [
          {"terms": ["内存利用率", "内存"], "value": "memory_usage"},
          {"terms": ["CPU利用率", "cpu"], "value": "cpu_usage"}
        ]
      }
    ]
  }
}
```

抽取结果形如：

```json
[
  {"metric": "memory_usage", "operator": ">", "value": 50.0},
  {"metric": "cpu_usage", "operator": ">", "value": 90.0}
]
```

### templates

模板字段：

- `template_id`：唯一标识
- `query_mode`：模板所属能力大类
- `description`：模板说明
- `utterances`：示例表达，用于召回和相似度
- `required_slots`：必填槽位
- `optional_slots`：可选槽位
- `required_one_of`：一组槽位至少命中一个
- `conditional_required`：条件必填规则
- `mutually_exclusive_slots`：互斥槽位组
- `must_terms`：必须出现的语义组；每组任一词命中即可
- `negative_terms`：模板级负向词
- `slot_constraints`：槽位约束
- `slot_extractors`：该模板自己的参数抽取规则
- `slot_defaults`：最终输出前补齐的默认执行参数
- `derived_slots`：由多个语义槽位派生出的后端槽位
- `slot_validations`：列表型槽位或集合槽位的数量约束
- `llm_slot_extraction`：该模板的可选 LLM 补参配置
- `metadata`：业务透传字段

重点字段解释：

- `query_mode`
  这是粗粒度能力分类
  在当前工程里通常固定写 `metric_query`
  它的作用主要是告诉下游“这是问数模板”
  不要把它当成 `count / topn / list` 这类查询形态字段
- `must_terms`
  这是模板的语义锚点
  每组里命中任意一个词就算该组通过
  如果一个模板很容易和别的模板打架，先加固这里
  每个词既可以直接写字符串，也可以写成带 `match_mode` 的对象
- `slot_constraints`
  用来限制抽出来的槽位值必须落在模板允许范围内
  比如 `query_operator` 必须是 `list`，或者 `severity` 必须是 `critical`
- `query_operator`
  这是细粒度查询形态
  真正区分“问数量 / 问排行 / 问列表”的是它
  所以在问数场景里，它通常比 `query_mode` 更重要
- `slot_extractors`
  当前推荐的参数定义位置
  模板需要什么参数，就在模板内定义什么参数
  不追求全局统一参数字典
- `llm_slot_extraction`
  控制该模板是否允许 LLM 补参，以及补哪些槽位
  这是模板级开关，不是全局一刀切
- `slot_defaults`
  只在最终输出和 SQL 路由前补齐
  它不会参与早期召回，避免默认值把不该命中的模板垫高
- `derived_slots`
  适合表达“同一个语义值在不同资源类型下对应不同后端枚举”
  例如 `resource_type=disk + health_status_label=normal -> healthStatus=2`
- `slot_validations`
  适合限制集合槽位数量
  例如单指标模板要求 `metric_conditions.exact_items = 1`
  多指标模板要求 `metric_conditions.min_items = 2`

模板示例：

```json
{
  "template_id": "device.offline.count",
  "query_mode": "metric_query",
  "description": "查询离线设备数量",
  "utterances": [
    "昨天华东离线设备数",
    "近24小时广州离线设备数量"
  ],
  "required_slots": ["time_range", "region_id", "query_operator"],
  "optional_slots": [],
  "must_terms": [
    ["设备"],
    ["离线", "掉线"],
    ["多少", "有多少", "数量", "数", "总数"]
  ],
  "negative_terms": ["原因", "根因", "报告", "总结", "预测"],
  "slot_constraints": {
    "entity_type": ["device"],
    "metric": ["offline_count"],
    "query_operator": ["count"]
  },
  "slot_extractors": {
    "time_range": {
      "extractors": []
    }
  },
  "slot_defaults": {
    "page_no": 1,
    "page_size": 100
  },
  "derived_slots": [],
  "slot_validations": {},
  "llm_slot_extraction": {
    "enabled": true,
    "slots": ["time_range", "region_id", "query_operator"],
    "instructions": "仅补充时间、区域和数量算子，不要发明额外条件。"
  },
  "metadata": {
    "metric_code": "device_offline_count"
  }
}
```

多属性联动示例：

```json
{
  "template_id": "resource.health.list",
  "required_slots": [
    "resource_type",
    "health_status_label",
    "healthStatus",
    "query_operator"
  ],
  "derived_slots": [
    {
      "slot_name": "healthStatus",
      "source_slots": ["resource_type", "health_status_label"],
      "mapping": [
        {
          "when": {"resource_type": "storage_pool", "health_status_label": "normal"},
          "value": 1
        },
        {
          "when": {"resource_type": "disk", "health_status_label": "normal"},
          "value": 2
        }
      ]
    }
  ]
}
```

这里不要直接让“正常”抽成 `healthStatus`。先抽语义槽位，再由 `derived_slots` 做确定性派生，才能处理不同资源类型下枚举值不同甚至相反的问题。

单/多指标模板示例：

```json
{
  "template_id": "storage.single_metric.list",
  "required_slots": ["query_operator", "metric_conditions"],
  "slot_validations": {
    "metric_conditions": {"exact_items": 1}
  }
}
```

```json
{
  "template_id": "storage.multi_metric.list",
  "required_slots": ["query_operator", "metric_conditions"],
  "slot_validations": {
    "metric_conditions": {"min_items": 2}
  }
}
```

这样 `查询今天内存利用率大于50% CPU利用率大于90%的分布式存储列表` 不会被单指标模板吞掉。

如果你需要边界匹配，也可以这样写：

```json
{
  "matcher": {
    "blocked_terms": [
      {"term": "idc", "match_mode": "whole_word"}
    ]
  },
  "templates": [
    {
      "must_terms": [
        [{"term": "idc", "match_mode": "whole_word"}],
        ["设备"]
      ],
      "negative_terms": [
        {"term": "分析", "match_mode": "exact"}
      ]
    }
  ]
}
```

`llm_slot_extraction` 字段说明：

- `enabled`
  该模板是否允许走 LLM 补参
- `slots`
  允许 LLM 补的槽位白名单
  不在这个列表里的槽位，即使缺失也不让 LLM 填
- `instructions`
  给 LLM 的模板级补充说明
  这里最好写成非常窄的约束，而不是泛泛的自然语言说明

## 匹配算法

当前不是严格字符串匹配，而是混合算法：

1. `BM25F` 多字段词法召回
2. `char ngram` 样本相似度
3. `vector search` 向量召回
4. 命中候选后，按模板本地 extractor 抽参
5. `RRF` 融合多路召回排名
6. 动态权重重排
7. `slot_fit_score` 槽位覆盖度
8. `constraint_score` 模板约束得分
9. `structure_score` 结构一致性得分
10. `slot_validations` 集合约束校验
11. `derived_slots` 派生联动槽位
12. `slot_defaults` 补齐执行默认值
13. 可选模板级 LLM 补参或 compact LLM 快速提参
14. SQL 路由入口按高置信门槛决定 `template / nl2sql`

最终分数：

```text
total_score =
  lexical * w1
  + sample * w2
  + vector * w3
  + fusion * w4
  + slot_fit * w5
  + constraint * w6
  + structure * w7
```

为什么这样设计：

- `BM25F` 比单文本 BM25 更适合模板多字段匹配
- `char ngram` 对中文短句、语序变化、口语化更稳
- `vector search` 处理更弱的表达改写
- `RRF` 能把多路召回的优势合并起来，减少单路偏置
- 动态权重会在多条件 query 上自动提高 `structure` 和 `slot_fit` 权重
- `slot_fit` 保证模板参数完整性
- `constraint` 防止“看起来像，但其实不是这个模板”
- `structure_score` 会同时惩罚两类问题：
  query 里多出的条件模板接不住；模板要求的关键过滤条件 query 没给全
- `slot_validations` 解决单/多指标模板混淆
- `derived_slots` 解决资源类型与状态枚举等联动映射
- `slot_defaults` 解决可选槽位缺失时的 SQL 执行默认值

## 向量接口接入

向量后端定义在 [vector_index.py](D:/GitHub/chat_pre_check_blank/src/template_capability/vector_index.py)。

当前默认实现：

- `LocalTfidfVectorProvider`
- `LocalHashVectorProvider`
- `InMemoryVectorIndex`

说明：

- `LocalTfidfVectorProvider`
  这是当前默认本地实现
  会基于模板语料统计 IDF，并把高价值 term 放进显式维度，剩余 term 走 overflow hashing
  对当前这种模板数不大、领域词明确的问数场景，比单纯 hashing 更稳
- `LocalHashVectorProvider`
  这是保留的轻量兼容实现
  适合极简本地测试或需要完全无状态 provider 的场景
- `RemoteEmbeddingProvider`
  用于后续接真实远端 embedding 接口
  主流程仍然保持 `512` 维可替换设计

后续替换真实向量服务时，建议保持同样的职责边界：

1. 离线为模板文档建立向量索引
2. 在线只对 query 编码一次
3. 返回 `[(template_id, score)]`
4. 向量维度保持 `512`

最低需要实现的接口：

```python
class VectorSearchBackend(Protocol):
    def build(self, documents: dict[str, str]) -> None:
        ...

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        ...
```

接入建议：

- 不要在 `search()` 里重新为所有模板编码
- 不要把外部向量服务耦合到 `engine.py` 里
- 保持 `engine` 只依赖统一 backend 接口

## 开发流程

### 新增一个模板

推荐顺序：

1. 在 [templates.json](D:/GitHub/chat_pre_check_blank/configs/templates.json) 添加模板定义
2. 直接在该模板下定义 `slot_extractors`
3. 给模板补 `matched / partial / unmatched` 样本
4. 运行批量评测
5. 根据失败样本调模板和阈值

### 新增一个槽位

推荐顺序：

1. 先判断这个槽位是否能通过 `keyword_value / regex / time_range / metric_conditions` 表达
2. 如果可以，优先只改配置
3. 如果配置表达不了，再扩 `extractors.py`
4. 补抽取测试和回归样本

不要做的事：

- 不要把具体区域、指标、协议硬编码回 Python
- 不要把模板追问文案混入能力层
- 不要为了一个模板改全局逻辑
- 不要先做全局通用抽参，再去套模板；优先按模板内规则抽参
- 不要把依赖资源类型的后端枚举直接写进单个语义 extractor；优先用 `derived_slots`

### 调权重和阈值

优先调：

- `match_threshold`
- `ambiguity_margin`
- `weights.structure`
- `weights.slot_fit`
- `weights.lexical`
- `weights.fusion`
- `lexical_field_weights.must_terms`
- `must_terms`
- `slot_constraints`

不建议优先靠大量堆同义词解决误匹配。先看是不是模板约束不够、负向词不够、评测样本不够。

## 批量评测

评测入口是 [evaluate_matcher.py](D:/GitHub/chat_pre_check_blank/tools/evaluate_matcher.py)。

命令：

```bash
python tools/evaluate_matcher.py
python tools/evaluate_matcher.py --json --output reports/evaluation_report.json
python tools/evaluate_matcher.py --fail-on-errors
```

报告内容：

- 总体通过率
- `matched / partial / unmatched / not_full_match` 分桶结果
- 每个模板的样本覆盖情况
- 没有任何评测样本的新模板
- 失败样本明细

适合接入：

- 本地开发回归
- CI 校验
- 模板发布前检查

## 评测语料

静态评测集在 [evaluation_cases.json](D:/GitHub/chat_pre_check_blank/tests/fixtures/evaluation_cases.json)。

分四类：

- `matched`：必须完整命中
- `partial`：必须命中模板但缺少槽位
- `unmatched`：必须返回 `-1`
- `not_full_match`：不能返回 `matched`

当前测试覆盖：

- 槽位抽取
- 最长 keyword 匹配
- 通用时间抽取
- 多属性联动派生槽位
- 可选槽位默认值
- 单/多指标条件区分
- compact LLM 路由串接
- 多表达泛化
- 非问数意图拦截
- 歧义模板返回 `-1`
- `1000` 模板规模回归
- 可替换向量后端
- 批量评测报告统计

## LLM 造样本

LLM 默认只用于离线造测试样本，不用于在线匹配主链路。

如果需要在线兜底，当前代码支持两类 LLM 钩子：

1. `llm_fallback`：模板选择兜底
2. `llm_slot_fallback`：模板已命中后的定向补参

模板级补参更适合你当前这种“参数定义强依赖模板”的场景。

在线使用时建议遵守以下约束：

1. 默认关闭
2. 只在 `partial` 或接近阈值的 `unmatched` 上触发
3. 只看前 `N` 个候选模板，不做全量模板推断
4. 优先用于补关键槽位或在 top 候选里做裁决
5. 不要覆盖 `blocked_terms` 命中的 query

当前已提供一个兼容 OpenAI 协议的模板级补参实现：

- [fallback.py](D:/GitHub/chat_pre_check_blank/src/template_capability/fallback.py) 里的 `OpenAICompatibleFallbackResolver`
- [fallback.py](D:/GitHub/chat_pre_check_blank/src/template_capability/fallback.py) 里的 `OpenAICompatibleTemplateSlotResolver`

当前实现方式：

- 使用 `openai` Python SDK
- 通过 `base_url` 指向兼容 OpenAI 协议的供应商
- 当前默认示例就是 DashScope 的兼容地址
- 在线调用时会优先请求 `json_schema` 结构化输出
- 如果供应商或模型不支持，再自动回退到 `json_object`
- 即使回退了，仍然会复用当前的 `<think> / fenced json / balanced json` 清洗逻辑兜底

最小可用方式：

1. 设置 `DASHSCOPE_API_KEY`
2. 使用 [main.py](D:/GitHub/chat_pre_check_blank/main.py) 的 `--llm-fallback` 或 `--llm-slot-fallback`
3. 只让它在模板已经比较稳定命中、但缺少少量关键参数时补参

当前 `main.py` 额外支持：

- `--llm-fallback`
- `--llm-base-url`
- `--llm-model`
- `--llm-timeout`
- `--llm-slot-fallback`
- `--llm-slot-base-url`
- `--llm-slot-model`
- `--llm-slot-timeout`

脚本在 [generate_eval_corpus.py](D:/GitHub/chat_pre_check_blank/tools/generate_eval_corpus.py)。

命令：

```bash
$env:DASHSCOPE_API_KEY="***"
python tools/generate_eval_corpus.py
```

默认参数：

- `base_url = https://coding.dashscope.aliyuncs.com/v1`
- `model = qwen3-coder-plus`

这个脚本会：

1. 读取当前模板配置
2. 调用兼容 OpenAI 协议的模型
3. 生成 `matched / partial / unmatched / not_full_match` 候选样本
4. 输出到 `tests/fixtures/`

注意：

- 生成样本后不要直接全量信任
- 需要人工筛掉不合理样本
- 最终应该把确认后的样本固化到静态评测集

## LLM 补参评测

已经提供一个在线 benchmark 脚本：

- [benchmark_llm_slot_fallback.py](D:/GitHub/chat_pre_check_blank/tools/benchmark_llm_slot_fallback.py)

命令：

```bash
$env:DASHSCOPE_API_KEY="***"
python tools/benchmark_llm_slot_fallback.py
python tools/benchmark_llm_slot_fallback.py --json
```

这个脚本会对一组“规则抽参容易漏，但模板语义其实已经命中”的样例做对比：

- 不开 LLM 时的结果
- 开 LLM 补参后的结果
- 每条 query 的总耗时
- 每次 LLM 调用自身的耗时

另外还提供了一条默认跳过的在线测试：

- [test_live_llm_slot_fallback.py](D:/GitHub/chat_pre_check_blank/tests/test_live_llm_slot_fallback.py)

手动运行：

```bash
$env:RUN_LIVE_LLM_TESTS="1"
$env:DASHSCOPE_API_KEY="***"
python -m pytest -q tests/test_live_llm_slot_fallback.py
```

## Compact LLM 路由

除了旧的模板选择 fallback 和模板级补参，现在还提供了一个更适合线上低延迟场景的 compact LLM 串接模块：

- [LLM/compact_router.py](D:/GitHub/chat_pre_check_blank/src/template_capability/LLM/compact_router.py)

它的目标不是让 LLM 做全量模板推理，而是：

1. 先用当前多路召回快速得到 top1/top3
2. 只把极简候选 payload 给 LLM
3. LLM 只返回模板 id 和槽位
4. 系统再做确定性校验
5. 不满足模板就回退 NL2SQL

固定规则放在 `COMPACT_SLOT_ROUTER_SYSTEM_PROMPT` 中，每次请求只需要传：

```json
{
  "q": "查询硬盘健康状态为正常的列表",
  "g": {
    "resource_type": "disk"
  },
  "c": [
    {
      "id": "resource.health.list",
      "sc": 0.82,
      "st": 0.91,
      "req": ["resource_type", "health_status_label", "query_operator"],
      "cur": {"resource_type": "disk"},
      "miss": ["health_status_label"],
      "schema": {
        "health_status_label": ["normal", "fault"],
        "query_operator": ["list"]
      },
      "derive": {
        "healthStatus": ["resource_type", "health_status_label"]
      }
    }
  ]
}
```

LLM 只需要返回：

```json
{"r":"template","id":"resource.health.list","s":{"health_status_label":"normal","query_operator":"list"},"m":[],"cf":0.93}
```

代码接入：

```python
from template_capability.engine import TemplateCapabilityEngine
from template_capability.LLM import route_with_compact_llm

engine = TemplateCapabilityEngine.from_file("configs/templates.json")

def real_llm_call(system_prompt, user_payload):
    # 替换为真实模型调用；返回 dict 或 JSON string
    return None

decision = route_with_compact_llm(
    engine,
    "查询硬盘健康状态为正常的列表",
    llm_call=real_llm_call,
)
```

如果 `real_llm_call` 返回 `None`、选择未知模板、结构分不够、或最终规则校验不通过，都会返回 `route_to = nl2sql`。

## 代码入口说明

主要文件职责：

- [main.py](D:/GitHub/chat_pre_check_blank/main.py)：CLI 入口
- [template-studio/server.js](D:/GitHub/chat_pre_check_blank/template-studio/server.js)：Node.js 本地模板工作台服务
- [template-studio/public/app.js](D:/GitHub/chat_pre_check_blank/template-studio/public/app.js)：模板工作台前端交互
- [engine.py](D:/GitHub/chat_pre_check_blank/src/template_capability/engine.py)：主匹配流程
- [compact_router.py](D:/GitHub/chat_pre_check_blank/src/template_capability/LLM/compact_router.py)：compact LLM 路由和极简 payload 构造
- [extractors.py](D:/GitHub/chat_pre_check_blank/src/template_capability/extractors.py)：标准化和槽位抽取
- [scoring.py](D:/GitHub/chat_pre_check_blank/src/template_capability/scoring.py)：召回、相似度、约束打分
- [vector_index.py](D:/GitHub/chat_pre_check_blank/src/template_capability/vector_index.py)：向量后端抽象和默认实现
- [evaluation.py](D:/GitHub/chat_pre_check_blank/src/template_capability/evaluation.py)：批量评测和报告
- [config.py](D:/GitHub/chat_pre_check_blank/src/template_capability/config.py)：配置加载
- [validation.py](D:/GitHub/chat_pre_check_blank/src/template_capability/validation.py)：配置 lint 和启动期校验
- [models.py](D:/GitHub/chat_pre_check_blank/src/template_capability/models.py)：核心数据模型

## 常见问题

### 为什么 query 看起来像模板，却返回 `-1`

常见原因：

- 命中了全局 `blocked_terms`
- top1 分数低于 `match_threshold`
- top1 和 top2 太接近，触发 `ambiguity_margin`
- 模板约束不足导致被误判成歧义

### 为什么 query 命中了模板，但只是 `partial`

因为模板语义已经比较明确，但缺少必填槽位，比如：

- 缺时间
- 缺 `topn`
- 缺区域
- 缺数量意图

### 为什么要把 `query_operator` 当作必填槽位

这是为了把“看指标”与“问数量/问排行”区分开。否则很多非问数 query 会被错误当成完整命中。

### `query_mode` 和 `query_operator` 有什么区别

- `query_mode` 是能力大类
  例如当前仓库里基本都是 `metric_query`
- `query_operator` 是问法形态
  例如 `count / topn / list`

在你当前这个只做问数的项目里：

- `query_mode` 更多是对外透传的保留字段
- `query_operator` 才是真正影响模板区分和查询执行形态的关键槽位

## 当前状态

当前仓库已具备：

- 配置驱动模板匹配
- 通用槽位抽取
- 高置信 SQL 路由
- compact LLM 快速提参串接
- 多属性联动派生槽位
- 可选槽位默认值
- 单/多指标条件校验
- 可替换 `512` 维向量后端
- 静态评测集
- 批量评测报告
- LLM 离线造样本脚本

当前本地回归：

```bash
python -m pytest -q
python tools/evaluate_matcher.py
npm run studio:test
```

当前回归结果：

- Python：`191 passed, 1 skipped`
- Node Template Studio：`20 passed`
- 静态评测：`75 / 75`

## Template Studio

新增了一个本地 Node.js 模板工作台，位置在 [template-studio](D:/GitHub/chat_pre_check_blank/template-studio)。

它提供这几件事：

- 导入现有 `templates.json`
- 首页优先做批量测试、单条调试和仓库评测集回归
- 支持导入 `txt / json / jsonl` 批量测试文件
- 支持批量导入问句或问答文件，批量生成模板建议
- 支持对一批问句或问答做“持续优化”，自动复测、识别失败样本、生成候选模板，并只接受能提升目标指标的改动
- 图形化编辑常用模板字段，高级 JSON 折叠收纳
- 用兼容 OpenAI 协议的大模型把一句真实 query 拆成模板建议
- 返回“为什么这样设计”的说明，方便人工微调模板
- 支持一键跑仓库内置评测集
- 支持把模板原始 JSON 放到弹层查看，避免编辑区过长
- 支持编辑 `slot_defaults / derived_slots / slot_validations`
- 提供 `/api/route-for-sql` 用于调试 SQL 路由决策
- 导出当前模板文件
- 保存一份工作区快照，便于下次继续编辑

推荐的本地调试顺序：

1. 先在首页跑批量测试或仓库评测集，看哪些 query 没命中
2. 用单条测试看前置改写、抽槽和 top candidates
3. 再去模板编辑器微调模板
4. 如果是新场景，再用单句或批量模板生成补模板骨架
5. 如果已经有一批真实问句或问答，可以直接跑“持续优化”，让工作台按目标通过率自动迭代到达标或达到轮数上限

批量导入当前支持这些格式：

- 纯文本：一行一条 query
- `json` 数组：每条记录支持 `text / query / question`
- `jsonl`：每行一个对象
- 结构化测试样例：额外支持 `expected_template_id / expected_status`
- 当前仓库的 [evaluation_cases.json](D:/GitHub/chat_pre_check_blank/tests/fixtures/evaluation_cases.json) 也能直接导入

批量模板生成和持续优化时，如果输入记录里有 `answer / response / output` 字段，工作台会把它当成意图解释和字段边界的辅助信息，一起喂给提示词。

持续优化当前支持三种目标指标：

- `pass_rate`：适合带 `expected_template_id / expected_status` 的回归集
- `coverage_rate`：适合只有问句或问答、没有标准标签的覆盖率观察
- `matched_rate`：适合你明确希望把 `partial` 也继续压成完整命中的场景

持续优化当前支持三种策略：

- `balanced`：默认模式，`partial` 更偏向修模板，`unmatched` 更偏向补模板
- `slot_completion_first`：优先修已有模板的 `utterances / must_terms / slot_extractors / required_slots`
- `new_template_first`：优先新增模板，适合你确认这批问句就是新意图时使用

持续优化的执行方式是：

1. 先跑当前配置在这批数据上的基线结果
2. 每轮挑失败样本或未覆盖样本
3. 调用模型生成“新增模板”或“替换现有模板”的候选
4. 立刻复测整批数据
5. 只接受能提升目标指标的改动
6. 直到达标或达到轮数上限

如果你当前最大的痛点是“已经命中了模板，但总是 `partial`”，推荐直接选：

- `target_metric = matched_rate`
- `optimization_strategy = slot_completion_first`

工作台示例文件：

- [batch_match_cases.jsonl](D:/GitHub/chat_pre_check_blank/template-studio/examples/batch_match_cases.jsonl)
- [batch_generate_questions.jsonl](D:/GitHub/chat_pre_check_blank/template-studio/examples/batch_generate_questions.jsonl)
- [optimization_qa_cases.jsonl](D:/GitHub/chat_pre_check_blank/template-studio/examples/optimization_qa_cases.jsonl)

模板生成这条链路额外做了两层兼容：

- 兼容 Qwen / GLM 这类 OpenAI-compatible 模型返回的额外思考文本、代码块包裹和 `<think>` 标签
- 支持在内部调试时关闭 SSL 校验

内部调试如果要默认关闭 SSL 校验，可以设置：

```bash
$env:TEMPLATE_STUDIO_INSECURE_SSL="1"
```

Python 侧兼容协议调用也支持：

```bash
$env:TEMPLATE_CAPABILITY_INSECURE_SSL="1"
```

推荐启动方式：

```bash
start-template-studio.cmd
```

默认地址：

```text
http://127.0.0.1:3847
```

### 已验证的 OpenAI-compatible 模型

下面这些模型已经在 `https://coding.dashscope.aliyuncs.com/v1` 下做过真实连通性验证：

- `qwen3.5-plus`
- `glm-5`
- `glm-4.7`
- `qwen3-coder-next`

验证范围包括：

- 普通 `chat.completions`
- `response_format: {"type": "json_object"}` 的 JSON 输出

推荐配置方式：

```bash
$env:DASHSCOPE_BASE_URL="https://coding.dashscope.aliyuncs.com/v1"
$env:DASHSCOPE_MODEL="qwen3.5-plus"
$env:DASHSCOPE_API_KEY="你的真实 key"
```

也可以在 Template Studio 页面里直接填写：

- `API Key`
- `Base URL`
- `Model`

不建议把真实 API key 明文写进仓库或 README。更稳的做法是：

- 用环境变量
- 或只保存在本地工作台输入框 / 本地存储里
