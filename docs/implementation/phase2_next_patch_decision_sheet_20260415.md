# Phase 2 Next Patch 决策单（2026-04-15）

本文档用于承接 `dev5_patch` 结果后的下一轮修复决策。它不是完整实验报告，而是面向实现推进的短决策单：先说当前最重要的判断，再说下一轮 patch 应优先改什么、看什么。

## 0. 执行状态（2026-04-15）

本决策单中的 4 个直接任务，当前已经完成了本地实现侧的第一轮落地：

- `get_goal_none`：
  - 已补 `raw frontier fallback`
  - 已补 `relaxed distance fallback`
  - 已补 `grounding_noop_reason / no_goal_reason / fallback_mode` 汇总
- `pending created/promoted`：
  - 已补 `pending_created_and_promoted_same_step`
  - created 不再因为当步 promotion 被冲掉
- monitor：
  - 已补 `llm_escalation` 候选策略
  - 当前 `smoothnav-full` 默认已切到 `heuristic-first / LLM-only-for-escalation`
- baseline regression：
  - 已完成一轮共享路径审计
  - 审计文档见 [phase2_shared_path_regression_audit_20260415.md](/Users/xin/Code/research/SmoothNav/docs/implementation/phase2_shared_path_regression_audit_20260415.md)

当前还没完成的是：

- 新一轮远端 gate 验收
- 用新结果判断这 4 个修复是否真正改善 `baseline-periodic / smoothnav-full / smoothnav-no-monitor`

## 1. 当前结论

当前系统已经越过了“profile 没有穿透到行为层”的阶段。

`smoothnav-full / smoothnav-no-monitor / smoothnav-rules-only / smoothnav-no-prefetch` 在 `SR / SPL / avg_high_level_calls / avg_low_level_calls / grounding_noop_rate / pending_* / override_ratio` 上已经明显分化，这说明：

- controller profile 差异已经能传到 primitive-action 层。
- 先前那种 “planner/monitor 有日志、下游没行为” 的 profile collapse 已被部分打破。

但当前新的主问题也很明确：

- `smoothnav-full` 不是“没起作用”，而是“起作用了，但净效果为负”。
- 当前 family 里最健康的参考线不是 `full`，而是 `smoothnav-no-monitor`。
- 当前最需要优先修的，已经不是“让 monitor 更活跃”，而是“让 grounding 更稳定、让 monitor 不要把本来有效的 heuristic 路径压坏”。

## 2. 这轮 `dev5_patch` 最关键的 4 个事实

### 2.1 `smoothnav-full` 的 grounding failure 现在主要不是 same-frontier

从 `step_traces/*.jsonl` 聚合出的 `grounding_noop_reason` 分布看：

- `baseline-periodic`: `goal_updated=25`, `get_goal_none=8`
- `smoothnav-full`: `goal_updated=8`, `get_goal_none=23`
- `smoothnav-no-monitor`: `goal_updated=15`, `get_goal_none=19`, `out_of_local_window=3`, `same_frontier_as_prev=4`
- `smoothnav-rules-only`: `goal_updated=34`, `same_frontier_as_prev=336`, `get_goal_none=190`
- `smoothnav-no-prefetch`: `goal_updated=10`, `get_goal_none=19`, `out_of_local_window=3`, `same_frontier_as_prev=4`

进一步看 `graph_debug.no_goal_reason`：

- `smoothnav-full`: `no_frontiers=22`, `no_candidate_frontiers=1`
- `smoothnav-no-monitor`: `no_frontiers=16`, `no_candidate_frontiers=3`
- `smoothnav-rules-only`: `no_frontiers=168`, `no_candidate_frontiers=22`

这说明 `full` 当前高 `grounding_noop_rate` 的主因，已经不是旧式的 same-frontier collapse，而是 `get_goal()` 返回 `None`，且绝大部分是 `no_frontiers`。

### 2.2 `same_frontier_rate` 不能直接解释 `grounding_noop_rate`

当前 `same_frontier_rate` 的定义是：

- 分母：所有 `grounding_events`
- 分子：其中 `selected_frontier_same_as_prev == True` 的事件数

因此：

- `get_goal_none`
- `projection_invalid`
- `out_of_local_window`

这些 grounding no-op 都会计入 `grounding_noop_rate`，但不会计入 `same_frontier_rate`。

所以 `smoothnav-full` 当前出现：

- `grounding_noop_rate = 0.742`
- `selected_frontier_same_as_prev_rate = 0.0`

并不矛盾，反而说明它主要坏在 `get_goal_none`。

### 2.3 `pending_promoted > pending_created` 说明统计口径仍有缺口

当前 `pending_created` 在 step trace 里的定义是：

- step 开始前 `pending is None`
- step 结束后 `pending is not None`

而 `pending_promoted` 的定义允许：

- step 开始前有 pending
- step 结束后 pending 被吃掉
- current strategy 变成原 pending strategy

这会漏掉一种情况：

- 某一步里 `pending` 被创建后又在同一步立刻 promotion

在这种情况下：

- `pending_created = False`
- `pending_promoted = True`

因此 `rules-only` 里出现：

- `pending_created_count = 1`
- `pending_promoted_count = 13`

不应被解释成方法能力，而应先视为 instrumentation 语义不完整。

### 2.4 baseline 自己也发生了明显回退

与上一轮 `dev5 baseline-periodic` 相比，这轮 `dev5_patch baseline-periodic` 明显变差：

- 旧：`SR = 0.8`, `SPL = 0.3847`
- 新：`SR = 0.4`, `SPL = 0.0854`

而且逐 episode 索引对照也能看到：

- 旧：`1, 1, 1, 1, 0`
- 新：`0, 1, 1, 0, 0`

当前结果包没有保存 Habitat 外部 `episode_id`，所以还不能写成“已 100% 证明是同一外部 episode 回退”；但在同一 `dev5` 队列口径下，这已经是一个足够强的 shared-path regression 信号。

## 3. 下一轮 patch 的 4 个直接任务

### 任务 1：先修 `get_goal_none` 主因

当前优先级最高。

目标不是继续泛泛地“增强 semantic bias”，而是先回答：

- 为什么 `full` 的 `get_goal_none` 主要是 `no_frontiers`？
- 这是地图/前沿生成的问题，还是更频繁的 monitor/planner 切换把系统带到了“当前无有效 frontier”的状态？
- baseline regression 是否也来自这条共享路径？

本轮 patch 应直接做：

- 在 `summary.json` 中补 `grounding_noop_reason` 聚合统计，至少区分：
  - `get_goal_none`
  - `same_frontier_as_prev`
  - `out_of_local_window`
  - `projection_invalid`
- 对 `get_goal_none` 再聚合 `graph_debug.no_goal_reason`，至少区分：
  - `no_frontiers`
  - `no_candidate_frontiers`
  - `no_bias_candidates`
  - `empty_bias_filtered_subset`
- 增加 per-step / per-episode frontier availability 统计：
  - `num_frontiers`
  - `num_candidate_frontiers`
  - `candidate_frontier_count_after_bias_filter`

重点文件：

- `base_UniGoal/src/graph/graph.py`
- `smoothnav/strategy_grounding.py`
- `smoothnav/control_metrics.py`

第一轮验收信号：

- `smoothnav-full` 的 `grounding_noop_rate` 明显下降
- 并且我们能清楚回答 `noop` 主要是哪一种，而不是只看到一个总比例

### 任务 2：修 `pending created/promoted` 统计

当前需要把“真实控制流”和“统计语义”彻底对齐。

本轮 patch 应直接做：

- 区分：
  - `pending_created_this_step`
  - `pending_promoted_this_step`
  - `pending_created_and_promoted_same_step`
- 在 `summary.json` 里分别输出：
  - `pending_created_count`
  - `pending_promoted_count`
  - `pending_created_and_promoted_count`

如果不这样做，`promoted > created` 会持续污染我们对 prefetch 的判断。

重点文件：

- `smoothnav/main.py`
- `smoothnav/control_metrics.py`
- 对应测试文件

第一轮验收信号：

- `pending_created_count >= pending_promoted_count` 不再因为统计缺口而被破坏
- 能区分 “pending 根本没活” 和 “活了，但当步即 promotion”

### 任务 3：把 monitor 改成 heuristic-first / LLM-only-for-escalation 的候选设计

这轮结果说明：

- `smoothnav-no-monitor` 是当前 family 里最健康的参考线
- `smoothnav-full` 的 monitor 已经不再“毫无后果”，但当前净效果为负
- `smoothnav-rules-only` 则说明纯高强度事件驱动已经超过当前 grounding/adoption 的可吸收能力

因此下一轮不建议继续把 monitor 当成默认主控器来加强，而应该先试一个更保守的候选设计：

- heuristic-first：
  - 常规 `CONTINUE`
  - 常规 `PREFETCH`
  - 常规 `direction reuse` / frontier-near 逻辑
  - 先由规则完成
- LLM-only-for-escalation：
  - 只有在真正 ambiguous / contradictory / high-value switch 时才调用 monitor
  - monitor 的职责先收缩到 `ESCALATE`

换句话说，下一轮最值得试的不是“让 monitor 多发言”，而是“让 monitor 只在真正需要时发言”。

重点文件：

- `smoothnav/controller_logic.py`
- `smoothnav/low_level_agent.py`
- `smoothnav/main.py`
- `smoothnav/controller_config.py`

第一轮验收信号：

- `smoothnav-full` 与 `smoothnav-no-monitor` 的差距缩小
- `full` 的 `avg_low_level_calls` 明显下降
- `full` 不再因为少量 monitor 决策而系统性掉到 `SR=0`

### 任务 4：专门检查 baseline 回退对应的 shared-path regression

这一项不能再顺带看，必须单独审。

当前最可能牵涉的共享文件正是这轮改动量最大的几处：

- `base_UniGoal/src/graph/graph.py`
- `smoothnav/strategy_grounding.py`
- `smoothnav/controller_logic.py`
- `base_UniGoal/src/agent/unigoal/agent.py`
- `smoothnav/control_metrics.py`

其中真正可能伤到 baseline 行为的，优先怀疑：

- `graph.get_goal()` 两阶段 bias 过滤后的 candidate 行为变化
- `apply_strategy()` 的投影与 no-op 处理
- executor side 的 epoch / stale temp-goal 清理

下一轮不要先拿大批 `full` 实验继续硬跑，而应先做 baseline 专项 sanity：

- old baseline vs new baseline
- 共享路径单独对比
- 优先回答“是 `graph/grounding` 回退，还是 executor state 回退”

第一轮验收信号：

- baseline 至少回到上一轮的大致健康区间
- 如果 baseline 仍明显退化，就暂停放大 `full` 的实验解释

## 4. 下一轮实施顺序

建议的最小顺序：

1. 先补 `grounding_noop_reason` / `no_goal_reason` 的 summary 聚合
2. 修 `pending created/promoted` 统计口径
3. 跑 baseline-only sanity，先确认 shared-path regression
4. 再做 heuristic-first / LLM-only-for-escalation monitor 原型
5. 最后再跑新一轮 `full / no-monitor / no-prefetch`

## 5. 当前主持口径

下一轮实验前，建议统一按下面的口径对外描述当前状态：

- 当前 SmoothNav 已经摆脱了“profile 没有穿透到行为层”的旧问题。
- 现在的主问题不是“没有控制”，而是“控制已经产生行为后果，但 `full` 这条主线的控制价值函数还没对齐”。
- 下一轮 patch 不应再泛化扩 monitor，而应先：
  - 修 `get_goal_none`
  - 修 `pending` 统计
  - 审 shared-path regression
  - 再把 monitor 收缩成更保守、更可信的模块
