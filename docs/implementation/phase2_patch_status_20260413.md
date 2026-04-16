# Phase 2 Patch Status And Doc Consolidation (2026-04-13)

本文用于回答一个更现实的问题：

**在第一轮 Phase 2 诊断之后，哪些问题已经真正改进到代码里了，哪些还只是“被观察清楚了但尚未彻底修完”。**

这份文档同时承担一次文档整理工作：

- 它吸收了此前 3 份细粒度 probe / trace 材料里仍然有效的核心证据
- 它明确标出哪些旧结论已经被新代码状态追平
- 它给出当前应保留的主文档与已合并删除的从属材料

关联主文档：

- [phase2_dev5_failure_taxonomy_20260413.md](/Users/xin/Code/research/SmoothNav/docs/implementation/phase2_dev5_failure_taxonomy_20260413.md)
- [phase2_code_level_diagnostic_20260413.md](/Users/xin/Code/research/SmoothNav/docs/implementation/phase2_code_level_diagnostic_20260413.md)
- [smoothnav_implementation_master_plan_20260412.md](/Users/xin/Code/research/SmoothNav/docs/planning/smoothnav_implementation_master_plan_20260412.md)

## 0.1 最新同步状态（2026-04-13，补齐清单版）

本文件前半部分保留了“第一轮 patch 后”的历史判断，但到 2026-04-13 这一轮补丁后，下面这些点已经需要按最新代码状态理解：

- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py) 现在已经不是单纯的 `base_score + weight * bias_score` 单阶段排序。
  - 当前实现先按 bias 做候选子集筛选，再在子集内按最终分数排序。
  - 对应的纯逻辑 helper 在 [frontier_scoring.py](/Users/xin/Code/research/SmoothNav/smoothnav/frontier_scoring.py)。
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py) 现在已经有 `consecutive_grounding_noops / same_frontier_reuse_count / forced_replan_due_to_grounding_failure` 这条显式控制回路，不再只是记录 `grounding_events`。
- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py) 里的 `GraphDelta.node_caption_changed` 已经由 `prev_node_captions -> node_captions_snapshot` 真正计算，不再只是占位字段。
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py) 已补上 `adopted_goal_before / adopted_goal_after / goal_epoch`，并通过 [executor_adoption.py](/Users/xin/Code/research/SmoothNav/smoothnav/executor_adoption.py) 统一处理 stale `temp_goal` 与 adoption transition。
- trace 现在支持 `enable_controller_trace` 开关。
  - `dev5/dev20` 可打开完整 controller trace。
  - 长跑可关闭 planner/monitor JSONL 与重型 step fields，避免结果目录膨胀。
- 清单要求的 6 个 generation-2 专项测试文件已经补齐：
  - [test_strategy_grounding_gen2.py](/Users/xin/Code/research/SmoothNav/tests/test_strategy_grounding_gen2.py)
  - [test_graph_goal_scoring_gen2.py](/Users/xin/Code/research/SmoothNav/tests/test_graph_goal_scoring_gen2.py)
  - [test_executor_adoption_gen2.py](/Users/xin/Code/research/SmoothNav/tests/test_executor_adoption_gen2.py)
  - [test_pending_promotion_gen2.py](/Users/xin/Code/research/SmoothNav/tests/test_pending_promotion_gen2.py)
  - [test_graph_delta_gen2.py](/Users/xin/Code/research/SmoothNav/tests/test_graph_delta_gen2.py)
  - [test_control_metrics_gen2.py](/Users/xin/Code/research/SmoothNav/tests/test_control_metrics_gen2.py)

当前最准确的口径是：

- P0/P1/P2/P3/P4 在代码层已经补到“清单要求的模块级实现”。
- 还没完成的不是代码缺口，而是新一轮远端 `dev5/dev20/dev100` gate 验收。

## 1. 这次整理后的文档口径

保留为主文档的有：

- `phase2_dev5_failure_taxonomy_20260413.md`
  - 作用：记录 184 服务器第一轮 `dev5` 的**修复前实验基线**
- `phase2_code_level_diagnostic_20260413.md`
  - 作用：记录基于修复前结果得到的**函数级问题定位**
- `phase2_patch_status_20260413.md`
  - 作用：记录**当前代码状态**，说明哪些诊断已经落实到实现中，哪些还没有

已合并删除的文档有：

- `phase2_get_goal_topk_probe_20260413.md`
- `phase2_smoothnav_full_ep0_step40_60_trace_20260413.md`
- `phase2_temp_goal_lifecycle_trace_20260413.md`

删除原因：

- 这 3 份文档的有效结论已经被当前代码中的 trace / metric / instrumentation 直接吸收
- 其中多处“当前还没有 `strategy_epoch` / `adopted_goal_source` / `semantic_bias_weight`”之类的表述已经过时
- 继续保留原文件会让读者混淆“修复前证据”和“当前代码状态”

## 2. 三份补充材料真正坐实了什么

虽然原始 probe 文档已合并，但它们建立的核心证据链仍然有效，而且应该保留为当前诊断的起点。

### 2.1 P0 的主故障点确实在 grounding，而不是 planner prompt

修复前的 probe 已经证明：

- `apply_strategy()` 并不是直接把语义 strategy 落到目标点
- 它只是把 `strategy.bias_position` 传给 `graph.get_goal(goal=bias)`
- `graph.get_goal()` 再对 frontier 做排序
- 原始实现里 `base_score` 明显强于 `bias_score`

因此存在两类坏情况：

- dead bias：bias 太远，语义项基本不起作用
- live-but-insufficient bias：bias 已经发生变化，但还不足以改写 top frontier 的身份

这条结论仍然成立，而且正是当前 P0 修改的依据。

### 2.2 修复前最关键的 room switch 失败发生在 executor 之前

修复前 `smoothnav-full ep0 step 46` 的关键事实是：

- `new_room_discovered`
- planner 已被调用
- strategy 从 `unexplored north` 切到 `bedroom`
- 但 `goal_before == goal_after`
- 同一步里 `temp_goal_override / stuck_goal_override / visible_target_override` 全是 false

这说明最先失效的不是 executor override，而是更早的：

`strategy -> apply_strategy() -> graph.get_goal() -> global_goals`

这条结论也仍然成立。

### 2.3 修复前 `temp_goal_override` 的持续性是真问题，但属于第二层

修复前 `episode_000002 step 982~988` 的窗口显示：

- `PREFETCH -> pending_promoted` 偶尔真的会发生
- 但 promotion 之后仍然可能先死在 grounding no-op
- 随后 `temp_goal_override` 会连续为真
- 而且它可以跨 strategy switch 持续存在

因此第二层问题并不是伪问题，只是优先级低于 P0。

## 3. 当前代码已经完成的修复

## 3.1 P0：grounding 已从黑盒变成可观测链路

已经完成的修改：

- [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py)
  - `apply_strategy()` 现在返回结构化 `GroundingResult`
  - 不再静默 no-op
  - 明确区分：
    - `get_goal_returned_none`
    - `local_window_out_of_range`
    - `frontier_same_as_before`
    - `goal_unchanged_after_projection`
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py)
  - `graph.get_goal()` 现在记录：
    - `selected_frontier`
    - `selected_frontier_same_as_previous`
    - `selected_frontier_score`
    - top-k frontier 的 `base_score / bias_score / final_score`
  - `semantic_bias_weight` 已参数化为 `graph_semantic_bias_weight`
  - `tuple` 类型的 `bias_position` 现在也会被正确识别
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py)
  - 每步 trace 会记录 `grounding_events`
  - 每步统计 `grounding_attempt_count / grounding_noop_count / grounding_changed_count`
- [control_metrics.py](/Users/xin/Code/research/SmoothNav/smoothnav/control_metrics.py)
  - 新增 `grounding_noop_rate`

这意味着：

- 现在 `step 46` 这类 case 不再只是“goal 没变”
- 我们已经可以直接看到它到底是：
  - `get_goal` 返回了空
  - 还是 frontier 没换
  - 还是 local projection 没变

### 3.2 P1：executor adoption 已经可观测，而且开始受高层 epoch 约束

已经完成的修改：

- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py)
  - `instance_discriminator()` 现在会记录：
    - `adopted_goal_source`
    - `adopted_goal_summary`
    - `adopted_goal_changed`
    - `strategy_epoch`
    - `temp_goal_epoch`
    - `stale_temp_goal_cleared`
    - `stuck_override_suppressed`
  - 高层 strategy epoch 变化时，会清理旧 epoch 的 `temp_goal`
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py)
  - 把 `strategy_epoch` 传进 `agent_input`
  - 加入 controller 侧 stuck suppression window
- [control_metrics.py](/Users/xin/Code/research/SmoothNav/smoothnav/control_metrics.py)
  - 新增 `executor_adoption_delay_steps`
  - 新增 override ratio：
    - `temp_goal_override_ratio`
    - `stuck_goal_override_ratio`
    - `global_goal_override_ratio`

这意味着：

- 旧文档里“当前 trace 没有 `adopted_goal_source` / `strategy_epoch`”的表述已经不再成立
- 现在我们终于能区分：
  - 是高层 goal 根本没落地
  - 还是落地了，但 executor 没采纳

### 3.3 P3：pending 路径不再只允许极窄的 direction -> room 早晋升

已经完成的修改：

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)
  - `maybe_promote_pending()` 现在按 strategy specificity 决定能否 promotion
  - 不再只支持 “current 非 room, pending 是 room”
  - 现在 `direction -> room/object`、`room -> object` 都可以提前晋升
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py)
  - auto-prefetch 已经放宽，不再只绑定 “当前必须是 room target”
  - 当 direction strategy 遇到 `new_room` 或 `room_object_count_increase` 时，也允许 prefetch
- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)
  - `frontier_reached` 现在有 `direction_reuse_limit`

### 3.4 P2：monitor 已经不再只盯着 new nodes

已经完成的修改：

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)
  - `maybe_call_monitor()` 现在会被以下事件触发：
    - `new_nodes`
    - `new_rooms`
    - `room_object_count_increase`
    - `frontier_near`
    - `no_progress`
    - `stuck`
- [low_level_agent.py](/Users/xin/Code/research/SmoothNav/smoothnav/low_level_agent.py)
  - monitor prompt 新增：
    - `CURRENT STRATEGY TYPE`
    - `NEW ROOMS`
    - `EVENT TYPES`
    - `FRONTIER NEAR`
    - `NO PROGRESS STEPS`
- [low_level_agent.py](/Users/xin/Code/research/SmoothNav/smoothnav/low_level_agent.py)
  - 规则 monitor 已经变成 consequential monitor：
    - stuck/no-progress 可触发 `ESCALATE`
    - frontier near 可触发 `PREFETCH`
    - room + 新对象可触发 `ADJUST`

### 3.5 P4：planner 只做了轻量增强，不是这轮主收口点

已经完成的修改：

- [planner.py](/Users/xin/Code/research/SmoothNav/smoothnav/planner.py)
  - prompt 明确要求优先 room/object evidence
  - object bias 现在取离 agent 最近的匹配对象
  - direction bias 现在优先落到该方向上的 frontier cluster，而不是只做固定偏移

但这一项仍然只是轻量增强，不是架构级重写。

## 4. 当前还没有彻底完成的部分

## 4.1 P0 还没有到“强语义落地”的最终形态

虽然 `semantic_bias_weight` 和 top-k 日志已经加上了，但 [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L875) 到 [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L903) 现在仍然是：

`final_score = base_score + semantic_bias_weight * bias_score`

也就是说，当前还没有做成更强的“两阶段选择”：

1. 先按 bias 缩候选 frontier 子集
2. 再在子集内按几何分排序

这意味着：

- P0 已经从“猜测”推进到了“可量化 + 可调权”
- 但还不能说已经完全解决 frontier-collapse

## 4.2 P1 已经可观测，但 executor flattening 是否显著下降还没验收

`strategy_epoch`、`adopted_goal_source`、stale temp-goal cleanup 都已经接入代码，但目前还没有新的远端结果证明：

- `temp_goal_override_ratio` 已明显下降
- `executor_adoption_delay_steps` 已明显下降
- full 与 no-monitor 的 primitive-action totals 已真正分岔

所以这一层的状态是：

- instrumentation 已就位
- 机制已补一轮
- 但还没有新的实验闭环

## 4.3 P2 的 GraphDelta 仍然是“更丰富了”，不是“完全成熟了”

[controller_events.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_events.py#L7) 里的 `GraphDelta` 现在已经有：

- `new_rooms`
- `room_object_count_increase_rooms`
- `frontier_near`
- `frontier_reached`
- `no_progress`
- `stuck`
- `event_types`

但它还不是完整的 rich semantic event layer。

目前最明显的缺口是：

- `node_caption_changed` 还只是字段占位
- 当前构造逻辑并没有真正去比较 caption 的变化来源

## 4.4 文档层也还有一个需要明确的事实

`phase2_dev5_failure_taxonomy_20260413.md` 和 `phase2_code_level_diagnostic_20260413.md` 依然有保留价值，但它们描述的是：

- 第一轮 `dev5` 的修复前实验表现
- 基于修复前结果得到的函数级定位

它们不是“当前代码已经修完后的状态说明”。

因此今后引用它们时，必须配合本文一起看。

## 5. 这次整理后，最值得保留的验收指标

当前最应该盯的指标是：

- `grounding_noop_rate`
- `executor_adoption_delay_steps`
- `temp_goal_override_ratio`
- `pending_created_count`
- `pending_promoted_count`
- `direction_reuse_count`

原有指标仍保留，但口径必须明确：

- `decision_delay`
  - 控制层 acknowledgment 指标
- `goal_update_delay`
  - 主循环 `global_goals` 变化指标
- 它们都不是 executor adoption 指标

## 6. 下一轮实验前，文档上最重要的共识

把当前代码和这批文档合在一起，最准确的一句话是：

**SmoothNav 当前已经完成了第一轮“把主故障显式化并接上修复钩子”的工作，但还没有完成第二轮“用新实验验证这些修复真的压低了 grounding collapse 和 executor flattening”。**

因此后续工作不应该再回到“大范围猜问题”的状态，而应该围绕下面的验证问题推进：

1. `smoothnav-full` 的 `grounding_noop_rate` 是否低于修复前基线
2. `executor_adoption_delay_steps` 是否下降
3. `temp_goal_override_ratio` 是否下降
4. `full / no-monitor / no-prefetch` 是否开始真正拉开
