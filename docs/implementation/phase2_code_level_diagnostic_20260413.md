# Phase 2 Code-Level Diagnostic Sheet (2026-04-13)

状态说明：

- 本文记录的是**第一轮 Phase 2 修复之前**，基于 `dev5` 结果做出的函数级定位。
- 其中很多诊断结论仍然有效，但文中个别“当前代码还没有某字段/某机制”的表述已经被后续实现追平。
- 阅读当前代码状态时，应与 [phase2_patch_status_20260413.md](/Users/xin/Code/research/SmoothNav/docs/implementation/phase2_patch_status_20260413.md) 配合使用。

本文把当前 `dev5` 的 failure analysis 收紧为代码级诊断单，目标不是重复讲故事，而是回答下面这个更具体的问题：

**问题到底卡在什么函数、什么条件、什么变量上，以及下一步该改哪里。**

关联文档：

- [phase2_dev5_failure_taxonomy_20260413.md](/Users/xin/Code/research/SmoothNav/docs/implementation/phase2_dev5_failure_taxonomy_20260413.md)
- [smoothnav_implementation_master_plan_20260412.md](/Users/xin/Code/research/SmoothNav/docs/planning/smoothnav_implementation_master_plan_20260412.md)

## 0. 总结先行

当前最核心的代码级结论是：

- 主故障点更像是 `strategy -> apply_strategy() -> graph.get_goal() -> global_goals` 这一段的**语义落地过弱**
- 次级故障点是 `UniGoal_Agent.instance_discriminator()` 中 `global_goal / temp_goal / stuck_goal` 的 override 继续**抹平 profile 差异**
- `monitor` 和 `prefetch` 不是完全没接入主循环，而是**触发条件太窄、输出太保守、落地后果太弱**
- 当前指标里的 `decision_delay` 和 `goal_update_delay` 还只是**控制层和主循环变量层**的指标，不是执行层 adoption 指标

因此下一轮实现，最应该围绕的是：

1. 让 `apply_strategy()` 的语义落地可观测、可判定、可增强
2. 让 strategy 切换能显式清理 stale override
3. 让 `pending` 真正进入工作态
4. 让 monitor 产生有后果的动作，而不只是 `CONTINUE`

## 1. `main.py` 中控制层是否真的“吃进了” planner/monitor 的结果

### 1.1 monitor 的调用条件

真实代码：

- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L531)
- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L85)

当前条件是：

- `args.controller_enable_monitor == True`
- `controller_state.current_strategy is not None`
- `controller_state.needs_initial_plan == False`
- `graph_delta.has_new_nodes == True`

代码级含义：

- 现在 monitor 只在“有新 object node”时触发
- `new_rooms`、`room_object_count_changes`、`frontier_near`、`stuck` 本身不会直接触发 monitor
- 所以 monitor 对“事件”的感知仍然很窄，更多是在 object growth 上被动响应

### 1.2 `PREFETCH -> pending_strategy` 的创建逻辑

真实代码：

- monitor 触发路径：[main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L560)
- auto-prefetch 路径：[main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L639)

当前实现有两条路：

1. monitor 返回 `PREFETCH`
2. 自动 prefetch 条件满足时直接调用 planner

但 auto-prefetch 的门槛很高：

- `enable_prefetch == True`
- `replan_policy == "event"`
- `dist_to_goal < 10`
- 当前没有 `pending_strategy`
- 当前动作不是 monitor 刚触发的 `PREFETCH`
- 当前不是 `frontier_reached`
- 已完成 initial plan
- 当前 strategy 必须是 `room target`

代码级含义：

- `pending` 不是“接近 frontier 就会提前生成”
- 它更像“接近 frontier 且当前 strategy 已经是 room search 时，才有机会生成”
- 在 planner 大量输出 `direction` 的现实下，这条链天然难触发

### 1.3 `pending_strategy` promotion 逻辑

真实代码：

- early promotion：[controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L106)
- frontier reached 时 promotion：[controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L127)

当前 early promotion 条件：

- `pending_strategy` 存在
- `current_strategy` 存在
- `current_strategy` 必须不是 room
- `pending_strategy` 必须是 room

当前 frontier reached 分支：

- 如果有 pending，直接应用 pending
- 如果当前还是 direction target，则直接 `Direction reuse`
- 如果当前是 room target 且没有 pending，才重新规划

代码级含义：

- 当前 pending 不是一个普适的“下一步计划缓存”
- 它更像“方向探索期间缓存 room plan”
- 这会显著压低 `pending_created` 和 `pending_promoted`

### 1.4 `frontier_reached` 分支

真实代码：

- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L621)
- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L127)

一个非常关键的分支是：

- 如果没有 pending，且当前 `target_region` 还是 `unexplored ...`
- 那么系统不会强制 replan，而是直接复用这个 direction strategy

代码级含义：

- 即使到达 frontier，profile 也不一定发生分岔
- 很多 episode 会继续沿“同一个方向语义”前进
- 这会加剧 profile delta collapse

### 1.5 stuck 分支

真实代码：

- controller 侧 stuck replan：[controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L185)
- executor 侧 stuck override：[agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L252)
- collision 触发 `been_stuck`：[agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L516)

当前有两套 stuck 机制同时存在：

- controller 看到 `graph_delta.stuck` 后，重新 `plan_strategy()`
- executor 自己看到 collision / no-progress 后，会直接采样 `stuck_goal`

代码级含义：

- controller 的 stuck replan 并不天然比 executor stuck override 更高优先级
- 如果高层刚换完 strategy，低层仍可能因为 `been_stuck` 先去追 `stuck_goal`

### 1.6 `goal_updated` 的打点定义

真实代码：

- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L780)
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L805)

当前定义：

```python
goal_before = list(global_goals)
...
goal_after = list(global_goals)
goal_updated = goal_after != goal_before
```

代码级含义：

- `goal_updated` 只说明主循环里的 `global_goals` 变了
- 它不说明 `instance_discriminator()` 之后 executor 真的采用了这个 goal
- 它也不说明 primitive actions 已经发生分岔

这就是为什么当前 `goal_update_delay` 仍然可能偏乐观。

## 2. `apply_strategy()` 和 `graph.get_goal(goal=bias)` 的真实语义

### 2.1 `bias` 是怎么传进去的

真实代码：

- [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py#L4)

当前实现很简单：

```python
bias = strategy.bias_position if strategy else None
goal = graph.get_goal(goal=bias)
```

这意味着：

- planner 并没有直接给出“必须去这个语义点”
- planner 只给出一个 `bias_position`
- 最终仍然要经过 frontier selection 再落成可执行目标

### 2.2 `graph.get_goal()` 的 frontier score 组成

真实代码：

- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L791)

分数由两部分叠加：

1. **agent 到 frontier 的基础分**

```python
distances_16_inverse = 10 - (np.clip(distances_16, 0, 10 + threshold) - threshold)
scores += distances_16_inverse
```

对应代码：

- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L829)
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L841)

量级大约是 `0~10`。

2. **bias 到 frontier 的附加分**

```python
distances_16_inverse = 1 - (np.clip(distances_16, 0, 10 + threshold) - threshold) / 10
scores += distances_16_inverse
```

对应代码：

- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L842)
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L855)

量级大约只有 `0~1`。

### 2.3 这意味着什么

这不是一个“semantic target selection”函数，而是一个：

**frontier selection with weak semantic bias**

所以代码层面已经可以明确说：

- 当前 frontier score 里，几何可达性占主导
- semantic bias 只是弱加分
- 不同 strategy 很可能因为 bias 权重不够强而塌缩到同一 frontier

### 2.4 会不会反复选到同一个 frontier

会，而且不仅会，代码上还缺少显式检测。

原因有三层：

1. `bias` 的分值权重弱
2. `apply_strategy()` 不返回“这次 grounding 是否真的改变了 frontier”
3. 主循环没有“strategy changed but selected frontier unchanged”的检测逻辑

这就是典型的 frontier-collapse 风险。

## 3. `UniGoal_Agent.step()` / `instance_discriminator()` 的 override 逻辑

### 3.1 override 优先级

真实代码：

- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L242)

优先级顺序是：

1. `global_goal`
2. `been_stuck -> stuck_goal`
3. `found_goal == 1` 分支里的 visible target / temp_goal
4. 默认 `exp_goal`

也就是说，高层传下来的主 goal 不一定是最终执行目标。

### 3.2 `temp_goal` 在什么条件下覆盖高层目标

真实代码：

- visible target 场景下复用 `temp_goal`：[agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L286)
- 默认场景下继续使用 `temp_goal`：[agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L352)

当前逻辑是：

- 只要 `self.temp_goal` 还在
- 就可能在后续多步中继续覆盖当前 `exp_goal`
- strategy 切换本身不会自动清理旧的 `temp_goal`

### 3.3 strategy 切换后旧 `temp_goal` 会不会继续保留

会。

代码里没有 “on strategy switch -> clear temp_goal / stuck_goal” 的机制。

`temp_goal` 目前只会在这些条件下被清：

- unavigable
- 被 `goal_map_mask` 吃掉
- 走近后被判定为需要作废
- 升级成 `global_goal`

对应代码：

- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L306)
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L324)
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L354)

这意味着：

- 高层换 strategy 之后，旧 temp goal 完全可能继续拖很多步
- 这是 override-dominated execution 的直接代码根因之一

### 3.4 `stuck_goal` 与 `global_goal` 的优先级

`global_goal` 更高。

因为判断顺序在代码里就是：

- 先 `self.global_goal is not None`
- 再 `self.been_stuck`

对应代码：

- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L245)

### 3.5 `visible target` / `found_goal` 何时接管

真实代码：

- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L416)
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L324)
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L556)

流程是：

1. 每步先根据检测框设置 `agent_input["found_goal"]`
2. 如果目标可见且距离足够近，就把它升级成 `global_goal`
3. 只有 `stop and found_goal == 1` 时才真正执行 stop

对 text goal 来说：

- 可见目标 + `goal_dis < 15` 就足以接管

## 4. 指标定义：`decision_delay / goal_update_delay / strategy_switch_count`

### 4.1 `decision_delay`

真实代码：

- [control_metrics.py](/Users/xin/Code/research/SmoothNav/smoothnav/control_metrics.py#L43)

当前定义：

- semantic event = `new_node_count > 0` 或 `new_rooms` 非空
- reaction = `planner_called == True` 或 `monitor_decision in {ADJUST, PREFETCH, ESCALATE}`

所以它量的是：

**控制层承认并响应事件所需的时间**

不是：

- goal 真正更新的时间
- executor 采纳该 goal 的时间
- primitive actions 改变的时间

这就是为什么 `smoothnav-no-monitor` 也能有很低的 `decision_delay`。

### 4.2 `goal_update_delay`

真实代码：

- [control_metrics.py](/Users/xin/Code/research/SmoothNav/smoothnav/control_metrics.py#L66)

当前定义：

- 从 `planner_called` 开始
- 到后续第一次 `goal_updated == True`

所以它量的是：

**planner 调用到主循环 `global_goals` 更新**

不是：

- planner 调用到 executor goal adoption
- planner 调用到 trajectory divergence

### 4.3 `strategy_switch_count`

真实代码：

- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L792)

当前定义：

```python
strategy_switched = (
    current_strategy_before is not None
    and current_strategy_after != current_strategy_before
)
```

它只比较 `target_region` 字符串。

这意味着：

- 同一个 `target_region` 下如果 `bias_position` 变了，不算 switch
- 同一个房间但换了不同 anchor，也可能不算 switch

所以这个指标会低估真实的策略变化次数。

## 5. 两条对照 trace：`baseline-periodic ep0` vs `smoothnav-full ep0`

远端原始文件：

- `baseline-periodic`
  - `/home/nebula/xxy/SmoothNav/results_phase2/dev5/baseline-periodic/20260413/smoothnav_text_162522_f81b8910/step_traces/episode_000000.jsonl`
- `smoothnav-full`
  - `/home/nebula/xxy/SmoothNav/results_phase2/dev5/smoothnav-full/20260413/smoothnav_text_163652_bc68c6fd/step_traces/episode_000000.jsonl`

下面只摘最关键的 step。

### 5.1 `baseline-periodic ep0`

`step 0`

- `current_strategy = unexplored north`
- `planner_called = true`
- `goal_before = [120, 120]`
- `goal_after = [136, 138]`
- `goal_updated = true`

`step 40`

- `planner_called = true`
- `planner_reasons = ["fixed_interval_refresh"]`
- `bias_position` 已经从 `[120, 360]` 变成 `[169, 381]`
- 但 `goal_before == goal_after == [112, 79]`
- `goal_updated = false`

这说明：

- 即使高层重新规划、甚至 bias 变了
- `apply_strategy() -> get_goal()` 仍可能落到同一个 frontier

`step 80`

- `current_strategy` 切到 `bedroom`
- `strategy_switched = true`
- `goal_updated = true`

这说明：

- 当前系统并不是“所有 strategy change 都落不了地”
- 但落地并不稳定

`step 200`

- `planner_called = true`
- `current_strategy = object: cabinet`
- `goal_updated = false`
- `temp_goal_override = true`

这说明：

- 在后半段，executor override 已经开始接管

### 5.2 `smoothnav-full ep0`

`step 0`

- 和 baseline 一样：初始 planning 落地成功

`step 46`

- `planner_called = true`
- `planner_reasons = ["new_room_discovered"]`
- `current_strategy` 切到 `bedroom`
- `monitor_called = true`
- `monitor_decision = CONTINUE`
- 但 `goal_before == goal_after == [112, 79]`
- `goal_updated = false`

这是整轮分析里最关键的一步。

它说明：

- graph event 已经发生
- planner 已经做出更具体的 room decision
- monitor 也已经被调用
- 但高层语义决策没有改变真正的 frontier goal

而且这一步里：

- `temp_goal_override = false`
- `stuck_goal_override = false`
- `global_goal_override = false`

所以这一步的问题**发生在 executor override 之前**，更像是 frontier-collapse / weak grounding。

`step 48`

- monitor 再次 `CONTINUE`
- `goal` 仍不变

说明 monitor 的“响应”没有产生行为后果。

`step 53`

- `planner_called = true`
- `planner_reasons = ["frontier_reached"]`
- `current_strategy` 又回到 `unexplored north`
- 这次 `goal_updated = true`

说明：

- 当前 event-driven loop 会在 room-target 没落地时，很快退回 direction exploration

### 5.3 这两条 trace 支持什么结论

对这两条 `ep0` 来说，主故障点更偏向：

- `frontier-collapse / weak strategy grounding`

而不是：

- 一开始就完全被 executor override 吃掉

但 executor override 不是假的，它更像第二层问题：

- 在 episode 后半段、尤其是 temp goal 已经建立后
- override 会进一步压平高层差异

## 6. “问题-证据-具体修改点”清单

下面是最值得直接进入实现的修改单。

### P0. `apply_strategy()` 必须返回 grounding 结果，而不是静默 no-op

问题：

- 当前 `apply_strategy()` 不返回是否成功落地
- 主循环不知道这次 strategy grounding 是否真的改变了 frontier goal

证据：

- [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py#L4)
- `step 46` 的 `smoothnav-full ep0`：strategy 已切换，goal 没变

具体修改点：

- 修改 [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py#L4)
- 让 `apply_strategy()` 返回结构体或 dict，例如：
  - `success`
  - `selected_frontier`
  - `selected_frontier_score`
  - `selected_frontier_dist_to_bias`
  - `goal_before`
  - `goal_after`
  - `selected_same_as_previous`
- 主循环在 [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L805) 不再只用 `goal_after != goal_before`，而是明确区分：
  - `strategy_changed`
  - `goal_grounded`
  - `goal_adopted`

### P1. 增强 `graph.get_goal()` 中 semantic bias 的权重，并把它参数化

问题：

- 当前 bias 分值量级远弱于基础 frontier 分

证据：

- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L829)
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L842)

具体修改点：

- 在 [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py#L791) 增加可配置权重：
  - `frontier_distance_weight`
  - `semantic_bias_weight`
- 至少先把当前的 `0~1` bias bonus 提高到和基础分同量级
- 额外增加 tie-break：
  - 当 `strategy_switched == true` 时，优先避免选到与前一步相同的 frontier
- 最好把 top-k frontier 候选也写进 trace，便于判断是否真的塌缩

### P2. strategy change 后应主动清理 stale override

问题：

- 当前高层换 strategy，不会自动清掉旧 `temp_goal` / `stuck_goal`

证据：

- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L286)
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L352)
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py#L252)

具体修改点：

- 给 [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py) 增加一个显式接口，例如：
  - `on_strategy_change(target_region, bias_position, strategy_epoch)`
- 在 `strategy_switched` 或 `goal_grounded` 时调用它
- 默认行为至少应包括：
  - 清理不兼容的 `temp_goal`
  - 清理 `stuck_goal`
  - 记录 `override_epoch`

### P3. 扩大 monitor 触发条件，让 monitor 真正看见“语义事件”

问题：

- monitor 现在只在 `has_new_nodes` 时触发

证据：

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L85)

具体修改点：

- 把 `maybe_call_monitor()` 的触发条件扩展为：
  - `has_new_nodes`
  - `has_new_rooms`
  - `room_object_count_changes`
  - `frontier_near`
  - `stuck`
- 输入里增加更强字段：
  - `strategy_type` (`room/object/direction`)
  - `new_rooms`
  - `room_count_changes`
  - `frontier_reached`
  - `is_current_room_target_consistent`
- 只有这样 monitor 才有机会输出真正有后果的 `ESCALATE / ADJUST / PREFETCH`

### P4. 让 pending 路径更容易进入工作态

问题：

- `pending` 的创建和 promotion 门槛都太窄

证据：

- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L560)
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py#L639)
- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L106)

具体修改点：

- 在 `new_room_discovered` 且当前 strategy 是 direction 时，允许直接创建 pending room strategy
- 不要把 early promotion 限死在“current 非 room 且 pending 是 room”
- 可以改成：
  - `dist_to_goal < threshold`
  - pending 的语义优先级高于 current
  - pending grounding 成功且与 current 选出的 frontier 不同

### P5. `frontier_reached` 的 direction reuse 需要收紧

问题：

- 当前 direction strategy 在 frontier reached 后可直接复用

证据：

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py#L148)

具体修改点：

- 加一个 reuse 上限，例如连续 `N` 次 direction reuse 后必须 replan
- 或者记录 directional sector，避免反复 reuse 同一方向
- 否则 event-driven controller 很容易在 semantics 尚未充分生效前退回重复探索

### P6. 指标需要从“控制层响应”升级到“执行层 adoption”

问题：

- 现在的延迟指标还不足以回答“执行层到底有没有跟上”

证据：

- [control_metrics.py](/Users/xin/Code/research/SmoothNav/smoothnav/control_metrics.py#L33)

具体修改点：

- 新增 3 个指标：
  - `grounding_delay_steps`: strategy change -> `goal_grounded`
  - `adoption_delay_steps`: `goal_grounded` -> executor override source 变更 / local goal 改变
  - `trajectory_divergence_steps`: strategy change -> primitive action 序列首次偏离前一策略
- 同时新增 trace 字段：
  - `executor_goal_source` (`global_goal/temp_goal/stuck_goal/controller_goal`)
  - `executor_goal_changed`
  - `selected_frontier`
  - `selected_frontier_same_as_prev`

## 7. 当前最值得优先实现的三件事

如果只能先改三件，我建议是：

1. 改 [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py) 和 [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py)
   - 先解决 semantic grounding 太弱的问题
2. 改 [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py)
   - 在 strategy change 时清 stale override
3. 改 [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py) 和 [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py)
   - 让 pending/monitor 更容易进入真正工作态

一句话收束：

**现在最该修的不是“planner 还不够聪明”，而是“planner 的结果没有被可靠地 ground 成不同 frontier，并且切换后还会被旧 override 拖回去”。**
