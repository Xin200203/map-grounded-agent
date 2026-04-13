# Controller Profile And Trace Alignment Notes (2026-04-13)

## 为什么要补这份说明

今天在对照 `baseline` 与 `smoothnav` 的 smoke 结果时，暴露出了两个会直接影响 Phase Gate 判断的问题：

1. `baseline-periodic` 的实现语义和计划书中的语义不一致。
2. trace 目录中出现了 phantom episode 文件，导致 `summary.json` 与 raw dump 条数对不上。

这两件事如果不写清楚，后续做 `dev20 / dev100` 时很容易再次误读结果。

---

## baseline family 的当前约定

### 1. `baseline-explore`

这是旧的 UniGoal frontier baseline，对应：

- `mode = baseline`
- `monitor = off`
- `prefetch = off`
- `replan_policy = baseline_explore`
- 不走 semantic periodic replanning

它的作用是：

- 保留一个最朴素的 frontier explore 参考线
- 对齐历史 UniGoal 执行链

### 2. `baseline-periodic`

从 2026-04-13 开始，这个名字专门表示：

**same-backbone semantic periodic baseline**

对应：

- `mode = smoothnav`
- `monitor = off`
- `prefetch = off`
- `replan_policy = fixed_interval`
- `stuck_replan = off`

它的作用是：

- 作为 `smoothnav-full` 的公平对照
- 回答“event-driven 相比 periodic semantic planning 是否更及时、更省”

### 3. 为什么必须拆开

如果把 `baseline-periodic` 继续用作旧的 UniGoal frontier baseline，就会出现一个概念混淆：

- 计划书里讨论的是 periodic semantic baseline
- 实际代码里跑出来的却是 frontier explore baseline

这样做出来的 Phase 2 对照不成立。

---

## trace / summary 对齐的当前约定

### 1. 之前的问题

之前主循环在最后一个 episode 完成后，没有立即退出当前 loop iteration，而是继续执行了一轮后续逻辑。

这会导致：

- `step_traces/episode_000005.jsonl` 这类 phantom episode 文件
- `planner_calls` / `monitor_calls` 中出现额外一条 next-episode 初始调用
- `summary.json` 用 `episode_results` 汇总，而 raw dump 枚举却把 phantom 文件也算进去

表现上就是：

- `summary.num_episodes = 5`
- 但 trace 目录里会出现 6 个 `episode_*.jsonl`

### 2. 当前修复

当前实现采用两层修复：

1. 在最后一个 episode 完成后，主循环立即 `break`，不再继续写 phantom trace。
2. `compute_run_control_metrics(...)` 新增 `episode_ids` 过滤口径，只统计真正完成的 episode。

### 3. 当前对齐口径

以后判断 run 是否对齐时，统一按：

- `episode_results.json` 中实际完成的 episode id 集合
- 对应 episode 的 `step_traces / planner_calls / monitor_calls`

来做核对。

不要再直接把目录中所有 `episode_*.jsonl` 无差别相加。

---

## 对 Phase 2 的直接影响

从这次修复开始：

- `baseline-explore` 不再冒充 `baseline-periodic`
- `baseline-periodic` 真正成为 periodic semantic baseline
- `summary` 与 raw dump 的对齐口径明确下来

因此后续 `dev20` 的这几组：

- `baseline-periodic`
- `smoothnav-full`
- `smoothnav-no-monitor`
- `smoothnav-rules-only`
- `smoothnav-no-prefetch`

才具备进入 Phase 2 因果验证的最低可解释性。
