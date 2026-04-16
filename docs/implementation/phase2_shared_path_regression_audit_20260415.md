# Phase 2 Shared-Path Regression Audit（2026-04-15）

本文档用于单独回答一个问题：

**为什么 `dev5_patch` 里连 `baseline-periodic` 也回退了，以及这件事在代码上最可能落在哪些共享路径上。**

它不是完整实验报告，而是一次实现侧审计。

## 1. 已确认的现象

`dev5_patch` 的 `baseline-periodic` 与上一轮 `dev5 baseline-periodic` 相比，出现了明显回退：

- 上一轮：`SR = 0.8`, `SPL = 0.3847`
- 本轮：`SR = 0.4`, `SPL = 0.0854`

在当前结果包里还没有保存 Habitat 外部 `episode_id`，因此不能写成“同一外部 episode 已 100% 对齐”；但在同一 `dev5` 队列口径下，这已经足够作为 shared-path regression 风险来单独处理。

## 2. 最可能受影响的共享路径

本轮实现前，改动量最大的共享文件是：

- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py)
- [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py)
- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)
- [agent.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py)
- [control_metrics.py](/Users/xin/Code/research/SmoothNav/smoothnav/control_metrics.py)

真正可能伤到 `baseline-periodic` 行为的，优先怀疑下面两条：

### 2.1 `graph.get_goal()` 的 frontier 筛选过严

在 `dev5_patch` 结果里，`baseline-periodic` 的 grounding no-op 主要来自：

- `get_goal_none = 8`

再往下看 `graph_debug.no_goal_reason`，全部是：

- `no_frontiers = 8`

这说明 baseline 的回退，并不是 monitor 或 pending 直接造成的，而更像共享的 frontier selection 路径本身过严，导致 `apply_strategy()` 根本拿不到可用 goal。

### 2.2 `apply_strategy()` 的 no-op 现在被更完整地记录了，但 baseline 也一起吃到了这些 no-op

`baseline-periodic` 同样走：

- [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py)
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py)

因此，只要 `graph.get_goal()` 更容易返回 `None`，baseline 也会一起退化。

## 3. 本轮已经落实的修复

针对这条 shared path，本轮已经补了两类防护：

### 3.1 raw frontier fallback

如果：

- `diff == 1` 的 raw frontier 存在
- 但 `remove_small_frontiers(min_size=graph_frontier_min_size)` 过滤后为空

现在不再立刻返回 `no_frontiers`，而是会回退到 raw frontier 集合继续选 goal。

对应实现：

- [frontier_scoring.py](/Users/xin/Code/research/SmoothNav/smoothnav/frontier_scoring.py)
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py)

### 3.2 relaxed distance fallback

如果：

- frontier 存在
- 但 `distance >= graph_goal_distance_threshold` 的 candidate frontier 为空

现在不再立刻返回 `no_candidate_frontiers`，而是会退回到全部 frontier，并以更宽松的距离阈值继续打分。

对应实现：

- [frontier_scoring.py](/Users/xin/Code/research/SmoothNav/smoothnav/frontier_scoring.py)
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py)

## 4. 当前新增的审计字段

为了让下一轮 baseline sanity 能直接解释 shared-path 是否改善，trace 和 summary 里现在已经补了：

- `grounding_noop_reason_counts`
- `grounding_no_goal_reason_counts`
- `raw_frontier_count`
- `filtered_frontier_count`
- `frontier_filter_fallback_mode`
- `candidate_distance_fallback_mode`
- `used_raw_frontier_fallback`
- `used_relaxed_distance_fallback`

这意味着下一轮如果 baseline 再退化，我们可以直接回答：

- 是 raw frontier 本来就没有
- 还是 small-frontier 过滤太强
- 还是 distance threshold 太强

## 5. 这轮审计的结论

当前对 baseline regression 最合理的判断是：

- 它不是 monitor 路径直接造成的
- 它优先属于 shared `graph/grounding` 路径的回退风险
- 本轮已经在代码里加上了两层 fallback 来缓解：
  - raw frontier fallback
  - relaxed distance fallback

因此下一步不需要再泛化猜测，而应该直接跑：

1. `baseline-periodic` 单独 sanity
2. 看 `grounding_no_goal_reason_counts`
3. 看 `frontier_filter_fallback_mode / candidate_distance_fallback_mode`

如果 baseline 仍明显回退，再继续向：

- executor state
- stale temp-goal cleanup
- strategy epoch 传播

这些共享路径排查。
