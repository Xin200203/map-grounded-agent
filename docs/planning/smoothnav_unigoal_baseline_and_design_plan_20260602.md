# SmoothNav 设计计划：从 UniGoal baseline 到在线语义建图显式导航

**日期**：2026-06-02  
**状态**：项目设计 / 计划整理稿  
**用途**：统一 SmoothNav 当前实现、与 UniGoal baseline 的关系、改造动机、落地路径和当前进度。本文不是最终论文稿；论文表述仍需以后续完整实验和数据集可用性审计为准。

---

## 1. 背景与核心 idea

SmoothNav 当前的核心目标不是重新训练一个导航模型，而是在 UniGoal 的零样本目标导航骨架上，重新组织“在线建图—语义理解—前沿选择—执行”的控制闭环，使 agent 能够围绕实时增长的地图和场景图做更可解释、更及时的显示导航决策。

一句话概括当前 idea：

> 保留 UniGoal 的 BEV mapping、scene graph、目标识别和 FMM 执行器，把原本偏周期式、偏一次性语义推理的导航流程，改造成“map-grounded semantic replanning”系统：高层 agent 负责生成语义策略，低层 monitor / arbiter 负责在线触发重规划，grounder 把语义策略落到可执行的地图前沿或目标锚点上。

因此，SmoothNav 的贡献更偏系统和工程路线：

1. **在线地图驱动**：每一步都从 BEV map / scene graph / frontier / executor 状态构造 world state。
2. **语义策略与几何执行解耦**：LLM / MLLM 不直接输出裸坐标，而输出房间、物体、方向或 branch 级别的语义意图，再由 grounder 和 graph scorer 转成可执行目标。
3. **事件触发重规划**：由 graph delta、frontier reached、stuck、out-of-local-window、grounding failure 等事件触发，而不是只依赖固定周期。
4. **可诊断控制链路**：planner、monitor、grounder、frontier scorer、executor adapter 均有 trace，便于定位失败来自语义、几何、局部窗口还是执行器。

---

## 2. UniGoal baseline：大致实现

### 2.1 外部链接

- Paper：<https://arxiv.org/abs/2503.10630>
- Project page：<https://bagh2178.github.io/UniGoal/>
- GitHub：<https://github.com/bagh2178/UniGoal>

### 2.2 UniGoal 要解决的问题

UniGoal 的目标是 **Universal Zero-shot Goal-oriented Navigation**：在不针对每类目标重新训练的情况下，用统一的图表示处理多种目标形式，包括文本目标、图像目标和实例目标。项目 README 中明确强调它是零样本方案，并在 Habitat / HM3D 上运行。

在当前 SmoothNav repo 中，UniGoal baseline 位于 `base_UniGoal/`。它提供了 SmoothNav 仍然复用的核心基础设施：

| 模块 | 文件位置 | 职责 |
| --- | --- | --- |
| 主循环 | `base_UniGoal/main.py` | 初始化环境、地图、图、agent；周期更新图和目标；执行动作 |
| BEV 建图 | `base_UniGoal/src/map/bev_mapping.py` | 将 RGB-D、语义、位姿投影到全局 / 局部 BEV map |
| Scene graph / goal graph | `base_UniGoal/src/graph/graph.py` | 维护房间、物体、frontier、目标相关关系；选择 mid-term goal |
| UniGoal agent | `base_UniGoal/src/agent/unigoal/agent.py` | 目标实例判别、局部 FMM planner、动作输出 |
| LLM / graph matching | `base_UniGoal/src/graph/tools.py` 等 | 房间预测、group correlation、语义相关性打分 |

### 2.3 Baseline 运行闭环

从 `base_UniGoal/main.py` 看，baseline 的关键运行逻辑大致是：

1. **初始化**：创建 `BEV_Map`、`Graph`、`UniGoal_Agent`。
2. **每步感知和建图**：读取 Habitat observation，将 RGB-D / semantic / pose 投影到 BEV map。
3. **周期性 scene graph 更新**：每隔若干步调用 `graph.update_scenegraph()`，把观测到的物体、房间和关系写入图中。
4. **周期性探索 / 目标选择**：当局部步数到达阈值，或 agent 接近当前 local goal 时，调用 `graph.explore()`。
5. **语义推理与 frontier 选择**：`graph.explore()` 内部会进行目标相关的语义推理，包括 room prediction、group correlation、frontier extraction、FMM 距离和 bias 计算，最后通过 `graph.get_goal()` 选出新的 full-map goal。
6. **局部执行**：将 full-map goal 转换到 local map，再交给 `UniGoal_Agent.step()`，由 FMM planner 输出 Habitat action。

可以把 UniGoal baseline 简化成下面这条链：

```text
observation
  -> BEV_Map.update_map
  -> Graph.update_scenegraph
  -> Graph.explore / Graph.get_goal
  -> full-map goal -> local-map goal
  -> UniGoal_Agent.step / FMM planner
  -> action
```

### 2.4 UniGoal 中的关键设计

#### 2.4.1 BEV map 是所有几何决策的基础

`BEV_Map` 把深度、语义分割、agent 位姿、visited path、obstacle、explored area 等投影到 BEV 栅格上。后续 scene graph、frontier selection 和 FMM local planner 都依赖这个地图。

#### 2.4.2 Scene graph 把在线观测组织成语义图

`Graph` 会维护已观测物体节点、房间节点、frontier、候选目标和关系。文本目标会先经过房间预测和 group-level correlation，决定目标更可能出现在什么房间或物体组附近。

#### 2.4.3 Frontier 是探索动作的中期目标

`Graph.get_goal()` 会从 map 中抽取 frontier，并结合：

- FMM reachable distance；
- semantic bias；
- novelty；
- actionability；
- repeat / recent penalty；
- target progress；
- 可选 MLLM prior；

返回一个 full-map 坐标，作为下一段导航的 mid-term goal。

需要注意：上面部分 scoring 逻辑在当前 repo 中已经被 SmoothNav 扩展过，因此它代表的是当前 baseline fork 的实现状态，而不是原始 UniGoal 论文代码的纯净版本。

#### 2.4.4 最终执行器仍然是局部 FMM planner

`UniGoal_Agent.step()` 会处理目标可见性、实例判别、stuck recovery、临时目标等逻辑，然后调用 FMM planner 产生低层动作。也就是说，UniGoal 和 SmoothNav 的最终运动控制都不是 LLM 直接控制，而是地图目标 + FMM planner 控制。

---

## 3. UniGoal baseline 的局限性，以及与 SmoothNav 改造的对应关系

下面的局限性不是泛泛批评 UniGoal，而是与 SmoothNav 当前已做或计划做的改造一一对应。

| UniGoal / baseline 局限 | 对导航的影响 | SmoothNav 对应改造 |
| --- | --- | --- |
| 周期式 `explore()` 触发，决策粒度较粗 | 新语义证据出现、frontier 到达、stuck 或局部窗口失败时，不能总是及时重规划 | `graph_delta.py`、`controller_logic.py`、`tactical_arbiter.py` 引入事件触发重规划 |
| 语义推理和 frontier 选择耦合在 `Graph.explore()` / `get_goal()` 中 | LLM 推理、图匹配、frontier scoring 混在一起，失败时难以判断是语义问题还是几何问题 | `planner.py` 负责语义策略，`strategy_grounding.py` / `frontier_scoring.py` 负责落地到几何前沿 |
| LLM 调用偏阻塞，容易拖慢在线导航 | 高层语义思考会卡住 action loop，尤其是多 group correlation 或 room reasoning 时 | `low_level_agent.py`、pending proposal、prefetch，把部分推理转成异步 / 预取式策略 |
| 目标锚点和 local window 之间缺少充分保护 | full-map 目标可能落在当前 local map 外，出现 `out_of_local_window` 类失败 | `GeometricGrounder`、`handle_out_of_local_window()`、local projection trace 和 recovery |
| frontier scoring 缺少足够可解释的策略接口 | 很难把“我想去厨房边缘 / 桌子附近 / 左侧走廊”稳定转成 frontier 分数 | `Strategy`、`resolve_bias_position()`、semantic bias、target progress、planner prior、branch prior |
| scene graph 持续增长会带来关系稠密和推理成本问题 | 长 episode 中图会变大，相关性推理和 frontier 选择成本增加 | graph delta、relation pruning / 诊断计划、episode 296 graph growth 诊断 |
| 执行器内部临时目标 / stuck recovery 可能覆盖上层语义意图 | 高层策略看似改变了目标，但底层可能仍在追逐旧 temp goal | `executor_adapter.py`、strategy epoch、goal adoption trace、stale temp-goal clearing |
| 原始 BEV debug 图更偏可视化，不是完整决策状态 | MLLM 如果直接看图选点，容易输出不可验证坐标或幻觉目标 | `semantic_bev.py`、`bev_decision_state.py`、`frontier_branching.py`、`mllm_frontier_contract.py` 要求输出 branch / evidence，而不是裸坐标 |

---

## 4. SmoothNav 的优化与设计改造

### 4.1 总体架构

SmoothNav 的设计可以看成在 UniGoal 上加了一层在线 agent 控制系统：

```text
UniGoal backbone
  BEV_Map + Graph + UniGoal_Agent/FMM

SmoothNav controller
  WorldStateBuilder
  + MissionProgressManager
  + HighLevelPlanner
  + LowLevelMonitor
  + TacticalArbiter
  + GeometricGrounder
  + ExecutorAdapter
  + Trace / replay / experiment IO
```

其中，UniGoal 仍然负责“我在哪里、地图长什么样、局部怎么走”；SmoothNav 负责“现在是否应该改变策略、语义上应该去哪里、如何把这个语义目标变成安全的地图目标”。

### 4.2 修改点 1：主循环从固定探索改成在线控制器

核心文件：`smoothnav/main.py`

SmoothNav 的主循环不再只是周期性调用 `graph.explore()`，而是每步构造 world state，检查事件，再决定是否规划、监控、预取或继续执行。

简化流程如下：

```text
reset episode
  -> build initial map / graph
  -> initial high-level strategy
  -> apply_strategy to graph/local goal

for each step:
  -> update BEV map
  -> update scene graph
  -> compute graph delta / progress / executor state
  -> tactical arbiter decides triggers
  -> maybe high-level replan
  -> maybe low-level monitor ADJUST / PREFETCH / ESCALATE
  -> maybe promote pending proposal
  -> ground semantic strategy to frontier/object/local goal
  -> executor adapter sends goal to UniGoal agent
  -> FMM action
  -> trace step
```

这使 SmoothNav 可以在以下事件发生时改变决策：

- 新目标候选出现；
- 新房间或关键物体出现；
- frontier reached；
- no progress / stuck；
- 当前策略的 grounding 失败；
- 目标投影出 local window；
- monitor 判断当前区域与目标不匹配。

### 4.3 修改点 2：高层 planner 输出语义策略，而不是坐标

核心文件：`smoothnav/planner.py`

`HighLevelPlanner` 的输出是 `Strategy`，主要字段包括：

- `target_region`：想探索的房间、区域或方向；
- `bias_position`：可选的地图偏置位置；
- `anchor_object`：可选语义锚点，例如 table、bed、sofa；
- `reasoning`：策略理由；
- `explored_regions`：已经考虑过的区域。

这有两个好处：

1. **减少 LLM 幻觉坐标**：LLM 不直接生成像素或地图坐标。
2. **方便几何校验**：grounder 可以判断该语义策略是否真的能落到已知物体、房间中心或 frontier branch 上。

### 4.4 修改点 3：语义策略由 grounder 落到图和地图上

核心文件：

- `smoothnav/strategy_grounding.py`
- `smoothnav/geometric_grounder.py`
- `smoothnav/frontier_scoring.py`
- `base_UniGoal/src/graph/graph.py`

`apply_strategy()` 会把高层策略转成可执行目标，典型路径包括：

1. 如果目标物体已经被可靠观测到，优先尝试 object anchor。
2. 如果策略指向某个房间或区域，则根据已知图节点 / 房间内物体中心估计 bias。
3. 如果只有方向或区域倾向，则把策略转换为 frontier scoring bias。
4. 如果可选 MLLM branch prior 存在，则以 prior 形式影响 frontier 分数，而不是绕过 graph scorer。
5. 最后将 full-map goal 投影到 local map，并记录 projection / grounding trace。

这条设计约束很重要：SmoothNav 不让 LLM / MLLM 绕过地图直接控制 agent，而是让它们只提供“偏好”，最终仍由地图可达性和 FMM 执行器决定动作。

### 4.5 修改点 4：低层 monitor 和 tactical arbiter 负责事件触发

核心文件：

- `smoothnav/low_level_agent.py`
- `smoothnav/tactical_arbiter.py`
- `smoothnav/controller_logic.py`
- `smoothnav/graph_delta.py`
- `smoothnav/mission_state.py`
- `smoothnav/world_state.py`

低层 monitor 的动作空间是：

- `CONTINUE`：继续当前策略；
- `ADJUST`：轻量调整目标；
- `PREFETCH`：预先生成下一策略，但不立刻切换；
- `ESCALATE`：触发高层重规划。

`tactical_arbiter` 和 `controller_logic` 则负责把实际事件转成重规划条件。例如 frontier reached、stuck、grounding failure、out-of-local-window 等都可以触发 replan 或 repair。

目前 repo 中存在多种 profile：

| Profile | 目的 |
| --- | --- |
| `baseline-explore` | 近似原始 UniGoal explore 流程 |
| `baseline-periodic` | 周期式语义规划 baseline，用于和事件式 SmoothNav 对比 |
| `smoothnav-no-monitor` | 保留事件 / prefetch / grounding，但去掉 LLM monitor |
| `smoothnav-full` | 完整 SmoothNav：event replanning + monitor escalation + prefetch + stuck / grounding recovery |
| `smoothnav-rules-only` | 用规则 monitor 替代 LLM monitor，隔离 LLM 贡献 |
| `smoothnav-fixed-interval` | 固定间隔策略，用于控制变量 |

### 4.6 修改点 5：BEV / MLLM 变成可验证的 branch-level prior

核心文件：

- `smoothnav/semantic_bev.py`
- `smoothnav/bev_decision_state.py`
- `smoothnav/frontier_branching.py`
- `smoothnav/bev_frontier_planner.py`
- `smoothnav/mllm_frontier_contract.py`

当前设计不是让 MLLM 直接看图输出坐标，而是：

1. 从 map 和 scene graph 构造 decision-state BEV；
2. 将 frontier 聚类成稳定 branch ID；
3. 给 MLLM 的 prompt 中只暴露 branch candidates、语义对象、目标信息和质量标记；
4. MLLM 必须输出合法 branch ID、ranking、evidence level 和 reason；
5. contract validator 检查输出合法性；
6. 合法输出只作为 `planner prior` 注入 frontier scoring。

这能降低 MLLM 幻觉坐标的风险，也方便做 replay 和离线 evaluator。

### 4.7 修改点 6：保留 UniGoal 执行器，但增加 adapter 和 trace

核心文件：

- `smoothnav/executor_adapter.py`
- `base_UniGoal/src/agent/unigoal/agent.py`
- `smoothnav/tracing.py`

SmoothNav 没有重写低层运动控制，而是复用 UniGoal 的 FMM planner。改造重点是：

- 记录 strategy epoch 和 goal adoption；
- 当新策略生效时清理 stale temp goal；
- 对 stuck、suppress、visible target、local goal projection 等状态打 trace；
- 将 executor 的行为暴露给 monitor / arbiter，而不是让底层状态成为黑盒。

---

## 5. 当前优化思想如何落地

### 5.1 运行时落地

当前落地方式可以概括为“四层控制”：

1. **World model 层**：BEV map + scene graph + frontier + semantic objects。
2. **Strategy 层**：高层 planner 根据目标和 world state 选择语义方向。
3. **Grounding 层**：把语义方向落到 object / room / frontier branch / full-map goal。
4. **Execution 层**：由 UniGoal agent 和 FMM planner 执行动作，并把 stuck / temp goal / progress 回传给上层。

这条链路的关键不是单次 LLM 决策质量，而是每一步都有地图约束和失败恢复：

```text
semantic intent
  -> graph-grounded bias
  -> frontier / object anchor
  -> local goal projection
  -> FMM action
  -> progress / failure signal
  -> replan or continue
```

### 5.2 实验落地

当前实验不只比较最终 SR / SPL，还在比较不同控制部件的贡献：

- `baseline-periodic` vs `smoothnav-full`：测试事件式重规划 + monitor + prefetch 的整体收益。
- `smoothnav-no-monitor` vs `smoothnav-full`：隔离 LLM monitor 的独立收益。
- `smoothnav-rules-only` vs `smoothnav-full`：比较规则 monitor 与 LLM monitor。
- `baseline-explore` vs `baseline-periodic`：区分原始 UniGoal explore 与周期式语义 baseline。
- BEV / MLLM branch planner replay：检查视觉先验是否真的改善 frontier ordering。

### 5.3 Trace / replay 落地

SmoothNav 目前大量依赖 trace 来避免“只看成功率猜原因”：

- step trace：记录每步 planner / monitor / grounding / executor 状态；
- suite summary：汇总 SR、SPL、Done、path length 等指标；
- branch decision trace：记录 MLLM branch 选择和 evidence；
- grounding trace：记录 full-map goal、local projection、failure reason；
- graph growth diagnosis：检查 graph size、edge relation 和 episode 级失败模式。

---

## 6. 当前进度

### 6.1 已完成 / 已实现

**Claim（基于当前 repo 代码与已有 suite summary）**：SmoothNav 的主体控制框架已经实现，且可以通过 profile 切换与 UniGoal 风格 baseline 做对照。

已落地内容包括：

1. **SmoothNav controller 主循环**：`smoothnav/main.py`。
2. **高层策略 planner**：`smoothnav/planner.py`。
3. **低层 monitor / escalation / prefetch**：`smoothnav/low_level_agent.py`。
4. **事件检测与控制逻辑**：`smoothnav/graph_delta.py`、`smoothnav/controller_logic.py`、`smoothnav/tactical_arbiter.py`。
5. **策略 grounding 和 frontier scoring**：`smoothnav/strategy_grounding.py`、`smoothnav/geometric_grounder.py`、`smoothnav/frontier_scoring.py`。
6. **执行器适配与 tracing**：`smoothnav/executor_adapter.py`、`smoothnav/tracing.py`。
7. **BEV / MLLM branch planner 原型**：`smoothnav/semantic_bev.py`、`smoothnav/bev_decision_state.py`、`smoothnav/frontier_branching.py`、`smoothnav/bev_frontier_planner.py`、`smoothnav/mllm_frontier_contract.py`。
8. **对照 profile 和实验脚本**：`smoothnav/controller_config.py`、`scripts/` 下的 runner / evaluator / replay 工具。

### 6.2 已观察到的实验现象

以下结果按项目 AGENTS.md 的证据规范表述：完成 suite summary 的结果标为 Claim；小样本或诊断性结果标为 Observation / Hypothesis。

#### Claim：单场景 intact dev suite 中 SmoothNav 有正向迹象

已有文档和 suite summary 显示，在部分 intact scene dev 设置中，`smoothnav-full` 相对 `baseline-periodic` 有 SR / SPL 提升迹象，例如：

- dev5 intact scene：`baseline-periodic` SR 约 0.4，`smoothnav-full` SR 约 0.6。
- dev15 intact scene：`baseline-periodic` SR 约 0.533，`smoothnav-full` SR 约 0.6。

这说明事件式重规划、grounding recovery 和策略控制在部分场景中能够改善 baseline 行为。

#### Observation：跨场景结果仍不稳定

跨场景 suite 中，`smoothnav-full` 并没有稳定压过 baseline。部分 minimatrix 结果显示 SmoothNav 的 SPL 可能更好，但 SR 仍持平或不足。当前还不能把 SmoothNav 表述为“跨场景普遍优于 UniGoal”。

#### Observation：monitor 的独立贡献尚未充分证明

在一些 minimatrix 结果中，`smoothnav-full` 与 `smoothnav-no-monitor` 的结果接近，说明当前增益可能主要来自 grounding / event logic / scoring，而不是 LLM monitor 本身。后续需要更严格的 ablation 来隔离 monitor 的价值。

#### Observation：hard episode 仍暴露 target anchoring 和 graph growth 问题

例如 episode 296 诊断中，主要问题集中在：

- graph growth 后关系和候选变多，推理 / scoring 成本和噪声上升；
- 目标锚点可能引导到 local window 之外；
- frontier target progress 与真实目标区域仍可能错位；
- 语义 evidence 与几何 reachable frontier 的耦合仍不够稳定。

### 6.3 当前 blocker / 风险

1. **HM3D val asset integrity blocker**：项目 AGENTS.md 明确指出当前存在 HM3D val assets corrupted 的硬阻塞。在该问题解决并完成 loadability / benchmark audit 前，不应声称系统已经 top-tier 或 benchmark-ready。
2. **跨场景泛化尚未证明**：已有结果支持“部分 intact scene 有改善”，但不足以支持强 leaderboard claim。
3. **BEV / MLLM branch planner 仍需在线验证**：当前设计已经有 contract 和 branch prior 机制，但需要证明它在真实 online loop 中稳定提升 frontier ordering。
4. **monitor 价值需要重新隔离**：如果 `smoothnav-no-monitor` 与 `smoothnav-full` 接近，则论文和系统叙述中应避免过度强调 monitor LLM 的独立贡献。
5. **graph growth / relation pruning 需要系统处理**：长 episode 中图增长仍可能带来成本和噪声，需要 relation budget、node aging 或 evidence-based pruning。
6. **局部窗口和目标锚点仍是关键失败面**：`out_of_local_window`、stale temp goal、anchor mismatch 需要作为后续修复重点。

---

## 7. 下一步计划

### 7.1 实验与证据优先级

1. **先做数据集可用性审计**  
   确认 HM3D val / dev assets 的 loadability，明确哪些 split 可用于正式结论。

2. **固定 matched profile suite**  
   在同一批 episode / scene 上跑：
   - `baseline-explore`
   - `baseline-periodic`
   - `smoothnav-no-monitor`
   - `smoothnav-rules-only`
   - `smoothnav-full`

3. **分开报告 SR / SPL / Done / invalid / timeout / grounding failure**  
   只报告 SR / SPL 不足以解释系统改造价值，需要同时报告失败归因。

4. **对 hard episode 做 replay audit**  
   对 episode 296、228 等失败案例，逐步检查 strategy、grounding、frontier scoring、local projection、executor adoption。

### 7.2 系统改造优先级

1. **强化 target anchor 到 local window 的安全落地**  
   对所有 object / room / branch anchor 增加 reachable / local-window / stale-goal gate。

2. **收紧 graph growth 和 relation budget**  
   引入关系裁剪、node aging、evidence score，防止长 episode 中 scene graph 噪声压过有效语义。

3. **把 BEV branch planner 从 replay 推进到 online ablation**  
   先确保 contract validator、branch ID 稳定性、quality flags 和 fallback 都可靠，再比较是否提升 frontier ordering。

4. **重新定位 monitor**  
   如果 LLM monitor 独立收益不明显，可以把它定位为“安全 escalation / failure detector”，而不是主要性能来源。

5. **论文表述保持系统导向**  
   当前更稳妥的 claim 是：SmoothNav 提供了一个 map-grounded、可诊断、可 ablate 的在线语义重规划框架；在部分 intact scene 上显示正向改善，但跨场景和 benchmark-ready 仍需补证。

---

## 8. 建议的论文 / 报告叙事

一个较稳妥的叙事结构是：

1. **Baseline**：UniGoal 已经证明零样本目标导航可以通过统一图表示和 LLM-assisted semantic reasoning 实现。
2. **Gap**：原流程中语义推理、frontier selection 和执行反馈耦合过紧，在线失败难以及时修复，也难以诊断。
3. **SmoothNav**：把导航重构为 map-grounded semantic replanning loop：strategy、grounding、monitor、executor 分层。
4. **Design principle**：LLM / MLLM 不直接控制坐标，只提供语义策略或 branch prior；所有动作必须经过地图可达性和 FMM 执行器。
5. **Evidence**：先展示单场景和小规模 suite 的正向结果，再诚实报告跨场景不稳定、monitor ablation 和 hard episode 诊断。
6. **Future work**：数据集完整性、graph pruning、BEV branch planner online validation、跨场景规模化实验。

---

## 9. 关键文件索引

| 主题 | 文件 |
| --- | --- |
| UniGoal baseline 入口 | `base_UniGoal/main.py` |
| UniGoal README / paper links | `base_UniGoal/README.md` |
| UniGoal graph / frontier | `base_UniGoal/src/graph/graph.py` |
| UniGoal BEV map | `base_UniGoal/src/map/bev_mapping.py` |
| UniGoal executor | `base_UniGoal/src/agent/unigoal/agent.py` |
| SmoothNav 主循环 | `smoothnav/main.py` |
| Controller profiles | `smoothnav/controller_config.py` |
| 高层 planner | `smoothnav/planner.py` |
| 低层 monitor | `smoothnav/low_level_agent.py` |
| 事件检测 | `smoothnav/graph_delta.py` |
| 控制逻辑 | `smoothnav/controller_logic.py` |
| Tactical arbiter | `smoothnav/tactical_arbiter.py` |
| Strategy grounding | `smoothnav/strategy_grounding.py` |
| Frontier scoring | `smoothnav/frontier_scoring.py` |
| Semantic BEV / branch planner | `smoothnav/semantic_bev.py`, `smoothnav/bev_decision_state.py`, `smoothnav/frontier_branching.py`, `smoothnav/bev_frontier_planner.py` |
| MLLM contract | `smoothnav/mllm_frontier_contract.py` |
| Trace | `smoothnav/tracing.py` |
| 当前重点诊断文档 | `docs/implementation/episode296_grounding_and_graph_growth_diagnosis_20260421.md` |

