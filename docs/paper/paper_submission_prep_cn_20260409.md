# SmoothNav 投稿准备统一文档

日期：2026-04-09
用途：内部论文投稿准备、统一定位、方法与实验边界说明
适用范围：`/home/nebula/xxy/SmoothNav`

## 0. 核心改写

这不是一篇“smoothness / action smoothing”论文，而是一篇“在线显式地图上的事件驱动语义重规划 agent”论文。

当前本地代码已经实现的是第二代三层架构：

- 高层 planner 稀疏地产生语义策略；
- 低层 monitor 在 scene graph 增量事件上只做 `CONTINUE / ADJUST / PREFETCH / ESCALATE`；
- 底层继续保留 UniGoal 的几何执行器、`graph.get_goal()` 和 FMM。

与此同时，当前 runtime 仍然只支持：

- `ins-image`
- `text`
- `InstanceImageGoal_Env`

因此当前系统故事已经成熟，但 benchmark 叙事还没接成正式的 R2R / RxR / VLN-CE 论文版本。

## 1. 一句话定位

英文定位：

Map-grounded Event-Driven Agent for Zero-shot Vision-and-Language Navigation in Continuous Environments

中文定位：

一种面向连续环境 VLN 的、基于在线显式地图的双时标事件驱动 agent。

关键词必须统一为：

- `map-grounded`
- `event-driven`
- `dual-timescale`
- `semantic-geometric decoupling`

不再用以下词作为论文核心：

- `smoothness`
- `action smoothing`
- `asynchronous replanning`

其中：

- `map-grounded` 表示系统有外部显式记忆，而不是纯 token 内隐记忆；
- `event-driven` 表示不是固定步长重规划，而是 scene graph 语义增量触发；
- `dual-timescale` 表示高层稀疏规划、低层轻量监测；
- `semantic-geometric decoupling` 表示 LLM 只给语义策略，不直接输出 primitive action。

## 2. 推荐标题方向

论文标题不建议继续用 `SmoothNav` 作为主标题，仓库名可以保留，但论文主名应从“平滑”转向“显式地图 + 事件驱动重规划”。

推荐标题方向：

1. Event-Driven Semantic Replanning on Online Scene Graphs for Zero-shot VLN
2. A Dual-timescale Map-grounded Agent for Zero-shot Vision-and-Language Navigation
3. Online Scene-Graph Replanning for Explicit-map Zero-shot VLN in Continuous Environments

## 3. 论文故事主线

### 3.1 背景起点

VLN-CE 是更接近真实机器人的 instruction-following 任务：

- agent 只有第一视角观测与自然语言指令；
- 需要在连续空间里自己完成低层控制；
- benchmark 上通常关心是否成功到达、路径效率以及是否忠实跟随路径。

当前相关 benchmark 主轴应聚焦：

- R2R-VLNCE v1-3
- RxR-VLNCE English Guide

在论文里，R2R 的作用是标准连续 VLN 主入口，RxR 的作用是放大：

- 指令更长、
- 路径更复杂、
- shortest-path 假设更弱、
- 更能体现语义更新粒度的重要性。

### 3.2 现有工作的分化

近年的 zero-shot / training-free VLN 可以粗分为几类：

- instruction decomposition / progress estimation：
  CA-Nav、Open-Nav、LAW
- structured execution without explicit maps：
  EmergeNav、DreamNav
- explicit memory / scene graph navigation：
  SG-Nav、UniGoal、SpatialNav

从这些工作中可以抽出三条共识：

1. instruction progress 很重要；
2. execution structure 很重要；
3. explicit memory 对 zero-shot navigation 很有帮助。

但当前仍缺一条关键链路：

- 在未预探索环境中，
- 当 online scene graph 出现有意义语义增量时，
- instruction-aligned strategy 能否及时更新。

### 3.3 这篇稿子的真正缺口

这篇稿子的缺口不是“动作不够平滑”，而是：

`semantic response latency`

也就是：

- 看到了新房间、
- 看到了关键地标、
- 看到了与当前策略矛盾的证据、
- 进入了新的语义上下文，

之后系统是否能及时更新 instruction-aligned strategy。

当前代码和内部分析已经完成了这一转向：

- 不再用 custom executor 覆盖 UniGoal 的低层动作；
- 改动集中在高层语义规划和中层事件监测；
- 低层继续保留成熟的几何执行器。

### 3.4 主 claim

建议主 claim 统一写为：

我们提出一个基于在线 scene graph 的双时标 VLN agent。高层 planner 稀疏地产生 instruction-aligned semantic strategy；低层 monitor 仅在 graph delta 上判断是否继续、微调、预取或升级；LLM 不直接输出 frontier 或 primitive action，而是输出语义目标，再通过 bias grounding 映射到几何可达目标，由成熟执行器完成运动控制。

这条 claim 与相关工作的差异是：

- 不只是 instruction decomposition；
- 不只是 structured execution；
- 不依赖任务前完整预探索；
- 不让 LLM 直接控制 primitive action；
- 关键机制是 online explicit memory 上的 event-driven replanning。

### 3.5 论文要证明什么

这篇稿子不需要证明“我们全面大幅超过所有方法”，而应证明下面这条更稳的因果链：

显式地图 + graph-delta 触发 + 语义—几何解耦接口
→ 更及时的策略更新
→ 更少的重型规划调用 / 更低 token 成本
→ 在 R2R-CE / RxR-English 上取得不劣于甚至优于 periodic baseline 的 SR / SPL / nDTW。

这是最稳的证据结构。

## 4. 相关工作章节建议

建议只分四节。

### 4.1 Continuous VLN benchmarks and evaluation

这一节讲：

- R2R
- VLN-CE
- RxR

重点不是历史综述，而是交代：

- R2R 是经典指令导航；
- VLN-CE 把问题转成连续低层控制；
- RxR 更长、更复杂、更强调路径忠实性；
- 官方 VLN-CE repo 是 R2R 与 RxR 的统一 Habitat benchmark 入口。

### 4.2 Zero-shot VLN with instruction decomposition or progress estimation

这一节放：

- CA-Nav
- Open-Nav
- LAW

核心写法：

- 这些工作说明 instruction progress / sub-instruction state 很关键；
- 传统导航指标并不直接反映 agent 完成了多少 instruction；
- 因而 instruction-progress diagnostics 是合理的辅助分析方向。

### 4.3 Explicit memory and scene-graph navigation

这一节放：

- SG-Nav
- UniGoal
- SpatialNav

强调：

- SG-Nav 与 UniGoal 证明了 online scene graph 对 zero-shot navigation 的价值；
- SpatialNav 说明显式 spatial scene graph 对 zero-shot VLN 有明显帮助；
- 但 SpatialNav 允许任务前完整预探索，和本工作的“在线建图执行一体化”设定不同。

### 4.4 Structured execution without explicit maps

这一节放：

- EmergeNav
- DreamNav

强调：

- 这些方法说明 execution structure 很重要；
- 但它们不依赖在线显式地图或 graph search；
- 本工作要做的是把 execution structure 与 explicit map 结合起来。

### 4.5 你的定位句

相关工作最后建议用一句话收束：

我们位于“online explicit memory”与“instruction-aware structured execution”的交叉点：不像 SpatialNav 那样依赖预探索，不像 EmergeNav / DreamNav 那样不维护显式地图，也不像 UniGoal / SG-Nav 那样停留在 object / image / text-goal，而是把 online scene graph、instruction progress 和 event-driven replanning 接到连续 VLN benchmark 上。

## 5. 方法章节组织

### 5.1 方法主题

统一主题：

`Dual-timescale Map-grounded Agent`

必须强调：

- 外部显式记忆；
- 事件触发；
- 语义—几何接口；

而不是泛化成抽象 LLM agent 叙事。

### 5.2 统一记号

建议全文统一用下面的记号：

- 指令：`I`
- 在线地图记忆：`M_t = {BEV_t, G_t}`
- instruction progress state：`k_t`
- graph delta：`ΔG_t`
- 高层策略：`z_t`
- 中层决策：`m_t ∈ {CONTINUE, ADJUST, PREFETCH, ESCALATE}`
- grounding bias：`b_t`
- 几何目标：`g_t`

统一写成：

```text
z_t = H(I, G_t, k_t)
m_t = L(I, z_t, ΔG_t, d_t, p_t)
b_t = B(z_t, G_t)
g_t = graph.get_goal(goal=b_t)
a_t = E(o_t, g_t)
```

这里最重要的是 reviewer 一眼看出：

- LLM 不直接输出 primitive action。

### 5.3 模块 1：Online scene graph memory

这一节直接承认基础来自 UniGoal 风格的：

- online scene graph
- BEV map

然后强调本工作不重写 scene graph 本体，而是研究：

- scene graph 何时触发语义更新

当前代码事实基础已经存在：

- `smoothnav/main.py` 会把 `full_map` 和 `full_pose` 写回 `graph`
- `_apply_strategy()` 会通过 `graph.get_goal(goal=bias)` 调用图与几何模块

### 5.4 模块 2：High-level Planner

当前 planner 最值得保留的设计是：

- constrained choice set

它不是直接选 frontier index，而是在：

- object
- room
- direction

的约束集合里选语义目标，再解析成 `bias_position`。

投稿版建议把当前 `Strategy` 扩成：

- `strategy_id`
- `instruction_focus`
- `target_region`
- `expected_landmarks`
- `anchor_object`
- `bias_position`
- `stop_condition`
- `confidence`
- `explored_regions`

其中真正新增、且与 VLN 强相关的字段主要是：

- `instruction_focus`
- `expected_landmarks`
- `stop_condition`

它们负责把当前 text-goal 逻辑扩展成 instruction-progress state。

当前代码已有骨架：

- `target_region`
- `bias_position`
- `reasoning`
- `explored_regions`
- `anchor_object`

所以这不是推倒重来，而是顺着已有结构向 VLN 补字段。

### 5.5 模块 3：Low-level Monitor

当前 low-level 动作空间已经很合适：

- `CONTINUE`
- `ADJUST`
- `PREFETCH`
- `ESCALATE`

论文中应把它写成：

`narrow semantic monitor`

而不是 second planner。

现阶段触发条件还是：

- `new_nodes = graph.nodes[prev_node_count:]`

也就是基于 graph list growth 判断事件。

实现上够用，但论文里最好抽象成：

`GraphDelta API`

不要把 list-growth 细节暴露成方法本体。

推荐在文稿中定义的事件类型为：

- `new_room`
- `relevant_landmark`
- `room_transition`
- `contradiction`
- `frontier_near`
- `no_progress`

### 5.6 模块 4：Semantic–geometric grounding

这是方法最核心的点之一。

当前 `_apply_strategy()` 做的事情非常值得写：

- 不让 LLM 直接选 frontier；
- 而是把 `strategy.bias_position` 喂给 `graph.get_goal(goal=bias)`；
- 继续保留 UniGoal 的 frontier extraction 和 FMM-based execution。

这个接口设计比：

- LLM 直接选目标点
- LLM 直接出 primitive action

都更稳，也更容易解释可控性与鲁棒性。

### 5.7 模块 5：Pending strategy cache

当前代码中已经有：

- `pending_strategy`

因此论文里不要再说：

- `asynchronous planning`

而应该写成：

- `anticipatory cached planning`

也就是：

- 继续执行当前策略；
- 同时提前计算下一策略；
- 在 frontier 到达或当前策略失效时快速 promotion。

这个说法比“异步规划”准确，也不会被 reviewer 追问线程级实现。

### 5.8 局限

建议诚实写三条局限：

1. 当前 event trigger 仍然基于 graph list growth，工程上偏脆弱。
2. room / object grounding 仍有 substring matching 风险。
3. 当前系统是 hybrid LLM + heuristic controller，而且是 synchronous cache，不是真正 async。

这三条都不是致命缺陷，但必须写准。

## 6. 创新点写法

建议只保留四条主创新。

### 创新点 1：问题重定义

我们不再把“动作平滑”当作核心问题，而把在线显式地图中的 `semantic response latency` 定义为关键瓶颈。

### 创新点 2：Graph-delta-driven replanning

不是固定间隔重规划，而是让 scene graph 的语义增量直接驱动策略更新。

### 创新点 3：Semantic–geometric decoupling

LLM 只给 room / object / direction 级别的语义策略，再通过 bias grounding + `graph.get_goal()` 映射到几何目标，从而显著降低 LLM 直接控制低层运动的风险。

### 创新点 4：Dual-timescale map-grounded agent

高层强模型稀疏规划，低层快模型窄任务监控，底层执行器保持成熟几何模块不变。

可选第五条，只有在实验做出来之后再写：

### 创新点 5：Semantic efficiency diagnostics

在标准导航指标之外，加入：

- heavy planner calls
- response delay
- token efficiency

等语义效率分析。

## 7. 实验设置建议

### 7.1 主 benchmark 选择

主文建议只放两套：

#### R2R-VLNCE v1-3

作用：

- 连续 VLN 标准入口；
- 数据规模合适；
- baseline 成熟；
- 适合作为主要 benchmark。

#### RxR-VLNCE English Guide

作用：

- 指令更长、更复杂；
- 路径更不服从 shortest-path；
- 更能放大“语义更新粒度”的差异。

当前项目还没有正式接上 VLN-CE episode loader，因此：

- 当前 UniGoal text-goal 实验建议保留为 supplementary / appendix controlled study；
- 主文必须以 VLN-CE benchmark 为中心。

### 7.2 主指标与辅助指标

#### R2R-CE 主指标

- `TL`
- `NE`
- `OS`
- `SR`
- `SPL`

#### RxR-English 主指标

- `nDTW`
- `SR`
- `SPL`

其中：

- `nDTW` 对 RxR 很关键；
- `NE / TL` 可以作为辅助诊断表。

#### 语义效率诊断指标

建议必须加入：

- `heavy_planner_calls_per_episode`
- `fast_monitor_calls_per_episode`
- `decision_delay_steps`
- `goal_update_delay_steps`
- `strategy_switch_count`
- `tokens_per_episode`
- `success_per_1k_tokens`

这些指标负责证明：

- event-driven 比 periodic 更合理；
- semantic response latency 确实降低；
- 代价控制确实更好。

#### smoothness 指标如何处理

以下指标不要进主表：

- `sigma_v`
- `sigma_omega`
- `jerk`
- `pause_count`
- `smoothness_score`

统一放 appendix diagnostics。

原因很明确：

- 项目内部分析已经证明 pause 类指标在离散 turn-based setting 下容易误导。

#### 可选 instruction-progress 指标

如果实现成本可控，可以增加轻量 instruction-progress diagnostic。

它不必成为主指标，但能在讨论部分更完整地说明：

- 语义重规划是否真的帮助了 instruction-following 过程。

### 7.3 对比方法与消融

#### 内部对比必须有

- `baseline-periodic`
- `full-model`
- `no-monitor`
- `fixed-interval-replanning`
- `rules-only-monitor`
- `no-prefetch`
- `no-adjust`

如果工程可承受，再加：

- `direct-frontier-selection vs bias-grounding`

这个消融非常关键，因为它直接解释：

- 为什么语义—几何解耦接口更稳。

#### 外部对比建议分三组

第一组：

- official benchmark baselines
- 至少包含 VLN-CE repo 提供的 CMA baseline

第二组：

- compatible zero-shot VLN methods
- 优先 CA-Nav、Open-Nav、EmergeNav、DreamNav

第三组：

- upper-bound trained methods
- 单独成组，不与 zero-shot 混表

关于 SpatialNav / SG-Nav / UniGoal：

- SpatialNav 建议放 discussion 或带 caveat 的对比，因为它允许 pre-exploration；
- SG-Nav 与 UniGoal 更适合放在方法血缘与设计来源，不一定适合作为主 benchmark 同表 baseline。

### 7.4 公平性与协议

这一节必须写硬。

对于 RxR-Habitat / VLN-CE 类 benchmark，必须严格锁定：

- turn angle
- step size
- observation resolution
- 评测 split
- challenge-valid configuration

另外必须保存：

- config hash
- prompt hash
- git hash
- result directory
- benchmark validator output

当前本地项目并不是正式 benchmark 版本，因此论文实验一旦开始，必须同步建立这一整套实验记录。

## 8. 结果展示组织

主文建议只放四个核心结果对象。

### Figure 1：整体架构图

应突出：

- planner
- monitor
- grounding
- executor
- explicit map
- graph delta

### Table 1：R2R-CE 与 RxR-English 主结果

只放主指标。

### Table 2：semantic efficiency 结果

放：

- heavy calls
- tokens
- response delay
- success per token

### Table 3：消融

至少包含：

- `no-monitor`
- `fixed-interval`
- `rules-only`
- `no-prefetch`
- `no-adjust`

### Figure 2：定性案例

成功类型建议展示：

- `new_room`
- `frontier_near + prefetch`
- `contradiction + escalate`
- `stuck recovery`

失败类型建议展示：

- `ambiguous room label`
- `bad substring grounding`
- `late event trigger`
- `instruction phase drift`

这样整篇稿子会更像系统论文，而不是单纯堆指标。

## 9. 不要这样写

### 不要把主线写成 smoothness

当前真正有科学意义的是：

- response latency
- replanning grain

而不是：

- pause ratio
- action smoothing

### 不要说 asynchronous replanning

准确说法是：

- `anticipatory cached planning`

### 不要 claim “first explicit-map VLN”

因为 SpatialNav 已经把 explicit spatial scene graph 用到 zero-shot VLN，只是设定不同。

### 不要说“当前系统已经是完整 VLN benchmark”

因为当前代码事实还不是。

### 不要把 hybrid heuristic 当成缺点

当前 low-level 本来就是：

- hybrid LLM + heuristic controller

这反而是优点，因为它让 decision scope 更窄、系统更稳定。

## 10. 投稿定位建议

对于这篇“显式表示 + 系统设计 + benchmark 实验”的稿子：

- journal-first 更自然；
- conference 版本则适合压缩成更聚焦的系统故事。

更自然的目标类型是：

- representation / structured reasoning / embodied system / explicit memory / benchmarked agent design

当前从材料成熟度上看，更像一篇：

- 明确系统定位、
- 给出 clean method interface、
- 用 benchmark 和效率诊断支撑因果链

的 B 类稳定稿。

## 11. 最终定稿版定位

建议全文最终压缩成下面这个版本。

### 论文主题

在线显式地图上的事件驱动语义重规划 VLN agent。

### 核心问题

固定间隔重规划导致 `semantic response latency`。

### 方法主轴

`online scene graph + dual-timescale planner/monitor + bias grounding + preserved geometric executor`

### 核心创新

不是“更平滑”，而是：

- 更及时
- 更省
- 更稳

### 主实验

- `R2R-CE val_unseen`
- `RxR-English val_unseen`

### 主指标

R2R：

- `TL`
- `NE`
- `OS`
- `SR`
- `SPL`

RxR：

- `nDTW`
- `SR`
- `SPL`

### 辅助证据

- heavy planner calls
- decision delay
- token efficiency
- qualitative event cases

### 相关工作定位

位于：

- explicit-memory zero-shot navigation
- instruction-aware structured execution

的交叉处，但强调：

- online graph delta

而不是：

- pre-exploration
- 无图 execution structure

## 12. 与当前代码的映射关系

当前本地代码已经能支撑的论文说法：

- 双时标结构：
  `smoothnav/main.py`
- 高层语义策略：
  `smoothnav/planner.py`
- 窄动作空间的中层 monitor：
  `smoothnav/low_level_agent.py`
- bias grounding：
  `_apply_strategy()` + `graph.get_goal(goal=bias)`
- 保留几何执行器：
  `UniGoal_Agent.step()` + FMM
- pending strategy cache：
  `pending_strategy`

当前本地代码还不能直接支撑的说法：

- 已完整接入 R2R / RxR / VLN-CE 主 benchmark
- 已有 instruction progress state `k_t`
- 已有 benchmark-valid main table
- 已有 response delay / token efficiency 正式实验

因此：

- 方法叙事已经成熟；
- benchmark 叙事仍需补接。

## 13. 下一步最自然的工作

最自然的下一步不是继续扩展“smoothness”故事，而是直接把这份准备文档展开成论文初稿：

- 标题
- 摘要
- Introduction
- Related Work
- Method
- Experiment
- Discussion / Limitation

同时配套推进三项工程落地：

1. 接入正式 VLN-CE episode loader 与评测指标；
2. 在 `Strategy` 中补齐 instruction-progress 相关字段；
3. 建立 semantic efficiency 诊断统计链路。

## 14. 一句话结论

SmoothNav 当前最值得写成论文的，不是“平滑导航”，而是：

一种建立在 online explicit memory 之上的、graph-delta 驱动的、双时标语义重规划连续 VLN agent。

