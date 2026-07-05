# Cross-scene Semantic Target 理论模型与下一阶段实验指南

Date: 2026-04-23  
Status: execution-ready theory + experiment guide  
Scope: 指导 SmoothNav 在 repaired cross-scene 设置下，围绕“如何稳定地产生高质量、可行动、持续更新的 semantic target”开展下一阶段探索实验。

---

## 0. 目的

当前项目已经证明：

- `smoothnav-full` 在 intact-scene `dev5` / `dev15` 上优于 baseline；
- 旧的执行层 hijack / override 主问题已经基本被排除；
- repaired cross-scene 验证已经真正跑起来。

但 repaired cross-scene 优势尚未稳定建立。  
因此下一阶段的目标，不再是“继续盲修 controller”，而是：

> 建立一套关于 **semantic target 生成** 的理论模型，并用最小实验确认：
> 1. bottleneck 到底在 target 质量、geometric actionability，还是更新机制；
> 2. 应该优先增强哪一层：planner、grounding、frontier/value map、还是执行器接口。

---

## 1. 当前实验事实压缩

### 1.1 已成立的事实

#### intact-scene
- `smoothnav-full` 在完整 explicit intact-scene `dev5` 和 `dev15` 上优于 `baseline-periodic`。
- 这说明系统主线不是 toy idea，而是已有 baseline-beating evidence。

#### repaired cross-scene
- HM3D `val` repaired bundle 已可用，多 scene loader 已恢复。
- cross-scene 现在不再卡在数据坏掉，而是真正进入方法比较。

#### 已基本排除的旧问题
- `temp_goal_override`
- `stuck_goal_override`
- `visible_target_override`
- `global_goal_override`
- planner budget exhausted 直接终止
- north-only parse-failure fallback
- room/object caption 污染的一条关键分支

这些问题曾经是主问题，但现在已经不是跨 scene 失败的主解释。

### 1.2 仍然存在的当前问题

在 repaired cross-scene hard cases（尤其 `ep228`, `ep527`, `ep717`）上，当前主要残留问题是：

- `out_of_local_window`
- `same_frontier_as_prev`
- planner `empty_response`
- search efficiency 不足
- semantic target 虽然更稳定，但没有稳定转化成高价值 frontier / region 选择

### 1.3 当前最有代表性的样本

#### 正向 repaired cross-scene 样本
- `ep661` repaired rerun
  - `SR = 1.0`
  - `SPL ≈ 0.8529`
  - `executor_override_ratio = 0.0`

说明：当前方法并不是“跨 scene 必败”，而是存在可转正的 case。

#### 负向但已净化的 hard case
- `ep228`
  - 最新几轮 patch 后仍 fail
  - 但 `executor_override_ratio = 0.0`
  - `goal_update_delay_steps` 已从高值降到接近 0~1 量级
  - 剩余主要是 `out_of_local_window` + `same_frontier_as_prev`

说明：它已经是一个很干净的 semantic-target / grounding hard case。

#### 正式线近期新信号
- `formal v2` 的 `ep228` / `ep527`
  - 已恢复 room-level semantic planning
  - 不再主要被 fallback / empty-response 支配
  - 当前仍需等待最终 `summary`

---

## 2. 问题的理论建模

---

### 2.1 我们真正想生成的不是 label，而是 target contract

semantic target 不能被看成一个简单的：

- room label
- object caption
- direction token

真正有用的 target，应该是一个 **target contract**：

\[
g_t = (\text{semantic hypothesis}, \text{support region}, \text{actionability}, \text{confidence})
\]

它必须同时满足：

1. **语义正确**：与任务文本真正相关  
2. **几何可执行**：能落到 map/frontier/local window 上  
3. **时间稳定**：不会每步无意义抖动  
4. **可更新**：有新证据时能转移 posterior

### 2.2 第一性原理：这是 belief-driven search，而不是 label selection

在跨 scene 下，目标初始不可见，系统真正应该维护的是：

\[
p(z \mid x, M_t, E_t)
\]

其中：
- \(x\)：文本任务
- \(M_t\)：当前 partial map / graph
- \(E_t\)：当前观测到的对象、房间、关系等证据
- \(z\)：最值得搜索的 support region（room/frontier/cell/cluster）

也就是说，semantic target 的本质不是 “当前最喜欢哪个 room”，而是：

> 当前 belief 下，**哪一片空间最值得搜**。

### 2.3 高质量 target 的效用函数

可以把 target 质量写成：

\[
U_t(z) = \alpha S_t(z) + \beta R_t(z) + \gamma I_t(z) - \lambda C_t(z) - \mu K_t(z)
\]

其中：

- \(S_t(z)\): semantic relevance  
  文本与区域/对象/关系的匹配度

- \(R_t(z)\): reachability / actionability  
  这个 target 是否 map-valid、local-valid、frontier-valid

- \(I_t(z)\): information gain  
  搜这个地方能否显著增加“发现目标”的 posterior

- \(C_t(z)\): switch cost  
  目标切换的代价，避免无意义跳变

- \(K_t(z)\): control cost  
  这个 target 会不会引发 planner churn / budget pressure / repeated noop

### 2.4 这五项对当前系统意味着什么

当前 SmoothNav 在 cross-scene 的问题，恰好对应这 5 项里的缺口：

- `S_t(z)`：文本属性 grounding 还弱，仍偏 room common-sense / category prior
- `R_t(z)`：`out_of_local_window` 说明 semantic target 仍会掉出可执行区
- `I_t(z)`：重复同一个 frontier / room，说明信息增益建模不够强
- `C_t(z)`：旧版曾有强 churn，现在已大幅改善
- `K_t(z)`：planner empty-response / budget 问题说明 control cost 仍敏感

### 2.5 由第一性原理推出的 5 条必要原则

#### Principle 1 — 文本不能过早塌缩成粗类别
如果一开始就把 text goal 压成粗类（chair / sofa / bed），会丢掉：
- 材质
- 颜色
- 邻接关系
- 场景上下文

跨 scene 时这会直接让 room prior 压过真正的属性线索。

#### Principle 2 — 证据不应直接等于行动目标
visible object、room caption、scene graph object，都是证据，不应自动成为主目标。

#### Principle 3 — 行动目标的落点最好在 map/frontier/value 层
room/object label 语义上有用，但执行上不可直接依赖。  
真正稳的是：
- frontier
- value map
- region posterior

#### Principle 4 — 更新必须反映 posterior shift，而不是 event reflex
当前系统不应因为“有个 event”就切，而应因为 belief 明显变化才切。

#### Principle 5 — budget exhaustion 不应直接判死
已经从实现上修正：budget 应降 planner 频率，而不是直接终止。

---

## 3. 当前 bottleneck 的层级定位

### 3.1 已被基本排除：执行层 hijack

现在 cross-scene fail 已经不能再主要归因于：
- temp/stuck goal takeover
- visible/global override

这些问题已经基本不是主矛盾。

### 3.2 当前 bottleneck A：semantic target 质量不足

最新 formal run 已经证明 planner 开始健康地产生：
- `living room`
- `lounge`
- `bedroom`
- `office room`
- `dining room`

这种 room-level reasoning。

但问题在于：

> 这些 target 虽然“看起来合理”，却还没有稳定转化成更高 SR/SPL。

也就是说，semantic target 的**质量**仍然不够高。

### 3.3 当前 bottleneck B：semantic target 和 geometric target 之间仍有 actionability gap

证据：
- `out_of_local_window`
- `same_frontier_as_prev`

说明：
- 语义上合理的选择
- 仍可能在几何上不可执行，或者不能造成真正 frontier 分岔

### 3.4 当前 bottleneck C：uncertainty 下的 search efficiency 不够

即使 controller 已更稳、更新已更快、override 已归零，
hard case 仍 fail。

这意味着问题已经转成：

> 在弱证据条件下，系统仍然没有足够高效地选择下一个“最值得搜索”的区域。

### 3.5 当前 bottleneck D：planner reliability 仍高度影响结果

shadow 线已反复证明：
- planner `empty_response` 仍然会发生

formal 线恢复后又说明：
- 当 planner 恢复稳定输出时，行为质量明显改善

因此 planner 目前仍是 cross-scene 成败的核心敏感因子。

---

## 4. 成熟工作的做法：它们如何解决同类问题

---

### 4.1 SemExp

#### 核心做法
- episodic semantic map
- high-level semantic policy 选 long-term goal
- analytic local policy 执行

#### 它解决的问题
把 semantic reasoning 与 execution 解耦，保证 low-level 只执行 map-valid goal。

#### 对我们的启发
当前 SmoothNav 仍偏 symbolic target 驱动，缺一个真正稳定的 map-goal interface。

---

### 4.2 PONI

#### 核心做法
在 partial semantic map 上预测 potential function，然后选 best frontier。

#### 它解决的问题
避免把 visible instance 直接变成搜索目标。

#### 对我们的启发
语义应该先转成 frontier-level utility，而不是直接转成 room/object target。

---

### 4.3 PEANUT

#### 核心做法
在全局 map 上预测 unexplored cells 的 target probability。

#### 它解决的问题
在目标尚未可见时，仍然能稳定给出“下一片值得搜的区域”。

#### 对我们的启发
当前 hard case 的主问题恰恰是目标不可见时的 global search prior 不够强。

---

### 4.4 VLFM

#### 核心做法
- 先提 frontier
- 再用 VLM 给 frontier 打语言相关性分数

#### 它解决的问题
避免用 symbolic room/object 直接主导 search，而是让语言驱动 frontier ranking。

#### 对我们的启发
这和我们现在最缺的层非常一致：

> semantic target 应该先变成 frontier/value，而不是直接变成 symbolic target。

---

### 4.5 SG-Nav

#### 核心做法
- online scene graph
- graph reasoning
- re-perception / verification before commitment

#### 它解决的问题
防止错误 visible object 或局部 perception noise 直接 hijack 全局策略。

#### 对我们的启发
我们现在已经做了一部分 suppression / sanitization，但还没做到显式 re-perception gate。

---

### 4.6 ZSON / open-vocab CLIP

#### 核心做法
不把 text goal 压成固定粗类别，而是直接在 embedding space 对齐。

#### 它解决的问题
避免 text goal 在跨 scene 时丢失属性细节。

#### 对我们的启发
如果后续继续强化 text target quality，这条路线非常关键。

---

## 5. 理论分析导出的当前结论

### Claim
当前 cross-scene 下的主问题，不是“系统不会推理”，而是：

> **semantic reasoning 还没有稳定地产生 frontier/value-level、可执行、可更新的 search target。**

换句话说：
- 它现在已经能给出 room-level semantic plan
- 但还不是一个 robust semantic search system

---

## 6. 对应下一阶段探索实验的路线

下面的路线按优先级排序，目标是尽快把 repaired cross-scene 从“过程健康”推进到“指标转正”。

---

### Phase 1 — Planner reliability stabilization

#### Goal
确认 planner empty-response 是否还是最主要噪声源，并将其对结果的影响最小化。

#### Experiment 1
继续 formal `ep228 / ep527 / dev3` rerun，记录：
- planner empty-response rate
- fallback ratio
- strategy switch count
- `goal_update_delay_steps`

#### Experiment 2
对比：
- formal line
- cheap shadow line

看 planner quality 对最终 `SR/SPL` 的影响是否显著。

#### Go / No-Go
- 如果 planner output 恢复后指标显著改善，则 planner reliability 是主因
- 如果过程健康但指标仍不上升，进入 Phase 2

---

### Phase 2 — Frontier / value anchoring

#### Goal
把 semantic target 从 room/object symbolic target，转成 frontier/value-level target。

#### Minimal patch direction
- 保留 room/object 作为 evidence
- 不再让其直接成为最终行动主目标
- 对每个 candidate frontier 计算：
  - semantic relevance
  - novelty / information gain
  - geometric actionability

形成 lightweight frontier scoring

#### Validation
先在：
- `ep228`
- `ep527`
- `ep717`

做 matched ablation。

#### Success signal
- `same_frontier_as_prev` 明显下降
- `out_of_local_window` 明显下降
- `SR/SPL` 开始上升

---

### Phase 3 — Re-perception / commit gate

#### Goal
防止局部语义证据在弱上下文下过早变成高层策略。

#### Minimal patch direction
- 对 room/object target 在 commit 前增加 lightweight verification
- 可以是：
  - graph evidence threshold
  - repeated observation threshold
  - caption consistency threshold

#### Success signal
- semantic target 切换次数下降
- room/object choice 的误导率下降

---

### Phase 4 — Text-goal representation strengthening

#### Goal
减少粗类别 room prior 对 text goal 的支配。

#### Minimal patch direction
- 保留现有 text mission
- 增加 attribute-sensitive relevance score
- 先在 planner object/room ranking 中使用，不必一开始重写全套 embedding pipeline

#### Success signal
- hard cases 中 room ordering更贴近 text attributes
- same-category但无关区域被降权

---

## 7. 下一阶段最小实验矩阵

### 必做最小矩阵
在 repaired cross-scene 上持续维护：

- `ep228`
- `ep527`
- `ep661`

profiles:
- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

### 为什么是这三条
- `ep228`：净化后的 hard case
- `ep527`：budget / planner quality 敏感 case
- `ep661`：已成功的 positive anchor case

这三条足以判断：
- 修复是在 hard case 上真正改善
- 还是只在 easy / already-good case 上有效

---

## 8. 当前阶段的成功标准

当前阶段还不是“直接冲顶会表格”，而是先过下面这几个 gate：

### Gate A
formal line 不再主要被：
- `empty_response`
- `budget_exhausted`
主导失败

### Gate B
`ep228` / `ep527` 至少有一条从 fail 转成 success  
或明确出现非偶然的 SPL 提升

### Gate C
`same_frontier_as_prev` / `out_of_local_window` 下降到不再是主导 failure

### Gate D
在 small repaired suite 上，`smoothnav-full` 至少不劣于 baseline / no-monitor

只有 Gate A-D 过了，才值得往“顶会可投稿 aggregate”继续扩。

---

## 9. 一句话结论

### Bottom line
下一阶段最该探索的，不是继续修 executor，而是：

> **把 semantic reasoning 稳定地落成 frontier/value-level、可执行、可更新的 search target。**

当前所有 hard case 都在把我们推向这个方向。


---

## 6. 2026-04-24 审查结论：当前问题边界、成熟工作对照、以及后续路线是否正确

### 6.1 当前问题的最清晰表述

#### Observation — 当前主问题已经不是“看不到 target”，而是“target branch 不稳定且不可复现”

从 `s12`–`s14` 可以把问题明确拆开：

1. `s12` 已证明：
   - `target_candidate_detected` 可以触发；
   - planner 可以切到 `unexplored target:tv`；
   - `target_anchor_local_map_refresh` hook 确实会触发；
   - 也暴露出 stale `pending_strategy` 会把 target-anchor 挤掉的 controller bug。

2. `s13` / `s14` 已证明：
   - 即使 Habitat seed 固定、`SMOOTHNAV_LLM_TEMPERATURE=0`，在线 rerun 仍然可能完全进不到 target branch；
   - 因此当前最大的实验风险已经变成 **branch reproducibility**，而不是单纯 target detection / refresh 逻辑。

#### Observation — 一旦进入 target branch，当前系统仍缺少“approach progress semantics”

即使在 `s12` 中成功进入 `unexplored target:tv`：

- 系统仍然没有一个明确的 `target_anchor_attempt` 生命周期；
- 没有把“正在接近 target support region”与“只是换了一个 frontier”区分开；
- frontier/value 仍偏局部几何可执行性，而不是 target approach utility。

#### Claim — 当前问题可以压缩成两个层次

**层次 A：在线分支不可复现**
- 同一个 `ep228`，相同 seed，`temperature=0`，仍然可能完全不进入 target branch。

**层次 B：进入分支后，缺少稳定的 target-approach contract**
- 即使进入 `unexplored target:tv`，也仍然可能因为 pending / frontier churn / progress definition 不清晰而失败。

### 6.2 与成熟工作的对照

#### Observation — SemExp / PONI / VLFM / SEEK 的共同点都不是“更早 commit object”，而是“把搜索目标做成空间分布 / potential / frontier value”

对照成熟工作可以看到一条稳定主线：

1. **SemExp**：semantic map + high-level long-term goal + deterministic local policy。它的核心不是把当前看到的 object 直接当执行目标，而是把语义信息落到长时程 map-goal 上。

2. **PONI**：明确把问题拆成 `where to look?` 和 `how to navigate to (x, y)?`，并把前者做成 potential function，而不是离散 label commit。

3. **VLFM**：直接构建 language-grounded frontier value map，用 value 来选 frontier，而不是先把 observation 压缩成 caption/object 再 hard commit。

4. **SEEK**：用 DSG + relational semantic network 维护目标分布 / 概率，而不是把语义证据直接变成动作目标。

#### Claim — 这几类成熟工作整体上支持我们已经形成的理论判断

它们都支持以下判断：

- 语义证据应该先进入 **belief / potential / value map**；
- 执行动作目标应落到 **frontier / region / waypoint** 层；
- 真正重要的是 **target likelihood / semantic frontier utility / probabilistic support region**；
- 不是“更激进地 object commit”，而是“更稳定地 spatially-grounded search”。

### 6.3 对当前实验路线的审查

#### Claim — 当前路线中“继续缩小实验、先抓单个 hard case”是正确的

原因：
- `ep228` 已经成为一个干净的 hard case；
- 它能同时暴露 target uptake、target branch persistence、frontier grounding、local map refresh、pending promotion 等多个核心问题；
- 在 branch reproducibility 没解决前，扩大 suite 只会把噪声放大。

#### Claim — 当前路线中“继续 blind rerun 在线实验”是不正确的

原因：
- `s13` / `s14` 已经说明 blind rerun 无法稳定覆盖 target branch；
- 这种设置下，online rerun 更多是在抽样不同轨迹，而不是验证同一个 hypothesis；
- 因此再继续跑很多 `ep228` online rerun，研究价值会迅速下降。

#### Claim — 当前路线需要从“在线重跑”切换到“受控 replay / branch capture”

这一步已经不是可选优化，而是为了让实验真正可证伪：

1. 先把 `s12` step `564` 之后的 target branch 固定下来；
2. 在同一 branch 上验证 pending-fix 是否阻止 stale direction 覆盖 target-anchor；
3. 再在同一 branch 上验证 target-anchor attempt / stalled decommit / target-approach value 是否真的改善行为；
4. 只有离线或半离线 counterfactual 成功，再回到在线完整 episode。

### 6.4 接下来最合理的 3 步

#### Step 1 — 做 branch capture / replay harness（最高优先级）

目标：把 `s12` 的 target branch 变成一个可重复验证的固定测试面。

最小要求：
- 能重放 `s12` 从 step `564` 之后的 planner input / graph_delta / grounding context；
- 能在不依赖完整在线 episode 漫游的情况下，验证 controller patch 是否保住 `unexplored target:tv`。

#### Step 2 — 引入 target-anchor attempt state，而不是继续堆事件规则

目标：把当前 `unexplored target:<name>` 从“字符串策略名”升级成“有生命周期的搜索假设”。

最小要求：
- attempt id / enter step / last progress step；
- stalled 判定；
- decommit reason；
- refresh count；
- 与 pending promotion 的优先级约束。

#### Step 3 — 让 frontier scoring 显式包含 target-approach utility

目标：不再只用“局部可执行 + 距 bias 近”，而是更接近：

\[
U(z)=\alpha \cdot semantic\_support(z)+\beta \cdot approach\_progress(z)+\gamma \cdot info\_gain(z)-\lambda \cdot churn(z)
\]

最小要求：
- 至少能监控“目标 anchor 附近 frontier 是否真的在减少目标不确定性”；
- 如果没有，就应该允许 target-anchor attempt stalled，而不是无限 refresh。

### 6.5 最终审查结论

#### Final Claim

目前后续实验路线需要做一个关键修正：

- **正确的部分**：继续聚焦单个 hard case、继续围绕 semantic target / frontier/value / grounding 做窄实验；
- **需要纠偏的部分**：停止 blind online rerun，把主要精力转到 `s12` target branch 的受控 replay / branch capture；
- **理论上更正确的方向**：向成熟工作的共同范式靠拢——从 hard object commit 转向 belief / potential / value-map 驱动的 target search。

也就是说，后续路线总体方向是对的，但执行顺序需要调整为：

> 先解决“可重复验证” → 再验证 patch → 再做 online episode 回归。
