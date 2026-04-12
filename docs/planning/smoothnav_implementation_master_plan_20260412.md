# SmoothNav 分阶段实施计划书（供 Codex 逐步执行）

版本：v1.0  
日期：2026-04-12  
用途：将当前 SmoothNav 原型，按**可观测、可测试、可比较、可扩展到标准 VLN benchmark**的顺序，逐步推进到可做正式实验与论文验证的状态。

---

## 0. 文档目的与使用方式

这不是一份“想法清单”，而是一份**带阶段门控（phase gate）的实施计划书**。  
Codex 的执行原则应当是：

1. **严格按阶段推进，不跨阶段跳做。**
2. **每个阶段必须先满足验收标准，再进入下一阶段。**
3. **Phase 0 和 Phase 1 以“证据化、状态机显式化”为主，不追求算法增强。**
4. **在 R2R/RxR benchmark 接线之前，必须先在当前 same-backbone 任务上证明控制器机制是可观测且有正向信号的。**
5. **禁止在没有 trace 和测试支撑的情况下，一边接 benchmark、一边改控制器核心逻辑。**

本计划书默认：
- 当前最真实的系统定义是：**基于 UniGoal backbone 的第二代层次语义重规划原型**；
- 当前最可信的论文主线是：**map-grounded dual-timescale agent**；
- 当前 runnable scope 仍是 `smoothnav|baseline × ins-image|text`，而不是正式 VLN benchmark；
- 当前最该补的是：**trace、状态机、测试、same-backbone 因果实验**；
- 当前不应重新回到：**custom executor / override_action / smoothness-first** 的老方向。

---

## 1. 当前项目统一共识（执行前必须统一）

### 1.1 当前系统的真实状态

当前代码已经真实存在并可作为后续工作的基础：

- 高层 `planner`：`smoothnav/planner.py`
- 低层 `monitor`：`smoothnav/low_level_agent.py`
- 底层 `executor`：`base_UniGoal/src/agent/unigoal/agent.py`
- 语义落地接口：`smoothnav/main.py` 中 `_apply_strategy()` 通过 `graph.get_goal(goal=bias)` 影响目标
- scene graph / frontier / FMM：仍由 UniGoal backbone 提供
- `pending_strategy` 已经存在，说明 anticipatory cached planning 已有原型

### 1.2 当前最强的研究资产

当前最强资产不是“smoothness”，而是：

1. **同 backbone 的因果比较条件已经天然具备**  
   `smoothnav` 与 `baseline` 共用同一套 scene graph、BEV、executor，只是高层控制机制不同。

2. **语义—几何解耦已经成立**  
   LLM 不直接输出 primitive action，而是输出 semantic strategy，再经 `graph.get_goal(goal=bias)` 落到几何目标。

3. **双时标 agent 雏形已经存在**  
   planner 稀疏规划，monitor 事件驱动判断，executor 每步执行。

### 1.3 当前最主要的短板

当前真正缺的不是“新模块”，而是：

1. 没有 step-level trace  
2. `main.py` 中状态机仍然隐式  
3. 没有 generation-2 测试  
4. benchmark adapter 尚未接入主工程  
5. benchmark-locked config 尚未建立  
6. 日志、实验 manifest、run 可追踪性不足  
7. smoothness 指标仍停留在 summary 主输出里，和论文主线不一致  

### 1.4 当前明确的非目标

在进入正式 benchmark 实验前，以下内容**不作为优先方向**：

- 不先做 RxR，多于 R2R
- 不先做复杂 instruction decomposition
- 不恢复 custom executor / override_action
- 不再以 pause ratio / smoothness score 作为核心优化目标
- 不在缺 trace 的情况下直接做“大规模 benchmark 跑分”

---

## 2. 总体路线图

整体路线分为 7 个阶段：

- **Phase 0：项目护栏与证据化基础设施**
- **Phase 1：控制状态机显式化（不改行为）**
- **Phase 2：当前 text-goal 任务上的 same-backbone 因果验证**
- **Phase 3：第一轮高概率收益优化**
- **Phase 4：R2R-CE benchmark adapter 与标准评测接线**
- **Phase 5：RxR-English adapter 与完整 benchmark 扩展**
- **Phase 6：论文导向的实验收束与结果资产化**

其中：
- **Phase 0–2 是必须先完成的底座阶段**；
- **Phase 3 是第一轮真正有望带来性能提升的优化阶段**；
- **Phase 4 以后才进入正式 benchmark 主线**。

---

## 3. 项目级执行红线

以下规则适用于所有阶段。

### 3.1 行为稳定性红线

在 Phase 0–1 中，**不允许引入刻意改变导航行为的大改动**。  
目标是“显式化”和“留证据”，不是“改算法”。

### 3.2 结果可追踪性红线

任何 run 必须生成最小结果包：

- `manifest.json`
- `summary.json`
- `step_traces/<episode_id>.jsonl`
- `planner_calls/<episode_id>.jsonl`
- `monitor_calls/<episode_id>.jsonl`

如果 run 没有这些文件，则该 run 不视为有效实验。

### 3.3 benchmark 公平性红线

任何进入论文主表的结果，必须使用**benchmark-locked config**。  
禁止使用 UniGoal 默认配置直接充当 R2R/RxR 正式结果。

### 3.4 控制器边界红线

保持以下设计不变，除非有明确立项：

- 不让 LLM 直接输出 primitive action
- 不恢复 `override_action`
- 不替换 UniGoal FMM executor
- 不绕开 `graph.get_goal(goal=bias)` 的语义—几何接口

### 3.5 安全与配置红线

- 移除明文 API key
- 同名实验结果不得覆盖旧结果
- 所有有效实验必须保存 final effective config

---

## 4. Phase 0：项目护栏与证据化基础设施

### 4.1 阶段目标

把当前原型变成一个**可观测、可回放、可追踪**的系统。  
此阶段不追求性能提升，只追求“任何一次 run 都能解释发生了什么”。

### 4.2 本阶段必须回答的 5 个问题

任何一步 step，都应能从日志中恢复：

1. 当前 active strategy 是什么？
2. 当前是否发生 graph event？
3. monitor 有没有被调用，输出了什么？
4. 目标有没有更新？
5. 当前动作最终是由 strategy grounding 主导，还是被 visible target / temp goal / stuck goal 覆盖？

### 4.3 主要任务

#### 任务 0.1：建立 run manifest 与输出目录隔离

目标：解决实验覆盖、配置不可追踪、运行上下文丢失的问题。

实施内容：
- 为每次 run 生成唯一 `run_id`
- 输出目录改为：`results/<date>/<run_id>/...`
- 新增 `manifest.json`
- 保存 final effective config
- 保存命令行、模型名、prompt schema 版本、goal_type、mode、num_eval、git hash（若不可得则写 `nogit`）

重点文件：
- `smoothnav/main.py`
- 新增 `smoothnav/experiment_io.py` 或等价模块

验收标准：
- 连续运行两次 `baseline` 不会覆盖旧结果
- 两次 run 都有各自独立 `manifest.json`
- `manifest.json` 能还原当前 run 的主要参数

#### 任务 0.2：移除敏感配置与硬编码运行依赖

目标：解决安全与可移植性问题。

实施内容：
- 将 API key / base_url 改为环境变量读取
- 清理配置中的明文 secrets
- 处理 `run.sh` 中写死路径问题

重点文件：
- `base_UniGoal/configs/config_habitat.yaml`
- `configs/setting.json`
- `run.sh`
- LLM wrapper 读取配置的位置

验收标准：
- 仓内不再存在明文 API key
- 缺失环境变量时能给出明确报错
- 在不同机器路径下可通过配置而非硬编码运行

#### 任务 0.3：增加 step-level trace

目标：让 step 级控制流可回放。

实施内容：
- 新增 `step_traces/<episode_id>.jsonl`
- 每 step 至少记录：
  - `episode_id`
  - `step_idx`
  - `pose_before`, `pose_after`
  - `graph_node_count`
  - `new_node_count`
  - `new_node_captions`
  - `current_strategy_*`
  - `pending_strategy_*`
  - `planner_called`
  - `monitor_called`
  - `monitor_decision`
  - `goal_before`, `goal_after`
  - `visible_target_override`
  - `temp_goal_override`
  - `stuck_goal_override`
  - `action`

重点文件：
- `smoothnav/main.py`
- 新增 `smoothnav/tracing.py`

验收标准：
- 至少 1 个 episode 的 trace 可完整落盘
- 可以通过 trace 恢复某一步是否发生了目标更新

#### 任务 0.4：planner / monitor 原始输入输出落盘

目标：让 LLM 行为可离线分析。

实施内容：
- 新增 `planner_calls/<episode_id>.jsonl`
- 新增 `monitor_calls/<episode_id>.jsonl`
- 记录：
  - prompt hash
  - raw prompt（必要时保存裁剪版与全文版）
  - raw response
  - parsed result
  - fallback 是否触发
  - resolved bias

重点文件：
- `smoothnav/planner.py`
- `smoothnav/low_level_agent.py`
- `smoothnav/main.py`

验收标准：
- 每次 planner/monitor 调用都可在独立 JSONL 中找到
- 对一次异常 episode，可以回看 raw response

#### 任务 0.5：建立最小 smoke pipeline

目标：确保日志基础设施可在最小实验上工作。

实施内容：
- 跑：`baseline + text + 5 episodes`
- 跑：`smoothnav + text + 5 episodes`
- 检查所有结果包是否完整

验收标准：
- 两个 run 都成功生成完整结果目录
- 每个 episode 都有 trace 文件
- summary 与 trace 的 planner/monitor call 数可以对齐

### 4.4 本阶段产出物

- `manifest.json`
- `step_traces/*.jsonl`
- `planner_calls/*.jsonl`
- `monitor_calls/*.jsonl`
- 输出目录隔离逻辑
- secrets 改造

### 4.5 本阶段禁止事项

- 不修改 planner/monitor 语义逻辑
- 不接 benchmark
- 不引入新 prompt schema
- 不做性能导向调参

### 4.6 Phase Gate（进入下一阶段前必须满足）

必须同时满足：

1. `baseline` 与 `smoothnav` 都可跑 5-episode smoke  
2. 每个 run 都有完整结果包  
3. 每个 planner/monitor 调用都有 raw dump  
4. 不再覆盖旧结果  
5. 不再有明文 API key  

---

## 5. Phase 1：控制状态机显式化（不改行为）

### 5.1 阶段目标

把 `main.py` 里的隐式状态机拆出来，形成**可测试的控制逻辑层**。  
此阶段仍然不追求性能提升，重点是：**行为保持大体不变，但结构变得可测、可维护**。

### 5.2 本阶段核心原则

- 先抽象，不优化
- 先显式化，不改语义
- 先保持行为一致，再谈控制器升级

### 5.3 主要任务

#### 任务 1.1：定义 `ControllerState`

建议字段：
- `current_strategy`
- `pending_strategy`
- `explored_regions`
- `prev_node_count`
- `prev_room_object_counts`
- `no_progress_steps`
- `last_position`
- `last_goal`
- `planner_call_count`
- `monitor_call_count`

重点文件：
- 新增 `smoothnav/controller_state.py`

验收标准：
- `main.py` 中不再散落多个平行状态变量
- 关键状态都可从 `ControllerState` 读取

#### 任务 1.2：定义最小版 `GraphDelta`

第一版只忠实包住当前已有触发：
- `new_nodes`
- `new_rooms`
- `room_object_count_changes`
- `frontier_near`
- `frontier_reached`
- `no_progress`
- `stuck`

重点文件：
- 新增 `smoothnav/controller_events.py`

验收标准：
- `main.py` 中不再直接散落 `graph.nodes[prev_node_count:]` 逻辑
- 所有 event trigger 都通过 `GraphDelta` 构造函数生成

#### 任务 1.3：抽取核心控制 helper

至少抽出以下 helper：
- `build_graph_delta(...)`
- `maybe_call_monitor(...)`
- `maybe_promote_pending(...)`
- `handle_frontier_reached(...)`
- `handle_stuck_replan(...)`

重点文件：
- 新增 `smoothnav/controller_logic.py`
- `smoothnav/main.py`

验收标准：
- `main.py` 只保留 orchestration，不直接包含大块策略分支
- 关键逻辑都可单测

#### 任务 1.4：建立 generation-2 测试骨架

至少新增：
- `tests/test_planner_gen2.py`
- `tests/test_monitor_gen2.py`
- `tests/test_controller_state.py`
- `tests/test_apply_strategy.py`

测试至少覆盖：
- planner parse/fallback
- monitor parse/action mapping
- pending promotion
- frontier reached 分支
- `_apply_strategy()` 坐标映射

验收标准：
- 不再依赖 generation-1 的 `test_planner_executor.py`
- generation-2 基础单测可跑

### 5.4 本阶段产出物

- `ControllerState`
- `GraphDelta` v0
- `controller_logic.py`
- generation-2 单测骨架

### 5.5 本阶段禁止事项

- 不新增复杂事件语义
- 不更换 planner/monitor prompt
- 不引入 benchmark adapter

### 5.6 Phase Gate

必须同时满足：

1. `main.py` 控制流已拆分  
2. `ControllerState` 与 `GraphDelta` 已建立  
3. 至少 4 份 generation-2 单测存在并可运行  
4. 重构前后 `dev5` 行为无明显偏移  

---

## 6. Phase 2：当前 text-goal 任务上的 same-backbone 因果验证

### 6.1 阶段目标

在当前 runnable scope 上先证明：

**双时标 event-driven agent 相比 periodic baseline，确实更及时、更省，且不明显伤害效果。**

注意：这一步不是 benchmark positioning，而是**内部因果验证**。

### 6.2 本阶段核心问题

1. `full` 相比 `baseline-periodic`，heavy planner calls 是否下降？
2. `full` 相比 `no-monitor`，decision delay 是否下降？
3. `full` 的 `SR/SPL` 是否至少不明显变差？

### 6.3 主要任务

#### 任务 2.1：把 baseline family 制度化为配置

建议通过 config/flags 控制，而不是复制代码路径。

建议 flags：
- `controller.enable_monitor`
- `controller.monitor_policy = llm | rules | off`
- `controller.enable_prefetch`
- `controller.replan_policy = event | fixed_interval | baseline_explore`
- `controller.enable_stuck_replan`

建议首批对照组：
- `baseline-periodic`
- `smoothnav-full`
- `smoothnav-no-monitor`
- `smoothnav-rules-only`
- `smoothnav-no-prefetch`
- （可选）`smoothnav-fixed-interval`

验收标准：
- 不需要复制 main loop 代码即可切换 ablation
- 每个 ablation 都可通过 manifest 标识

#### 任务 2.2：建立当前任务的指标管线

主看指标：
- `SR`
- `SPL`
- `avg_high_level_calls`
- `avg_low_level_calls`
- `strategy_switch_count`
- `decision_delay_steps`
- `goal_update_delay_steps`
- `pending_promotion_rate`

注意：
- smoothness 相关指标保留为 legacy diagnostics
- 不再作为主导优化目标

验收标准：
- `summary.json` 中能直接得到上述核心指标
- `decision delay` 与 `goal update delay` 可从 trace 自动计算

#### 任务 2.3：建立 dev20 / dev100 内部验证流程

建议流程：
- `dev20`：快速 smoke
- `dev100`：小规模 pilot

输出要求：
- 每个设置都有统一 `summary.json`
- 失败 case 可从 trace 回看

验收标准：
- 5 个对照组至少完成 `dev20`
- 关键组（baseline/full/no-monitor/rules-only/no-prefetch）完成 `dev100`

#### 任务 2.4：建立错误分类表

首版错误分类建议：
- `missed_event`
- `bad_grounding`
- `executor_override_conflict`

验收标准：
- 至少人工分析 10 个失败 episode
- 每个失败都能归入上述某类或新增明确类别

### 6.4 本阶段产出物

- baseline family config 化
- 当前任务主指标 summary
- `dev20 / dev100` 结果集
- 第一版 error taxonomy

### 6.5 本阶段禁止事项

- 不进入 R2R/RxR benchmark
- 不大改 planner prompt
- 不做复杂 instruction progress modeling

### 6.6 Phase Gate

必须同时满足：

1. `full` 相比 `baseline-periodic`，planner calls 或语义成本有下降  
2. `full` 相比 `no-monitor`，decision delay 有改善  
3. `full` 的 `SR/SPL` 不明显劣化  
4. 已建立首版失败类型表  

若不满足，则不得进入 benchmark adapter 阶段，必须先在当前任务上修控制器。

---

## 7. Phase 3：第一轮高概率收益优化

### 7.1 阶段目标

在已有 trace 与当前任务证据基础上，进行**最有可能带来正向收益**的第一轮优化。  
本阶段不追求方法复杂化，只追求高收益、低风险、可解释的改进。

### 7.2 优化方向 1：`GraphDelta` 从 list growth 升级为控制事件

#### 任务内容

在 `GraphDelta` v0 基础上，补充以下事件：
- `new_room_detected`
- `room_object_count_increase`
- `node_caption_changed`

实现原则：
- 先做最小快照 diff，不做复杂语义图编辑距离
- 保持与当前数据结构兼容

验收标准：
- 能在 trace 中明确区分“节点数增加”和“节点语义变化”
- missed event 类型的失败占比下降

### 7.3 优化方向 2：提高 `ADJUST` 的真实落地率

#### 任务内容

改进 anchor resolve，而不是上来换大模型：

1. room-aware object resolve  
   若当前 strategy 有 `target_region`，优先在该 room 下找 object。

2. distance-aware tie break  
   多个同名 object 时，不再取“第一个匹配”，而用当前 bias 附近 / target room 内最近 / agent 前方最近做 tie-break。

验收标准：
- `ADJUST` 成功导致 goal 变化的比例上升
- `goal_update_delay_steps` 改善

### 7.4 优化方向 3：把 `PREFETCH`/promotion 机制制度化

#### 任务内容

统一 prefetch 与 promotion 条件：

允许 prefetch：
- `frontier_near == True`
- 当前 strategy 为 room/object，且图中出现新的相关 room/object 证据

允许 promotion：
- frontier reached
- 当前 strategy 为 direction-like，而 pending 更具体（room/object）

验收标准：
- `pending_strategy` 的生成与晋升不再分散在多个隐式分支
- `pending_promotion_rate` 有明确统计

### 7.5 建议暂不做的优化

- 不上 embedding-heavy grounding
- 不做复杂 instruction parser
- 不增加新的 LLM 角色
- 不换 executor

### 7.6 Phase Gate

必须同时满足：

1. `GraphDelta` v1 已可记录控制事件  
2. `ADJUST` 落地率显著提升或可解释提升  
3. `PREFETCH`/promotion 已形成可统计机制  
4. 在 `dev100` 上相较 Phase 2 有至少一个核心指标改善  

---

## 8. Phase 4：R2R-CE benchmark adapter 与标准评测接线

### 8.1 阶段目标

将当前 controller 架构最小成本地接到**正式的 R2R-CE benchmark**。  
注意：第一版目标不是“充分优化 instruction modeling”，而是**先跑通标准 benchmark 主循环**。

### 8.2 核心原则

- 先 adapter，后增强
- 先 `goal_description = instruction_text`，后补 instruction-progress
- 先 R2R，再 RxR

### 8.3 主要任务

#### 任务 4.1：梳理 vendor tree 中现有 VLN 基础

重点阅读并确认：
- `habitat/tasks/vln/vln.py`
- `habitat/datasets/vln/r2r_vln_dataset.py`
- `config/habitat/task/vln_r2r.yaml`
- `config/habitat/dataset/vln/mp3d_r2r.yaml`

产出：
- `R2R adapter 设计说明.md`

验收标准：
- 明确 reset/step 所需字段
- 明确 metrics 导出路径

#### 任务 4.2：新增 `vln_instruction_env.py`

建议位置：
- `base_UniGoal/src/envs/vln_instruction_env.py`

目标：尽量复用 `InstanceImageGoal_Env` 接口风格，但新增：
- `instruction_text`
- `reference_path`
- `trajectory_id`
- `scene_id`

验收标准：
- 可加载 R2R episode
- reset 后能得到 instruction 文本与基础元信息

#### 任务 4.3：建立 benchmark-locked R2R config

新增：
- `configs/config_r2r_vlnce.yaml`

要求：
- 明确 task / dataset / split
- 与当前 UniGoal config 分离
- 建立 `validate_benchmark_config.py`

验收标准：
- 错误配置会被 blocker 阻止运行
- 主工程不再用 `config_habitat.yaml` 直接充当 R2R config

#### 任务 4.4：最小方式接 planner context

第一版做法：
- `goal_description = instruction_text`

暂不做：
- clause parser
- progress-aware instruction slicing

验收标准：
- planner/monitor 可以直接读取 instruction text
- 先跑通，再优化

#### 任务 4.5：导出 R2R 主指标

至少导出：
- `NE`
- `OSR`
- `SR`
- `SPL`

验收标准：
- 有独立 benchmark summary
- 可在 `val_unseen` 上得到完整结果

### 8.4 执行顺序

1. `dev20` on R2R wrapper  
2. `dev200` on `val_unseen`  
3. full `val_unseen` on core groups  

### 8.5 本阶段主对照组

- `baseline-periodic`
- `smoothnav-full`
- `smoothnav-no-monitor`
- `smoothnav-rules-only`
- （必要时）`smoothnav-no-prefetch`

### 8.6 Phase Gate

必须同时满足：

1. R2R adapter 可稳定跑 `dev20`  
2. `dev200` 已能输出标准 R2R 指标  
3. 至少 3 个主对照组完成 `val_unseen` 结果  
4. benchmark config blocker 已就位  

---

## 9. Phase 5：RxR-English adapter 与完整 benchmark 扩展

### 9.1 阶段目标

在 R2R 跑通后，将系统扩展到 **RxR-English**。  
注意：只做 English，不同时碰 Hindi / Telugu。

### 9.2 主要任务

#### 任务 5.1：建立 benchmark-locked RxR config

新增：
- `configs/config_rxr_vlnce_en.yaml`

要求：
- 明确 30° turn
- 0.25m step
- 30° look up/down
- 480×640 RGBD

验收标准：
- validator 能明确指出与当前 UniGoal config 的差异
- 错误配置不可用于正式结果

#### 任务 5.2：接入 RxR-English episodes

要求：
- 支持 `train / val_seen / val_unseen`
- 第一版仅 English guide

验收标准：
- 能稳定加载并运行 `dev20`

#### 任务 5.3：导出 RxR 主指标

至少导出：
- `NE`
- `SR`
- `SPL`
- `nDTW`
- `SDTW`

验收标准：
- `val_unseen` English 可得到完整 benchmark summary

#### 任务 5.4：建立 RxR 分析维度

建议新增两类分析：
- long instruction / long trajectory 分桶
- `response delay` 与 `nDTW` 的相关性

验收标准：
- 至少有 1 张相关性分析图的数据资产

### 9.3 Phase Gate

必须同时满足：

1. RxR-English `dev20` 跑通  
2. `dev200` 可输出完整 benchmark 指标  
3. `val_unseen` English 上完成核心组比较  
4. 配置完全锁定并经 validator 检查  

---

## 10. Phase 6：论文导向的实验收束与结果资产化

### 10.1 阶段目标

把工程结果整理成论文可用的主表、消融表、误差分析与可视化资产。

### 10.2 必做主表

#### 主表 1：same-backbone causal comparison

数据集：
- 当前 text-goal / 或 R2R-CE `val_unseen`

行：
- baseline-periodic
- ours-full
- ours-no-monitor
- ours-rules-only
- ours-no-prefetch

列：
- NE / SR / SPL（若是 benchmark）
- heavy planner calls / ep
- decision delay
- goal update delay

用途：
- 证明 event-driven dual-timescale agent 的因果价值

#### 主表 2：R2R-CE benchmark positioning

数据集：
- R2R-CE `val_unseen`

分组：
- trained / official baselines
- zero-shot / training-free comparable methods
- ours

#### 主表 3：RxR-English benchmark positioning

数据集：
- RxR-CE `val_unseen` English

分组：
- trained / official baselines
- full-protocol comparable zero-shot methods
- ours

### 10.3 必做消融

- event-driven vs fixed-interval
- with vs without monitor
- with vs without prefetch
- rules-only vs hybrid monitor
- bias grounding vs direct frontier selection
- （可选）preserved executor vs override-style executor

### 10.4 必做定性资产

至少准备：
- 4 个成功案例
- 4 个失败案例
- strategy timeline 图
- graph delta 触发示意
- `decision delay` vs performance 分析图

### 10.5 阶段验收标准

必须同时满足：

1. 三张主表的数据已齐  
2. 核心消融结果已齐  
3. 至少 8 个高质量案例可视化素材已齐  
4. 所有结果都能追溯到 manifest 与 trace  

---

## 11. 跨阶段重点关注事项（项目管理视角）

### 11.1 重点关注一：不要让实验先于证据基础

表现形式：
- 没有 trace 就开始大规模跑 benchmark
- 没有 state machine 测试就开始改控制器

风险：
- 跑不稳、看不清、解释不明白

应对：
- Phase Gate 严格执行

### 11.2 重点关注二：不要让 benchmark adapter 和方法改造交织失控

表现形式：
- 一边接 R2R，一边改 planner prompt，一边换 monitor 逻辑

风险：
- 任何结果都无法归因

应对：
- 先 adapter 后优化
- adapter 期只允许最小语义接入

### 11.3 重点关注三：不要让 smoothness 重新成为主目标

表现形式：
- 调参只看 pause ratio
- 主表继续放 smoothness_score

风险：
- 论文主线再次偏离

应对：
- 主线指标固定为 response / efficiency / benchmark performance
- smoothness 只保留为 appendix diagnostics

### 11.4 重点关注四：不要让 baseline 不清晰

表现形式：
- baseline 与 ours 路径混杂
- ablation 需要复制 main loop

风险：
- same-backbone 因果比较失效

应对：
- baseline family 全部配置化
- 所有对照共享同一 executor

### 11.5 重点关注五：不要忽视 executor override 问题

表现形式：
- 以为 strategy 没起作用，实际上是 visible target / temp goal / stuck goal 覆盖了它

风险：
- 错误分析失真

应对：
- step trace 必须记录 override flags

### 11.6 重点关注六：不要忽视 benchmark 公平性

表现形式：
- 用当前 `turn_angle: 15` 直接跑 RxR
- 用 sampled subset 结果充当 full benchmark 主表

风险：
- 结果不可比较，论文风险极高

应对：
- validator + benchmark-locked configs + protocol note

---

## 12. 风险清单与应对策略

### 风险 1：Phase 0 做不干净，后面所有实验无法解释

应对：
- 将 trace / manifest 作为强制输出
- 未产出则视为失败 run

### 风险 2：Phase 1 重构改变行为

应对：
- 先跑 `dev5` 对照
- 重构前后比较关键计数与 episode 走势

### 风险 3：Phase 2 没有任何正向信号

应对：
- 暂不进入 benchmark port
- 优先排查：missed event / grounding / executor override

### 风险 4：Phase 3 的优化收益不稳定

应对：
- 只保留能在 `dev100` 上稳定改善的改动
- 其余进入实验分支，不进主线

### 风险 5：R2R adapter 工程量失控

应对：
- 只做最小 wrapper
- 尽量复用 `InstanceImageGoal_Env` 的接口风格
- 不在 adapter 期引入复杂 instruction progress

### 风险 6：RxR benchmark 配置不公平

应对：
- 单独 validator
- config blocker
- English-only first

---

## 13. 交付物清单（每阶段）

### Phase 0 交付物
- output manifest
- step trace
- planner/monitor dump
- secrets 改造
- smoke run 结果

### Phase 1 交付物
- `ControllerState`
- `GraphDelta` v0
- `controller_logic.py`
- generation-2 单测

### Phase 2 交付物
- baseline family config
- current task summary
- `dev20/dev100` 结果
- error taxonomy

### Phase 3 交付物
- `GraphDelta` v1
- ADJUST grounding 优化
- PREFETCH/promotion 机制统一
- 优化前后对照结果

### Phase 4 交付物
- `vln_instruction_env.py`
- `config_r2r_vlnce.yaml`
- benchmark validator
- R2R `val_unseen` 结果

### Phase 5 交付物
- `config_rxr_vlnce_en.yaml`
- RxR-English adapter
- RxR `val_unseen` English 结果
- 长指令/轨迹分桶分析

### Phase 6 交付物
- 3 张主表
- 核心消融表
- 8 个定性案例
- 论文图表原始数据

---

## 14. 建议的首周执行顺序（可直接交给 Codex）

### Day 1
- 建立 `run_id` 与输出目录隔离
- 生成 `manifest.json`
- 移除明文 secrets

### Day 2
- 新建 `tracing.py`
- 增加 step-level trace
- 增加 planner/monitor raw dump

### Day 3
- 建立 `ControllerState`
- 建立 `GraphDelta` v0
- 抽 `controller_logic.py`

### Day 4
- 写 generation-2 单测骨架
- 先覆盖 parser / promotion / apply strategy

### Day 5
- 把 baseline family 做成 config/flags
- 跑 `dev5` 比较行为是否稳定

### Day 6
- 跑 `text-goal dev20`
- 输出 summary 与 trace

### Day 7
- 人工看 10 个失败 case
- 产出第一版 error taxonomy

---

## 15. 结语：本计划书的总原则

本项目当前最应该守住的，不是“方法越来越复杂”，而是：

**先把当前 agent 控制闭环做成可观测、可测试、可比较的系统，再逐步推到标准 VLN benchmark。**

只要 Phase 0–2 做扎实，后面的 benchmark 接线、实验主表和论文叙事都会自然收束；反之，如果在没有 trace、没有状态机、没有 baseline family 的情况下直接跳到 R2R/RxR，很容易陷入“跑不稳、看不清、也解释不明白”的状态。
