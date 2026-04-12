# SmoothNav 分层导航架构设计

## 1. 问题定义

### 1.1 Baseline (UniGoal) 的结构与瓶颈

UniGoal 的导航循环：

```
每步:
  ① BEV_map.mapping(rgbd)                    地图更新
  ② graph.update_scenegraph()                 场景图更新 (每2步, ~1 LLM call)
  ③ [每25步末尾 或 到达目标] graph.explore()   目标选择 (5-20 LLM calls)
  ④ agent.step(goal_maps)                     FMM执行 + instance_discriminator
```

**explore() 内部的 LLM 调用分解:**

| 函数 | LLM 调用数 | 模型 | 用途 |
|------|-----------|------|------|
| overlap() → GraphMatcher | 1×N_common | sonnet | 判断场景图与 goal graph 匹配度 |
| insert_goal() → room_predict | 1 | sonnet | 预测目标在哪个房间 |
| graph_corr() × N_groups | 4×N | sonnet | 对每个 group 做语义相关性评分 |
| get_goal() | 0 | — | FMM 空间评分（纯算法） |

典型一次 explore(): 有 3 个 group → 1+1+12 = **14 次 sonnet 调用**。

**核心瓶颈:**

1. **决策真空**: explore() 之间的 ~25 步盲跑期间无智能判断。即使看到新物体、进入新房间，也不会调整目标。
2. **模型-任务不匹配**: graph_corr 问的是"A和B一起出现的概率是多少"——简单常识推理，却用了 sonnet 级模型，且每个 group 重复问 4 遍。
3. **阻塞式调用**: explore() 执行期间 agent 完全停滞，所有 LLM 调用串行完成后才能继续移动。

### 1.2 SmoothNav 的目标

> 通过分层模型架构 + 事件驱动触发 + 预判式规划，消除决策真空期和阻塞等待，在保持/提升导航成功率的同时实现无停顿导航。

**不是**减少 LLM 调用总量（虽然这是副产品），**而是**让正确级别的模型在正确的时间做正确的事。

---

## 2. 分层架构

### 2.1 总体结构

```
┌─────────────────────────────────────────────────────────────┐
│ 高层 Planner (sonnet)                                        │
│ 深度场景理解 + 搜索策略                                        │
│ 触发: 底层 ESCALATE / PREFETCH                               │
│ 频率: 3-8 次/episode                                         │
│ 执行: 可提前触发，结果缓存待用                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ Strategy (语义方向 + bias坐标)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 底层 Agent (haiku)                                           │
│ 轻量评估 + 触发判断                                           │
│ 触发: 场景图发生实质变化时                                     │
│ 频率: 事件驱动                                                │
│ 输出: CONTINUE / ADJUST / PREFETCH / ESCALATE               │
└────────────────────────┬────────────────────────────────────┘
                         │ bias_position (传给 FMM)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 执行层 (FMM + instance_discriminator, 每步)                   │
│ get_goal(bias) → FMM 路径规划 → 离散动作                      │
│ instance_discriminator: 目标检测/锁定 (已有, 不修改)           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 信息流

```
Episode 开始
    │
    ▼
高层 Planner (sonnet, 1 call)
    │ → Strategy_0 = {target_region, bias_position, reasoning}
    ▼
get_goal(bias_position) → frontier_A
    │
    ▼
每步循环 ─────────────────────────────────────────────
    │
    ├─ ① BEV_map.mapping(rgbd)             [每步]
    ├─ ② graph.update_scenegraph()          [每2步, 与baseline相同]
    │     └→ 返回: has_new_nodes (bool)
    │
    ├─ ③ [has_new_nodes=true 时] 底层 Agent  [事件驱动]
    │     输入: Strategy + delta_nodes + dist_to_goal
    │     输出: CONTINUE / ADJUST(new_bias) / PREFETCH / ESCALATE
    │     │
    │     ├─ CONTINUE → 不做任何事
    │     ├─ ADJUST   → get_goal(new_bias) → 更新 frontier
    │     ├─ PREFETCH → 异步触发高层 (agent继续用旧目标)
    │     └─ ESCALATE → 触发高层 (agent继续用旧目标直到新策略ready)
    │
    ├─ ④ [高层结果ready时] 更新 Strategy → get_goal(new_bias)
    │
    └─ ⑤ agent.step(goal_maps)             [每步, FMM执行]
          └→ instance_discriminator (已有, 每步, 不修改)
```

---

## 3. 各层详细设计

### 3.1 高层 Planner

**模型**: sonnet (强推理模型)

**触发时机**:
- Episode 开始（首次规划）
- 底层发出 ESCALATE（策略已过时）
- 底层发出 PREFETCH（预判即将需要新策略）

**输入**:
```
- 完整场景图文本 (graph.scenegraph 序列化)
- 目标物体描述 (intrinsic: 类别/外观, extrinsic: goal graph)
- 探索历史: 已访问的 room 列表, 已排除的区域
- [如有] 底层 escalate 的 reason
```

**输出 — Strategy**:
```python
@dataclass
class Strategy:
    target_region: str              # 语义描述: "kitchen area" / "unexplored corridor to the east"
    bias_position: Tuple[int, int]  # 地图坐标: 语义推断的目标大致位置
    reasoning: str                  # 搜索理由 (用于底层判断策略是否过时)
    explored_regions: List[str]     # 已探索区域 (累积)
```

**LLM prompt 设计原则**:
- 不给 frontier 列表（空间决策由 FMM 做）
- 给 room 结构 + 物体列表 + 空间关系（语义决策）
- 要求输出"目标最可能在哪个区域"及推理

**bias_position 的生成**:
- 如果 planner 判断目标在某个已知 room 中 → 取该 room 内 group 的中心坐标
- 如果 planner 判断目标在未探索区域 → 取对应方向的地图边界坐标
- 本质上是给 `get_goal()` 提供一个语义 anchor，让 FMM 在其附近选 frontier

**与 baseline `insert_goal()` 的对应关系**:
```
Baseline insert_goal():
  1. room_predict (1 LLM)  → 预测房间
  2. graph_corr × N (4N LLM) → 对每个 group 评分
  → mid_term_goal = 最高分 group 的 center

SmoothNav 高层 Planner:
  1. 综合 prompt (1 LLM) → 直接输出 target_region + bias_position
  → bias_position ≈ mid_term_goal 的作用

区别: baseline 用 4N 次 sonnet 做 group 评分,
      SmoothNav 用 1 次 sonnet 直接给出结论
      (代价: 可能没有逐 group 评分精确, 但配合底层 ADJUST 可以修正)
```

### 3.2 底层 Agent

**模型**: haiku (轻量快速模型)

**触发条件 — 场景图实质变化**:

基于 baseline 的 `update_node()` 机制，"实质变化"的判定：
```python
# 在 update_node() 执行后检查
new_node_count = len([n for n in graph.nodes if n.is_new_node])
# is_new_node 在节点首次创建时为 True，在 update_edge() 中被设为 False

# 触发条件: 有新节点产生
has_new_nodes = (new_node_count > 0)
```

注意：baseline 中新节点需要物体被检测到 ≥ 3 次（`obj_min_detections=3`）且 3D 空间上与已有物体不匹配才会产生。所以新节点不会非常频繁，典型情况是 agent 移动到新区域后连续几步会产生一批新节点，然后在已知区域移动时不产生。

**输入**:
```python
{
    "strategy": current_strategy,        # 高层策略 (target_region, bias_position, reasoning)
    "new_nodes": [...],                  # 本次新增的场景图节点 (caption + position)
    "current_goal": (gx, gy),            # 当前 frontier 目标坐标
    "dist_to_goal": float,               # agent 到当前目标的 FMM 测地距离
    "agent_position": (ax, ay),          # agent 当前位置
    "total_nodes": int,                  # 场景图总节点数
}
```

**输出 — 四种 Action (互斥)**:

| Action | 含义 | 后续处理 |
|--------|------|---------|
| **CONTINUE** | 当前策略和目标仍然合理 | 不做任何事 |
| **ADJUST(new_bias)** | 在当前策略框架内微调方向 | `get_goal(new_bias)` 更新 frontier |
| **PREFETCH(reason)** | 预判即将需要新策略 | 提前触发高层，当前继续执行旧目标 |
| **ESCALATE(reason)** | 策略已过时，需要高层重规划 | 触发高层，当前继续执行旧目标 |

**约束**:
- ADJUST 的 new_bias 应基于新发现的物体位置做局部调整，不能完全改变搜索方向
- 底层不能自主切换到完全不同的区域
- PREFETCH 和 ESCALATE 不阻塞执行（agent 继续朝旧目标走）

**haiku prompt 设计原则**:
```
当前搜索策略: {strategy.reasoning}
目标区域: {strategy.target_region}
新发现的物体: {new_nodes 的 caption 列表}
距当前目标: {dist_to_goal} 格

判断:
1. 新发现的物体是否提供了关于目标位置的线索? → ADJUST
2. 新发现的物体是否表明当前策略方向错误? → ESCALATE
3. 距目标是否已经很近 (< 15格)? → PREFETCH
4. 以上都不是 → CONTINUE

输出 JSON: {action, reason, [new_bias]}
```

**PREFETCH 的提前量**:

不用固定阈值（如 "距离 < 15 格"），而是交给 haiku 综合判断。haiku 能看到 dist_to_goal，当它判断"快到了且当前区域看起来不像目标所在地"时自然会输出 PREFETCH。但作为兜底，如果 dist_to_goal < 10 格且底层未触发 PREFETCH，由规则强制触发 PREFETCH。

### 3.3 执行层

**完全复用 baseline**，不修改：

**组件 A — `get_goal(bias_position)`**:
- 从当前 full_map 提取 frontier map
- 过滤距 agent < 1.2 倍 map_resolution 的 frontier
- 用 FMM 计算 agent → 每个 frontier 的测地距离 → 距离评分 (0~10)
- 如果有 bias_position: 用 FMM 计算 bias → 每个 frontier 的距离 → 评分 (0~1)
- 综合评分，选最优 frontier
- **副作用**: 更新 `graph.frontier_locations_16`

注意：每次底层 ADJUST 或高层策略更新后都要调用 `get_goal()`，确保 frontier 数据始终新鲜。

**组件 B — FMM local planner** (每步):
- 根据 global_goals 做局部路径规划 → 离散动作

**组件 C — instance_discriminator** (每步, 不修改):
- 检测目标物体是否在视野中
- 检测到时通过 temp_goal → global_goal 机制接管导航

---

## 4. 与 Baseline 的时序对比

### 4.1 Baseline 时序

```
step 0-24:
  [explore() 阻塞: 14× sonnet] → frontier_A → [盲跑 FMM 24步]

step 12: 看到 stove, refrigerator → 无反应（没到25步）
step 24: 到达 frontier_A → 触发 explore()

step 25-49:
  [explore() 阻塞: 14× sonnet] → frontier_B → [盲跑 FMM 24步]
```

### 4.2 SmoothNav 时序

```
step 0:
  高层 Planner (1× sonnet): "目标可能在 kitchen, 向东探索"
  → Strategy_0, bias=(300,200)
  → get_goal(bias) → frontier_A

step 1-7:
  FMM 执行, 场景图无新节点 → 底层不触发

step 8:
  场景图新增 "dining_table", "chair"
  → 底层 Agent (1× haiku):
    "dining_table 和 chair 说明在 dining area, 与 kitchen 策略一致"
    → CONTINUE

step 14:
  场景图新增 "stove", "refrigerator"
  → 底层 Agent (1× haiku):
    "stove 和 refrigerator 说明 kitchen 在附近, 微调方向"
    → ADJUST(new_bias=(280,190))
  → get_goal(new_bias) → frontier_B (偏南)

step 20:
  dist_to_frontier_B = 8 格
  场景图新增 "bathroom_sink"
  → 底层 Agent (1× haiku):
    "bathroom_sink 说明走到了 bathroom 而非 kitchen, 快到 frontier 了"
    → PREFETCH("到达 frontier 后需要新策略, 当前区域是 bathroom")
  → 触发高层 Planner (异步, 1× sonnet)
  → agent 继续走向 frontier_B (不等)

step 25:
  到达 frontier_B
  → 高层结果已 ready: Strategy_1 = {target_region="west corridor", bias=(150,350)}
  → get_goal(new_bias) → frontier_C (无缝切换, 0 等待)
```

### 4.3 LLM 调用对比 (典型 100 步 episode)

| | Baseline | SmoothNav |
|---|---|---|
| 场景图构建 (update_edge) | ~50× sonnet | ~50× sonnet (不变) |
| 目标选择: 高层 | 4× explore() = ~56× sonnet | 4× planner = **4× sonnet** |
| 目标选择: 底层 | — | ~8× haiku |
| 总计 (目标选择部分) | **~56× sonnet** | **4× sonnet + 8× haiku** |

---

## 5. 关键设计决策

### 5.1 为什么用事件驱动而不是固定间隔

固定 N 步触发（无论 N=5 还是 N=25）和 baseline 是同一个思维模式。
在已知区域行走时，场景图无变化，底层调用纯属浪费。
在进入新区域时，可能连续几步都有新节点，需要密集响应。
事件驱动自然适配这种非均匀分布。

触发事件 = `update_node()` 后存在 `is_new_node=True` 的节点。

### 5.2 为什么底层不直接选 frontier

底层 haiku 的输入是文本（物体名称 + 策略描述），它没有能力理解地图坐标和可达性。
让 haiku 选 frontier index 和让 sonnet 选 frontier index 一样不合理。
正确分工: haiku 提供/调整语义 bias，FMM 做空间评分。

### 5.3 为什么 PREFETCH 而不是真异步

Habitat 是同步仿真器，没有真实的 wall-clock 并行。
"提前触发"的实现方式:
1. 底层在 step N 发出 PREFETCH → 立即调用高层 planner
2. 高层 planner 执行完毕，结果缓存
3. Agent 继续用旧目标执行 step N+1 ... N+K
4. 到达旧 frontier 时，新策略已 ready，无缝切换

在论文中表述为 "anticipatory planning"，不强调异步。
在真实机器人部署时，可以自然扩展为真异步。

### 5.4 get_goal() 每次都重新计算 frontier

这解决了当前实现的核心 bug: frontier_locations_16 过期。
每次 ADJUST 或策略更新后都调用 `get_goal()`，确保:
- frontier map 从当前 full_map 重新提取
- FMM 距离重新计算
- frontier_locations_16 重新赋值

### 5.5 instance_discriminator 不修改

Baseline 的 instance_discriminator 是独立于 explore() 的目标检测/锁定机制，每步运行。
它处理: 检测到目标 → temp_goal → 逼近 → feature match → global_goal → 导航到目标。
这个机制与我们的分层架构正交，无需修改。

---

## 6. 实现计划

### 6.1 需要新增/重写的文件

| 文件 | 内容 |
|------|------|
| `smoothnav/planner.py` | **重写**: 高层 Planner (Strategy 输出, 不再选 frontier) |
| `smoothnav/low_level_agent.py` | **新增**: 底层 Agent (haiku, 4种action) |
| `smoothnav/main.py` | **重写**: 分层主循环 |
| `smoothnav/smooth_executor.py` | **删除**: 不再需要独立的 monitor (底层 Agent 承担了监控职责) |

### 6.2 需要复用不修改的模块

| 模块 | 来源 |
|------|------|
| BEV_map | baseline (地图更新) |
| graph.update_scenegraph() | baseline (场景图构建) |
| graph.get_goal(bias) | baseline (FMM 空间评分) |
| agent.step() + instance_discriminator | baseline (执行 + 目标锁定) |
| metrics.py | 已实现的平滑性指标 |

### 6.3 实现顺序

```
Phase 1: 高层 Planner
  - 重写 planner.py: Strategy 数据结构 + sonnet prompt
  - 输出 bias_position → 验证 get_goal(bias) 能选出合理 frontier
  - 测试: 单独跑 planner + get_goal, 检查 frontier 选择质量

Phase 2: 底层 Agent
  - 新增 low_level_agent.py: haiku prompt + 4 action 解析
  - 实现事件驱动触发 (hook into update_node 的 is_new_node)
  - 测试: 在已有 Strategy 下, 检查 ADJUST/PREFETCH/ESCALATE 的触发时机

Phase 3: 主循环集成
  - 重写 main.py: 分层信息流
  - 实现 PREFETCH 的提前触发逻辑
  - 端到端测试: 跑 5 episode, 检查 SR/SPL + 底层/高层调用日志

Phase 4: 对比实验
  - Baseline vs SmoothNav 同 episode 对比
  - 关注: SR, SPL, 高层调用次数, 底层调用次数, 决策真空步数
```

---

## 7. 评估指标

### 7.1 核心指标

| 指标 | 含义 | 来源 |
|------|------|------|
| SR | 成功率 | Habitat 标准 |
| SPL | 路径效率 | Habitat 标准 |
| LLM_calls_high | 高层调用次数 | 新增 |
| LLM_calls_low | 底层调用次数 | 新增 |
| decision_vacuum | 连续无智能判断的最长步数 | 新增 |
| avg_response_delay | 从"需要新策略"到"新策略生效"的平均步数 | 新增 |

### 7.2 平滑性指标 (已实现)

| 指标 | 公式 | 含义 |
|------|------|------|
| σ_v | Var(step_distance) | 速度方差 |
| σ_ω | Var(heading_change) | 角速度方差 |
| J (jerk) | mean(\|a_{t+1} - a_t\|) | 加速度变化率 |
| PC (pause count) | 停顿→运动转换次数 | 停顿频率 |
| PDR (pause ratio) | 停顿步数/总步数 | 停顿占比 |

### 7.3 预期结果

| 指标 | Baseline | SmoothNav 预期 |
|------|----------|---------------|
| SR | 0.80 | ≥ 0.80 |
| SPL | 0.337 | ≥ 0.35 (消除盲跑应提升路径效率) |
| 目标选择 LLM (sonnet) | ~56/episode | ~4-6/episode |
| decision_vacuum | 25 步 | 0 步 (事件驱动填补) |
| avg_response_delay | 0 (阻塞等待) → 但浪费时间 | 0-5 步 (PREFETCH 提前准备) |
