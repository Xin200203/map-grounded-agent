# 文献调研：零样本VLN + 显式建图 + 平滑导航

> 调研日期：2026-03-08
> 项目代号：SmoothNav

---

## 一、领域全景

### 1.1 分类框架

当前零样本视觉语言导航方法可以按**地图表示**和**决策范式**两个维度划分：

|  | 隐式/无显式地图 | 显式建图 |
|---|---|---|
| **Reactive（逐步决策）** | NaVid, NavGPT | VLFM, ESC, OpenFMNav, CA-Nav |
| **Deliberative（预规划）** | DreamNav | SayPlan, SG-Nav, SpatialNav |

**我们的定位**：显式建图 + Deliberative，但区别于 SpatialNav（不需要预探索）和 SG-Nav（做 VLN 而非仅 ObjectNav），并首次将**导航平滑性**作为核心优化目标。

### 1.2 关键趋势

1. **地图表示从 dense 向 structured graph 演进**：VLMaps(dense feature map) → HOV-SG(hierarchical graph) → SG-Nav(online scene graph)
2. **LLM 从"工具"变"大脑"**：ESC 只用 LLM 做 frontier scoring → SG-Nav 用 CoT 做层次推理 → SpatialNav 用全局 SSG 做空间推理
3. **从离线到在线**：ConceptGraphs/HOV-SG 离线建图 → SG-Nav 在线增量 → 但尚无在线增量 + VLN 的结合
4. **从离散到连续环境**：R2R(离散图) → R2R-CE(连续控制) → 真实机器人部署
5. **Trajectory-level planning 萌芽**：DreamNav 首次提出轨迹级规划替代点级决策

---

## 二、核心论文深度分析

### 2.1 SG-Nav (NeurIPS 2024)

**论文**: "SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation"
**作者**: Hang Yin, Xiuwei Xu, Zhenyu Wu, Jie Zhou, Jiwen Lu (清华)
**链接**: https://arxiv.org/abs/2410.08189

#### Pipeline

1. **在线 3D Scene Graph 构建**：
   - **Object 节点**：基于 ConceptGraphs 的在线开放词汇 3D 实例分割（2D VLM 逐帧 + 跨帧融合）
   - **Group 节点**：语义相关物体聚类（用预计算的 LLM 字典定义相关性，如 Bed-Nightstand）
   - **Room 节点**：用开放词汇分割模型查询 room 类别
   - **边**：物体间空间关系（next to, above, opposite to 等）
     - 短距离边：VLM (LLaVA-1.6) 验证共视物体关系
     - 长距离边：几何约束过滤（无遮挡 + 同房间）
   - **效率优化**：批量 LLM prompt 一次处理所有新边，复杂度从 O(m(m+n)) 降到 O(m)

2. **层次化 CoT 推理**：
   - 将场景图拆分为以物体为中心的子图
   - 4 步 CoT：预测距离 → 生成问题 → 用子图回答 → 最终距离估计
   - Frontier 评分：$P_{fro}(i) = \sum_j P_{sub}(j) / D_{ij}$

3. **Re-perception 机制**：
   - 检测到目标后不立即停止，而是多视角验证
   - 累积可信度分数超阈值(0.8)才停止，否则放弃该检测继续探索

4. **导航决策**：**Reactive 逐步决策** — 每步更新图 → 重评 frontier → Fast Marching 执行

#### 定量结果

| Benchmark | SR | SPL |
|---|---|---|
| MP3D | 40.2 | 16.0 |
| HM3D | 54.0 | 24.9 |
| RoboTHOR | 47.5 | 24.0 |

超越所有零样本方法 3-14% SR，在 MP3D 上甚至超越监督方法。但 **SPL 提升远小于 SR**。

#### 关键不足

1. **仅做 ObjectNav，不做 VLN** — 无法处理自然语言路径指令
2. **仍然是 reactive 逐步决策** — 每步都需要 LLM CoT 推理，无法 smooth 执行
3. **SPL 差** — re-perception 机制增加路径长度（接近假阳性验证），导致效率低
4. **感知瓶颈** — 依赖非端到端的 2D VLM 逐帧分割 + 跨帧融合，非 3D-aware
5. **Group 节点依赖预定义字典** — 不自适应
6. **GPT-4 vs LLaMA-7B 几乎无差异（0.1% SR）** — 说明瓶颈在感知而非推理

---

### 2.2 SpatialNav (arXiv 2026.01)

**论文**: "SpatialNav: Leveraging Spatial Scene Graphs for Zero-Shot Vision-and-Language Navigation"
**作者**: Jiwen Zhang, Zejun Li, Siyuan Wang, Xiangyu Shi, Zhongyu Wei, Qi Wu
**链接**: https://arxiv.org/abs/2601.06806

#### Pipeline

1. **Spatial Scene Graph (SSG) 构建**（离线，需预探索）：
   - Floor 分割：高度直方图 + DBSCAN
   - Room 分割：几何启发式方法，**>20㎡的区域需手动修正**
   - Room 分类：收集区域内图像，用 **GPT-5** 分类
   - Object 检测：微调 **SpatialLM** 在 Matterport3D 上预测 bbox + 标签

2. **Agent-centric Spatial Map**：
   - 以 agent 位置为中心，7.68m 半径内的 room 投影到 top-down map
   - Agent 朝向始终对齐向上

3. **Compass-like Visual Observation**：
   - 全景离散为 8 个方向视角 → 组合成 3×3 网格图像（中心为指南针）
   - ~640 visual tokens（vs 顺序 8 图 ~1700+ tokens）

4. **Remote Object Localization**：
   - 查询 SSG 获取候选导航点同房间内的物体信息
   - 提供"预见"能力：agent 知道移到某处会看到什么

#### 定量结果

| Benchmark | SR | SPL |
|---|---|---|
| R2R val-unseen (discrete) | 57.7 | 47.8 |
| R2R-CE val-unseen (continuous) | 64.0 | 51.1 |
| RxR-CE val-unseen | 32.4 | 24.6 |

零样本 SOTA，+9.3% SR / +11.7% SPL vs SpatialGPT。连续环境也大幅领先。

#### 关键不足

1. **需要预先完整探索环境** — 这是最大限制。假设 3D 点云已可用，不评估 SLAM 鲁棒性
2. **Room 分割需手动修正** — 开放空间自动分割不可靠
3. **SSG 是静态的** — 构建一次，无法处理环境变化
4. **Object 检测需要微调 SpatialLM** — 不是完全零样本（检测器在 MP3D 训练集上训练）
5. **使用 GPT-5/5.1** — 推理成本高，per-step latency 未报告
6. **Ground-truth 标注始终优于预测** — 空间标注质量是瓶颈
7. **不讨论导航平滑性** — 完全不关注执行层面的连续性

---

### 2.3 DreamNav (arXiv 2025.09)

**论文**: "DreamNav: A Trajectory-Based Imaginative Framework for Zero-Shot Vision-and-Language Navigation"
**作者**: Yunheng Wang et al.
**链接**: https://arxiv.org/abs/2509.11197

#### Pipeline

1. **EgoView Corrector**：
   - Macro-Adjust Expert (GPT-4o)：初始方向校准
   - Micro-Adjust Controller：每步后微调视角

2. **Trajectory Predictor**：
   - 基于扩散策略模型生成 24 个未来 waypoint
   - Farthest-first 算法选择最不相似的候选轨迹
   - **轨迹级规划而非点级决策** — 核心创新

3. **Imagination Predictor**：
   - Dream Walker：沿候选轨迹做视觉 rollout
   - Narration Expert (Qwen-VL-Max)：将视觉 rollout 转为文本描述
   - 提供长视野预见能力

4. **Navigation Manager**：
   - Navigator (GPT-4o)：选择最佳轨迹
   - Execution Expert (GPT-4o)：监控执行进度

#### 定量结果

| Benchmark | SR | SPL |
|---|---|---|
| R2R-CE val-unseen | 32.79 | 28.95 |
| Real-world (20 trials) | 60% | - |

零样本 egocentric SOTA，超越 CA-Nav +7.5% SR。真实世界 12/20 成功。

#### 重要设计细节

- **Trajectory Generator** 使用**扩散策略模型**（首次应用于零样本 VLN）生成 24 个未来 waypoint
- **Imagination Predictor** 使用**可控视频生成模型（Sora-like）**做 rollout + 视频理解模型解读
- 想象输出转为**结构化文本描述**（非原始视觉）以控制 API token 成本
- 最优候选轨迹数 CTN=4（farthest-first 选择），imagination rollout 最优长度 IRL=18 步
- EgoView Corrector 两阶段：MAE 粗校准（+10% SR）+ MAC 精校准（+6% SR）

#### 关键不足

1. **无显式地图** — 完全靠"想象"补偿空间理解，缺乏几何约束；目标丢失后无法回溯
2. **NE 偏高（7.06m）** — 纯 egocentric 输入，目标丢失后难以重定向，累积漂移
3. **重度依赖 API**（GPT-4o + Qwen-VL + 视频生成模型）— 成本高，延迟未报告
4. **未量化 smoothness** — 虽然轨迹级规划天然更 smooth，但论文未定义或测量 smoothness 指标
5. **扩散模型需要预训练** — Trajectory Predictor 不是完全零样本
6. **视频生成模型的幻觉风险** — imagination 质量依赖生成模型对室内场景的准确性

---

### 2.4 CA-Nav (arXiv 2024.12)

**论文**: "Constraint-Aware Zero-Shot Vision-Language Navigation in Continuous Environments"
**作者**: Zhangze An et al.
**链接**: https://arxiv.org/abs/2412.10137

#### Pipeline

1. **CSM (Constraint-Aware Sub-instruction Manager)**：
   - LLM 将指令分解为子指令，每个子指令绑定约束类型：
     - Object 约束：Grounding DINO 检测，5m 内满足
     - Location 约束：BLIP2 VQA 场景识别
     - Direction 约束：里程计数据对比位姿
   - 最小/最大步数阈值（10/25步）防止过早/过晚切换

2. **CVM (Constraint-Aware Value Mapper)**：
   - BLIP2 计算观察与约束的余弦相似度
   - 置信度掩码 + 历史衰减因子(γ=0.5) + 轨迹掩码(λ=0.95)
   - 超像素聚类(SLIC)选择高值区域中心作为 waypoint
   - Fast Marching Method 生成低层动作

#### 定量结果

| Benchmark | SR | SPL |
|---|---|---|
| R2R-CE val-unseen | 25.3 | 10.8 |
| RxR-CE val-unseen | 19.0 | 5.0 |
| Real-world (80 trials) | 40-80% | - |

在当时（2024.12）是零样本 VLN-CE SOTA。

#### 重要设计细节

- **LLM 仅在开头调用一次**做指令分解（GPT-4），之后 per-step 只跑 BLIP2 相似度计算（轻量）
- 因此 CA-Nav 实际上是**"deliberative 分解 + reactive 执行"的混合范式**
- 效率极高：0.45s/step（vs NavGPT 1.29s/step），$0.04/episode（vs $0.85/episode）

#### 关键不足

1. **执行仍是 reactive** — 虽然 LLM 只调一次，但 per-step 仍需 BLIP2 + Grounding DINO + value map 更新
2. **SPL 极低（10.8）** — egocentric 视角导致大量冗余探索，路径效率差
3. **Value map 缺乏结构化语义** — 只有像素级相似度，没有物体/房间级理解
4. **LLM 指令分解会出错** — 遗漏约束导致跳过关键子指令；5+ 子指令时 SR 显著下降
5. **无 re-planning 能力** — 固定子指令队列，无法根据探索中发现的新信息调整计划
6. **离散动作空间（0.25m/30°）** — 非连续速度控制，本质上不 smooth
7. **超像素选点仍是局部贪心** — 高值区域不一定是全局最优方向

#### 亮点（值得借鉴）

- **超像素选点**比 frontier 点更稳定，避免 value 突变导致的抖动
- **约束感知的子指令分解**思路可以复用到我们的 re-plan trigger 设计
- **"LLM 只调一次"思路**证明了可以大幅降低推理成本
- **真实机器人部署经验**（QiZhi robot + Kinect V2 + Hector SLAM）可参考

---

## 三、其他相关工作简要

### 3.1 显式建图类

| 论文 | 会议 | 地图类型 | 关键贡献 | 局限 |
|---|---|---|---|---|
| ConceptGraphs | ICRA 2024 | Object-centric Scene Graph | SAM+CLIP+LLaVA+GPT-4 开放词汇 3D 场景图 | 建图重（多次 LLM 调用），非实时 |
| HOV-SG | RSS 2024 | 层次化 OV Scene Graph (floor→room→object) | 比 dense map 小 75%，open-vocab 精度高 | 主要做 scene understanding |
| SayPlan | CoRL 2023 | 3D Scene Graph (预建) | 层次搜索 + 经典路径规划 + 迭代 re-planning | 依赖预建静态图，环境变化需重建 |
| VLMaps | CoRL 2023 | Dense VL Feature Map | 开放词汇空间索引的先驱 | Dense，不可扩展，无结构 |

### 3.2 Frontier-based 零样本导航

| 论文 | 会议 | 关键贡献 | 局限 |
|---|---|---|---|
| VLFM | ICRA 2024 | VLM 对 frontier 打分，零样本 ObjectNav SOTA | 仅 ObjectNav，无语义结构 |
| ESC | ICML 2023 | LLM 常识约束选 frontier | 视觉→文本瓶颈 |
| OpenFMNav | NAACL 2024 | LLM 提取目标 + VLM 检测 + VSSM | 仍是 frontier scoring |
| VoroNav | ICML 2024 | Voronoi 图提取拓扑路径 + LLM 选 waypoint | 有拓扑但非 scene graph |

### 3.3 端到端 VLA

| 论文 | 会议 | 关键贡献 | 局限 |
|---|---|---|---|
| NaVid | RSS 2024 | 视频 VLM 直接输出动作，无需地图 | 需 550k 训练数据，每步推理延迟大 |
| Uni-NaVid | RSS 2025 | 统一多种导航任务的 VLA | 同 NaVid，VLA 范式固有瓶颈 |
| NavGPT | 2023 | 首个 GPT-4 零样本 VLN | 延迟极高(1.29s/step)，不可部署 |

### 3.4 2025-2026 最新工作

| 论文 | 时间 | 关键创新 |
|---|---|---|
| UniGoal (SG-Nav 后续) | CVPR 2025 | 统一不同目标导航任务 |
| VLN-Zero | 2025.09 | 快速探索 + cache + neurosymbolic 规划 |
| Fly0 | 2025 | 解耦语义 grounding 与几何规划 |
| OpenFrontier | 2025 | VLM-grounded frontier 探索 |
| MSNav | 2025 | 动态记忆 + 全景零样本 VLN |

---

## 四、Gap 分析与精准定位

### 4.1 已发现的三个核心 Gap

#### Gap 1: 在线增量 Scene Graph + VLN 的结合空白

| 方法 | 在线建图？ | 做 VLN？ | 二者兼备？ |
|---|---|---|---|
| SG-Nav | ✅ 在线增量 | ❌ 仅 ObjectNav | ❌ |
| SpatialNav | ❌ 需预探索 | ✅ VLN | ❌ |
| DreamNav | ❌ 无显式地图 | ✅ VLN | ❌ |
| CA-Nav | ❌ 仅 value map | ✅ VLN | ❌ |
| **Ours** | **✅ 在线增量** | **✅ VLN** | **✅** |

**没有任何已有工作同时实现：在线增量建 Scene Graph + 零样本 VLN。**

#### Gap 2: 导航平滑性从未被系统研究

- SG-Nav：SPL 差，re-perception 增加路径长度，未讨论 smoothness
- SpatialNav：完全不关注执行层面
- DreamNav：提了 trajectory-level 但没量化 smoothness
- CA-Nav：提到超像素比 frontier 更稳定，但仅定性描述
- **没有任何工作定义、量化或优化导航的执行平滑性（速度方差、jerk、停顿次数等）**

#### Gap 3: 规划与执行的解耦不彻底

| 方法 | 决策频率 | 执行方式 | 问题 |
|---|---|---|---|
| SG-Nav | 每步 LLM CoT | Fast Marching | 每步都等 LLM |
| SpatialNav | 每步 MLLM | 选 nav point | 每步都等 MLLM |
| CA-Nav | 每步评估约束 | Fast Marching | 每步都跑 VLM |
| DreamNav | 轨迹级但仍需 GPT-4o | 扩散策略 | 多次 GPT-4o 调用 |
| **Ours** | 仅在关键节点 re-plan | 连续执行预规划路径 | ✅ 解耦 |

### 4.2 我们的精准定位

```
┌─────────────────────────────────────────────────────────────┐
│              SmoothNav: 核心差异化                            │
│                                                             │
│  vs SG-Nav:                                                 │
│    - SG-Nav 仅做 ObjectNav → 我们做 VLN（自然语言指令）        │
│    - SG-Nav 每步 LLM → 我们预规划 + 异步执行                  │
│    - SG-Nav SPL 差 → 我们优化路径效率和平滑性                  │
│                                                             │
│  vs SpatialNav:                                             │
│    - SpatialNav 需预探索全环境 → 我们在线增量建图               │
│    - SpatialNav SSG 是静态的 → 我们动态更新                   │
│    - SpatialNav room 分割需手动修正 → 我们自动化               │
│                                                             │
│  vs DreamNav:                                               │
│    - DreamNav 无显式地图 → 我们有 Scene Graph 支撑            │
│    - DreamNav 目标丢失后难恢复 → 我们有地图可回溯              │
│    - DreamNav 扩散模型需预训练 → 我们完全零样本                │
│                                                             │
│  vs CA-Nav:                                                 │
│    - CA-Nav value map 无结构 → 我们有层次化 Scene Graph       │
│    - CA-Nav SPL 极低(10.8) → 我们预规划提升路径效率            │
│    - CA-Nav 仍是逐步决策 → 我们轨迹级规划                     │
│                                                             │
│  独有贡献:                                                   │
│    ★ 首次将导航平滑性作为核心指标定义和优化                     │
│    ★ 首个在线增量 Scene Graph + 零样本 VLN 系统               │
│    ★ Plan-then-Execute 解耦：大模型仅在关键节点调用            │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 需要进一步明确的技术问题

1. **Smoothness Metric 如何定义？**
   - 候选：速度方差 (σ_v)、角速度方差 (σ_ω)、jerk (da/dt)、停顿次数、停顿总时长
   - 需要在 VLN-CE 仿真中可计算

2. **在线 Scene Graph 的构建效率？**
   - SG-Nav 的批量 LLM prompt 可借鉴
   - 是否可以用轻量本地 VLM 替代 GPT-4？

3. **何时触发 Re-plan？**
   - 检测到未见过的大面积区域？
   - 子指令约束超时未满足？
   - 感知到与预期不符的物体布局？

4. **VLN-CE vs 离散 R2R？**
   - 连续环境更能体现 smoothness 优势
   - 但实验难度更大

---

## 五、建议精读优先级

| 优先级 | 论文 | 原因 |
|---|---|---|
| P0 | SG-Nav | 在线 scene graph 建图的直接工程参考 |
| P0 | SpatialNav | scene graph + VLN 的最新 SOTA，理解 SSG 设计 |
| P1 | DreamNav | trajectory-level planning 的思路借鉴 |
| P1 | CA-Nav | 约束感知子指令分解 + value map + 真实部署参考 |
| P2 | ConceptGraphs | 建图 pipeline 工程参考 |
| P2 | HOV-SG | 层次化场景图结构设计 |
| P3 | VLFM | frontier-based 方法的 baseline 理解 |
| P3 | VoroNav | Voronoi 拓扑路径规划的灵感 |

---

## 六、参考文献

1. SG-Nav: https://arxiv.org/abs/2410.08189
2. SpatialNav: https://arxiv.org/abs/2601.06806
3. DreamNav: https://arxiv.org/abs/2509.11197
4. CA-Nav: https://arxiv.org/abs/2412.10137
5. ConceptGraphs: https://arxiv.org/abs/2309.16650
6. HOV-SG: https://github.com/hovsg/HOV-SG (RSS 2024)
7. SayPlan: https://arxiv.org/abs/2307.06135 (CoRL 2023)
8. VLFM: https://arxiv.org/abs/2312.03275 (ICRA 2024)
9. ESC: ICML 2023
10. OpenFMNav: https://arxiv.org/abs/2402.10670 (NAACL 2024)
11. VoroNav: https://arxiv.org/abs/2401.02695 (ICML 2024)
12. NaVid: https://arxiv.org/abs/2402.15852 (RSS 2024)
13. Uni-NaVid: https://arxiv.org/abs/2412.06224 (RSS 2025)
14. NavGPT: https://arxiv.org/abs/2305.16986 (2023)
15. VLMaps: CoRL 2023
16. VLN Survey (TMLR 2024): https://github.com/zhangyuejoslin/VLN-Survey-with-Foundation-Models
17. UniGoal: CVPR 2025 (SG-Nav follow-up)
18. VLN-Zero: https://arxiv.org/abs/2509.18592
19. AO-Planner: https://chen-judge.github.io/AO-Planner/
20. Fly0: https://arxiv.org/abs/2602.15875
