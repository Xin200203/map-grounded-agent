# SmoothNav ICRA 2027 论文故事重设计与实验矩阵

日期：2026-07-05
角色：投稿负责人工作文档（story 定版 + 实验设计 + gate + 时间倒排）
状态：v1，随实验结果滚动修订

---

## 0. 投稿目标与约束

- 目标会议：ICRA 2027（首尔，2027-05-24~28）。CFP 尚未公布，按历年惯例假定截稿 **2026-09-15** 前后；内部硬截止定为 **2026-09-08**（留一周缓冲）。
- 页数：6 页正文 + 引用。系统类论文，评审关注：清晰的机制贡献、matched 消融、诚实的失败分析。
- 可用算力：73 服务器 4×24G GPU（当前全空闲）；LLM 经 Sonnet 4.5（主）/ DashScope deepseek-v4-pro（省预算探针）。
- 关键事实约束（2026-07-05 审查结论）：
  - intact-scene（dev5/dev15）SR 有优势；cross-scene SR 持平、**SPL 大幅占优**（0.284 vs 0.101）。
  - monitor 无独立增益（full 与 no-monitor 并列）。
  - 失败根因收敛于「目标不可见时上游语义证据稀疏」，非 controller bug。
  - BEV/MLLM branch planner 代码闭环但在线未验证（E0 决定去留）。

## 1. 故事重设计

### 1.1 放弃的故事

- ~~“事件驱动重规划全面超越 periodic baseline（含跨场景 SR）”~~ —— 跨场景 SR 无证据，且赌注押在 10 周内翻转 hard-case 感知问题上，风险不可接受。
- ~~“LLM monitor 带来收益”~~ —— 消融不支持；monitor 降级为安全升级器（escalation-only），不进贡献列表。

### 1.2 选定的故事：证据分级的语义权限（Evidence-Gated Semantic Authority）

**一句话主张**：在零样本目标导航中，LLM/MLLM 的语义指导只有在其**控制权限与地图证据等级挂钩**时才可靠地帮助导航；不设门控的语义先验会劣化为幻觉驱动的探索。SmoothNav 把这一原则实现为一个覆盖三层的统一门控体系，在冻结的 UniGoal 执行器之上取得**一致的效率提升（SPL/路径/平滑度）与证据充分时的成功率提升**，并保证证据不足时优雅退化为几何探索（非劣于 baseline）。

这个故事的关键优势：**核心主张的验证不依赖跨场景 SR 翻转**。主对照是「有门控 vs 无门控」的内部对比 + 对 baseline 的非劣性/效率优势，这两者当前证据已部分支持，剩余部分可以用 config 级消融在 2-3 周内补齐。跨场景 SR 若翻转是锦上添花，不翻转不毁故事。

### 1.3 创新点表述（Contribution bullets 草案）

1. **分层事件驱动重规划控制器**（frozen backbone 之上）：把周期式语义推理改造为证据事件触发（新目标候选/新房间/frontier 到达/卡死/落地失败/出局部窗口），每类失败有显式 failure code 与恢复路径（局部窗口重投影、锚点停滞退委、落地 noop 重规划）。
2. **证据分级语义权限（graded semantic authority）**：统一的门控原则贯穿三层——(a) planner 只能从可解析菜单选择，object 选项经目标相关性阈值过滤；(b) 落地层的直接对象执行需相关性 + 局部可投影双重门控，否则降级为 frontier 偏置或 `unexplored target:` 搜索锚点（配 PONI-lite 目标进度价值项保持锚定存活）；(c) MLLM frontier 先验按证据等级（direct/anchor/room_prior/geometry_only）缩放权重，geometry_only 强制归零，且只能输出 branch ID 不能输出坐标。语义假设**偏置**地图价值，永不**绕过**它。
3. **matched-backbone 评估 + 可回放归因**：同一 backbone 上 7 种 controller profile 消融 + 逐 episode 配对分析 + trace 合约审计，把收益分解到机制（效率来自事件驱动重规划；成功率来自证据充分时的目标锚定；门控保证证据不足时非劣）。

### 1.4 叙事分叉预案（按 gate 结果降级）

- **Plan A**（G2 过）：如上，gating 为第一贡献。
- **Plan B**（G2 不过，即无门控变体并不更差）：降级为「高效可诊断的语义重规划系统」——主打 SPL/路径/平滑度效率 + intact SR + 失败归因体系；gating 改为设计原则而非实验主张。
- **Plan C**（E1 连 SPL 优势都不稳）：收缩为 intact-scene 系统贡献 + 机制诊断，同时评估改投 RA-L / IROS 2027；此为最坏预案。
- BEV/MLLM 支线：E0 通过 → 保留为第二贡献的 (c) 项 + 一个小节实验；不通过 → 移入 future work，正文只保留 LLM 层门控。

## 2. 实验矩阵

### 2.0 套件定义（已定版，2026-07-05）

- **intact-15**：episodes **286–300**（scene MHPLjHsuG27，与 4 月 s3_dev15 完全一致，保证历史可比）。
- **cross-12**：4 月 post-repair loadability audit 全部 12 个通过项，**12 个 episode 分属 12 个不同场景**、5 类目标（比原计划 5 场景×3 集的场景多样性更强）：

| ep | 类目 | 场景 | 备注 |
|---|---|---|---|
| 228 | tv_monitor | 00862-LT9Jq6dN3Ea | 历史 hard case |
| 527 | chair | 00821-eF36g7L6Z9M | 历史 hard case |
| 661 | chair | 00814-p53SfW6mjZe | positive anchor（须保护） |
| 64 | chair | 00823-7MXmsvcQjpJ | |
| 159 | sofa | 00824-Dd4bFSTQ8gi | |
| 358 | tv_monitor | 00871-VBzV5z6i1WS | |
| 486 | chair | 00891-cvZr5TUy5C5 | |
| 574 | plant | 00815-h1zeeAwLh9Z | |
| 717 | plant | 00844-q5QZSEeHe5g | |
| 778 | plant | 00813-svBbv1Pavdk | |
| 859 | plant | 00831-yr17PDCnDDW | |
| 955 | bed | 00839-zt1RVoi7PcG | |

- 所有 profile 跑**同一 episode 集**（matched pairing），报告逐 episode win/loss/tie + 聚合 SR/SPL。
- 统一运行配置：controller trace 开、`grounding_snapshot_policy=target`；LLM 通道见 2.0.1。

### 2.0.1 LLM 通道（2026-07-06 已定版：官方 DeepSeek API）

- 排查结论：Clauddy 两侧全灭（远端 503 组策略 / 本地 401 失效）；用户提供的 key 实测为**官方 DeepSeek API key**（DashScope 拒绝、api.deepseek.com 通过）。
- **定版通道**：`https://api.deepseek.com/v1`，模型 `deepseek-chat`，配置 `config_habitat_deepseek_official.yaml`，key 部署在 73:`.local/deepseek.env.sh`（经 clauddy.env.sh source）。
- **已知限制**：官方 deepseek-chat 拒绝图像输入 → graph 视觉 relation 判别失败（边无 relation 标签）。影响面：planner 场景文本少一段 SPATIAL RELATIONSHIPS；**全 profile 同等受影响，matched 对照内部公平**。E3e（MLLM 分支消融）需要独立视觉通道，保持条件项——若后续找到 4 月的 DashScope key（多模态 deepseek-v4-pro）即可解锁。
- 纪律：E1/E2/E3 全部走本通道本模型，不中途切换。

### 2.0.2 E1 执行记录

- **2026-07-06 00:07 已点火**：cross-12 × {baseline-explore, baseline-periodic, smoothnav-full} → GPU 0/1/2，`results/e1_main_cross12_20260706/`；00:10 intact-15 × 同三 profile → GPU 4/5/6，`results/e1_main_intact15_20260706/`；E0b（ep228 真 planner 探针）→ GPU 3。服务器实有 8 卡，7 卡在用。
- 首报健康信号：planner 空响应文件数 = 0。

### E0：决定性在线验证（阻塞 BEV 支线，第 1 优先；已拆分）

- **E0a（感知侧，LLM 无关，2026-07-05 已启动）**：新代码在 ep228 跑 1 个在线 episode，capsule/snapshot policy=target，`SMOOTHNAV_LLM_MAX_RETRIES=0` 快速失败（planner 走确定性 fallback，不影响 footprint 判据）。判据：capsule 中 `semantic_instance_footprints` 非零、`object_footprint_pixel_count>0`。运行目录 `results/e0_footprint_probe_20260705/`。
- **E0b（planner 通道健康，依赖 2.0.1 通道决策）**：通道恢复后重跑 ep228 完整探针，判据：planner 空响应率 <10%、target branch covered。
- 分叉：E0a 过 → E3e 纳入矩阵；不过 → BEV/MLLM 移 future work，问题记录为上游感知稀疏（引用诊断，不再投入修复）。

### E1：主表（3 profile × 2 套件 = 90 episodes）

| Profile | 说明 |
|---|---|
| baseline-explore | UniGoal 风格周期 explore（共享 backbone 修复） |
| baseline-periodic | 周期式语义规划 |
| smoothnav-full | 完整系统 |

- 指标：SR / SPL / path length / smoothness（σ_v, jerk, pauses, reversals，代码已产出）/ planner+monitor 调用数 / terminal outcome 分布。
- **G1 gate**：full 对 baseline-periodic 在两套件 SPL 均有优势（配对多数 win）且 SR 非劣 → 故事成立，进入 E2/E3。SR 出现劣化 → 先归因再前进。

### E2：机制消融（4 profile × cross-15）

no-monitor / rules-only / fixed-interval / no-prefetch —— 把 E1 的 full 收益分解到事件驱动、monitor、prefetch。预期结论（按已有小样本）：收益主要来自事件驱动 + grounding 恢复，monitor 中性——照实写。

### E3：门控消融（故事的核心实验，cross-15，全部 config 级）

| 变体 | 开关 | 验证的门控 |
|---|---|---|
| E3a gate-off:relevance | `graph_text_goal_direct_relevance_threshold: 0.0` | planner 菜单/对象锚相关性门 |
| E3b gate-off:direct-exec | `graph_enable_direct_object_goal: true` + `graph_text_goal_use_frontier_anchor: false` | 直接对象执行的双重门 |
| E3c gate-off:target-progress | `graph_target_progress_weight: 0.0` | PONI-lite 锚定存活项 |
| E3d gate-off:stall-decommit | `controller_target_anchor_stall_patience_updates: 9999` | 锚点停滞退委 |
| E3e gate-off:mllm-prior（条件于 E0） | 需加一个小开关强制 evidence_gate_strength=1.0；对照 mllm-on-gated / mllm-off | MLLM 证据分级门 |

- **G2 gate**：≥2 个 gate-off 变体在 SR 或 SPL 上可测地劣化（配对多数 loss 或失败模式指标恶化：out_of_local_window 次数、frontier 重复率、grounding noop 率）→ gating 主张成立。不足 2 个 → 切 Plan B。

### E4：效率/平滑度分析（无新跑）

从 E1/E2 trace 汇总：SPL 分解（成功 episode 的路径比）、smoothness 指标、重规划次数 vs 效率散点。SmoothNav 之名由此坐实。

### E5：失败归因表（无新跑）

`audit_trace_contracts.py` 14 层归因 + terminal outcome 分布，产出论文的 failure taxonomy 表和 1 个 hard-case 定性图（capsule BEV 渲染）。

### 规模与预算估计

- E1+E2+E3 ≈ 90 + 60 + 60~75 ≈ **210~225 episodes**；单 episode 10~20 min（GPU + LLM 延迟），4 GPU 并行 → 每 GPU 约 55 集 ≈ 12~18 h 纯运行；含排障按 **2~3 周**排。
- LLM 成本主要在 Sonnet 4.5 planner 调用（预算约 8 calls/百步 × ~500 步 × 225 集）；E3 变体可考虑 DashScope 通道降本，但**主表 E1 必须与消融同模型**，默认全 Sonnet 4.5。

## 3. 与 baseline 公平性的处理（写作决策）

backbone 共享改动（frontier 价值评分、关系裁剪、文本可见接管抑制）对**所有 profile 生效**，包括 baseline。写作口径：
1. 明确声明 baseline = “UniGoal backbone + 共享基础设施修复”，非上游原版；上游原版数字引用 UniGoal 论文作参照行。
2. 论证方向对我们有利：共享改进**增强了 baseline**，在其上取得的 delta 只能来自 controller 层差异，属保守比较。
3. 若审稿风险仍在意：备选补一组 `--legacy-shared-path` 开关实验（时间允许才做，优先级最低）。

## 4. 时间倒排（内部截止 2026-09-08）

| 周 | 日期 | 里程碑 |
|---|---|---|
| W1 | 07/06–07/12 | 远端环境恢复 + E0 + 套件定版 + E1 启动 |
| W2 | 07/13–07/19 | E1 完成 → **G1 决策**；E2/E3 启动 |
| W3 | 07/20–07/26 | E2/E3 完成 → **G2 决策（故事定版，Plan A/B 二选一）** |
| W4 | 07/27–08/02 | 补测（规模扩充/E3e/legacy 开关按需）；方法节初稿 |
| W5 | 08/03–08/09 | **机制冻结**（此后只测量和写作）；实验节 + 图表 |
| W6 | 08/10–08/16 | 全文初稿（intro/related/method/exp/limitation） |
| W7 | 08/17–08/23 | 内审 + 补图 + 数字复核（对照 suite_summary 溯源） |
| W8 | 08/24–08/30 | 打磨 + 外部意见 |
| W9 | 08/31–09/08 | 缓冲 + 提交 |

## 5. 执行纪律

- 每个 suite 完成即跑 `summarize_explicit_suite.py` 聚合并把 Observation/Hypothesis/Claim 分级结论写回本文档修订记录；聚合可复核后允许清理远端大体积中间产物（保留 summary/trace）。
- 遵守 AGENTS.md 证据层级：单 episode 不作 claim；样本不齐标 partial。
- 所有远端 run 用显式 episode 套件（不用 num_eval 隐式切片），运行目录名带日期 + 代号（e1_main_20260707 等）。
- 代码改动随做随提交推送，远端只 pull 不手改。

## 6. 修订记录

- 2026-07-05 v1：初版（故事定版为 evidence-gated semantic authority；E0–E5 矩阵；G1/G2 gate；Plan A/B/C 分叉）。
- 2026-07-06 v2.1：**E0b 判定通过**（Observation，run `results/e0b_planner_channel_probe_20260706/`，ep228，官方 DeepSeek 通道）：20 次 planner 调用 **0 空响应**（4 月 Clauddy 为 13/16）、fallback 2/20、planner 主动选中目标 tv 2 次、23 个 capsule 覆盖 target 分支（painting→tv 锚定切换序列，是 gating 分析的好样本）；episode 本身仍失败（ep228 感知受限 hard case，符合预期）。晚期 capsule 语义基座继续增长（true 1546px / obj_fp 1213px）。**跟进项确认**：`semantic_instance_footprints` 在 E0a/E0b 全部 capsule 中恒为 0，`_project_semantic_instance_footprints` 触发条件待查（非阻塞；graph pcd footprint 是可用的可信来源）。E0 整体关闭，E3e 保持条件项（视觉通道）。
- 2026-07-06 v2：**E0a 判定通过**（Observation，run `results/e0_footprint_probe_20260705/`，ep228 764 步，LLM 快速失败模式）：6 个 capsule 全部在 target_tv 锚定分支触发（step 668+，与 4 月观察吻合）；`object_footprint_pixel_count=995`（30 个对象，全部 `graph_pcd_bbox_cells` 可信来源；旧 s22b capsule 恒为 0）；`true_semantic_pixel_count=218`、正阈值前 7 通道 1341 正像素（旧 capsule 0~50）；target heatmap source=`direct_plus_prior`（直接证据出现）。**4/30 悬置 P0 关闭：BEV 语义证据基座在线可用，旧 capsule 稀疏是因为先于管线存在。** 保留项（非阻塞）：检测器实例侧 `semantic_instance_footprints=0`，待查 `_project_semantic_instance_footprints` 触发条件；replay warning `target_value_final_branch_unexplained_drift` 复现（进 E5 归因）。LLM 通道定版官方 DeepSeek；E1 双套件已全部点火（见 2.0.2）。
