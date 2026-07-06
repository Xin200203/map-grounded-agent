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

## 5.0 G1 正式判定（2026-07-06 01:20，基于 periodic+full 双套件完整数据；explore 尚在收尾）

**判定：G1 未通过（outcome 层），但机制层出现大效应分离；按预案调整证据结构，不改故事内核。**

- Outcome 层（Observation）：cross-12 SR 平局（2/12 vs 2/12），聚合 SPL full 2×（0.087 vs 0.042）但由 ep661 单集驱动，配对 1W/2L/9T；intact-15 SR 6 vs 7（1 集差）、SPL≈平。n=12/15 无法分辨 1–2 集差异——**回溯确认 4 月的"优势"同为 1 集差，整个项目至今的 SR 层证据（正负两向）都欠功效**。
- 机制层（Observation，大效应）：cross-12 上 `out_of_local_window` **43（periodic）vs 3（full）**（14×）；executor override 率 **0.0285 vs 0.0053**（5×；intact 侧 0.018 vs 0.0058，3×）；planner 调用数相同（10.4 vs 10.3）——同预算下层间冲突与不可执行目标大幅减少。intact 场景两 profile 出窗均为 0 → **出窗是跨场景特有失败模式**。
- 机制归因（Observation，trace 级，`scripts/analyze_out_of_window_mechanism.py`）：**优势来自"避免"而非"修复"**。periodic 触发共享修复路径 20 次但当步成功仅 1 次、19 次搁置且仅 3 次事后恢复（16 次目标永久滞留）；full 全程仅产生 3 次出窗（grounding 总次数相当 193 vs 184）。因果链：事件驱动 bias 时效性 + target-anchor 局部窗口重投影 → 出窗目标从源头不产生；周期式陈旧远距 bias 则让事后修复也无力回天。这是论文机制表的核心行。
- 证据结构调整（故事内核不变，仍是 evidence-gated authority）：
  1. 主证据 = 机制效应表（出窗/override/noop 分布，大效应）+ outcome 非劣；
  2. cross-48 扩样（12 场景 × 4 集）检验 outcome 效应是否在更高功效下显形——**已启动**（periodic/full × 36 新集，GPU 3/7，01:17；episodes: 65 79 93 160 170 180 229 248 267 359 372 385 487 500 513 528 543 558 575 585 595 662 674 686 718 734 750 779 788 797 860 886 912 956 970 984）；
  3. E2 机制消融（no-monitor/fixed-interval/rules-only/no-prefetch × cross-12）**已全部在跑**；
  4. E3 gate-off 消融押大效应（若去门控导致灾难性震荡，n=12 可分辨），待 GPU；
  5. LLM 强度轴（Sonnet 条件复测，把"April vs 现在"的通道差异变成受控实验轴）——待用户批准官方 Anthropic key（预算约 $50–150）。

## 5.0.1 E1 定稿 + E2 消融分解（2026-07-06 02:00，Claim 候选——E2 差 no-prefetch 收尾）

**E1 cross-12 定稿**（DeepSeek 通道，SR 全部 2/12 持平）：explore SPL 0.048 / periodic 0.042（oow=43）/ full **0.087**（oow=3）。intact-15：explore 0.333/0.089、periodic 0.467/0.128、full 0.400/0.122。（注意：baseline-explore 走 baseline 模式管线，无 apply_strategy 记账，其 oow=0 与 smoothnav 系 profile **不可比**；oow 对比仅限 smoothnav-mode profiles。）

**E2 消融分解（cross-12，SR 全部 2/12 持平）——双因子结构**：

| profile | 调度 | 恢复机制 | monitor | SPL | oow | hi_calls |
|---|---|---|---|---|---|---|
| baseline-periodic | 周期 | 无 | 无 | 0.042 | 43 | 10.4 |
| smoothnav-fixed-interval | 周期 | 有 | 无 | 0.037 | 7 | 8.0 |
| smoothnav-no-monitor | 事件 | 有 | 无 | **0.085** | 3 | 10.2 |
| smoothnav-rules-only | 事件 | 有 | 规则 | 0.088 | 7 | 15.5 |
| smoothnav-full | 事件 | 有 | LLM-esc | **0.087** | 3 | 10.2 |

1. **恢复/锚定机制消灭出窗**（43→7），事件调度再抛光（→3）；
2. **事件驱动调度(+prefetch) 是 SPL 的来源**（0.037→0.085，2.3×；对照组恢复机制相同）；
3. **monitor 无独立增益**（full≈no-monitor；rules-only 同结果多花 50% planner 调用）——论文中 monitor 降为附录/工程细节，主贡献聚焦「事件调度 + gated grounding core」。
4. **no-prefetch 意外成为全场最佳**（SR 4/12=0.333、SPL 0.116、planner 调用近减半 5.6）：胜集 = full 胜集(661@0.853,717) ∪ baseline 胜集(717,859) + 358——锚定机制无损，而 prefetch 的 pending 晋升疑似在 frontier 到达时挤掉有效策略（`handle_frontier_reached` 的 stale-pending 保护只覆盖 specificity ≤ 当前的情形）。n=12 属 Observation；已加入 E1X 扩样（n=48 裁决）。若坐实，推荐默认 profile 改为 event+recovery+anchor 而不带 prefetch，且这本身是门控故事的又一例证：**未经证据门控的策略切换（pending 晋升）有害**。

## 5.0.2 G2 判定（2026-07-06 02:40，单门消融完成；e3e 组合证伪在跑）

**判定：G2（强形式）未通过——单门移除无一导致 outcome 退化；但绑定计数证明门在高强度工作，价值重定位为"churn 控制 + 纵深防御"。**

| 变体（cross-12，profile=full） | SR | SPL | hi_calls | 关键绑定计数 |
|---|---|---|---|---|
| full（全门参照） | 0.167 | 0.087 | 10.2 | obj 选择 33、低相关 0、直接执行 66 |
| e3a relevance off | 0.250 | 0.117 | **17.2** | obj 选择 **134**、低相关 **73**、被下层拦截 **44** |
| e3b direct-exec ungated | 0.250 | 0.121 | 8.5 | 锚定机制被结构性旁路（anchor_steps 0） |
| e3c target-progress off | 0.167 | 0.108 | 8.6 | 锚定活跃 814 步、价值项被置零 |
| e3d stall-decommit off | 0.167 | 0.109 | 7.9 | （行为分歧：LLM 高温随机性放大，见方法论教训） |

- **纵深防御被量化**：e3a 放开菜单门后 73 个低相关对象策略涌入，但落地层 no-match 守卫拦下 44 次直接执行，其余转为锚点被价值系统与退委机制消化——outcome 无损，代价是 planner 调用 +69%。**相关性门的可证明价值 = 计算/churn 控制，不是 SR**。
- **单门冗余 ≠ 全体冗余**：组合变体 e3e（四门全关）作为最后强证伪在跑（default temp，与对照一致）。若 e3e 也不退化 → 正式切 **Plan B**（效率+退化鲁棒叙事，E2 分解为主表，门控降为设计原则+纵深防御分析）；若 e3e 崩溃 → 门控故事以"冗余保护、联合必要"的形式保留。
- **方法论教训（入论文实验设置）**：默认温度下 LLM 选择的混沌放大使 n=12 的行为轨迹分歧巨大（e3d 的 obj 选择 3 vs full 的 33，仅 patience 一键之差不可能致此）；后续新条件（Sonnet 轴、复跑）统一钉 `SMOOTHNAV_LLM_TEMPERATURE=0`；已跑套件内部温度一致、结论不受影响。

## 5.0.3 n=48 终裁（2026-07-06 03:40，Claim）

**cross-48（12 场景 × 4 集，matched）：periodic SR 6/48=0.125、SPL 0.035；full SR 5/48=0.104、SPL 0.042；配对 4W/6L/38T。**

- Claim：在本基准（HM3D val 跨场景 text-goal）+ DeepSeek 通道下，**控制器复杂度不改变 outcome**——n=48 下 full 与 periodic 不可区分；成功率上限由感知/语义信号质量决定（基线 ~10–12%）。4 月的正向差与本轮早期正向差均为小样本涨落。
- 这决定了论文的两条可行路径：
  - **路径 α（首选，需资源）**：引入 LLM 强度轴。4 月 Sonnet 数据显示 outcome 在强 LLM 下分离（intact 0.6 vs 0.533、s6 SPL 2.8×）；若 Sonnet 条件复测证实"强语义信号下控制器把语义转化为 outcome 增益、弱信号下维持控制质量但无法拯救 outcome"，则论文主张变为**条件化的控制器价值定理**——机制表（已坐实）+ 双通道 outcome 对照。需官方 Anthropic key（约 $50–150）。
  - **路径 β（无新资源）**：纯机制/系统论文——outcome 诚实报告为 null，主张收缩为"同预算下控制质量大幅改善（出窗 14×、override 5×、churn 控制）+ 纵深防御量化 + 可回放归因体系"。ICRA 接收风险显著更高（审稿人问 so what）。
- 待决输入：e3e（联合必要性，661 已失守）与 no-prefetch n=48（若其 SR 优势保持，"减少 LLM churn → 更好 outcome"可成头条发现）。

## 5.0.4 E1X/E3 全部收线：弱通道最终图景（2026-07-06，Claim 候选）

**cross-48 四 profile 终表**（12 场景 × 4 集，matched，DeepSeek 通道）：

| profile | SR | SPL | 备注 |
|---|---|---|---|
| baseline-explore | 6/48=0.125 | 0.036 | |
| baseline-periodic | 6/48=0.125 | 0.035 | |
| smoothnav-full | 5/48=0.104 | 0.042 | |
| **smoothnav-no-prefetch** | **8/48=0.167** | **0.061** | 唯一双指标全面 ≥ 基线（SPL +74% vs periodic）；intact 侧与 full 持平（6/15） |

**e3e 全门关闭终判**：SR 2/12（与 full 同计数，但**失守正锚 661**）、SPL 0.056（−36% vs full 0.087）、oow 3→7、planner 调用 +65%（16.8 vs 10.2）。**G2 终形**：门控联合贡献 = 效率 + churn 控制 + 锚点利用能力（661 是锚定机制的 showcase），而非 SR；单门可被下游吸收（纵深防御）。无灾难性崩溃——诚实报告。

**弱通道系统结论（论文推荐配置）**：event-driven 调度 + grounding 恢复 + 证据门控锚定，**去掉投机性 prefetch、去掉 LLM monitor**——比 naive 周期基线和 kitchen-sink 完整版都好。叙事弧："构建全部机制、诚实测量、数据告诉你保留哪 60%"。prefetch 有害机制：pending 晋升在 frontier 到达时挤掉有效策略（358 案例），移除后 planner 调用近半省。

**强条件（Sonnet4.5+Haiku4.5 via vapeur）在跑**：判定 outcome 分离是否随语义质量出现（路径 α 的核心检验）。

## 5.0.5 强条件裁决 + 最终论点定稿（2026-07-06，Claim 候选）

**强条件（Sonnet4.5+Haiku4.5 via vapeur）cross-12**：periodic SR 2/12 / SPL 0.054 / oow **0**；full SR 2/12 / SPL **0.090** / oow **0**；配对 SPL **2W/0L/10T**（661: 0.853 vs 0.517；358: 0.226 vs 0.127）。

**LLM 强度轴三结论**：
1. SR 不随通道强度分离（双通道 null：弱 n=48、强 n=12 同集同数）——跨场景成功率为任务/感知上限；
2. SPL 在强通道干净分离（+68%，共同成功集全胜）——控制器把好语义转化为严格更优路径；弱通道下 SPL 配对混杂（1W/2L/9T）；
3. 出窗病理是弱语义特有（periodic：DeepSeek 43 次 → Sonnet 0 次）——恢复机制的价值形态 = 弱语义保险。

**最终论点（论文 thesis）**：控制器机制买不来感知受限任务的成功率；它在强语义下买效率、在弱语义下买退化保护；monitor 无价值、投机 prefetch 为负资产。推荐精益配置：事件调度 + 恢复 + 门控锚定（无 monitor/prefetch）。

**最后一波实验（2026-07-06 10:40 已发射）**：强条件 no-prefetch × cross-12（补主表对称）+ 强条件 {periodic, full, no-prefetch} × 36 扩样 → 双通道均达 n=48。收线后表 1 定稿、Abstract/Intro 按"条件化效率定理"改写。

## 5.0.6 强通道 n=48 终表：全面校准（2026-07-06 15:00，Claim）

**strong cross-48**：periodic 8/48=0.167/SPL 0.062；full 8/48=0.167/0.086（dSPL +0.023，CI[−0.034,+0.086] n.s.，配对 8W/5L/35T）；no-prefetch 5/48=0.104/0.046（n.s.）。

三项诚实校准：
1. **强通道 SPL 分离未活过 n=48**（+68%@n12 → +38% 聚合 n.s.）——Finding 6 强形式降级为"两通道聚合 SPL 一致偏向 full（+0.007/+0.023）但均不显著"。
2. **no-prefetch 优势在强通道反转**（弱通道最佳 8/48 → 强通道最差 5/48，两向均 n.s.）——精益推荐的依据改为**成本**："同等 outcome、planner 调用减半"在两通道均成立；性能表述撤回。
3. **终局 null（96 集×2 通道 matched）**：outcome 层无任何变体、任何方向、任何通道的显著差异。稳固的只有：机制效应（出窗 14×/override 5×/调用成本）、纵深防御计价、图召回根因（~6 caption/集）。
4. **G5 现在悬于召回探针**（15:00 已发射，3 profile × cross-12，门控 3→1/16→8/10→5）：outcome 若在健康 regime 显形 → 建设性正结果；SR 抬升无差异 → "召回是唯一瓶颈"成为主发现；均不动 → 纯审计+根因定位，主投策略转 RA-L 并行。

## 5.0.7 召回修复探针判决（2026-07-06 17:00，Observation→扩样中）

**修复生效**：唯一 caption ~6 → **15.7–17.1**（2.5×）；never_captioned 失败 55–70% → **1/30**。一行门控参数（obj_min_detections 3→1 等）完成感知召回修复。
**SR 不动（2/2/3 of 12）但失败上移一层**：残余失败对半分为 detected_no_anchor（4–6：目标在图中、planner 未承诺——控制器层可改进）与 anchored_failed（4：承诺未转化）。**瓶颈是分层的**：修复一层暴露下一层。
**修复 regime 中控制器 SPL 差距全程最大**：periodic 0.041 / full 0.104（2.6×）/ no-prefetch 0.126（3.1×，配对 3W/1L）@n=12 —— 与"控制器价值随世界模型质量增长"方向一致；**n=48 扩样已发射**（17:51，3 profile × 36 集），显著则为论文的建设性正结果。
**论文弧最终形态（候选）**：null（双通道 n=48）→ 根因（图召回 6 caption）→ 一行修复（2.5× 召回）→ 瓶颈上移（转化层）+ 控制器价值随 regime 健康度显现[待 n=48 检验]。

## 5.0.8 召回 regime n=48 终判 —— 三重复现的 regime-invariant null（2026-07-06，Claim，实验战役结束）

**召回 regime cross-48**：periodic SR 0.146/SPL 0.043；full 0.083/0.037（**名义更差**，n.s. p=0.51）；no-prefetch 0.125/0.048（n.s.）。**n=12 的"控制器价值最大"信号未活过 n=48**——这是同一模式第三次发生（弱 SPL、强 SPL、召回 SPL 全部 n12→n48 归零）。

**由此确立论文的中心 Claim（比任何"正向"版本都强）：outcome-level null 是 regime-invariant 的**——跨弱 LLM / 强 LLM / 修复感知三个正交轴、144 集 matched，**无任何控制器变体在任何 regime、任何方向上显著改变 SR/SPL**。这不是"我们没测出优势"，而是"在系统性充分功效下，控制器复杂度对该任务 outcome 的贡献可证伪地为零"。

**分层瓶颈被完整刻画（建设性诊断主结果）**：召回修复（1 参数）令 never_detected 占失败 55–70% → **21–29%**，caption 6→16；但 detected_no_anchor 立即升为最大失败桶（18–23/~30）→ **瓶颈从感知层移到语义-几何转化层，SR 不变**。控制器同样不修转化层。

**机制层正向结果不变（论文的正贡献所在）**：出窗 14×、override 5×、prefetch −50% 调用、门控纵深防御计价、召回修复解锁 ep159（全程不可解→首解）。这些是过程/成本/诊断层的真实效应，与 outcome null 并存不矛盾。

**G5 终裁触发**：论文确定为 **rigorous negative-result + layered-bottleneck audit** 定位。正向 outcome 结果已在三 regime 下证伪，不再追。venue 决策上交用户（见对话）。

## 5.0.9 C3 probe 判定：承诺粘性机制生效（2026-07-06 23:00，Observation → n=48 扩样中）

**C1 诊断 → C2 机制 → C3 验证的闭环**：从 34 次被驱逐承诺的解剖导出 target-commitment persistence（stagnation/stuck/pending 三驱逐器只换路径不弃目标 + 菜单毒化免疫 + 安全阀），probe 结果：

| 修复 regime cross-12 | SR | SPL |
|---|---|---|
| periodic / recall-full（对照） | 各 2/12=0.167 | 0.041 / 0.104 |
| **full+persistence** | **4/12=0.333** | **0.132** |
| **no-prefetch+persistence** | **4/12=0.333** | 0.129 |

- **战役首个真正 SR 提升**（翻倍，两宿主一致，全战役跨场景最高）；翻转集 159/358 正是诊断标本；859 安全阀耗尽回退（by design）。
- n=12 McNemar +2/−0 n.s. → **n=48 扩样已发射**（23:10，3 profile 含 periodic+persistence 对照，构成调度×粘性 2×2）。判据：若 SR 比例保持（~12-16/48 vs 6-7/48），McNemar 可达显著 → 论文获得"诊断驱动的正向主结果"，叙事升级为 audit→root-cause→repair(感知层)→repair(转化层)→outcome 增益的完整闭环。

## 5.1 E1 中期观察（2026-07-06 01:00，Observation，套件未全部完成）

- **intact-15 已完成两 profile**（DeepSeek 通道）：baseline-periodic SR 7/15=0.467、SPL≈0.128；smoothnav-full SR 6/15=0.400、SPL≈0.122。**与 4 月 Sonnet 结果（full 0.6 > periodic 0.533）排序翻转**，且两者绝对值都大幅低于 Sonnet 时代——通道质量对全系统影响显著。SR 差距为 1 集（6 vs 7，n=15），在噪声区间内，先按"平局"解读。逐集：full 独得 291/296，periodic 独得 289/293/299。
- **cross-12 前 3 共同集**：full SR 0.333/SPL 0.284（复现 4 月），两 baseline 全零（4 月 periodic 曾拿下 661，本轮丢失）。若后续保持，cross 将出现 4 月不存在的 SR 分化。
- **中期假设（Hypothesis，待 cross 完成检验）**：门控/恢复机制的价值随"LLM 语义决策可靠性下降 + 场景难度上升"而放大——intact+弱 LLM 下周期重规划已够用（平局），cross+弱 LLM 下 baseline 崩溃而 full 存活。若成立，论文主表叙事从"处处更强"调整为"弱语义条件下的优雅退化/鲁棒性"，且可考虑引入 LLM 强度作为实验轴（Sonnet 复测需官方 Anthropic key，约 $50–150，届时请示）。
- E2 已提前启动（no-monitor / fixed-interval × cross-12，GPU 5/6，2026-07-06 00:51）——两者在任何叙事分支下都需要。

## 6. 修订记录

- 2026-07-05 v1：初版（故事定版为 evidence-gated semantic authority；E0–E5 矩阵；G1/G2 gate；Plan A/B/C 分叉）。
- 2026-07-06 v2.1：**E0b 判定通过**（Observation，run `results/e0b_planner_channel_probe_20260706/`，ep228，官方 DeepSeek 通道）：20 次 planner 调用 **0 空响应**（4 月 Clauddy 为 13/16）、fallback 2/20、planner 主动选中目标 tv 2 次、23 个 capsule 覆盖 target 分支（painting→tv 锚定切换序列，是 gating 分析的好样本）；episode 本身仍失败（ep228 感知受限 hard case，符合预期）。晚期 capsule 语义基座继续增长（true 1546px / obj_fp 1213px）。**跟进项已关闭（2026-07-06 复查）**：`semantic_instance_footprints` 并非恒为 0——E0b step385 capsule 中 count=1（potted plant，真实投影，全图坐标）；该层是**逐帧瞬态**设计（只保留当前 mapping 步的新检测），后期 capsule 为 0 属正常。初判"恒为 0"是检查脚本读错 schema 键所致。含义：论文 dense BEV 图应以 graph-pcd footprint（随节点持久）为主层、实例 footprint 作新鲜证据叠加；跨步持久化累积为可选增强（W4 有余力再做）。E0 整体关闭，E3e 保持条件项（视觉通道）。
- 2026-07-06 v2：**E0a 判定通过**（Observation，run `results/e0_footprint_probe_20260705/`，ep228 764 步，LLM 快速失败模式）：6 个 capsule 全部在 target_tv 锚定分支触发（step 668+，与 4 月观察吻合）；`object_footprint_pixel_count=995`（30 个对象，全部 `graph_pcd_bbox_cells` 可信来源；旧 s22b capsule 恒为 0）；`true_semantic_pixel_count=218`、正阈值前 7 通道 1341 正像素（旧 capsule 0~50）；target heatmap source=`direct_plus_prior`（直接证据出现）。**4/30 悬置 P0 关闭：BEV 语义证据基座在线可用，旧 capsule 稀疏是因为先于管线存在。** 保留项（非阻塞）：检测器实例侧 `semantic_instance_footprints=0`，待查 `_project_semantic_instance_footprints` 触发条件；replay warning `target_value_final_branch_unexplained_drift` 复现（进 E5 归因）。LLM 通道定版官方 DeepSeek；E1 双套件已全部点火（见 2.0.2）。
