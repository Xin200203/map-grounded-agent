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

## 5.0.10 S8 漏斗战役：C456 终判 + C4567 中期 + C7' 机制定版（2026-07-08 01:00）

**成功机制勘误（Claim，run 解剖证实）**：HM3D text-nav 评测为 `end_on_success`——agent **进入目标 1 m 半径即成功结束，无需 stop 动作**（c4567 ep 解剖：末帧 `visible_target_temp_goal` 源、前进动作、无 stop/lock 即 success=1）。此前 S8 审计中"stop 被 `found_goal==1` 锁死"（截肢 2）对 SR 实际无关紧要；**真正致命的是接近链路被切**（截肢 1：接管恒禁；截肢 3：5 m 处无验证槽位直接清除+拉黑）。文档 `current_bottlenecks` 原因 2 需按此勘误解读。

**C456 n=48 终判（Claim）**：full 4/48 vs C45 参照 8/48——重开接管但无验证，**净负**。C4567 probe-12 中读同向：full 2/12、np 1/12 vs C45 3/12。归因：3 个 probe 成功全部由接管驱动（机制在正确目标上有效），但**未经验证的接管把步数预算烧在假 sighting 上**。接管路径存活 20–38 步（adapter watchdog 非杀手）。

**C4567 n=48 终判（Claim，2026-07-08 收线，96/96）**：
| 变体 | full | np | vs C45 McNemar |
|---|---|---|---|
| C45（漏斗切断，参照） | 8/48=0.167 | 6/48=0.125 | — |
| C456（接管无验证） | 4/48=0.083 | — | +2/−6，p=0.289 n.s. |
| C4567（接管+近程 keep，无验证） | 4/48=0.083 | 5/48=0.104 | full +1/−5，p=0.219；np +2/−3，p=1.0，均 n.s. |

单项均 n.s.，但**方向在两宿主 × 两波次上一致为负**：无验证的漏斗重开稳定折损约一半成功（SPL 同向：dSPL −0.013~−0.019）。归因：9 个 c4567 成功中 7 个接管参与（机制对真目标做功），但 C45 独得集（ep797、ep956 等）被假 sighting 追逐烧掉。**判定：任务 #23 关闭，C4567 作为消融中间行定格；正向希望全部押在 C7v（验证漏斗）**。

**结论：漏斗缺的不是管道（C6/C7 已通），是验证器**——ins-image 在同一槽位有 LightGlue，text 是空洞。与调研报告 R3（SG-Nav/TriHelper/VLFM 等 propose-verify 模式，+3–6 SR）及 Haiku/Sonnet 验证 probe（Haiku 8/12 ≥ Sonnet 7/12，模型不是瓶颈）三方汇合。

**C7' 机制定版（2026-07-08 实现，commit 4db245c）**：接管发起点（`found_goal==1` 远距分支）VLM 验证闸：
- sighting crop（检测框 +20% margin，480×640 原帧）→ Haiku 判定 `yes/no/unsure`；
- 分层判据（probe 教训）：**只判类目+内在属性**（颜色/材质/形状），周边一律忽略（数据集描述噪声大）；
- `no`=本帧跳过接管、**不拉黑**（近处再看可能翻案）；`unsure`/解析失败/通道故障=放行（fail-open，验证器宕机不得切断漏斗）；
- 节流：10 步间隔 + 每集 ≤8 次调用（超限沿用末次判定）；独立 vapeur Haiku 通道（主通道 deepseek 纯文本不受影响）；
- flag `executor_takeover_vlm_verify` 默认关；trace 字段 `takeover_verify_verdict` 逐步落盘；纯逻辑（节流/解析）10 项单测，全套 244 绿。

**c7v 套件设计**：config `_c7v`（=c4567+验证），cross-48 matched（probe12+exp36 × full/np），四行消融链 C45（无接管）→C456（接管无 keep）→C4567（接管+keep 无验证）→C7v（全漏斗+验证）。判据：C7v SR 显著回升过 C45（McNemar n=48）→ 论文获得"propose-verify 是文本目标漏斗的必要件"的正向主张；若仅回到 C45 水平 → 验证器只能中和接管毒性，级联继续。

## 5.0.11 C7v 终判：S8 战役收官——被切断的接口是承重墙，不是断裂（2026-07-08，Claim）

**c7v n=48 终值（96/96 收线，零崩溃；判定 105 yes / 103 no / 20 unsure；接管放行 765 步 / 拦截 1184 步）**：

| 变体 | full | np | vs C45（同宿主） |
|---|---|---|---|
| C45 漏斗切断（参照） | **8/48** | **6/48** | — |
| C456 接管无验证 | 4/48 | — | +2/−6，p=0.29 |
| C4567 +近程 keep 无验证 | 4/48 | 5/48 | +1/−5，p=0.22 / p=1.0 |
| **C7v +VLM 验证** | **4/48** | **3/48** | +1/−5，p=0.22 / +3/−6，p=0.51 |

C7v vs C4567（验证的净效应）：+2/−2，p=1.0，dSPL +0.0007——**验证对结局零净效应**。

**尸检（5 个 C45 独得集在 c7v 下的行为）**：ep229/358/717/797/956 全部有 'yes' 判定 + 接管步——**验证器放行了 sighting，执行器去追了，episode 反而失败**；同集在 C45（接管恒禁）下经锚点/探索路径成功。反向仅 ep486（41 接管步、5 yes）、ep64/674（np）三例验证接管制胜。

**机制结论（三行 4/48 平坦性 + 尸检 + end_on_success 勘误的合并推论）**：
1. 在 `end_on_success` 判定下，锚点路径本身就能把"坐标足够准"的集走进 1 m 半径完成——接管并不提供锚点路径没有的完成能力；
2. 接管目标是**单帧深度估计的椭圆**（compute_ins_dis），坐标质量低于 C5 门控的 ≥2 次检测节点中心；接管一旦发起就**顶掉**了更优的锚点接近，8 步 watchdog 再制造目标震荡；
3. 因此 4 月的"截肢"实际是**承重安全栏**：C6/C7/C7' 整个重开计划被 n=48×2 宿主×3 变体一致裁决为负向/无效。验证器工作正常（105 yes 里有真目标、no 拦掉 1184 步假追逐），但它验证的是"看见的是不是目标"，而系统真正缺的是"更准的目标坐标"。

**裁决与转向**：S8 执行器接口层关闭（C6/C7/C7' 全部 flag 归 off 为默认；代码保留作消融行）。剩余失败质量的约束在**锚点坐标误差**（anchored_failed 16/40，坐标偏差 1–3 m）与**候选质量**（detected_no_anchor 18/40）——两者都是感知/融合问题。下一波：**R2 多帧证据融合**（提升节点质心精度，直接攻击 anchored_failed）+ **R1 GroundedSAM 目标文本主动 grounding**（提升合格候选出现率，攻击 detected_no_anchor）。论文获得的不是正向 SR 主结果，而是一条**完整的三层审计弧**：S1 召回修复生效 → S3–7 承诺机械生效 → S8 接口重开被证伪（含验证版），瓶颈被压缩到感知坐标质量一个点上。

## 5.0.12 最近逼近距离基线 + R2 波次发射（2026-07-10，Observation）

新工具 `analyze_goal_distance.py`：从 step trace 位姿轨迹 + 数据集 GT 目标位置计算每集**最近逼近距离**（episodic→map 变换用成功集自校准：`x=+fwd,y=-right`，c45-full 8 成功中 7 个末位姿落 1.5 m 内，变换自验证）。C45 基线（46/48 集有 trace）：

| | ≤1 m | ≤2 m | ≤3 m | 中位最近逼近 |
|---|---|---|---|---|
| c45-full | 5/46 | 11/46 | 17/46 | **4.97 m** |
| c45-np | 5/46 | 13/46 | 21/46 | **3.36 m** |

**含义**：大多数失败集从未接近过目标（中位 3.4–5 m），"差最后一米"只是失败质量的一部分——瓶颈量化从 outcome 二值细化为连续距离谱。R2（多帧质心中位数融合，commit aedc605，flag `graph_robust_center`）已发射 cross-48 四车道；判据双层：(a) SR/McNemar vs C45；(b) **配对最近逼近距离差**（机制层——坐标改善即使不过 1 m 阈值也可测）。冒烟 ep661：robust center 在体应用 3→7→8 节点，零崩溃。

## 5.0.13 R2 终判：坐标精度不是最后一公里的约束——双层皆 null（2026-07-10，Claim）

**r2 n=48 收线（96/96，success 合计 14，与 C45 持平）。双层判据双双 n.s.**：

| | SR full | SR np | vs C45 McNemar |
|---|---|---|---|
| C45（参照） | 8/48=0.167 | 6/48=0.125 | — |
| **R2 鲁棒质心** | 7/48=0.146 | 7/48=0.146 | full +2/−3 p=1.0；np +3/−2 p=1.0 |

- **机制层（配对最近逼近距离差，10k bootstrap）**：full 均值 Δ=**−0.28 m** CI95=[−0.79,+0.19]；np 均值 Δ=**+0.44 m** CI95=[−0.11,+1.15]——**两宿主方向相反，CI 均跨零**。
- **分布**：r2-full ≤1m 4/46、≤2m 12/46、中位 4.23m（c45-full 5/46、11/46、4.97m）；r2-np 中位 4.63m（c45-np 3.36m）。一升一降，噪声。
- 机制**确实在体生效**（冒烟 ep661：robust center 应用到 3→7→8 个多视角节点），不是没跑起来——是跑了但改变量太小。

**结论（比控制器 null 更硬的一条）**：R2 有一个**专门为它设计的机制层度量**（1 m 阈值下也能测到的坐标改善），连这个度量都没动。因此 **anchored_failed 的 1–3 m 缺口不是"质心被坏帧带偏"**——中位数质心与均值质心之差本身就小于 closest-approach 能分辨的量级。**锚点坐标精度不是最后一公里的绑定约束**。合并 §5.0.12 基线（大多数失败集从未接近目标，中位 3.4–5 m），剩余杠杆（若有）在**目标发现**（detected_no_anchor，R1 域），不在坐标精度、也不在最后一公里执行。

**三条已耗尽的机制前线**：控制器（3 regime）/ S8 漏斗（C45→C7v 含 VLM 验证）/ R2 坐标融合。R1（目标文本主动 grounding）是唯一未试且被 closest-approach 数据指向的方向，同时也是"正向结果"路径的最后一次机制摆动。

## 5.0.14 R1 波次发射：目标文本主动 grounding（2026-07-10，Observation）

用户在 R2 双 null 后拍板：**R1 + 并行起草审计论文**（对冲——R1 中则升级故事，R1 null 则审计论文已在路上）。

**R1 机制（commit 48b0d3f，flag `graph_target_text_grounding` 默认关）**：把目标特异短语（主类目规范名 + intrinsic 描述，**排除 extrinsic 周边**以免 ground 上下文物）并入 GroundingDINO 的 `node_space` prompt。攻 detected_no_anchor 18/40 的两个洞：
1. **缺失类目**：node_space 15 词里**没有 toilet**——6 个目标类目中唯一缺的，GroundingDINO 从不被要求找马桶（chair/sofa/bed/plant/tv 都在）。
2. **通用 caption 相关性不足**：目标实例只带通用类目 token，相关性可能过不了 0.75 候选门；并入 intrinsic 描述后 caption 带目标属性词 → 相关性升。
- 单次 GroundingDINO 前向，仅文本 prompt 变长，无额外算力；纯短语构造 9 项单测（含 toilet 补缺、television→tv、去冠词、截断、去重、幂等），全套 260 绿；`target_grounding_phrase` 逐步落 trace。
- 冒烟 ep661（chair 集）：`node_space += 'chair. material...wood and leather...'` 正确触发、零崩溃。

**c48 判据**：cross-48 matched（probe12+exp36 × full/np）vs C45 参照，双层——(a) SR McNemar；(b) 若 SR 不动，则看 detected_no_anchor 桶是否缩小（合格候选出现率）+ 最近逼近距离谱是否左移。发射于 GPU 4/5/6/7（0-3 被外部占用）。

**这是"正向结果"路径的最后一次机制摆动。** R1 中 → 论文获诊断驱动的正向主结果（audit→localize→fix）；R1 null → 四条前线全证伪，审计论文定稿（脊柱已在 §4.6c synthesis 写好）。

## 5.0.15 GT 直接测量纠正瓶颈定位——"目标发现"是错的推断（2026-07-10，Claim，方法学转折）

**动因**：用户质问——每部分都有 trace 记录、都有 GT，为何瓶颈定位仍是*推断*（消融反推 + 图内部候选桶）而非*测量*？批评成立。此前 detected_no_anchor 18/40 是用**图自己的候选逻辑**判的，没拿 GT 核对。

**新工具 `analyze_failure_stage.py`**：逐集用 trace 决策链（承诺/候选事件/caption 打分）+ GT 位姿逼近距离，测出"死在哪一环"。trace schema 审计发现关键缺口——**逐节点地图坐标从未落盘**（正是 GT 化"目标定位"所需字段），已补 `record_graph_nodes`（commit 2217bc8，episode 末落盘 center/caption/num_detections）。

**c45-full 40 失败集 GT 测量（对比推断，结论反转）**：
| 环节（GT 测量） | 计数 |
|---|---|
| committed-ever（承诺过目标类目） | **36/40** |
| candidate-ever（合格候选事件触发） | **34/40** |
| caption≥0.75（caption 识别出目标） | **34/40** |
| 逼近 ≤2m（pose→GT，已校验成功集 7/8≤1.5m） | 仅 4/40 |
| **no_candidate（从没识别出目标）** | **仅 4/40** |

**结论（硬测量）**：**目标在 85–90% 失败集里被检测、caption、识别为候选、并承诺了——"找不到目标"只有 4/40。瓶颈不是目标发现。** 主桶是 **32/40：识别+承诺了目标，但 agent 从没走到 GT 2m 内**。这直接**证伪**了 §4.6c synthesis 里"剩余杠杆在 perception recall/目标发现"的推断。

**对 R1 的含义**：R1 攻 detected_no_anchor（找目标）——现测得该桶仅 4/40，**R1 上限只有 ~4 集**，瞄错了桶。R1 仍值得跑完（4/40 真实、已在跑、且落盘后可 GT 验证其行为），但**正向主结果不该押在 R1**。

**32/40 主桶的下一刀（正在测）**：需节点坐标才能 GT 化——承诺的目标节点**定位在 GT 附近**（→导航/可达性失败）还是**定位到错地方/错实例**（→定位/感知失败）？注：执行器 geometric goal 多为搜索 frontier（成功集 goal_dist 中位 5.9m 证实其非目标定位坐标），故此刀必须用节点落盘。R1（带落盘）+ c45 重采（带落盘）已并行发射（8 卡），收线后出配对定位诊断。

**方法学价值**：这正是"测量而非推断"的意义——GT 直接测量抓出了我们瞄错桶。审计论文的脊柱因此更硬：不仅逐层证伪机制，还用 GT 把失败质量精确归位到"承诺→到达"这一段。

## 5.0.16 GT 定位诊断收官：真正的瓶颈是**实例辨识**，不是前四条前线（2026-07-11，Claim，决定性）

**R1 终判：NULL（第四条机制前线证伪）**。对稳定原始 c45 参照：full SR 0.167 vs 0.167（p=1.0）、np 0.146 vs 0.125（p=1.0），dSPL 均 n.s.。（同批 c45redump-np 因重跑温度方差虚高到 12/48，那处 "sig 负 SPL" 是参照虚高产物。）如 GT 预测——R1 攻的 no_candidate 桶仅 4/40。

**GT 定位诊断（节点落盘 + GT，轴锁定 flip_y=True/x=+fwd,y=-right，与 pose→GT 校准一致）**：

| | 成功集 有目标节点≤R | 失败集 无任何节点≤R | 失败集 定位对但没到(NAV) |
|---|---|---|---|
| R=1.5m | 10/21 (48%) | **69/75 (92%)** | 3/75 |
| R=2.5m | 17/21 (81%) | 58/75 (77%) | 7/75 |

R1 同型（76/81、68/81 无 GT 附近节点）——机制层未动。

**判别信号（跨半径稳健）**：目标节点出现在 GT 附近的比率，成功集 48–81% vs 失败集 4–9%——**差 8–9 倍**。失败几乎不是"定位对了没走到"（NAV 仅 3–7/75）。

**决定性结论（四份 GT 测量自洽）**：
1. 系统承诺目标类目（36/40）、触发候选（34/40）、caption≥0.75（34/40）——**认出了"一个"目标类目物**；
2. 但**真目标位置从未进图**（69/75 失败集 GT 1.5m 内无任何节点），agent 从没去那儿（逼近中位 3.4–5m）；
3. 故承诺的是**别处的、不同实例**的同类目物。

**瓶颈 = 目标实例辨识（instance disambiguation）+ 目标导向探索**：文本描述本就是要在同类多实例中选**特定那个**（"窗边那把棕色椅子"），系统却承诺了**先遇到的任一同类实例**，于是走错方向、真目标区域永不被探索/建图。这解释了**为何前四条前线全 null**——它们都在错误的层做功：
- 控制器机械（证伪）／S8 最后一公里（证伪）／R2 坐标精度（证伪）／R1 类目发现（证伪）——**没有一条触及"选错实例"**。

**这是"测量而非推断"的终点**：GT 直接测出真瓶颈在实例辨识，而非任何已试机制。论文脊柱因此完整且硬：四条前线 matched-ablation 证伪 + GT 定位诊断精确定位真因。下一步机制（若攻正向）应是**用描述（内在属性+周边关系）辨识正确实例并持续探索直到找到**——`target_matching` 现在只按类目+token 承诺先遇者，从不校验实例特异性、从不为更优实例继续搜。

## 5.0.17 实例辨识为何结构性受阻：caption 类目盲（2026-07-11，Claim）

实现 D1（实例辨识）时查真实节点 caption——**全是类目裸词**（'chair'、'cabinet'、'bed'、'plant'、'painting picture'；无颜色/材质/形状/周边）。成因：`node_space` 是固定类目表，GroundingDINO caption 只回类目 token。

**含义（对 D1 决定性）**：
1. `score_caption_against_goal` 类目单命中即 1.0——**自动承诺对任何含目标类目的节点在 1.0 触发**，实例盲的根因就在这。
2. **图的语义表示是类目级的，无法区分同类两实例**（两把椅子 caption 都是 'chair'）。**文本属性门控不可行**（caption 无属性可匹配描述的内在属性）。
3. 故 D1 只能走**两条非 caption 路径**：(a) VLM crop 在承诺时按完整描述验证实例身份（复用 C7' 基建，no→不承诺续搜）；(b) 用图的**关系/房间结构**按描述的 extrinsic 属性辨识（"卧室里靠桌的椅子"→选 room=bedroom 且有边连 table 的椅子节点；但 deepseek 文本通道下关系边无标签，可行性存疑）。
4. **风险坦白**：C7'（VLM 承诺/接管验证）已 null；且 GT 诊断显示真目标 69/75 从未进图——即使完美实例验证，若真实例从未被观测也无从辨识。D1 是有依据但高风险的一搏。

**这本身强化审计结论**：不仅四条前线证伪 + GT 定位真因，还测出**冻结 backbone 的类目级表示使实例辨识结构性不可行**——这是根本限制而非调参问题，是审计论文有力的收尾论点。

## 5.0.18 根因锁定：类目级 found_goal 早停在错实例（2026-07-11，Claim，统一根因）

**存量数据决定性测量（c45redump n=96）**：
- **无一集用满 1000 步**（成功 max 558、失败 max 643）；41/75 失败集 <100 步；失败 total_steps 中位 95（成功 187）。`trace_recs==total_steps`（逐步记录）证实非抽样。
- 终止方式：`terminal.reason='environment_done_without_success'`——Habitat 在低步数返回 done。非成功非 1000 步 → **agent 早发 stop**。
- 歧义在成败两边都普遍（多实例：成功 18/21、失败 52/75），成功集反而映射更多目标类目节点（14 vs 7.6）——**问题不是"候选间辨识"，是真实例从未进候选集**。

**代码确认根因**（agent.py:724, 860）：`found_goal = (检测到 gt_goal_idx 的框)`，`gt_goal_idx` 是**类目**索引 → **任何目标类目物检测即 found_goal=1**；唯一 stop 路径是 `if stop and found_goal==1: action=0`。故 agent 停在**先遇到、可达的任一目标类目实例**，无实例校验。若非描述实例/不在真目标 1m 内 → 失败。

**统一根因**：类目级 found_goal 让 agent 在错实例上早停，烧掉 episode、永不探索到真目标。这**一条**解释了全部测量：committed 36/40（承诺错实例）、真目标 69/75 从未进图（停了不再探索）、逼近中位 3.4–5m、无一集用满预算、四条前线全 null（都不在"类目级早停"这一点做功）。

**D1 精确化**：不再是"候选间辨识"，而是**在 found_goal stop 闸处按完整描述做实例验证**——检测到目标类目物、FMM 将 stop 时，VLM crop 比对描述（内在属性+周边）；不匹配则**抑制 found_goal/不 stop、拉黑该实例区、继续探索**，用满预算找正确实例。复用 C7' VLM 基建，但注入点正确（stop 闸 vs 接管点）、判据正确（实例身份 vs 类目）、效应正确（阻止错实例早停 vs 追加接管）。C7' null 恰因注入点/判据错——此设计针对已测根因。

**注意公平性**：类目级 found_goal 早停是 UniGoal 原生行为，baseline 同样受此限——这也解释了发表 SR 仅 20.2。D1 是在冻结 backbone 外加的实例验证闸（flag 门控），需在写作中说明。

## 5.1 E1 中期观察（2026-07-06 01:00，Observation，套件未全部完成）

- **intact-15 已完成两 profile**（DeepSeek 通道）：baseline-periodic SR 7/15=0.467、SPL≈0.128；smoothnav-full SR 6/15=0.400、SPL≈0.122。**与 4 月 Sonnet 结果（full 0.6 > periodic 0.533）排序翻转**，且两者绝对值都大幅低于 Sonnet 时代——通道质量对全系统影响显著。SR 差距为 1 集（6 vs 7，n=15），在噪声区间内，先按"平局"解读。逐集：full 独得 291/296，periodic 独得 289/293/299。
- **cross-12 前 3 共同集**：full SR 0.333/SPL 0.284（复现 4 月），两 baseline 全零（4 月 periodic 曾拿下 661，本轮丢失）。若后续保持，cross 将出现 4 月不存在的 SR 分化。
- **中期假设（Hypothesis，待 cross 完成检验）**：门控/恢复机制的价值随"LLM 语义决策可靠性下降 + 场景难度上升"而放大——intact+弱 LLM 下周期重规划已够用（平局），cross+弱 LLM 下 baseline 崩溃而 full 存活。若成立，论文主表叙事从"处处更强"调整为"弱语义条件下的优雅退化/鲁棒性"，且可考虑引入 LLM 强度作为实验轴（Sonnet 复测需官方 Anthropic key，约 $50–150，届时请示）。
- E2 已提前启动（no-monitor / fixed-interval × cross-12，GPU 5/6，2026-07-06 00:51）——两者在任何叙事分支下都需要。

## 6. 修订记录

- 2026-07-05 v1：初版（故事定版为 evidence-gated semantic authority；E0–E5 矩阵；G1/G2 gate；Plan A/B/C 分叉）。
- 2026-07-06 v2.1：**E0b 判定通过**（Observation，run `results/e0b_planner_channel_probe_20260706/`，ep228，官方 DeepSeek 通道）：20 次 planner 调用 **0 空响应**（4 月 Clauddy 为 13/16）、fallback 2/20、planner 主动选中目标 tv 2 次、23 个 capsule 覆盖 target 分支（painting→tv 锚定切换序列，是 gating 分析的好样本）；episode 本身仍失败（ep228 感知受限 hard case，符合预期）。晚期 capsule 语义基座继续增长（true 1546px / obj_fp 1213px）。**跟进项已关闭（2026-07-06 复查）**：`semantic_instance_footprints` 并非恒为 0——E0b step385 capsule 中 count=1（potted plant，真实投影，全图坐标）；该层是**逐帧瞬态**设计（只保留当前 mapping 步的新检测），后期 capsule 为 0 属正常。初判"恒为 0"是检查脚本读错 schema 键所致。含义：论文 dense BEV 图应以 graph-pcd footprint（随节点持久）为主层、实例 footprint 作新鲜证据叠加；跨步持久化累积为可选增强（W4 有余力再做）。E0 整体关闭，E3e 保持条件项（视觉通道）。
- 2026-07-06 v2：**E0a 判定通过**（Observation，run `results/e0_footprint_probe_20260705/`，ep228 764 步，LLM 快速失败模式）：6 个 capsule 全部在 target_tv 锚定分支触发（step 668+，与 4 月观察吻合）；`object_footprint_pixel_count=995`（30 个对象，全部 `graph_pcd_bbox_cells` 可信来源；旧 s22b capsule 恒为 0）；`true_semantic_pixel_count=218`、正阈值前 7 通道 1341 正像素（旧 capsule 0~50）；target heatmap source=`direct_plus_prior`（直接证据出现）。**4/30 悬置 P0 关闭：BEV 语义证据基座在线可用，旧 capsule 稀疏是因为先于管线存在。** 保留项（非阻塞）：检测器实例侧 `semantic_instance_footprints=0`，待查 `_project_semantic_instance_footprints` 触发条件；replay warning `target_value_final_branch_unexplained_drift` 复现（进 E5 归因）。LLM 通道定版官方 DeepSeek；E1 双套件已全部点火（见 2.0.2）。
