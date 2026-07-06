# 审稿缺口补强计划（Reviewer-Gap Reinforcement Plan）

日期：2026-07-06
输入：`smoothnav_icra2027_story_and_experiment_design_20260705.md` §5 系列判定 + 模拟审稿评审（weak reject / borderline 判决）
原则：每个补强项对应一个明确的审稿反对意见；优先做**不需要新实验**的项；所有新实验限制在已验证的 12 场景集内。

---

## Gap 1：外部效度锚（对齐 UniGoal 发表协议与数字）——最高优先级

**审稿反对**："SR 10–17% 远低于已发表零样本系统 → 你们的复现可能是坏的，内部消融无意义。"

**行动**（无 GPU，约 1 天）：
- G1.1 取 UniGoal 论文（arXiv 2503.10630）与官方 README 的 text-goal 协议与数字：数据集/split、成功半径、步数预算、episode 数、SR/SPL。
- G1.2 与我们的 effective config 逐项对照：`success_dist=1.0`、`max_episode_length=1000`、HM3D v0.2 val、`val_text.json.gz`（795 attribute 条目）、检测器与阈值。产出协议对照表。
- G1.3 判定三分支：
  - (a) 协议同构且数字可比 → 论文直接引用为上游参照行；
  - (b) 协议有差异（如成功半径/步数/场景子集）→ 论文明确声明差异并解释方向性影响；
  - (c) 发现我们的配置显著偏离发表协议 → 修正配置后**只重跑 baseline-explore cross-12** 做 sanity（12 集，最小代价）。
- 产出：`docs/paper/protocol_alignment_unigoal_20260706.md`

## Gap 2：感知上限 Oracle 分析——第二优先级（零新实验，价值最高）

**审稿反对**："'SR 是感知上限'只是断言。"

**行动**（无 GPU，约半天）：
- G2.1 写 `scripts/analyze_perception_ceiling.py`：对每个失败 episode 从 step traces 统计——
  - 目标候选是否曾被检测（`target_candidate_detected` 事件 / `target_candidate_captions`）；
  - 若检测过：检测首现步数、之后的锚定尝试与结局（锚定失败 vs 到达失败 vs 识别失败）；
  - 汇总为"失败归因金字塔"：从未见过目标 / 见过但锚定失败 / 锚定成功但未达 / 到达但未判停。
- G2.2 在弱通道 cross-48 × 4 profile + 强通道全部 run 上运行，产出表格。
- 判定：
  - 若失败集中大部分"从未见过目标" → SR null 的感知上限解释坐实（表 4 入论文）；
  - 若大量"见过但失败" → 发现新的可修复层，反而是正向发现（进 Discussion 或快速修复）。
- 产出：论文表 4 + `analysis/perception_ceiling_20260706/`

## Gap 3：补齐条件矩阵（通道 × 机制消融的空格）

**审稿反对**："monitor 无用 / prefetch 有害只在弱通道验证过。"

**行动**（GPU，1–2 天，Sonnet+Haiku 便宜档）：
- G3.1 强通道 `smoothnav-no-monitor` × cross-12（12 集）——让 monitor 结论通道不变。【立即可发，GPU 空闲】
- G3.2 强通道 `smoothnav-no-prefetch` × cross-12 —— 已在跑。
- G3.3 强通道三 profile × 36 扩样 → 双通道 n=48 —— 已在跑。
- G3.4 强通道 run 上跑 `analyze_gate_binding.py` + `analyze_out_of_window_mechanism.py`（零成本），检验绑定计数与机制指标的通道不变性。
- 判定：monitor/prefetch 结论若跨通道一致 → 升级为通道不变 Claim；不一致 → 如实作为通道交互效应报告。

## Gap 4：统计处理成文

**审稿反对**："没有一个 outcome 差异做过显著性检验。"

**行动**（无 GPU，半天）：
- G4.1 写 `scripts/stats_significance.py`：配对 McNemar（SR）+ 逐集配对 bootstrap CI（SPL）+ 效应量报告；对主表全部对照运行。
- G4.2 论文表格标注规范：显著者标注，不显著者明写 "n.s."；机制指标（大效应）报告原始计数与比率。
- G4.3 方法节写入统计纪律段落（n≤15 不可分辨的回溯教训、温度混沌敏感性）。

## Gap 5：叙事保险（定位分叉预案）

**审稿反对**："创新点=工程报告。"

**行动**（写作期执行）：
- G5.1 若 Gap 2 坐实感知上限 + Gap 3 通道不变 → 维持"条件化效率定理 + 精益配置"主叙事（ICRA）。
- G5.2 若强通道 n=48 的 SPL 分离仍只靠 2–3 个成功集 → 重心切换为**控制器审计方法学**定位（matched-backbone 审计协议 + capsule/replay/14 层归因工具链开源），并行准备 RA-L 版本（无页限、利于完整审计表）。
- G5.3 无论哪个分支：贡献表述避免"我们提出了分层控制器"（不新），改为"我们首次系统量化了 LLM 导航控制器各机制的真实贡献"。

## 执行顺序与时间

| 日期 | 项 |
|---|---|
| 07/06 晚 | G3.1 发射（GPU 空闲）；G1.1–G1.2 协议对照；G2.1 脚本 |
| 07/07 | G2.2 oracle 表产出；G4.1 统计脚本；G1.3 判定 |
| 07/08 | G3 收线 → 通道不变性判定；主表定稿（双通道 n=48 + 显著性标注） |
| 07/09+ | 按 G5 分支进入写作 |

## 完成判据

- [ ] 协议对照表落档，分支判定明确
- [ ] 失败归因金字塔表产出（双通道）
- [ ] monitor/prefetch 结论有双通道数据
- [ ] 主表每个对照有显著性标注
- [ ] 叙事分支（ICRA 主投 vs 审计定位+RA-L 并行）做出决定
