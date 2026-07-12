# Abstract + Introduction 草稿 v0

## Abstract

LLM-guided zero-shot text-goal navigation systems keep accreting mechanism — semantic replanning, runtime monitors, speculative prefetch, commitment persistence, last-mile funnels, propose-verify perception — yet which mechanism moves success, and *where* systems actually fail, is rarely measured against ground truth. Over a frozen zero-shot backbone (UniGoal) we build a fully-instrumented, event-driven controller in which every mechanism is independently switchable and every decision is traced, then run what is, to our knowledge, the most systematic matched-backbone audit of such a system to date: episode-paired suites (48 episodes / 12 scenes) with exact McNemar / paired-bootstrap statistics and, crucially, a **ground-truth failure forensic** that measures per episode how far down the pipeline each target gets. We falsify **four successive mechanism fronts**, each targeting the layer the previous one exposed: (1) controller machinery (replanning/monitor/prefetch/gates) shifts SR at neither of two LLM strengths spanning two orders of model scale; (2) reopening the executor's severed last-mile approach funnel is net-negative, and adding VLM verification to it has zero net effect; (3) robust multi-view coordinate fusion is null on both SR and a purpose-built coordinate-error metric; (4) active target-text grounding is null. The forensic then explains *why* they all miss: the target is detected, captioned, recognized as a qualified candidate, and committed to in 34–36/40 failures — finding the target *category* is not the bottleneck. Instead, in 69/75 failures no scene-graph node is ever mapped within 1.5 m of the true goal (vs a target node near GT in 48–81% of successes, an 8–9× gap): the system commits to a category-matching object that is **not the described goal instance**, and the true goal is never observed. We trace this to a structural limit — the backbone's scene graph captions objects by category alone ('chair', 'sofa'), so it cannot represent, and therefore cannot disambiguate, instances that a text description exists precisely to distinguish. The contribution is a GT-grounded forensic methodology that replaces inference-by-ablation with direct measurement, a four-front falsification that redirects effort away from control and last-mile execution, and the localization of text-goal navigation's binding constraint to instance disambiguation under a category-only world model. We release the full diagnostic toolchain.

## 1. Introduction（段落级草稿）

**P1 — 现象与问题.** Zero-shot goal navigation has rapidly adopted LLM/VLM guidance: semantic frontier scoring, language-driven replanning, runtime monitors, hierarchical planners. Each new system adds mechanisms; ablations, when present, compare end scores on small suites. Two questions are rarely answered: *which* mechanism produces the gain, and *under what semantic-signal quality* does it hold? At the episode counts typical of these evaluations (10–15), we show even the sign of an SR difference is unstable — a methodological hazard we quantify and then avoid.

**P2 — 我们的载体系统.** We build SmoothNav, an event-driven semantic replanning controller over a frozen UniGoal backbone. It is deliberately *maximal*: six single-writer layers; event-triggered replanning; grounding-failure recovery with explicit failure codes; a target-anchor state machine; an LLM runtime monitor; speculative plan prefetch; and a four-surface "graded semantic authority" scheme in which every LLM/VLM output is a gated hypothesis, never a command. Crucially, every mechanism is independently switchable and every decision is traced to a replayable artifact.

**P3 — 测量而非主张.** Rather than claim the architecture is better, we measure what each piece does: episode-paired suites (48 episodes × 12 audited scenes + a single-scene control), two LLM channels (deepseek-chat vs Claude Sonnet 4.5 + Haiku 4.5; the original system used 7B open models), exact McNemar / paired-bootstrap statistics, and trace-level failure attribution down to individual gate-binding events.

**P4 — 三组矫正性发现.**（对应 Abstract 三点，各一句展开 + Finding 编号引用）

**P5 — 贡献列表.**
1. A fully-switchable, fully-traced event-driven semantic replanning controller with graded semantic authority over a frozen zero-shot backbone (§3).
2. The first systematic mechanism-level audit of such a controller: 10+ matched variants × 2 LLM strengths × paired statistics, with trace-level failure and gate-binding attribution (§4).
3. Corrective findings for the community: SR is graph-recall-bound, not controller-bound; monitors buy nothing; prefetch harms under weak semantics; gates are defense-in-depth priced in planner calls — distilled into a lean recommended profile (§4.2–4.6).
4. An open diagnostic toolchain (capsule replay, 14-layer contract audit, counterfactual grounding replay) enabling this style of measurement on other systems (§3.3).

**P6 — 定位句.** This is not a claim that controllers are useless: it is a map of where their value actually lies — efficiency when semantics are strong, degeneracy protection when they are weak — and a demonstration that the field's default evaluation practice cannot see this map.

## 待填

- [S] Abstract 中强通道 SPL 数字与显著性措辞（等 n=48 统计）。
- Fig.1 引用位置、相关工作引文编号。
