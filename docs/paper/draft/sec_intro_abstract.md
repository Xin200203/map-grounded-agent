# Abstract + Introduction 草稿 v0

## Abstract

LLM-guided zero-shot object navigation systems keep accreting control machinery — periodic semantic replanning, runtime monitors, speculative plan prefetching, multi-level semantic priors — yet the individual contribution of each mechanism is rarely measured. We build a fully-instrumented, event-driven semantic replanning controller over a frozen zero-shot backbone (UniGoal) in which every mechanism is independently switchable, and every semantic assertion carries graded, gated, and traceable authority. We then conduct, to our knowledge, the most systematic matched-backbone ablation of such a controller to date: 10+ variants over episode-paired suites (48 episodes / 12 scenes), at two LLM strengths spanning two orders of magnitude in model scale, with trace-level attribution of every failure. The results are corrective on three fronts. (1) *Success rate is not controller-bound*: no variant shifts SR significantly at either LLM strength — failure attribution shows the scene graph accumulates only ~6 unique object captions per 1000-step episode, so controllers reason over a nearly empty world model; upgrading the 7B planner of the original system to a frontier-scale LLM does not move SR either. (2) *What control does buy* is control quality: event-driven scheduling doubles path efficiency on solved episodes [S: +68% SPL, paired-significant at n=48]; recovery machinery eliminates a degenerate out-of-window failure mode 14-fold — by construction rather than repair; an LLM runtime monitor contributes nothing, and speculative prefetch is actively harmful under a weak planner while merely useless under a strong one. (3) *Semantic gates behave as insurance*: individually removable without outcome damage (junk admitted by one gate is absorbed by layers below), jointly load-bearing for efficiency, and their binding rate scales inversely with planner quality. We distill these into a lean recommended configuration and release the full diagnostic toolchain — capsule replay, contract auditing, and counterfactual grounding analysis — as a template for measuring, rather than assuming, what helps in LLM navigation systems.

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
