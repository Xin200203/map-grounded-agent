# Experiments 节草稿 v0（数字已锁定；标 [S] 处等强通道 n=48 收线填入）

## 4. Experiments

### 4.1 Setup

**Task & protocol.** Text-goal navigation on HM3D v0.2 val in Habitat, following UniGoal's published protocol exactly: max T=1000 steps, success radius r=1.0 m, actions {forward, left, right, stop}. The agent receives a free-form textual description of a target object instance.

**Episode suites.** We audited scene loadability on the repaired HM3D val assets and fixed two matched suites: **cross-48** (12 distinct scenes × 4 episodes, 5 goal categories, deliberately including known hard cases) and **intact-15** (15 episodes in one scene, historical dev block). Every controller variant runs the *same* episodes; all comparisons are episode-paired.

**Backbone & channels.** All variants share a frozen UniGoal backbone (BEV mapping, scene graph, FMM executor) plus shared infrastructure fixes (which strengthen the baseline; comparisons are thus conservative). Semantic reasoning runs on two channels: **weak** = deepseek-chat (text-only; visual relation proposals disabled equally for all variants) and **strong** = Claude Sonnet 4.5 planner + Haiku 4.5 captioner (full multimodality). UniGoal's published results use LLaMA-2-7B + LLaVA-1.6-7B.

**Statistics.** SR differences are tested with exact McNemar on discordant pairs; SPL with episode-paired bootstrap CIs (10k). We report n.s. explicitly. A cautionary methodological note: at n≤15, 1–2 episode differences are unresolvable — several "advantages" in our own early development runs (and, we suspect, in the wider literature) are of exactly this size.

**Profiles.** baseline-explore (UniGoal-style periodic explore), baseline-periodic (periodic semantic replanning), smoothnav-full (event-driven + recovery + anchoring + monitor + prefetch), and subtractive ablations (no-monitor / rules-only / fixed-interval / no-prefetch) plus four semantic-gate removals and their combination.

### 4.2 Main result: outcomes are perception-bound, not controller-bound (Table 1)

**Table 1: cross-48 (episode-paired).**

| Profile | SR | SPL | vs periodic (McNemar / dSPL CI) |
|---|---|---|---|
| UniGoal published (full val, 7B models) | 0.202 | 0.114 | — reference row |
| baseline-explore | 0.125 | 0.036 | p=1.00 n.s. / +0.001 [−0.037,+0.036] n.s. |
| baseline-periodic | 0.125 | 0.035 | — |
| smoothnav-full | 0.104 | 0.042 | p=1.00 n.s. / +0.007 [−0.043,+0.063] n.s. |
| smoothnav-no-prefetch | 0.167 | 0.061 | p=0.69 n.s. / +0.026 [−0.016,+0.077] n.s. |
| [S] strong-channel rows | [S] | [S] | [S] |

intact-15: explore 0.333/0.089, periodic 0.467/0.128, full 0.400/0.122, no-prefetch 0.400/0.121 (all pairwise n.s.).

**Finding 1 (null, twice-confirmed).** No controller variant changes SR significantly on either channel; swapping the 7B planner for a frontier-scale LLM does not move SR either (weak 0.104–0.167 vs strong [S], published 7B: 0.202). Success is bounded elsewhere.

### 4.3 Why: failure-attribution pyramid (Table 2)

Per-episode classification from step traces (priority: success > visible-failed > anchored-failed > detected-no-anchor > never-detected):

| Profile (cross-48, weak) | success | visible-failed | anchored-failed | detected-no-anchor | never-detected | never/failures |
|---|---|---|---|---|---|---|
| baseline-explore | 6 | 0 | 0 | 15 | 27 | 0.64 |
| baseline-periodic | 6 | 0 | 2 | 12 | 28 | 0.67 |
| smoothnav-full | 5 | 0 | 7 | 12 | 24 | 0.56 |
| smoothnav-no-prefetch | 8 | 0 | 5 | 12 | 23 | 0.58 |
| strong periodic / full (cross-12) | 2 / 2 | 0 / 0 | 2 / 2 | 1 / 1 | 7 / 7 | 0.70 / 0.70 |

**Finding 2.** 55–70% of failures never surface a target-like detection in the entire episode — stable across controllers and channels; zero episodes fail after a confirmed visible-target lock. Caption-level root-cause analysis sharpens this: the scene graph accumulates only **~6 unique object captions per 1000-step episode**, and several *chair* episodes fail with **zero chairs ever entering the graph** — implicating detection→graph-node throughput (the detector itself fires: capsules show active semantic channels), not exploration coverage. The SR ceiling is *graph-recall-bound*: controllers reason over a nearly empty world model. Controllers do shift failures up the pyramid (never-detected ↓, anchored-failed ↑): better coverage creates perception opportunities that then stall at anchoring/reachability. This also contextualizes the published upstream TN figure (20.2 with the same pipeline).

### 4.4 What the controller does buy: control quality (Table 3)

Mechanism ladder on cross-12 (weak channel; identical LLM budget ~10 planner calls/ep):

| Variant | scheduling | recovery | monitor | SPL | out-of-window | planner calls |
|---|---|---|---|---|---|---|
| baseline-periodic | periodic | — | — | 0.042 | 43 | 10.4 |
| fixed-interval | periodic | ✓ | — | 0.037 | 7 | 8.0 |
| no-monitor | event | ✓ | — | 0.085 | 3 | 10.2 |
| rules-only | event | ✓ | rules | 0.088 | 7 | 15.5 |
| full | event | ✓ | LLM-esc | 0.087 | 3 | 10.2 |

**Finding 3 (decomposition).** Recovery machinery eliminates the out-of-window degeneracy (43→7→3); trace-level attribution shows this is *avoidance-by-construction*, not repair: the shared repair path fires 20× for periodic yet resolves once (16 deferrals never recover), while the event-driven variants generate only 3 such events at comparable grounding-attempt counts (193 vs 184). Event-driven scheduling contributes the efficiency (SPL 0.037→0.085 at equal recovery machinery). Executor override ratio drops 5× (0.0285→0.0053). The LLM monitor contributes nothing (full ≈ no-monitor); a rule monitor matches outcomes at +50% planner calls.

**Finding 4 (prefetch is a net negative).** Removing speculative prefetch yields the only variant ≥ all baselines on both metrics (Table 1) at *half* the planner calls (5.6 vs 10.2/ep): its per-episode win set is the union of the wins of full and the baselines. Mechanism: pending-strategy promotion on frontier arrival displaces working target anchors.

### 4.5 Semantic gates: defense-in-depth, priced (Table 4)

Gate-binding event counts under gate removals (cross-12, weak):

| Variant | SR | SPL | planner calls | binding evidence |
|---|---|---|---|---|
| full (all gates) | 0.167 | 0.087 | 10.2 | 33 object choices, 0 low-relevance |
| relevance gate off | 0.250 | 0.117 | **17.2** | **73 low-rel choices admitted; 44 caught by the next layer** |
| direct-exec ungated | 0.250 | 0.121 | 8.5 | anchor machinery structurally bypassed |
| target-progress off | 0.167 | 0.108 | 8.6 | anchors active 814 steps, value term muted |
| stall-decommit off | 0.167 | 0.109 | 7.9 | (chaotic trajectory divergence; see §4.1 note) |
| **all gates off** | 0.167 | **0.056** | **16.8** | loses the anchor showcase episode; oow 3→7 |

**Finding 5.** No single gate removal degrades outcomes — junk semantics admitted by one gate are absorbed by the layers below (73 admitted, 44 blocked downstream, remainder digested as anchors), at the price of +69% planner calls. Joint removal costs −36% SPL, 2.3× out-of-window, +65% calls, and the loss of the one episode where anchor exploitation shines (SPL 0.853, reproduced identically across three channels). Gates are *churn control plus defense-in-depth*, not an SR mechanism.

### 4.6 Channel interaction: machinery as insurance

Under the strong channel the pathologies the machinery guards against largely vanish: out-of-window 43→0 for the *baseline* (trace-confirmed), low-relevance proposals 73→≤3, gates nearly dormant. Conversely the controller's efficiency edge appears where semantics are good: [S] strong-channel SPL comparison — full vs periodic +68% aggregate, paired 2W/0L on jointly-solved episodes (0.853 vs 0.517; 0.226 vs 0.127) at n=12, [S] n=48 pending.

**Finding 6 (conditional value theorem).** Controller machinery cannot buy success on a perception-bound task at either semantic strength; it buys *efficiency* when semantics are strong and *degeneracy protection* when semantics are weak. The premium paid by each protective mechanism scales inversely with planner quality.

### 4.7 Qualitative

[图 4：ep228 capsule BEV 时间线——painting→tv 锚定切换帧序列 + 出窗修复对比帧；渲染器现成]

## Limitations（要点）

Single backbone, single benchmark, simulation-only; 48-episode matched suites resolve large mechanism effects but not small SR deltas (reported n.s. accordingly); default-temperature LLM sampling amplifies trajectory divergence at small n (documented; one ablation affected); detection-recall ceiling suggests the next leverage lies in perception, not control.
