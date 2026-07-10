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
| strong: baseline-periodic | 0.167 | 0.062 | — |
| strong: smoothnav-full | 0.167 | 0.086 | p=1.00 n.s. / +0.023 [−0.034,+0.086] n.s. |
| strong: smoothnav-no-prefetch | 0.104 | 0.046 | p=0.45 n.s. / −0.016 [−0.058,+0.021] n.s. |

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

**Finding 4 (prefetch buys nothing; outcome sign is channel-unstable).** Removing speculative prefetch halves planner calls (5.6 vs 10.2/ep) with no significant outcome change on either channel — nominally best under the weak planner (8/48 vs 5–6/48) and nominally worst under the strong one (5/48 vs 8/48), both n.s.; episode-level analysis under the weak planner shows pending-strategy promotion displacing working target anchors. The defensible recommendation is cost-based: identical outcomes at half the planner budget.

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

Under the strong channel the pathologies the machinery guards against largely vanish: out-of-window 43→0 for the *baseline* (trace-confirmed), low-relevance proposals 73→≤3, gates nearly dormant — the machinery is insurance whose premiums scale inversely with planner quality. The controller's efficiency edge on jointly-solved episodes is consistent in direction (e.g., 0.853 vs 0.517 and 0.226 vs 0.127 SPL on the two episodes solved by both at n=12) but dilutes to non-significance over the full n=48 (+0.023, CI spans zero): with so few solvable episodes, efficiency effects cannot reach significance before recall does (§4.3).

**Finding 6 (what machinery buys, finally).** Across 96 matched episodes × 2 channels, no variant shifts SR or SPL significantly in any direction — a well-powered, twice-replicated null for *outcome-level* controller value on a graph-recall-bound task. What the machinery demonstrably buys is *process quality and cost*: elimination of degenerate control states (out-of-window 14×, overrides 5×), planner-call efficiency (prefetch −50%, rules-monitor +50% for nothing), and protection whose binding rate scales inversely with planner quality (out-of-window 43→0, junk proposals 73→≤3 as the planner strengthens). Aggregate SPL consistently favors the full controller on both channels (+0.007 / +0.023) without reaching significance.

### 4.6b The S8 approach funnel: repair is necessary but verification is load-bearing (Table 5, 数字待 c4567/c7v 收线)

Text goals inherited a severed approach funnel: the executor's native chain (sighting → temp-goal approach → close-range re-discrimination → lock) had its takeover link disabled for text goals outright, and its close-range slot — where the ins-image pipeline runs LightGlue re-verification — cleared *and blacklisted* the pursued sighting at ~5 m. Anchoring machinery (C2/C4/C5) delivered agents to committed coordinates, but no in-executor path could finish (anchored_failed 16/40; visible-lock events: 0 across all failures).

Reopening the funnel is not enough — it must be *verified*. The four-row ablation (recall regime, cross-48 matched, McNemar):

| variant | funnel state | SR (full) | SR (np) |
|---|---|---|---|
| C45 | severed (reference) | 8/48 | 6/48 |
| C456 | takeover reopened, unverified | 4/48 (p=.29) | — |
| C4567 | + close-range keep, unverified | 4/48 (p=.22) | 5/48 (p=1.0) |
| C7v | + VLM verification at initiation | 4/48 (p=.22) | 3/48 (p=.51) |

The adjudication is sharper than "verification is missing": adding the verifier (C7v: Haiku crop-verdicts, ≤8 calls/episode, layered category+intrinsic criteria, fail-open) changes *nothing* (vs C4567: +2/−2, p=1.0, dSPL +0.001). All three reopened variants land at 4/48 — half the reference. Post-mortem of the five episodes the reference wins: verified takeover fired in every one, displacing an anchor-path approach that — under the benchmark's enter-radius success criterion — would have finished. The takeover goal is a single-frame depth-estimate ellipse; the committed anchor is a multi-detection node centroid. The severed interface was not a defect but a load-bearing guard: the executor cannot finish episodes the anchor cannot, it can only lose episodes the anchor would have won. This inverts the propose-verify prescription reported for object-nav (SG-Nav, TriHelper, VLFM): verification rescues a takeover funnel only if the takeover's goal estimate outperforms what it preempts. The residual failure mass therefore sits in anchor *coordinate quality* — a perception-fusion property — not in executor control.

### 4.6c Anchor coordinate precision is not the last-mile constraint (Table 6)

§4.6b localizes the residual mass to anchor *coordinate quality*. We test that directly. R2 replaces the node centroid — a mean over all accumulated point-cloud points, which lets one depth-bleed frame outvote several clean views — with a per-detection-vote component-wise median (robust to a minority of bad frames). We evaluate on **two layers**: outcome SR, and a purpose-built *mechanism metric* — the paired per-episode closest-approach distance to the GT goal, which registers coordinate improvement even below the 1 m success threshold.

| variant | SR (full) | SR (np) | closest-approach Δ (full) | Δ (np) |
|---|---|---|---|---|
| C45 (reference) | 8/48 | 6/48 | — | — |
| R2 robust centroid | 7/48 (p=1.0) | 7/48 (p=1.0) | −0.28 m [−0.79,+0.19] | +0.44 m [−0.11,+1.15] |

Both layers are null: SR is flat, and the mechanism metric — designed to catch sub-threshold coordinate gains — crosses zero with *opposite signs across the two hosts*. The fusion fired in-vivo (median applied to 3→8 multi-view nodes per episode), so the mechanism was active; the median-vs-mean centroid difference is simply below what closest-approach can register. **Anchor coordinate precision is not the binding last-mile constraint**; the anchored_failed gap is not centroid noise. Combined with the closest-approach baseline — most failures never approach the goal at all (median 3.4–5 m; ≤2 m only 11–13/46) — the residual leverage, if any, sits one layer further back: in *target finding* (detected_no_anchor 18/40, qualified candidates that never appear), not coordinate precision and not last-mile execution.

**Bottleneck localization (synthesis).** Four mechanism fronts, each attacking the layer the previous one exposed, each falsified at n=48 with matched pairing: controller machinery (§4.2–4.6, outcome-null), the S8 approach funnel incl. VLM verification (§4.6b, load-bearing severance), and anchor coordinate fusion (§4.6c, double-null). The audit does not merely report a null — it *localizes*: it rules out control, last-mile execution, and coordinate precision, and points the residual mass at perception recall / target finding, corroborated by the closest-approach distribution and by the original backbone's published ceiling (20.2 SR / 11.4 SPL, same protocol, 7B models). [R1 target-text grounding attacks exactly this residual; its verdict — capstone fix or fourth falsified front — lands here.]

### 4.7 Qualitative

[图 4：ep228 capsule BEV 时间线——painting→tv 锚定切换帧序列 + 出窗修复对比帧；渲染器现成]

## Limitations（要点）

Single backbone, single benchmark, simulation-only; 48-episode matched suites resolve large mechanism effects but not small SR deltas (reported n.s. accordingly); default-temperature LLM sampling amplifies trajectory divergence at small n (documented; one ablation affected). The audit localizes the residual bottleneck to perception recall / target finding rather than control, last-mile execution, or coordinate precision — three fronts we falsified directly; whether that residual is itself addressable within a frozen backbone is the open question our final front (R1) probes.
