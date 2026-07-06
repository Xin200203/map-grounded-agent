# SmoothNav Project AGENTS

This file governs the repository rooted at `/Users/xin/Research/Code/research/SmoothNav`.

## Current Project Mission
- Treat the project as a **map-grounded semantic replanning system** whose current strongest paper framing is a **system / engineering claim**.
- The immediate research objective is not just to keep experiments running, but to establish whether `smoothnav-full` sustains a trustworthy advantage over the semantic baseline and key ablations.

## Current Evidence Hierarchy
Always read evidence in this order before making a new claim:
1. Completed suite aggregates (`suite_summary.json`)
2. Per-run `summary.json`
3. Step traces / planner calls / monitor calls
4. Launcher logs / stderr / environment diagnostics

Do not promote a conclusion based only on a single successful episode when a larger explicit suite exists.

## Experiment Interpretation Rules
- Label every result statement as one of:
  - `Observation`
  - `Hypothesis`
  - `Claim`
- `Claim` requires a completed comparison set or a direct matched control/ablation.
- If sample counts differ across profiles, mark the comparison as **partial**.
- If data corruption or missing assets block a validation path, record that as a **dataset blocker**, not a controller failure.

## Current State (updated 2026-07-06)
- HM3D `val` assets repaired long ago; the old data blocker is CLOSED.
- Weak-channel (DeepSeek) evidence is final: outcome null at n=48 for full-vs-periodic;
  mechanism-level large effects (out-of-window 14x, override 5x); no-prefetch is the
  best cross-scene variant; monitor contributes nothing.
- Strong-channel condition (Sonnet 4.5 planner via vapeur gateway) adjudicates whether
  outcome separation appears with better semantics.
- Statistical discipline: SR differences at n<=15 are unresolvable; use matched n=48
  pairing plus mechanism metrics; never claim from 1-2 episode deltas.

## Document Priorities
When updating documentation, keep these outputs current in priority order:
1. `docs/paper/smoothnav_icra2027_story_and_experiment_design_20260705.md` (living G-gate record)
2. `docs/paper/draft_outline_20260706.md` (paper skeleton)
3. phased execution / experiment plans under `docs/planning/`

## Non-goals For The Current Pass
- Do not draft the full English paper manuscript yet.
- Do not silently widen scope into unrelated benchmark integration unless it directly removes the current top blocker.
- Do not reframe the work as a pure leaderboard paper.

## Execution Preference
- Prefer explicit, reproducible episode suites over implicit `num_eval` slices when comparing controller profiles.
- Prefer adding tooling that makes experiment state inspectable and repeatable.
- For new validation, favor the smallest experiment that resolves the highest-leverage ambiguity.

## If You Need To Continue The Research Loop
The default loop is:
1. Analyze completed results
2. Identify the highest-leverage blocker or ambiguity
3. Design the smallest next experiment or repair
4. Run / verify / aggregate
5. Write the updated interpretation back into docs before moving on
