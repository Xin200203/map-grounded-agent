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

## Current Hard Blocker
- Cross-scene validation is currently blocked by corrupted HM3D `val` assets on the experiment server.
- Treat HM3D data integrity as a first-class research blocker for top-tier submission readiness.
- Do not declare the paper submission-ready while multi-scene validation remains blocked by broken assets.

## Document Priorities
When updating documentation, keep these outputs current in priority order:
1. `docs/implementation/episode296_grounding_and_graph_growth_diagnosis_20260421.md`
2. paper-gap assessment documents under `docs/paper/`
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
