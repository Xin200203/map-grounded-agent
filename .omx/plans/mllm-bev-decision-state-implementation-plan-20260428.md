# MLLM Planner BEV Decision-State Implementation Plan

Date: 2026-04-28
Scope: Convert the current planner BEV from a debug visualization into a decision-state representation that can support MLLM branch selection.

## Requirements Summary

Observation: The current BEV input is still mostly geometry + sparse debug overlays. It lacks dense semantic state, target-conditioned value, clear room spatialization status, and a strict coordinate contract.

Goal: Build a replay-first, minimally invasive pipeline where the planner receives:
1. canonical geometry state,
2. spatial semantic state,
3. target-conditioned value field,
4. branch decision table,
5. evidence-level constrained prompt contract.

Non-goals for this round:
- Do not expand to a larger online suite before replay passes.
- Do not add egocentric RGB/panorama yet.
- Do not train PONI/SemExp-style models yet.
- Do not let unlocalized room text act as strong spatial evidence.

## Acceptance Criteria

1. Existing s22b replay steps 31, 43, 98, 145 can be replayed offline and produce:
   - decision-state JSON,
   - multi-panel planner BEV image,
   - branch decision table,
   - evaluator scorecard.
2. The planner prompt exposes only one canonical coordinate frame and one branch direction definition.
3. Every branch has evidence level: direct / anchor / room_prior / geometry_only.
4. Target-conditioned value map exists even when target is not directly visible; source breakdown must be explicit.
5. Step 31 must be classified as geometry_only or weak-prior, not falsely semantic-grounded.
6. Step 145 must expose whether target-related choice is direct, anchor, or prior-only.
7. MLLM output cannot hard-gate a branch unless evidence level supports it.

## Phase 0 — Replay scorecard before behavior changes

Purpose: make the current failure measurable without another full online episode.

Files likely touched:
- `scripts/evaluate_semantic_bev_branch_decisions.py` or new `scripts/evaluate_planner_bev_decision_state.py`
- `scripts/replay_semantic_bev_capsules.py`

Implementation:
- Add a replay evaluator that reads existing capsules and reports:
  - semantic coverage,
  - target heat/value coverage,
  - room localized vs unlocalized counts,
  - branch table completeness,
  - contradictory coordinate/direction fields,
  - evidence-level consistency,
  - final branch vs value-top agreement.

Verification:
- Run evaluator on steps 31/43/98/145.
- Expected current result should clearly expose: sparse semantics, target heat absent/weak, unlocalized room priors, and branch-value mismatch risk.

## Phase 1 — Canonical coordinate contract

Purpose: prevent the MLLM from seeing mixed raw/repaired/local/full-map coordinates.

Files likely touched:
- `smoothnav/semantic_bev.py`
- `smoothnav/frontier_branching.py`
- `smoothnav/bev_frontier_planner.py`
- tests under `tests/`

Implementation:
- Define one canonical planner frame:
  - full-map pixel frame,
  - canonical robot pose,
  - canonical branch center,
  - canonical direction computed from robot pose to branch center.
- Move raw/repaired/local coordinate warnings into debug-only fields.
- Ensure MLLM prompt and branch table show only canonical coordinates/directions.

Acceptance:
- No branch prompt line contains conflicting direction or coordinate variants.
- Coordinate repair warnings may exist but are not mixed into decision fields.

## Phase 2 — Decision-state schema v2

Purpose: separate state representation from visualization.

Files likely touched:
- `smoothnav/semantic_bev.py`
- `smoothnav/task_frame_capsule.py`
- `scripts/replay_semantic_bev_capsules.py`

Implementation:
- Add a versioned decision-state object containing:
  - `geometry_state`,
  - `semantic_state`,
  - `room_state`,
  - `target_value_state`,
  - `branch_decision_table`,
  - `quality_flags`.
- Save this object beside replay images in capsules.

Acceptance:
- Each replay step has a machine-readable JSON decision state.
- The renderer consumes this schema instead of ad hoc side-panel fields.

## Phase 3 — Room spatialization split

Purpose: stop treating room names as spatial evidence unless they are localized.

Files likely touched:
- `smoothnav/semantic_bev.py`
- `smoothnav/strategy_grounding.py`
- `smoothnav/frontier_scoring.py`

Implementation:
- Split rooms into:
  - `localized_room_regions`: has center/bbox/mask/entry support,
  - `unlocalized_room_priors`: text/commonsense hypotheses only.
- Draw only localized rooms as spatial regions.
- Keep unlocalized rooms in side table / JSON weak prior.

Acceptance:
- `living room unlocalized` is not drawn as a region.
- Branch evidence can say `room_prior`, but not `direct` or `anchor`, when only unlocalized room text is available.

## Phase 4 — Target-conditioned value map v0

Purpose: replace simple target heat overlay with a decision field.

Files likely touched:
- new `smoothnav/target_value_map.py` or `smoothnav/semantic_bev.py` initially
- `smoothnav/frontier_scoring.py`
- `smoothnav/semantic_bev.py`
- tests under `tests/`

Implementation:
- Compute dense value field from:
  - direct target evidence,
  - anchor objects,
  - localized room/function-area priors,
  - frontier novelty / unexplored gain,
  - reachability/actionability,
  - revisit/dead-end penalties.
- Produce per-branch value breakdown:
  - `target_value_score`,
  - `direct_score`,
  - `anchor_score`,
  - `room_prior_score`,
  - `info_gain_score`,
  - `revisit_penalty`,
  - `evidence_level`.

Acceptance:
- If target is not visible, target value may still exist as search utility, but source is not labeled direct.
- Step 31 should not produce fake target evidence.
- Synthetic direct-target test gives `evidence_level=direct`.

## Phase 5 — Multi-panel planner BEV renderer

Purpose: make the image a decision-state visualization, not one overloaded debug map.

Files likely touched:
- `smoothnav/semantic_bev.py` or new `smoothnav/decision_bev_renderer.py`
- `scripts/replay_semantic_bev_capsules.py`

Implementation:
- Generate four panels:
  - A Geometry: obstacle/free/explored/unknown, pose, trajectory, frontier candidates.
  - B Semantic: semantic masks/objects/localized room regions/confidence.
  - C Target Value: target-conditioned value field and high-value zones.
  - D Branch Table: ID, direction, value, evidence, support, info gain, loop/dead-end risk.
- Keep full debug image separately.

Acceptance:
- Branch markers no longer dominate the map.
- Planner image clearly separates geometry, semantics, target value, and branch comparison.

## Phase 6 — Planner prompt contract v2

Purpose: force the MLLM to reason over evidence levels and branch table, not vague map labels.

Files likely touched:
- `smoothnav/bev_frontier_planner.py`
- `smoothnav/mllm_frontier_contract.py`
- tests under `tests/`

Implementation:
- Prompt inputs:
  - target/category,
  - canonical pose/frame note,
  - compact branch decision table,
  - evidence-level rules,
  - multi-panel BEV image.
- Output schema:
  - selected branch ID,
  - evidence_level,
  - target_visible boolean,
  - supporting evidence IDs,
  - risk flags,
  - short rationale.
- Hard rule: if no direct target/anchor/localized room support, output must not claim direct semantic grounding.

Acceptance:
- Schema validation rejects missing evidence level.
- Step 31 replay prompt leads to geometry-only or weak-prior reasoning.

## Phase 7 — MLLM-to-scorer coupling and gating

Purpose: make planner output and downstream grounding use the same evidence semantics.

Files likely touched:
- `smoothnav/strategy_grounding.py`
- `smoothnav/frontier_scoring.py`
- `smoothnav/bev_frontier_planner.py`

Implementation:
- Evidence-level to gate mapping:
  - direct: hard or strong gate,
  - anchor: strong soft prior,
  - room_prior: moderate/weak prior,
  - geometry_only: tie-break only.
- Downgrade gating if quality flags show coordinate warning, empty semantic map, or stale value map.

Acceptance:
- MLLM cannot force a low-value branch through a hard gate when evidence is geometry_only.
- Evaluator flags any mismatch between evidence level and applied gate strength.

## Phase 8 — Verification ladder

Order:
1. Static compile/import checks.
2. Unit tests for schema, coordinate contract, target value scoring, prompt validation.
3. Offline capsule replay for steps 31/43/98/145.
4. Human inspection of generated multi-panel BEV.
5. Only then run one previously failed online scene, not a full suite.

Suggested commands:
- `python -m py_compile smoothnav/semantic_bev.py smoothnav/bev_frontier_planner.py smoothnav/frontier_branching.py smoothnav/frontier_scoring.py smoothnav/strategy_grounding.py`
- `python -m pytest tests/test_semantic_bev*.py tests/test_bev_frontier*.py tests/test_frontier_scoring*.py`
- `python scripts/replay_semantic_bev_capsules.py --steps 31 43 98 145 ...`
- `python scripts/evaluate_planner_bev_decision_state.py ...`

## Risk Register

1. Semantic map remains sparse after renderer changes.
   - Mitigation: evaluator must report semantic coverage honestly; do not hide sparse semantics with prettier colors.
2. Target value map becomes over-handcrafted.
   - Mitigation: keep source breakdown and replay scorecard; compare branch value with actual downstream behavior.
3. Coordinate cleanup breaks existing downstream assumptions.
   - Mitigation: schema versioning and debug-only raw fields; replay before online.
4. Prompt becomes too long.
   - Mitigation: branch table compact form; image carries spatial fields, text carries only decision constraints.
5. MLLM overrules scorer unpredictably.
   - Mitigation: evidence-level gating and evaluator mismatch checks.

## Recommended execution order

Implement in this strict order:
1. Phase 0 evaluator,
2. Phase 1 coordinate contract,
3. Phase 2 decision-state schema,
4. Phase 3 room split,
5. Phase 4 target value map,
6. Phase 5 multi-panel renderer,
7. Phase 6 prompt contract,
8. Phase 7 gating,
9. Phase 8 replay and single-scene online verification.
