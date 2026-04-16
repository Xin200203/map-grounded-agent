# SmoothNav Module Refactor Status

Date: 2026-04-15

This note records the first module-level refactor pass based on
`/Users/xin/Research/project/VLN/smoothnav_module_refactor_design_doc_20260415.md`.

## Goal

The refactor moves SmoothNav away from a single large `main.py` control block and
toward an explicit layered agent:

- Layer 0: World Model
- Layer 1: Mission / Progress Manager
- Layer 2: Strategic Semantic Planner
- Layer 3: Tactical Arbiter
- Layer 4: Geometric Grounder
- Layer 5: Reactive Executor

This pass prioritizes interfaces, traceability, and behavior-preserving
migration. It does not claim benchmark convergence.

## New Contracts

The shared dataclasses and enums now live in:

- `smoothnav/types.py`

Key contracts:

- `WorldState`
- `MissionState`
- `StageGoal`
- `TacticalDecision`
- `GeometricGoal`
- `ExecutorCommand`
- `TacticalMode`
- `GeometricGoalType`

`StageGoal` intentionally keeps compatibility with the old `Strategy` shape via
the `reasoning` property and fields such as `target_region`, `bias_position`,
`explored_regions`, and `anchor_object`.

## New Modules

- `smoothnav/graph_delta.py`
  - Owns `GraphDelta`, graph-delta construction, room/object count snapshots,
    caption-change detection, and strategy-type helpers.
  - `smoothnav/controller_events.py` now re-exports `GraphDelta` for backward
    compatibility.

- `smoothnav/world_state.py`
  - Builds a per-step `WorldState`.
  - Emits compact frontier, room, object, visible-target, stuck, and graph-delta
    summaries.

- `smoothnav/mission_state.py`
  - Adds `MissionProgressManager`.
  - Tracks mission text, current stage, stage status, completed/blocked stages,
    required evidence, obtained evidence, and replan reason.

- `smoothnav/tactical_arbiter.py`
  - Adds a heuristic-first `TacticalArbiter`.
  - Produces `TacticalDecision` for initial planning, pending promotion,
    frontier reached, new evidence, recovery, and no-frontier holding.

- `smoothnav/geometric_grounder.py`
  - Wraps existing `apply_strategy()`.
  - Converts `GroundingResult` into the explicit `GeometricGoal` contract.

- `smoothnav/executor_adapter.py`
  - Wraps `UniGoal_Agent.step()`.
  - Builds `ExecutorCommand` and returns a normalized executor/adoption trace.

- `smoothnav/controller_runtime.py`
  - Builds the layered trace block used by the main loop.

## Main Loop Integration

`smoothnav/main.py` now wires the new layers into each step:

- creates `MissionProgressManager`, `TacticalArbiter`, `GeometricGrounder`, and
  `ExecutorAdapter`
- emits `world_state_summary`
- emits `mission_state_summary`
- emits `current_stage_goal` and `pending_stage_goal`
- emits `tactical_decision`
- emits `geometric_goal`
- emits `executor_command`
- routes grounding through `GeometricGrounder`
- routes executor calls through `ExecutorAdapter`
- uses `TacticalDecision` to gate the first migrated tactical branches:
  `new_room_discovered`, monitor evaluation, and stuck recovery

The previous control branches still exist for behavior preservation, but their
state is now represented through explicit layered contracts. This is the intended
Phase A/B migration shape, not the final reduced `main.py`.

## Planner Integration

`smoothnav/planner.py` keeps the existing `plan()` API and adds:

- `HighLevelPlanner.plan_stage_goal(mission_state, world_state, reason, ...)`

`controller_logic.plan_strategy()` can now call this Layer-2 API when
`MissionState` and `WorldState` are available. Existing tests and call sites that
depend on the old `Strategy` behavior remain compatible.

## What This Refactor Does Not Yet Do

- It does not fully remove tactical if/else branches from `main.py`.
- It does not rewrite `UniGoal_Agent.instance_discriminator()`.
- It does not make `full` a validated winning profile.
- It does not expand to new benchmarks.

Those are intentionally later steps after the contracts are stable.

## Verification

Targeted local tests passed:

```bash
python3 -m unittest \
  tests.test_layered_contracts \
  tests.test_controller_logic \
  tests.test_graph_delta_gen2 \
  tests.test_strategy_grounding_gen2 \
  tests.test_apply_strategy \
  tests.test_executor_adoption_gen2 \
  tests.test_tracing
```

Result:

```text
Ran 31 tests ... OK
```

Full local discovery still fails only because the system Python lacks `numpy`
for `tests/test_metrics.py`. This is the same local-environment issue observed
before this refactor.
