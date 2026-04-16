# SmoothNav MEP Refactor Status

Date: 2026-04-15

This note tracks the local code refactor against `smoothnav_mep_and_implementation_priorities_20260415.md`.

## Implemented Protocol Objects

- `TaskSpec` / `TaskConstraint`: static task skeleton with composition, terminal constraints, and candidate stop conditions.
- `WorldState`: now carries monotonic `world_epoch` through `WorldStateBuilder`.
- `TaskBelief`: dynamic task-state object with `task_epoch`, `belief_epoch`, open/completed constraints, contradictions, stop evidence, and recent evidence ids.
- `EvidenceLedger`: normalized proposition ledger with `supports`, `derived_from`, freshness, revocation, and contradiction propagation.
- `StageGoal`: keeps legacy strategy compatibility while serializing MEP fields such as `stage_id`, `task_epoch`, `belief_epoch`, `world_epoch`, `stage_epoch`, and `stage_selection_confidence`.
- `PendingStageProposal`: managed by `PendingProposalManager` with `K <= 2`, dominance/replacement, stale invalidation, and single adopted proposal enforcement.
- `TacticalDecision`: now separates legacy `mode` from MEP `steady_mode` and `transition_intent`, and includes dwell/review/fallback guards.
- `GroundingResult`: carries family, candidates, primary goal, actionability confidence, failure code, and fallback policy in addition to existing frontier diagnostics.
- `ExecutorFeedback`: returned by `ExecutorAdapter` every step with override reason, duration, detour estimate, escalation flag, and adopted goal source.

## Writer Boundaries

The writer boundary helpers live in `smoothnav/writer_guards.py`.

- `TaskSpec`: `task_spec.parse_task_spec`
- `WorldState`: `WorldStateBuilder`
- `TaskBelief` / `EvidenceLedger`: `TaskBeliefUpdater`
- `PendingStageProposal`: `PendingProposalManager`
- `TacticalDecision`: `TacticalArbiter`
- `GroundingResult`: `GeometricGrounder` / `strategy_grounding`
- `ExecutorFeedback`: `ExecutorAdapter`
- `BudgetState`: `BudgetGovernor`
- `TerminalDecision`: `TerminalArbiter`

The current implementation enforces writer ownership inside the new managers. Legacy controller paths still pass strategy objects for compatibility, but trace serialization now exposes the MEP objects as source-of-truth snapshots.

## Main Loop Integration

`smoothnav/main.py` now wires the MEP objects into the fast loop:

1. Parse `TaskSpec` at episode bootstrap.
2. Build `WorldState` with `world_epoch` after graph delta construction.
3. Update `TaskBelief` and `EvidenceLedger` from graph events and executor feedback.
4. Use `TacticalArbiter` decisions with explicit `steady_mode` and `transition_intent`.
5. Ground strategies through `GeometricGrounder`, preserving goal/task/belief/world epochs.
6. Execute through `ExecutorAdapter`, returning `ExecutorFeedback`.
7. Evaluate terminal state through `TerminalArbiter`.
8. Trace `task_epoch`, `belief_epoch`, `stage_epoch`, `mode_epoch`, `goal_epoch`, `world_epoch`, evidence ids, and pending proposal epoch.

## Terminal And Metrics

Terminal outcomes are represented by:

- `SUCCESS`
- `FAILURE_BUDGET_EXHAUSTED`
- `FAILURE_UNGROUNDED`
- `FAILURE_STUCK_PERSISTENT`
- `FAILURE_CONTRADICTION_UNRESOLVED`
- `FAILURE_NO_PROGRESS_TIMEOUT`

`control_metrics` now aggregates terminal outcome counts and reports missing epoch-trace steps.

## Current Gate Status

- Gate 1, protocol correctness: implemented at schema/manager level; runtime still keeps legacy strategy fields for compatibility.
- Gate 2, trace attribution: implemented for epoch/evidence/pending/terminal fields.
- Gate 3, executor feedback visibility: implemented through `ExecutorFeedback`; UniGoal protective logic is preserved.
- Gate 4, grounder failure separation: implemented through `failure_code`, `graph_no_goal_reason`, and fallback policy.
- Gate 5, algorithmic benefit: not claimed. The next remote experiments should validate protocol behavior before interpreting full/no-monitor benefit.

## Verification

Local verification command:

```bash
python3 -m unittest tests.test_mep_contracts tests.test_layered_contracts tests.test_controller_logic tests.test_graph_delta_gen2 tests.test_strategy_grounding_gen2 tests.test_apply_strategy tests.test_executor_adoption_gen2 tests.test_tracing tests.test_monitor_gen2 tests.test_planner_gen2 tests.test_controller_config_phase2 tests.test_control_metrics_gen2
```
