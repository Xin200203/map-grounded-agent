# SmoothNav Monitoring Implementation And s16 Single-episode Audit

Date: 2026-04-26.

## Observation

Implemented the post-hoc layered monitoring system as:

- `scripts/audit_trace_contracts.py`
- `tests/test_audit_trace_contracts.py`

The auditor reads completed run directories or results roots, verifies artifacts, parses step traces / planner calls / monitor calls, and emits a JSON report with these layers:

1. `suite_artifact_config`
2. `dataset_scene_episode`
3. `determinism_branch_reproducibility`
4. `perception_graph_target_evidence`
5. `planner_menu_choice`
6. `strategy_contract`
7. `grounding_frontier_value`
8. `local_map_lifecycle`
9. `controller_pending_prefetch`
10. `target_anchor_attempt`
11. `monitor_trigger_coverage`
12. `executor_low_level_control`
13. `terminal_metric_arbitration`
14. `replay_counterfactual_coverage`

It also computes cross-run branch reproducibility comparisons when multiple runs share the same episode/profile.

## Evidence

Validation:

- Local targeted test: `python3 -m unittest tests/test_audit_trace_contracts.py` -> `5 tests OK`.
- Local compile: `python3 -m py_compile scripts/audit_trace_contracts.py tests/test_audit_trace_contracts.py` -> OK.
- Remote targeted test on server 73: `/mnt/sdd/xxy/miniconda3/envs/unigoal/bin/python tests/test_audit_trace_contracts.py` -> `5 tests OK`.
- Remote full suite on server 73: `/mnt/sdd/xxy/miniconda3/envs/unigoal/bin/python -m unittest discover -s tests` -> `156 tests OK`.

Historical audit smoke:

- `s12` is classified as `controller_pending_prefetch` because the historical trace still contains the stale pending drop at step `610`.
- `s15` is classified as `determinism_branch_reproducibility` because it never covered the target branch.

## s16 online single-episode experiment

Result root:

- `results/phase2_revalidation/s16_contract_audit_ep228_20260426/suite_summary.json`

Run directory:

- `results/phase2_revalidation/s16_contract_audit_ep228_20260426/smoothnav-full/20260426/smoothnav_text_195315_7c1397e9`

Setting:

- profile: `smoothnav-full`
- episode: `228`
- `SMOOTHNAV_LLM_TEMPERATURE=0`

Aggregate outcome:

| Profile | Episode | SR | SPL | Outcome |
| --- | ---: | ---: | ---: | --- |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Generated audit artifacts:

- `results/phase2_revalidation/s16_contract_audit_ep228_20260426/contract_audit.json`
- `results/phase2_revalidation/s16_contract_audit_ep228_20260426/target_evidence_trace.jsonl`
- `results/phase2_revalidation/s10_s16_contract_audit_20260426.json`

## Observation: s16 did see target evidence, but planner did not consume it

`s16` target evidence trace:

- `first_target_step = 668`
- `target_event_count = 2`
- `first_planner_after_target = 668`
- `evidence_to_planner_latency = 0`
- `first_object_commit_after_target = null`
- `first_direct_goal_after_target = null`

Contract audit dominant layer:

- `dominant_failure_layer = planner_menu_choice`
- `dominant_failure_reason = target evidence was not consumed by planner`

Planner layer details:

- `planner_call_count = 16`
- `empty_response_count = 13`
- `fallback_triggered_count = 13`
- `target_in_prompt_after_target = true`
- `selected_target_after_target = false`

At step `668`, the planner prompt contained:

- `[object] tv`
- replanning reason: `Target-like object detected, decide whether to commit`

But the raw LLM response was empty, so deterministic parse-failure fallback selected a direction (`unexplored west` in the planner call), while the active stage remained generic direction search (`unexplored south`). The same pattern recurred at step `768`.

## Claim

The fresh failed single-episode run exposes a different first-failing layer than the captured `s12` branch:

- `s12`: target branch entered; historical first failure was stale pending promotion; downstream frontier-value semantic collapse is visible.
- `s16`: target evidence appeared, but the planner call returned empty responses and deterministic fallback prevented transition into `unexplored target:tv`.

Therefore, for the online path, the immediate blocker is now:

> `target_candidate_detected` reaches the planner, but empty LLM responses / parse-failure fallback can still override the target menu and keep the controller in generic direction search.

## Cross-run branch reproducibility audit (`s10`–`s16`)

Across seven `ep228/smoothnav-full` online runs:

- dominant layers:
  - `determinism_branch_reproducibility`: `5`
  - `controller_pending_prefetch`: `1` (`s12` historical trace)
  - `planner_menu_choice`: `1` (`s16`)
- target branch coverage rate: `1 / 7 = 0.1429`
- first target steps: `[null, null, 564, null, null, null, 668]`
- target event counts: `[0, 0, 371, 0, 0, 0, 2]`

## Hypotheses (<=3)

1. Online `ep228` has two gates before target-approach can even be tested: perception branch coverage and planner reliability. `s16` passed the first gate weakly (two target events) but failed the second gate due empty LLM responses.
2. Current deterministic parse-failure fallback is unsafe after target evidence appears; when the prompt contains `[object] tv`, fallback should preserve target evidence instead of selecting a generic direction.
3. Once planner reliability / target-preserving fallback is fixed, the next expected blocker remains the known `frontier_value_semantic_collapse` seen in `s12` captured branch.

## Decision

The monitoring system is now correctly added and verified. The next code intervention should be minimal and targeted:

- add a target-aware planner fallback: if LLM response is empty/parse-failed and the structured menu contains a target object with high relevance, return that object choice rather than deterministic direction fallback;
- keep the existing `nonlocal object -> unexplored target:<name>` conversion;
- validate with replay/unit tests first, then rerun the same `ep228` single-episode audit.
