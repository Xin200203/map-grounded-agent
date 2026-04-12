# SmoothNav Implementation Audit

Update Note (2026-04-09):
This audit should now be interpreted under the manuscript framing:
"online explicit-map event-driven semantic replanning agent."
It is no longer aligned with an "action smoothing" primary story.
For the unified paper-facing writeup, see:
`docs/paper/paper_submission_prep_cn_20260409.md`.

Date: 2026-04-08
Scope: code-grounded audit of current implementation state
Workspace: `/home/nebula/xxy/SmoothNav`

## 1. Audit Objective

This document records what is actually implemented in the codebase, what remains conceptual, where the code and docs disagree, and what evidence supports those conclusions.

It is intended as a discussion companion to `docs/planning/project_state_briefing_20260408.md`.

## 2. Active Runtime Path

### 2.1 Entry point

`run.sh` launches:

```bash
/home/nebula/miniconda3/envs/unigoal/bin/python -m smoothnav.main \
    --config-file configs/config_habitat.yaml \
    --mode $MODE \
    --goal_type $GOAL_TYPE \
    --num_eval $NUM_EVAL
```

Supported goals in the launcher:

- `ins-image`
- `text`

Supported modes:

- `smoothnav`
- `baseline`

Implication:

- the operational comparison is SmoothNav vs UniGoal-style baseline under the same backbone.

### 2.2 Main configuration

`base_UniGoal/configs/config_habitat.yaml` still defines the main system-level configuration.

Notable fields:

- `num_local_steps: 40`
- `turn_angle: 15`
- `llm_model: claude-sonnet-4-5-20250929`
- `llm_model_fast: claude-haiku-4-5`
- `vlm_model: claude-haiku-4-5`

At the time of this audit, the config contained hardcoded API information. That issue has since been moved to environment-variable-based configuration, but the audit finding remains historically important.

## 3. Current Architecture in Code

### 3.1 `smoothnav/main.py`

The file header explicitly states the intended architecture:

- High-Level Planner:
  sonnet, semantic strategy, sparse calls
- Low-Level Agent:
  haiku, event-driven monitor plus bias adjust
- Executor:
  unchanged from UniGoal, every step

This is consistent with the latest architecture document and inconsistent with the older executor-buffer design.

### 3.2 Strategy application

`_apply_strategy()` in `smoothnav/main.py`:

- sets current full map and full pose into the graph,
- extracts `strategy.bias_position`,
- calls `graph.get_goal(goal=bias)`,
- converts returned full-map coordinates into local-map `global_goals`.

This is a critical design choice:

- SmoothNav does not replace frontier scoring,
- SmoothNav biases UniGoal's existing frontier selection.

This is scientifically cleaner and lower risk than directly choosing frontier indices in the LLM.

### 3.3 High-level planning path

On initial planning, `main.py`:

- serializes scene graph state,
- calls `high_planner.plan(...)`,
- stores a `Strategy`,
- applies the strategy through `_apply_strategy()`.

Later replanning is triggered by:

- new room emergence while current strategy is direction-like,
- low-level `PREFETCH`,
- low-level `ESCALATE`,
- frontier reached,
- stuck condition.

Important observation:

- the code already supports `pending_strategy`,
- therefore the design does implement anticipatory planning semantics even though execution remains synchronous.

### 3.4 Low-level event trigger

The low-level agent is called only when:

- `new_nodes = graph.nodes[prev_node_count:]`
- `has_new_nodes = len(new_nodes) > 0`

This is a simple but important assumption:

- "graph growth" is treated as the event signal.

Potential issue:

- this depends on list growth rather than a robust node-level event API.
- if graph internals later change ordering or mutate in place, this trigger could become brittle.

### 3.5 Promotion and fallback logic

Current code contains several practical heuristics:

- early promotion of pending strategy when current strategy is vague and pending strategy is more specific,
- frontier-reached logic that either:
  applies pending strategy,
  reuses direction strategy,
  or requests a fresh high-level plan,
- auto-prefetch when approaching frontier under room/object strategies,
- rule-based stuck detection using consecutive low-progress steps.

This means the current system is not purely LLM-driven. It is already a hybrid LLM-plus-heuristic controller.

That is likely a strength, not a weakness.

## 4. Reused UniGoal Backbone

### 4.1 `Graph.explore()` vs `Graph.get_goal()`

UniGoal baseline uses:

- `graph.explore()`

Inside `explore()`, UniGoal:

- reasons over scene graph and goal graph,
- computes a semantic target,
- then calls `get_goal()` to turn that target into a frontier.

SmoothNav bypasses most of `explore()` and directly uses:

- `graph.get_goal(goal=bias)`

Meaning:

- SmoothNav replaces part of UniGoal's high-level semantic reasoning path,
- but deliberately keeps UniGoal's frontier extraction and FMM-based frontier scoring.

### 4.2 `UniGoal_Agent.step()`

`UniGoal_Agent.step()` still does:

- target visibility detection,
- `instance_discriminator()`,
- action generation through `get_action()` unless `override_action` is supplied.

Current SmoothNav does not pass `override_action`.

This is the key difference from the abandoned first-generation design and is one of the most important positive findings from this audit.

### 4.3 Environment

`construct_envs()` in `base_UniGoal/src/envs/__init__.py` constructs only `InstanceImageGoal_Env`.

`InstanceImageGoal_Env`:

- loads object-goal style episodes,
- supports `ins-image`,
- supports UniGoal-style text-goal descriptions,
- does not expose a VLN benchmark-specific instruction-following loop.

This proves that the current operational system is not yet a proper VLN benchmark implementation.

## 5. Planner and Low-Level Prompt Audit

### 5.1 High-level planner prompt design

Strengths:

- constrained output schema,
- explicit choice set,
- choice types map to resolvable coordinates,
- direct avoidance of free-form frontier indexing.

Good design invariants:

- every valid LLM choice should map to coordinates,
- room and object choices are grounded in existing graph nodes,
- direction choices are always resolvable.

Potential weaknesses:

- object matching is substring-based,
- room matching is substring-based,
- planner reliability depends heavily on graph room labels already being meaningful,
- ambiguous object names may resolve incorrectly if multiple similar nodes exist.

### 5.2 Low-level agent prompt design

Strengths:

- narrow decision space,
- directly tied to operational actions,
- uses semantically meaningful triggers rather than fixed intervals.

Potential weaknesses:

- decision quality depends on newly observed captions being discriminative enough,
- there is no explicit calibration or confidence model,
- `PREFETCH` vs `ESCALATE` boundary may be unstable in ambiguous rooms,
- no offline replay harness currently validates these decisions systematically.

## 6. Metrics Audit

### 6.1 What works

`tests/test_metrics.py` runs successfully under:

```bash
/home/nebula/miniconda3/envs/unigoal/bin/python tests/test_metrics.py
```

Observed result:

- all metric tests pass.

This means:

- the module is importable,
- numerical logic behaves as expected on synthetic trajectories.

### 6.2 What is conceptually inconsistent

Internal findings in `docs/archive/findings_action_analysis.md` argue:

- pause-related metrics are misleading in a turn-based discrete action space,
- high pause ratio often reflects normal turning behavior rather than poor navigation,
- smoothing FMM outputs can be actively harmful.

Yet `smoothnav/metrics.py` still computes and reports:

- `pause_count`
- `pause_duration_ratio`
- `sigma_v`
- composite `smoothness_score`

Problem:

- the code still treats these as meaningful top-line outputs,
- while the project's own analysis says they should not be interpreted naively.

This is not a bug in implementation. It is a mismatch in scientific interpretation.

## 7. Test and Validation Audit

### 7.1 Current test status

`tests/test_metrics.py`

- passes when run directly with Python.

`tests/test_planner_executor.py`

- fails immediately because it imports symbols that no longer exist:
  `Planner`, `Subgoal`, `PlanResult`, and `smoothnav.smooth_executor`.

This confirms:

- tests belong to the old generation-1 architecture,
- the current generation-2 architecture has no matching unit-test coverage.

### 7.2 Environment status

Observed during audit:

- shell did not have `pytest`,
- the `unigoal` environment also did not have `pytest`,
- direct execution via `python tests/test_*.py` was still possible.

Implication:

- ad hoc testing is possible,
- standardized test execution is not currently maintained.

## 8. Code-Doc Mismatch Inventory

### 8.1 Smooth executor removal

Latest architecture doc says:

- `smoothnav/smooth_executor.py` should be deleted.

Current state:

- active code no longer depends on it,
- stale tests still depend on it.

Interpretation:

- design migration happened in code,
- cleanup is incomplete.

### 8.2 VLN claims

Docs repeatedly describe SmoothNav as:

- zero-shot VLN,
- online scene graph plus VLN,
- a response to SpatialNav, DreamNav, CA-Nav.

Current implementation supports:

- `ins-image`
- UniGoal-style text-goal navigation

Interpretation:

- the high-level scientific aspiration is ahead of the runnable task adaptation.

### 8.3 Async language

Docs sometimes describe `PREFETCH` as asynchronous or non-blocking.

Code reality:

- planner is still called inline during the step loop,
- what is actually achieved is cached anticipatory planning, not true parallel execution.

This is acceptable, but the language should be precise.

## 9. Research Risks

### 9.1 Overclaim risk

The biggest risk in external or senior discussion is overstating current capability.

Safe claim:

- SmoothNav implements a hierarchical replanning prototype on top of UniGoal's online scene graph and FMM executor.

Unsafe claim in current state:

- SmoothNav is already a complete zero-shot VLN system with validated smoothness gains.

### 9.2 Metric-definition risk

If the project continues to emphasize `pause_ratio` or similar metrics without redefinition, senior reviewers may challenge the validity of the smoothness narrative immediately.

### 9.3 Evaluation gap

Without current end-to-end experiment outputs and updated test coverage, it is difficult to argue that the new architecture is empirically better rather than just cleaner on paper.

## 10. Strongest Positive Signals

There are several genuinely strong signs in the current codebase.

### 10.1 The team corrected a wrong path quickly

The project did not remain trapped in a flawed custom-executor design. The migration back to UniGoal's FMM executor is a good architectural correction.

### 10.2 The new role separation is principled

Semantic reasoning:

- handled by high-level and low-level LLM layers

Geometric reachability and target locking:

- kept in the proven UniGoal backbone

This is a sensible systems decomposition.

### 10.3 The low-level monitor is narrowly scoped

The low-level agent does not attempt full replanning or path generation. It only decides:

- continue,
- adjust,
- prefetch,
- escalate.

This keeps its task tractable.

## 11. Highest-Value Immediate Fixes

### 11.1 Replace stale tests

Add generation-2 tests for:

- `serialize_for_planner()`
- `build_choices_text()`
- `resolve_bias_position()`
- `HighLevelPlanner._parse()`
- `LowLevelAgent._parse()`
- `main.py` strategy promotion logic in isolated helpers

### 11.2 Clarify task scope in docs

Split wording into:

- current implemented scope,
- planned VLN extension scope.

### 11.3 Redesign the metrics section

Separate:

- diagnostic action statistics,
- primary scientific evaluation metrics.

### 11.4 Add experiment manifests

For each run, preserve:

- command,
- config hash,
- dataset split,
- model identifiers,
- output summary location.

### 11.5 Remove plaintext credentials

Move secrets to environment variables or local untracked config.

## 12. Key Questions to Resolve Before Large-Scale Experiments

Question 1:

- What is the actual target benchmark for the next milestone:
  UniGoal text-goal navigation or real VLN benchmark tasks?

Question 2:

- What is the primary claimed contribution:
  smoothness,
  LLM efficiency,
  semantic responsiveness,
  or online scene-graph replanning?

Question 3:

- Which outputs will be considered success:
  SR and SPL only,
  or also planning-call reduction,
  or new semantic-response metrics?

Question 4:

- Is the low-level agent best kept as an LLM, or can its trigger policy become partially rule-based?

Question 5:

- What specific ablation will demonstrate value over baseline UniGoal:
  fewer heavy calls,
  faster response to new evidence,
  better path efficiency,
  or all three?

## 13. Audit Conclusion

The current codebase is internally most consistent when interpreted as:

- a second-generation UniGoal-derived prototype for hierarchical semantic replanning with online scene graphs.

It is internally least consistent when interpreted as:

- a finished zero-shot VLN system with validated smoothness optimization.

The code is already substantially better than the earlier design documents in one important sense:

- it no longer makes the most damaging architectural mistake of overriding UniGoal's low-level executor.

The next major challenge is not inventing another architecture. It is aligning:

- scientific claim,
- task definition,
- evaluation metrics,
- and validation coverage.
