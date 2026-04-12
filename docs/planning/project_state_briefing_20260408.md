# SmoothNav Project State Briefing

Update Note (2026-04-09):
This document remains a codebase state briefing, but it should now be read under the newer manuscript framing:
"event-driven semantic replanning on online explicit maps,"
not as a "smoothness / action smoothing" paper.
For the consolidated paper-facing positioning, see:
`docs/paper/paper_submission_prep_cn_20260409.md`.

Date: 2026-04-08
Prepared for: internal technical discussion with senior scientists
Prepared from: `/home/nebula/xxy/SmoothNav`

## 1. Executive Summary

SmoothNav is a research prototype built on top of UniGoal. Its core thesis is that zero-shot navigation should not rely on a single heavy decision loop that blocks execution every fixed interval. Instead, navigation should be split into:

- a high-level semantic planner that decides which region or object to search next,
- a lightweight event-driven monitor that reacts when the scene graph meaningfully changes,
- an unchanged low-level executor based on UniGoal's FMM navigation and target-locking logic.

The project has already gone through one major conceptual pivot:

- Early direction: "smooth execution" via precomputed action buffers and a custom executor.
- Current direction: keep UniGoal's low-level FMM executor intact and improve only high-level replanning logic.

This pivot is justified by the internal action-analysis note in `docs/archive/findings_action_analysis.md` and a now-retired first-generation static analysis note that documented why override-style execution conflicted with UniGoal's native target-locking behavior.

Current status is best described as:

- the second-generation three-tier architecture is implemented at the main-loop level,
- some early-generation artifacts remain in tests and documentation,
- the project is not yet a complete VLN system despite VLN-oriented positioning in docs,
- evaluation and validation are incomplete,
- the most important open issues are now research-definition and system-integration issues rather than low-level action smoothing.

## 2. Repository Overview

Top-level structure:

- `base_UniGoal/`
  - upstream-like codebase used as the execution and scene-graph backbone
- `smoothnav/`
  - current SmoothNav implementation
- `docs/`
  - reorganized into `implementation/`, `planning/`, `paper/`, and `archive/`
- `tests/`
  - mixed-status tests; some are current, some target removed code paths
- `run.sh`
  - entrypoint for `smoothnav` vs `baseline`

Important implementation files:

- `smoothnav/main.py`
- `smoothnav/planner.py`
- `smoothnav/low_level_agent.py`
- `smoothnav/metrics.py`
- `base_UniGoal/src/graph/graph.py`
- `base_UniGoal/src/agent/unigoal/agent.py`
- `base_UniGoal/src/envs/__init__.py`
- `base_UniGoal/src/envs/instanceimagegoal_env.py`

## 3. Original Problem Statement

The project starts from a critique of UniGoal's exploration loop.

UniGoal baseline behavior:

- updates BEV map every step,
- updates online scene graph every 2 steps,
- performs semantic target selection through `graph.explore()` on a fixed interval or when near current goal,
- uses FMM for low-level navigation,
- uses `instance_discriminator()` every step for goal locking.

The internal design document identifies three baseline bottlenecks:

- decision vacuum:
  between two `explore()` calls, the agent can move many steps without reconsidering strategy even after seeing important new evidence;
- model-task mismatch:
  simple semantic common-sense judgments are repeatedly delegated to strong LLM calls;
- blocking execution:
  semantic planning is coupled to execution and can stall motion.

This framing appears in `docs/implementation/architecture_design.md` and is the conceptual foundation for SmoothNav.

## 4. Architecture Evolution

### 4.1 First-generation idea: plan-then-execute with custom executor

The first visible design direction, documented in an earlier first-generation UniGoal adaptation note that has since been retired from `docs/`, proposed:

- instruction decomposition,
- subgoal queue management,
- planner-generated waypoints,
- executor-maintained action buffer,
- explicit smoothing of action sequences,
- custom replan triggers.

This approach assumed that action-level smoothing was an important contribution and that it was safe to override UniGoal's internal action generation.

### 4.2 Failure analysis of first-generation idea

Two internal documents explain why this direction was largely abandoned.

`docs/archive/findings_action_analysis.md`:

- shows that high pause ratio mostly comes from normal turn actions in a discrete action space,
- shows that collision and oscillation were not the dominant issue in the sampled baseline runs,
- concludes that many smoothness metrics were structurally misaligned with the real navigation problem,
- explicitly warns against smoothing FMM heading outputs with EMA-like methods.

The retired first-generation static analysis note:

- identified a critical architectural conflict between custom `override_action` and UniGoal's `instance_discriminator()`,
- explained that once the target becomes visible, baseline UniGoal can reorient immediately, but buffered actions keep moving on the obsolete path,
- identified missing obstacle awareness in the old heading-only executor,
- concluded that SmoothNav should stop replacing low-level action generation and instead only control goal updates.

### 4.3 Current direction: three-tier hierarchical navigation

The current design in `docs/implementation/architecture_design.md` is:

- High-Level Planner:
  strong LLM, sparse calls, outputs semantic `Strategy`
- Low-Level Agent:
  fast LLM, event-driven, inspects newly added scene-graph nodes and decides whether to continue, adjust, prefetch, or escalate
- Executor:
  unchanged UniGoal low-level stack using `graph.get_goal()` and agent-side FMM execution

This is the architecture that is now reflected in `smoothnav/main.py`, `smoothnav/planner.py`, and `smoothnav/low_level_agent.py`.

## 5. What Is Implemented Today

### 5.1 Main loop

`smoothnav/main.py` is the current orchestration layer.

It:

- loads UniGoal config and environment,
- constructs UniGoal `BEV_Map`, `Graph`, and `UniGoal_Agent`,
- creates two LLM wrappers:
  one for high-level planning and one for fast low-level monitoring,
- maintains per-episode state:
  `current_strategy`, `pending_strategy`, `explored_regions`, node counters, no-progress counter, last position,
- runs either:
  `smoothnav` mode or `baseline` mode.

In `smoothnav` mode it performs:

- initial high-level planning at episode start,
- scene-graph updates every 2 steps,
- event-driven low-level evaluation when new nodes appear,
- strategy adjustment through `graph.get_goal(goal=bias)`,
- pending-plan prefetch and promotion,
- rule-based stuck-triggered replanning,
- low-level execution through unmodified `agent.step(agent_input)`.

This is the clearest sign that the project has already moved away from the older action-buffer design.

### 5.2 High-level planner

`smoothnav/planner.py` implements:

- `Strategy` dataclass:
  `target_region`, `bias_position`, `reasoning`, `explored_regions`, `anchor_object`
- scene-graph serialization for planning context,
- a constrained choice menu:
  objects, rooms, directions
- a prompt that asks the LLM to choose one of those choices,
- coordinate resolution from semantic selection to map bias coordinates.

Important properties:

- the planner does not pick a frontier directly,
- the planner picks a semantic target or direction,
- coordinates are translated to a bias point that is then fed into UniGoal's `graph.get_goal()`.

This is conceptually strong because it respects role separation:

- LLM handles semantic search,
- FMM handles reachability and geometry.

### 5.3 Low-level event-driven monitor

`smoothnav/low_level_agent.py` implements:

- `LowLevelAction` enum:
  `CONTINUE`, `ADJUST`, `PREFETCH`, `ESCALATE`
- `LowLevelResult` dataclass
- a prompt that compares:
  current strategy, newly observed objects, distance to current frontier, total observed nodes
- parsing and optional anchor-to-bias coordinate resolution for `ADJUST`.

Operational meaning:

- `CONTINUE`
  current strategy remains semantically consistent
- `ADJUST`
  keep same broad strategy but shift bias toward a newly informative object
- `PREFETCH`
  compute next strategy early while continuing motion
- `ESCALATE`
  current strategy is stale or contradicted by new observations

### 5.4 Smoothness metrics

`smoothnav/metrics.py` computes:

- `sigma_v`
- `sigma_omega`
- `jerk`
- `pause_count`
- `pause_duration_ratio`
- `direction_reversals`
- `smoothness_score`
- planning-call statistics

This module is self-contained and currently the most stable part of the repo from a software perspective.

### 5.5 Baseline compatibility

The project still depends heavily on UniGoal internals:

- scene graph update:
  `Graph.update_scenegraph()`
- semantic exploration primitive:
  `Graph.get_goal()`
- FMM-based low-level planner:
  inside `UniGoal_Agent.get_action()`
- target visibility handling:
  `instance_discriminator()`

This reuse is desirable. It reduces risk compared to the earlier custom-executor direction.

## 6. What Is Not Implemented Yet

### 6.1 True VLN task adaptation

Despite repeated VLN framing in docs and module docstrings, the runnable system is still tied to UniGoal-style tasks.

Current run interface:

- `smoothnav|baseline`
- `ins-image|text`

Current environment construction:

- only `InstanceImageGoal_Env` is instantiated.

This means:

- no R2R / RxR / VLN-CE episode loader,
- no VLN instruction-grounded evaluation loop,
- no official VLN metrics integration such as NE, nDTW, SDTW,
- no subinstruction progression logic in the current active codepath.

The Habitat subtree includes generic VLN-related code, but SmoothNav itself does not yet wire it up.

### 6.2 Evaluation pipeline maturity

The code writes per-episode and summary JSONs, but there were no ready-made output artifacts in the working directory at the time of inspection.

This suggests one of:

- no recent full experiment runs in this workspace,
- outputs stored elsewhere,
- implementation is still being stabilized before systematic experiments.

### 6.3 Updated test coverage

Tests are split into two groups:

- `tests/test_metrics.py`
  current and working
- `tests/test_planner_executor.py`
  stale and broken against the current codebase

This indicates that the project does not yet have a coherent validation layer for the new architecture.

## 7. Key Findings from Internal Docs

### 7.1 The project already disproved some of its initial assumptions

The strongest internal scientific insight so far may be negative rather than positive:

- naive action-level smoothing is not the right research target in this discrete navigation setting,
- pause-heavy trajectories are often not pathological,
- FMM heading changes should not be artificially smoothed,
- smoothness cannot be defined by raw "stationary steps" in a discrete turn-based environment.

This is important for senior discussion because it prevents the team from spending time optimizing the wrong objective.

### 7.2 The current contribution hypothesis has shifted

The more defensible contribution is now:

- sparse semantic replanning,
- event-driven strategy revision,
- reduced dependence on repeated heavy semantic calls,
- better alignment between semantic reasoning and geometric planning.

This is a stronger systems contribution than "we smoothed actions."

## 8. Current Technical Risks and Blockers

### 8.1 Positioning mismatch: "VLN project" vs "ObjectNav/TextNav prototype"

This is the largest scientific risk.

The docs position SmoothNav as:

- zero-shot VLN,
- online scene graph + VLN,
- a gap-filling contribution relative to SpatialNav, DreamNav, CA-Nav.

The runnable code currently implements:

- a modified UniGoal pipeline for `ins-image` and UniGoal-style `text` goals,
- not a proper VLN benchmark pipeline.

If this gap is not closed, discussion with senior scientists may quickly expose overclaiming risk.

### 8.2 The smoothness story is not yet settled

The project has correctly identified flaws in earlier smoothness metrics, but the active code still reports several of those same metrics. This creates a conceptual inconsistency:

- analysis says these metrics are structurally misleading,
- implementation still computes and logs them as headline outputs.

This should be addressed either by:

- redefining the metrics,
- or clearly separating "legacy diagnostics" from "main evaluation metrics."

### 8.3 Validation drift

The stale test file indicates architectural drift:

- code has moved to generation 2,
- tests still target generation 1.

This is a practical blocker for fast iteration because future refactors cannot be safely validated.

### 8.4 LLM system design still has open questions

Current planner and low-level prompts are reasonable, but there are unresolved issues:

- how reliable is room/object choice under sparse graph evidence,
- whether `new_nodes = graph.nodes[prev_node_count:]` is a sufficiently robust event trigger,
- whether the same fast model is suitable for all low-level semantic judgments,
- whether strategy promotion and explored-region bookkeeping are stable across edge cases,
- whether prefetch logic actually improves wall-clock or only reduces semantic delay conceptually.

### 8.5 Experimental reproducibility is weak

Observed issues:

- no local git repo initialized in the current working tree,
- no complete local experiment result bundle found during inspection,
- shell environment did not have `pytest`,
- the `unigoal` conda environment could run plain Python tests but did not have `pytest` installed.

This is normal for a research sandbox, but it lowers confidence in systematic ablation and reproducibility.

## 9. Software Hygiene Concerns

There are plaintext API credentials in local config files:

- `configs/setting.json`
- `base_UniGoal/configs/config_habitat.yaml`

This is a serious engineering hygiene problem and should be fixed before broader collaboration or repository publication.

The issue is not just security. It also makes the experiment environment tightly coupled to one local setup.

## 10. Empirical Status

What can be stated with confidence from local inspection:

- the current three-tier architecture is implemented in code;
- the repo still contains artifacts from an older executor-based design;
- the metrics module works as a standalone unit;
- the old planner/executor tests are broken against current code;
- there is no evidence in this workspace of completed recent end-to-end experiment output;
- the project remains much closer to "research prototype under active reframing" than "stable benchmarked system."

## 11. Most Useful Discussion Points for Senior Scientists

### 11.1 Contribution definition

Question:

- Is the real contribution "smooth navigation," or is it "hierarchical semantic replanning over online scene graphs"?

My assessment:

- the latter is stronger and more defensible.

### 11.2 Task definition

Question:

- Should the immediate milestone remain UniGoal-style text/object navigation, or should the project fully move to VLN-CE/R2R-style instruction-following?

My assessment:

- this must be decided early because it changes environment wiring, metrics, prompt design, and what counts as success.

### 11.3 Evaluation definition

Question:

- Which metrics should be primary once action-level pause metrics are acknowledged as misleading?

Candidate direction:

- SR / SPL / semantic replanning call count / strategy response delay / semantic efficiency diagnostics,
- plus a revised smoothness definition that does not penalize necessary turns.

### 11.4 Architecture scope

Question:

- Should the low-level monitor remain LLM-based, or should some of its decisions become rules or learned heuristics?

Reason:

- several low-level decisions may not require an LLM once graph-change features are formalized.

## 12. Recommended Near-Term Priorities

Priority 1:

- align claims with current implementation,
- explicitly state that current runnable system is built on UniGoal-style tasks, not yet full VLN.

Priority 2:

- replace stale generation-1 tests with tests for:
  `HighLevelPlanner`, `LowLevelAgent`, strategy application, prefetch promotion, and stuck-triggered replanning.

Priority 3:

- decide whether to:
  continue as a refined UniGoal extension first,
  or fully wire the system into a real VLN benchmark.

Priority 4:

- redesign the smoothness evaluation story so it is consistent with the internal findings.

Priority 5:

- remove hardcoded credentials and make experiment configuration portable.

## 13. Bottom-Line Assessment

SmoothNav is not a failed project. It is a project that has already successfully invalidated one weak design path and migrated toward a more principled systems design.

Its current strength is:

- the reframed architecture is more coherent than the original one.

Its current weakness is:

- the scientific claim, active code, tests, and evaluation protocol are not yet fully aligned.

For discussion with senior scientists, the most honest and productive framing is:

- SmoothNav is a promising second-generation prototype for hierarchical semantic replanning on top of UniGoal,
- the architecture has improved materially,
- the empirical and task-definition story now needs to catch up to that architectural change.
