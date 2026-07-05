# Episode 296 Grounding and Graph Growth Diagnosis (2026-04-21)

This note captures the first detailed diagnosis of the new blocker exposed while extending `B2 canary` from episode `286` to episode `296` on server 73.

Run under inspection:

- partial run id: `smoothnav_text_123510_8b32b23d`
- results root: `results/phase2_revalidation/s1_canary_intact_scene/20260421/`
- profile at failure time: `baseline-periodic`
- episode id: `296`
- scene: `00869-MHPLjHsuG27`
- target category: `plant`

## 1. Problem Definition

The blocker is not an infra failure. The pipeline starts correctly, reaches the episode, and keeps making LLM calls. The new issue is a runtime control/perception failure that has two coupled symptoms:

1. `grounding` repeatedly fails with `failure_code = out_of_local_window`.
2. graph relation maintenance becomes progressively slower because the object graph keeps growing and relation proposal workload grows with it.

These two symptoms appear in the same late-episode window and reinforce each other:

- the controller keeps choosing an object-centric strategy (`object: plant`);
- grounding keeps selecting a full-map frontier that projects outside the current local map;
- the local map is not moved to absorb that goal;
- the agent continues observing and adding objects/relations instead of making decisive progress;
- relation proposal cost rises because the graph is larger and new edges are still being proposed.

## 2. Concrete Evidence

### 2.1 The run is healthy before the failure

The run successfully:

- initializes Habitat / simulator
- loads the intact HM3D scene
- initializes detectron2 and segmentation
- enters `episode:296, cat_name:plant`
- starts `SmoothNav [smoothnav] starting, 1 episodes`

Therefore this is not a startup, API, or dataset-entry failure.

### 2.2 Grounding failure is explicit at step 320

In the partial trace, step `320` shows:

- `planner_called = true`
- `planner_reasons = ["fixed_interval_refresh", "grounding_failure_replan"]`
- `current_strategy.target_region = "object: plant"`
- `goal_before = [30, 82]`
- `goal_after = [30, 82]`
- `grounding_failure_reason = "out_of_local_window"`
- `graph_node_count = 31`
- `new_node_count = 0`

The corresponding `grounding_events` show the same full-map frontier twice:

- `selected_frontier = [327, 304]`
- `projected_goal = [-3, 13]`
- `failure_code = "out_of_local_window"`
- `fallback_policy = "hold_or_move_local_map"`

This means the controller has already identified a semantically plausible target frontier, but the projected local-map coordinate is outside the active local window, so the global goal is not updated.

### 2.3 The same frontier is reused, but the local map is not recentered

The same step-320 trace shows:

- `selected_frontier_same_as_prev = true`
- `consecutive_grounding_noops = 2`
- `same_frontier_reuse_count = 4`

So this is not just a one-off projection miss. The system is repeatedly selecting the same full-map target while staying in a local window that cannot represent it.

### 2.4 Graph growth is real and monotonic

The partial `step_traces` show `graph_node_count` growth over time:

- near start: `0`
- around early semantic grounding: `2`
- later: `7`
- then: `11`
- then: `17`
- then: `26`
- near failure window: `29`
- partial stop: `32`

The strategy also shifts from room-like search to object-specific anchors:

- `bedroom`
- `object: cabinet`
- `object: plant`

This means the graph is not staying compact after the target context becomes narrow.

### 2.5 Runtime symptoms confirm workload blow-up

The live rollout did not crash, but became clearly inefficient:

- by step `250`, the run was still ongoing with `H=10`
- by step `300`, the run was still ongoing with `H=11`
- relation discrimination counts in stdout repeatedly climbed into large batches, including `24`, `25`, `26`, `29`, `30`, `31`, and even larger spikes earlier in the episode

This is enough to classify the issue as a runtime scalability blocker rather than a harmless slowdown.

## 3. Code-Level Root Cause Candidates

### 3.1 Primary root cause: `out_of_local_window` is classified but not operationally handled

Relevant code:

- `smoothnav/strategy_grounding.py`
- `smoothnav/controller_logic.py`
- `base_UniGoal/src/map/bev_mapping.py`
- `smoothnav/tactical_arbiter.py`

`apply_strategy()` already returns a structured failure:

- `reason = "out_of_local_window"`
- `failure_code = "out_of_local_window"`
- `fallback_policy = "hold_or_move_local_map"`

However, the runtime does not actually execute a "move local map then retry" path when this happens.

`controller_logic.handle_grounding_failure()` only checks:

- consecutive grounding no-ops
- same-frontier reuse count

Then it triggers another planner call plus another `apply_strategy()`. It does **not**:

- shift the local map window
- hold the current strategy until the window catches up
- apply a frontier-local fallback

Also, `TacticalArbiter.post_grounding_patch()` contains failure-aware logic, but it is not wired into `main.py`. So the architecture already has the notion of a grounding patch, but the running loop does not consume it.

### 3.2 Secondary root cause: edge construction is dense, but pruning is absent

Relevant code:

- `base_UniGoal/src/graph/graph.py::update_edge`

Current behavior:

1. every newly registered node is connected to all old nodes and all other new nodes
2. all unresolved edges are collected
3. one LLM call proposes a relation for every unresolved edge
4. there is no geometric or semantic pruning before proposal generation

This is effectively dense object-object connection with no explicit edge budget.

The code comments and runtime behavior show:

- no same-room filter
- no visibility/obstruction gate
- no top-k nearest-neighbor restriction
- no hard cap on unresolved edge proposals per update

This makes runtime cost highly sensitive to graph size.

### 3.3 Upstream contributing cause: object deduplication is not strong enough in this episode

Relevant code:

- `base_UniGoal/src/graph/graph.py::mapping3d`
- `base_UniGoal/src/graph/graph.py::update_node`
- `base_UniGoal/src/graph/utils/mapping.py::merge_detections_to_objects`

The mapping path uses overlap-based matching and merges detections into existing objects, but the observed graph growth in episode 296 suggests that repeated views of the same semantic context are still producing too many distinct object nodes.

This does not appear to be the only problem, but it amplifies the relation workload significantly.

## 4. What Existing Work Does Differently

### 4.1 PONI: keep long-term goals on frontiers, not arbitrary map cells

PONI explicitly defines its potentials only on frontiers, arguing that this is sufficient because any path to unexplored locations must pass through a frontier first.

Source:

- [PONI (CVPR 2022)](https://www.cs.utexas.edu/~grauman/papers/poni-cvpr2022.pdf)

Relevant passage:

- lines around `199-204`: potentials are defined only at frontiers
- lines around `226-229`: the long-term goal is sampled from those frontier-based potentials

Why this matters here:

- our failure appears when an object-anchored semantic target is grounded into a full-map location that cannot be represented in the current local window
- frontier-only long-term goals reduce this class of failure because they preserve geometric actionability by construction

### 4.2 SemExp: replan continuously, but local policy always acts on a map-valid long-term goal

SemExp uses a semantic map and a deterministic local planner, but the long-term goal remains part of the map planning interface, and the local policy replans every step on the map.

Source:

- [Semantic Exploration (CVPR 2020)](https://devendrachaplot.github.io/papers/semantic-exploration.pdf)

Relevant passage:

- lines around `150-168`: semantic policy chooses long-term goals from the map; local policy replans every step to the long-term goal

Why this matters here:

- SemExp avoids the exact "goal exists but cannot be represented in current local window" ambiguity because the planner-local-policy interface is consistently map-centric

### 4.3 SG-Nav: dense edge proposals are followed by explicit pruning

SG-Nav is especially relevant because it also builds an online scene graph for object navigation.

Source:

- [SG-Nav (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/098491b37deebbe6c007e69815729e09-Paper-Conference.pdf)

Relevant passages:

- lines `255-263`: newly registered nodes are densely connected to previous nodes in one-shot
- lines `291-298`: those edges are then pruned
  - short-range edges are verified by VLM
  - long-range edges are kept only if unobstructed and in the same room
- lines `517-520`: group nodes reduce redundant information and graph complexity

Why this matters here:

- our current graph code uses the dense-connection part
- but does not implement the pruning part
- and does not introduce a group-level compression stage before heavy reasoning

So our current implementation is exposed to exactly the kind of edge growth that SG-Nav explicitly guards against.

### 4.4 SGM / PEANUT: reason over incomplete maps, but preserve global semantic context

Both SGM and PEANUT emphasize that object search should be performed over incomplete but accumulating semantic maps, rather than repeatedly collapsing to immediate local cues.

Sources:

- [SGM / Imagine Before Go (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Imagine_Before_Go_Self-Supervised_Generative_Map_for_Object_Goal_Navigation_CVPR_2024_paper.pdf)
- [PEANUT (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_PEANUT_Predicting_and_Navigating_to_Unseen_Targets_ICCV_2023_paper.pdf)

Relevant passages:

- SGM lines `104-113`: maintain a local semantic map and infer target location from incomplete context
- PEANUT lines `64-65`: use an incomplete global semantic map to predict a target probability map and then select long-term goals

Why this matters here:

- our current failure arises after the system already has strong semantic confidence in `object: plant`
- but the transition from semantic confidence to actionability is not stabilized by a map-level fallback

## 5. Refined Diagnosis

The new blocker is best understood as a two-stage failure:

### Stage A: semantic grounding becomes non-actionable

At late steps in episode 296:

- the planner locks onto `object: plant`
- `graph.get_goal()` finds a semantically plausible full-map frontier
- that frontier projects to `[-3, 13]` in the current local map
- `apply_strategy()` returns `out_of_local_window`
- the runtime does not move the local map window

So the system does not fail because semantic reasoning is absent. It fails because semantic reasoning is not translated into an actionably represented local goal.

### Stage B: the system keeps perceiving and graphing while not making geometric progress

Since the controller does not resolve the local-window mismatch:

- the agent keeps observing
- object nodes continue to accumulate
- unresolved object-object edges continue to be proposed
- runtime per step worsens

Thus graph growth becomes a secondary performance failure caused by the first liveness failure.

## 6. Most Likely Root Cause Ordering

### P0: runtime does not operationalize `hold_or_move_local_map`

This is the highest-confidence primary fault.

Symptoms it explains:

- repeated `out_of_local_window`
- same frontier reused
- planner re-called without goal change
- goal stays unchanged even when the target object is semantically obvious

### P1: relation pipeline has no pruning budget

This is the highest-confidence scalability fault.

Symptoms it explains:

- very large relation batches late in the episode
- strong dependence of runtime on graph size

### P2: object merging is too permissive / too weak in this episode

This is a contributing fault.

Symptoms it explains:

- node count rising from `17 -> 26 -> 29 -> 32` after the search has already narrowed to one target region

## 7. Next-Round Plan

The next round should not jump directly back into large experiments. It should first fix liveness, then fix graph scalability, then re-run canaries.

### Phase 1: fix `out_of_local_window` handling

Goal:

- when grounding selects a valid full-map target that falls just outside the current local map, do not treat it as a generic replan failure

Actions:

1. Add an explicit `out_of_local_window` branch in the runtime loop.
2. If `failure_code == out_of_local_window` and `primary_goal.full_map_coord` exists:
   - move / recenter the local map window via `bev_map.move_local_map()`
   - recompute local projection
   - retry grounding once before planner re-entry
3. If the retried projection is still invalid:
   - keep the current semantic strategy
   - mark the episode as `grounding_deferred`
   - do not trigger immediate planner churn
4. Wire `tactical_arbiter.post_grounding_patch()` into `main.py`, or remove the dead abstraction and implement equivalent runtime logic directly.

Validation target:

- step-320-like cases no longer yield consecutive `out_of_local_window` with unchanged goal

### Phase 2: cap relation growth

Goal:

- prevent relation reasoning cost from scaling with all object pairs

Actions:

1. Add pre-LLM edge pruning in `graph.update_edge()`:
   - same-room only for long-range object-object relations
   - optionally top-k nearest neighbors by Euclidean center distance
   - skip pairs farther than a configured threshold
2. Add a hard budget:
   - `max_new_relation_pairs_per_update`
   - overflow pairs are deferred, not processed immediately
3. Record runtime metrics:
   - `relation_candidate_count`
   - `relation_pairs_pruned`
   - `relation_pairs_processed`

Validation target:

- relation batch sizes stop growing with total node count

### Phase 3: strengthen object dedup / graph compaction

Goal:

- reduce node accumulation in semantically stable regions

Actions:

1. Audit overlap-based matching thresholds in `mapping.py`
2. Add trace fields:
   - `objects_pre_merge`
   - `objects_post_merge`
   - `new_object_count_this_step`
3. If needed, tighten merge thresholds or add caption-aware merge preference for repeated target-context objects

Validation target:

- graph node count does not continue rising sharply after the agent has already narrowed into one room/object cluster

### Phase 4: rerun canaries in this order

1. Re-run `episode 296`, `baseline-periodic`
2. If it completes cleanly, run:
   - `smoothnav-no-monitor`, `episode 296`
   - `smoothnav-full`, `episode 296`
3. Only after `B2` is fully closed should `B3 dev5` resume

## 8. Immediate Recommendation

Do not resume `dev5` yet.

The highest-leverage next move is:

- implement `out_of_local_window` handling first
- then add edge-budget pruning
- then re-run `episode 296` as the stress-test canary

This is the smallest change set that directly targets the new blocker exposed by the current experiments.

## 9. Follow-up Implementation Passes (Executed)

After the first diagnosis, the codebase was updated in multiple passes before returning to experiments.

### 9.1 Pass A: operationalize `out_of_local_window`

Implemented in:

- [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py)
- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py)
- [tactical_arbiter.py](/Users/xin/Code/research/SmoothNav/smoothnav/tactical_arbiter.py)
- [controller_state.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_state.py)

Behavioral change:

- `out_of_local_window` is no longer treated as a generic grounding noop
- runtime now:
  - moves the local map once
  - retries grounding
  - if still invalid, marks the state as deferred instead of immediately re-entering planner churn

Observed effect:

- the old step-320 pattern no longer causes repeated fatal stalling
- in later reruns, `out_of_local_window` appears transiently and then recovers into valid projections

### 9.2 Pass B: relation pruning and runtime tracing

Implemented in:

- [relation_pruning.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/relation_pruning.py)
- [graph.py](/Users/xin/Code/research/SmoothNav/base_UniGoal/src/graph/graph.py)
- [main.py](/Users/xin/Code/research/SmoothNav/smoothnav/main.py)

Behavioral change:

- relation pruning now happens before LLM prompting
- runtime traces now include:
  - `mapping_debug`
  - `relation_debug`

Observed effect:

- relation proposal batch sizes stop exploding into the very large counts seen in the original partial run
- however, this alone did not fully remove late-episode slowdown

### 9.3 Pass C: direct object-goal grounding

Implemented in:

- [strategy_grounding.py](/Users/xin/Code/research/SmoothNav/smoothnav/strategy_grounding.py)

Behavioral change:

- when strategy is `object: ...`, grounding first tries to match directly against existing graph object nodes
- only falls back to frontier-style graph grounding when no suitable object node is available

Observed effect:

- `selected_frontier` is no longer the only semantic-to-geometric path for object-centric strategies

### 9.4 Pass D: suppress false grounding-failure replans for stable goals

Implemented in:

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)

Behavioral change:

- `same_goal_as_prev` no longer counts as generic grounding failure
- this prevents one class of unnecessary planner churn

Observed effect:

- removed a large amount of pointless replanning where goal stayed stable and locally valid

### 9.5 Pass E: suppress semantic stuck replans on stable object targets

Implemented in:

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)

Behavioral change:

- when current target is already a concrete object target, the generic semantic `stuck_replan` path is suppressed

Observed effect:

- removed one major source of high-level oscillation, but did not fully solve late-stage control churn

### 9.6 Pass F: object-target frontier handling no longer treated as direction reuse

Implemented in:

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)

Behavioral change:

- `frontier_reached` no longer routes object targets through the direction reuse / direction-replan path

Observed effect:

- object targets stopped getting repeatedly interpreted as direction-like strategies

### 9.7 Pass G: stagnant object-target escape

Implemented in:

- [controller_logic.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_logic.py)

Behavioral change:

- if an object target holds the same goal too long without progress, it is marked as `stagnant`
- controller is then allowed to choose a new target region instead of orbiting the same anchor indefinitely

Observed effect:

- in cheap smoke reruns, the system successfully switched from repeatedly revisiting `object: cabinet` to new exploratory alternatives such as `unexplored east`

## 10. Cheap-Smoke Revalidation Results

To reduce token cost while testing pipeline liveness, a runtime-only config was used that swaps:

- `llm_model -> claude-haiku-4-5-20251001`
- `llm_model_fast -> claude-haiku-4-5-20251001`
- `vlm_model -> claude-haiku-4-5-20251001`

The goal of these runs was not final quality, but system-level liveness after each patch pass.

### 10.1 `smoothnav-no-monitor` cheap smoke succeeded

Result:

- run id: `smoothnav_text_174400_f7452772`
- summary: [summary.json](/mnt/sdd/xxy/SmoothNav/results/phase2_revalidation/s1_canary_intact_scene/20260421/smoothnav_text_174400_f7452772/summary.json)

Key metrics:

- `controller_profile = smoothnav-no-monitor`
- `SR = 1.0`
- `SPL = 0.15034257918122163`
- `avg_high_level_calls = 12.0`
- `strategy_switch_count = 7.0`
- `goal_update_delay_steps = 6.285714285714286`
- `grounding_noop_rate = 0.6521739130434783`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 4, 'same_goal_as_prev': 11}`
- `pending_created_count = 3`
- `pending_promoted_count = 1`
- `executor_override_ratio = 0.0`
- `terminal_outcome_counts = {'FAILURE_BUDGET_EXHAUSTED': 66, 'SUCCESS': 1}`
- `missing_epoch_trace_steps = 0`

Interpretation:

- the pipeline is no longer blocked in `episode 296`
- at least one realistic object-heavy canary now completes successfully end-to-end

### 10.2 `smoothnav-full` cheap smoke still lags behind

Current best completed reference:

- run id: `smoothnav_text_175019_ca4e6d93`
- summary: [summary.json](/mnt/sdd/xxy/SmoothNav/results/phase2_revalidation/s1_canary_intact_scene/20260421/smoothnav_text_175019_ca4e6d93/summary.json)

Key metrics from the completed full run:

- `controller_profile = smoothnav-full`
- `SR = 0.0`
- `SPL = 0.0`
- `avg_high_level_calls = 22.0`
- `avg_low_level_calls = 17.0`
- `strategy_switch_count = 14.0`
- `goal_update_delay_steps = 17.846153846153847`
- `grounding_noop_rate = 0.6744186046511628`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 4, 'same_goal_as_prev': 17, 'out_of_local_window': 8}`
- `pending_created_count = 8`
- `pending_promoted_count = 5`
- `executor_override_ratio = 0.13043478260869565`
- `terminal_outcome_counts = {'FAILURE_BUDGET_EXHAUSTED': 64, 'FAILURE_NO_PROGRESS_TIMEOUT': 1}`
- `missing_epoch_trace_steps = 0`

Interpretation:

- `full` is no longer blocked by infra, but it still has method-level control issues
- compared to `no-monitor`, it still triggers more control flow and produces worse outcomes

### 10.3 `smoothnav-full` later produced a successful smoke run

After the additional controller/planner fixes, `full` also produced a positive cheap-smoke result.

Result:

- run id: `smoothnav_text_185649_c029d27f`
- summary: [summary.json](/mnt/sdd/xxy/SmoothNav/results/phase2_revalidation/s1_canary_intact_scene/20260421/smoothnav_text_185649_c029d27f/summary.json)

Key metrics:

- `controller_profile = smoothnav-full`
- `SR = 1.0`
- `SPL = 0.7406789288874713`
- `avg_high_level_calls = 10.0`
- `avg_low_level_calls = 0.0`
- `strategy_switch_count = 6.0`
- `goal_update_delay_steps = 7.333333333333333`
- `grounding_noop_rate = 0.7058823529411765`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 4, 'same_goal_as_prev': 8}`
- `pending_created_count = 2`
- `pending_promoted_count = 1`
- `executor_override_ratio = 0.0`
- `terminal_outcome_counts = {'FAILURE_BUDGET_EXHAUSTED': 46, 'SUCCESS': 1}`
- `missing_epoch_trace_steps = 0`

Interpretation:

- `full` is now demonstrably capable of completing this episode successfully
- the remaining question is not “can full work?”, but “which control subpath still makes it unstable in some runs?”

### 10.4 Formal-configuration `smoothnav-full` eventually produced a positive run

After the additional stuck/object-choice/executor patches, `full` also produced a positive result under the normal configuration:

- run id: `smoothnav_text_211605_55ab8dda`
- summary: [summary.json](/mnt/sdd/xxy/SmoothNav/results/phase2_revalidation/s1_canary_intact_scene/20260421/smoothnav_text_211605_55ab8dda/summary.json)

Key metrics:

- `controller_profile = smoothnav-full`
- `SR = 1.0`
- `SPL = 0.2157720952950592`
- `avg_high_level_calls = 14.0`
- `avg_low_level_calls = 0.0`
- `strategy_switch_count = 8.0`
- `goal_update_delay_steps = 10.272727272727273`
- `grounding_noop_rate = 0.6`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 2, 'same_goal_as_prev': 16}`
- `pending_created_count = 4`
- `pending_promoted_count = 2`
- `executor_override_ratio = 0.019455252918287938`
- `terminal_outcome_counts = {'FAILURE_BUDGET_EXHAUSTED': 63, 'SUCCESS': 1}`
- `missing_epoch_trace_steps = 0`

Interpretation:

- this is the strongest result so far, because it is not only a cheap-smoke success but a normal-configuration success
- the two previously dominant failure symptoms are materially improved:
  - the episode now completes successfully
  - executor override drops from the earlier failed `full` run's `0.163686...` to `0.019455...`

## 11. Updated Root Cause Ordering

## 11. Updated Root Cause Ordering

The current root cause ordering has shifted since the original diagnosis.

### P0: `full` can now succeed, but still has a long-tail instability risk

This remains the top method-level issue to audit for future scale-up, but it is no longer a blocking condition for `episode 296`.

The system is no longer blocked by raw grounding failure. Instead:

- `no-monitor` can finish successfully
- `full` can now also finish successfully in both cheap-smoke and formal configuration
- however, some earlier `full` runs still generated more interventions than needed
- those earlier runs remain useful failure cases for future scaling work

### P1: late-stage object-target stagnation

This is now clearly observable in traces:

- controller settles on a plausible object anchor
- geometric goal becomes stable
- progress stalls
- without a special escape path, the controller would keep circling the same anchor

This is partially improved by the stagnation patch and was a key blocker in the earlier failed/slow runs.

### P2: relation growth is contained, but not fully cheap

The pruning pass clearly reduced the worst edge explosion, but relation work can still become non-trivial in long episodes.

### P3: object pool size is still large

Even in healthier reruns, trace snapshots show:

- `objects_pre_merge` can stay very high
- `objects_post_filter` still leaves a substantial graph

This means object association / compaction is still a real long-tail optimization issue.

## 12. Immediate Next Recommendation

The pipeline is now past the “hard blocked” phase for `episode 296`.

Immediate next work should focus on:

1. treat the successful formal `no-monitor` and formal `full` runs as the current healthy reference lines
2. consider `B2` functionally closed for `episode 296`
3. resume broader `B2/B3` work, while keeping the earlier failed `full` runs as regression tests
4. if later scale-up reintroduces instability, revisit:
   - object-target stagnation handling
   - temp/stuck override clearing
   - planner object-choice determinism

## 13. Follow-up Stability Check: `full` still has a long-tail override regression

After the first formal `full` success, I continued running additional formal reruns instead of immediately declaring the issue fully solved.

That follow-up was necessary: a later formal rerun,

- run id: `smoothnav_text_213542_6af14109`

showed that the old executor-layer failure mode can still reappear in a long-tail form.

Observed behavior from trace inspection:

- by `step ~390`, `object: table cabinet` was still the stable high-level target
- `goal_after` remained fixed at `[112, 33]`
- no planner, no monitor, and no grounding failure were firing
- but executor adoption switched to `stuck_goal`
- later in the same run, executor adoption also switched to `temp_goal_persisted`

Concrete evidence from the trace:

- `step 390`: `override_reason = stuck_goal`, `override_duration = 2`, `local_unreachable = True`
- `step 400`: `override_reason = stuck_goal`, `override_duration = 12`
- `step 600`: `override_reason = temp_goal`, `override_duration = 7`
- throughout this window, the command path still sent:
  - `allow_recovery = True`
  - `clear_temp_goal = False`

Interpretation:

- the earlier fixes were enough to make at least one formal `full` run succeed
- but they were not yet strong enough to suppress long-lived executor recovery when the target is a stable `object: ...` anchor
- this is no longer a grounding problem or monitor problem
- it is a narrower executor-command gating problem

## 14. Narrowed Code-Level Cause

The new failure is now localized to the object-target recovery gate in the executor adapter.

Relevant code path:

- [executor_adapter.py](/Users/xin/Research/Code/research/SmoothNav/smoothnav/executor_adapter.py)
- [main.py](/Users/xin/Research/Code/research/SmoothNav/smoothnav/main.py)
- [agent.py](/Users/xin/Research/Code/research/SmoothNav/base_UniGoal/src/agent/unigoal/agent.py)

What the trace showed:

- `main.py` computes `clear_temp_goal` via `should_clear_temp_goal_for_command(...)`
- `main.py` computes `disable_recovery` via `should_disable_recovery_for_command(...)`
- but the old helper policy only meaningfully tightened recovery for `unexplored ...` direction targets
- under a stable `object: table cabinet` target, the command path kept allowing recovery detours
- that let the mature UniGoal executor keep re-adopting:
  - `stuck_goal`
  - `temp_goal`

This is exactly the long-tail executor regression that remained after the earlier object-target stagnation fixes.

## 15. New Patch Direction

The new patch is intentionally narrow.

Instead of changing planner or grounding again, it strengthens executor gating specifically for stable object targets:

- `temp_goal` should be cleared earlier for `object: ...` targets
- recovery should be disabled sooner for `object: ...` targets once repeated override is observed

Current intended semantics:

- direction targets keep the older, looser thresholds
- object targets use tighter thresholds because their semantic anchor is already specific, so persistent executor detours are more likely to be harmful than helpful

This change is implemented and regression-tested locally in:

- [executor_adapter.py](/Users/xin/Research/Code/research/SmoothNav/smoothnav/executor_adapter.py)
- [test_layered_contracts.py](/Users/xin/Research/Code/research/SmoothNav/tests/test_layered_contracts.py)

Local evidence:

- object-target temp-goal clearing is now tested explicitly
- object-target recovery suppression is now tested explicitly
- the relevant local test bundle remains green

## 16. B3 Intact-Scene Protocol

To avoid slipping back into the corrupted-scene problem, `B3` is now being run as an explicit intact-scene dev5 suite rather than a raw `num_eval=5` slice.

Chosen episode pool (all from the intact scene `MHPLjHsuG27`):

- `286` → `tv_monitor`
- `287` → `chair`
- `291` → `sofa`
- `294` → `bed`
- `296` → `plant`

This choice gives:

- one intact scene
- five explicit episode ids
- category diversity
- reproducibility independent of default dataset prefix slicing

New helper tooling added:

- [run_b3_intact_dev5.sh](/Users/xin/Research/Code/research/SmoothNav/scripts/run_b3_intact_dev5.sh)
- [summarize_explicit_suite.py](/Users/xin/Research/Code/research/SmoothNav/scripts/summarize_explicit_suite.py)

Protocol:

- run each profile as five explicit single-episode jobs
- aggregate by reading `effective_config.json` for `episode_id`
- write a stable suite-level summary at `suite_summary.json`

## 17. Early B3 Expansion Status

The first B3 lane currently running is:

- `baseline-periodic`
- `smoothnav-fixed-interval`

on the intact dev5 suite above.

Early baseline results already written:

- episode `286`: fail
- episode `287`: success
- episode `291`: success

So even before the whole lane completes, we already know:

- the intact-scene suite is runnable
- results are landing correctly
- this is now a real expansion lane, not a synthetic smoke test

## 18. Formal `full v10` Repeat Success After Semantic-Stage Recovery Tightening

After observing that a later rerun could still fall back into executor recovery during a stable semantic stage, I tightened the executor gate again:

- not only `object: ...`, but all semantic targets now use stricter recovery suppression
- only `unexplored ...` retains the looser recovery behavior

This change was implemented in:

- [executor_adapter.py](/Users/xin/Research/Code/research/SmoothNav/smoothnav/executor_adapter.py)
- [test_layered_contracts.py](/Users/xin/Research/Code/research/SmoothNav/tests/test_layered_contracts.py)

and validated locally with the relevant unit bundle.

The resulting formal rerun was:

- run id: `smoothnav_text_220501_cb99bb16`

Result:

- `SR = 1.0`
- `SPL = 0.07728753292092762`
- `avg_high_level_calls = 11.0`
- `avg_low_level_calls = 0.0`
- `goal_update_delay_steps = 3.75`
- `executor_override_ratio = 0.02526315789473684`
- `pending_created_count = 4`
- `pending_promoted_count = 2`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 2, 'same_goal_as_prev': 11}`
- `terminal_outcome_counts = {'FAILURE_BUDGET_EXHAUSTED': 53, 'SUCCESS': 1}`
- `missing_epoch_trace_steps = 0`

Why this matters:

- this is no longer only one successful formal `full` run
- there are now repeated formal successes after multiple executor-side corrections
- the latest successful rerun also crossed the old bad window (`~390-420`) without reproducing `stuck_goal` / `temp_goal` takeover

So the earlier long-tail executor regression is no longer a one-off suspicion; it has now been countered with:

1. a concrete code-level diagnosis
2. a targeted patch
3. a fresh formal rerun that succeeds

## 19. Current B3 Parallel Expansion State

At this point the work has moved beyond the single `episode 296` failure loop and into parallel intact-scene validation.

Currently active B3 lanes:

1. `baseline-periodic + smoothnav-fixed-interval`
2. `smoothnav-no-monitor + smoothnav-full`

Both use the same intact episode suite:

- `286`
- `287`
- `291`
- `294`
- `296`

Latest observed partial B3 evidence:

- `baseline-periodic`
  - `286`: fail
  - `287`: success
  - `291`: success
  - `294`: fail
- `smoothnav-no-monitor`
  - `286`: fail

This is still partial, but it already confirms:

- the multi-profile intact-suite setup is running correctly
- results are being written per profile / per explicit episode
- the project is no longer stuck on a single failure case

## 20. First Partial B3 Comparison Snapshot

The intact-scene `dev5` expansion is now producing enough results to read the first partial comparison.

### 20.1 `baseline-periodic` (5/5 complete)

Current explicit results:

- `286`: fail (`FAILURE_NO_PROGRESS_TIMEOUT`)
- `287`: success
- `291`: success
- `294`: fail (`FAILURE_NO_PROGRESS_TIMEOUT`)
- `296`: fail (`FAILURE_BUDGET_EXHAUSTED`)

So the current partial baseline success count is:

- `2 / 5`

This is already useful because it anchors the intact-scene suite with a complete baseline profile.

### 20.2 `smoothnav-no-monitor` (3/5 complete so far)

Current explicit results:

- `286`: fail (`FAILURE_BUDGET_EXHAUSTED`)
- `287`: success
- `291`: success

So the current partial `no-monitor` success count is:

- `2 / 3`

This is not yet a final profile judgment, but it already suggests that the current healthy reference line remains competitive on the intact-scene suite.

### 20.3 `smoothnav-fixed-interval` and `smoothnav-full`

At the moment of this note:

- `smoothnav-fixed-interval` has started but has not yet written its first summary
- `smoothnav-full` has not yet started its explicit `dev5` episode sequence because the parallel lane is still finishing `no-monitor`

So the B3 interpretation is still incomplete.

### 20.4 Why this matters

This is the first time the project has had all of the following simultaneously:

- repeated formal `full` success on the key hard episode
- a stable intact-scene episode suite
- partial multi-profile expansion results landing under that suite

That means the project is now materially past the old “blocked on one failure case” stage and is properly in comparative validation mode.

## 21. First Partial Suite-Level Aggregate

I also ran the explicit-suite aggregator:

- [summarize_explicit_suite.py](/Users/xin/Research/Code/research/SmoothNav/scripts/summarize_explicit_suite.py)

against the current intact-scene `dev5` root:

- `results/phase2_revalidation/s2_dev5_intact_scene`

This produced a suite-level partial summary while some profiles are still running.

### 21.1 Current aggregate snapshot

#### `baseline-periodic` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2071`
- `avg_high_level_calls = 11.4`
- `executor_override_ratio = 0.1314`
- `grounding_noop_reason_counts = {'out_of_local_window': 53, 'same_goal_as_prev': 24}`

#### `smoothnav-fixed-interval` (2/5 complete)

- `SR = 0.5`
- `SPL = 0.2927`
- `avg_high_level_calls = 6.5`
- `executor_override_ratio = 0.2031`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 3, 'same_goal_as_prev': 8}`

#### `smoothnav-no-monitor` (4/5 complete)

- `SR = 0.5`
- `SPL = 0.2922`
- `avg_high_level_calls = 6.0`
- `executor_override_ratio = 0.1334`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 4, 'same_goal_as_prev': 13}`

#### `smoothnav-full`

- no explicit `dev5` summary has landed yet at the time of this note

### 21.2 How to read this partial aggregate

This is still incomplete, so it should not yet be treated as the final comparative claim.

But even this partial aggregate is already useful:

- `baseline-periodic` is now fully characterized on the intact suite
- `smoothnav-no-monitor` is currently tracking at least competitively with baseline
- `smoothnav-fixed-interval` has started to provide the event-vs-interval comparison that was previously missing
- the expansion loop is now generating profile-level evidence rather than only isolated episode narratives

The next important checkpoint is when:

1. `smoothnav-no-monitor` reaches `5/5`
2. `smoothnav-fixed-interval` has more than two samples
3. `smoothnav-full` starts writing explicit `dev5` summaries

## 22. First Explicit `smoothnav-full` Dev5 Results

The explicit intact-scene `smoothnav-full` lane has now started producing its own `dev5` summaries instead of only running in the background.

Current partial explicit results:

- `286`: fail
- `287`: success
- `291`: launched / running

So the current partial `smoothnav-full` success count is:

- `1 / 2` completed

This is an important milestone because the project now has all four profiles active on the same explicit intact-scene suite:

- `baseline-periodic`
- `smoothnav-fixed-interval`
- `smoothnav-no-monitor`
- `smoothnav-full`

## 23. Updated Partial Comparison Snapshot

At the current checkpoint:

### `baseline-periodic`

- completed: `5 / 5`
- successes: `2 / 5`

### `smoothnav-fixed-interval`

- completed: `3 / 5`
- successes: `2 / 3`

### `smoothnav-no-monitor`

- completed: `4 / 5`
- successes: `2 / 4`
- `episode 296` is still running
- latest trace snapshot shows:
  - stable strategy: `object: painting picture`
  - no executor override

### `smoothnav-full`

- completed: `2 / 5`
- successes: `1 / 2`
- `episode 291` has launched

## 24. What This Means

This is still not enough to make the final profile ranking claim.

But it is already enough to say something important:

- the new `full` path is no longer confined to one repaired episode
- it can already produce an explicit intact-suite success (`episode 287`)
- the comparison has now moved from “single hard case stabilization” to “multi-episode profile behavior”

The next decision-quality checkpoint will be reached when:

1. `smoothnav-no-monitor` finishes `episode 296`
2. `smoothnav-full` finishes at least `291` and one more episode
3. the suite-level aggregate is regenerated with those new runs

## 25. Updated Partial Suite Aggregate After `no-monitor` Completion and `full` Starts

I regenerated the suite-level summary after:

- `smoothnav-no-monitor` finished all `5 / 5`
- `smoothnav-full` produced three explicit intact-scene runs
- `smoothnav-fixed-interval` reached four explicit runs

Current partial aggregate:

### `baseline-periodic` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2071`
- `avg_high_level_calls = 11.4`
- `executor_override_ratio = 0.1314`

### `smoothnav-no-monitor` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2338`
- `avg_high_level_calls = 9.8`
- `executor_override_ratio = 0.1409`
- terminal failures now show a clearer long-tail issue on `episode 296`:
  - `FAILURE_BUDGET_EXHAUSTED`
  - `FAILURE_STUCK_PERSISTENT`
  - `FAILURE_NO_PROGRESS_TIMEOUT`

### `smoothnav-fixed-interval` (4/5 complete)

- `SR = 0.5`
- `SPL = 0.2650`
- `avg_high_level_calls = 8.0`
- `executor_override_ratio = 0.1446`

### `smoothnav-full` (3/5 complete)

- `SR = 0.6667`
- `SPL = 0.3687`
- `avg_high_level_calls = 9.0`
- `executor_override_ratio = 0.1847`
- explicit runs currently include:
  - one failure (`286`)
  - two successes (`287`, `291`)

## 26. First Real Comparative Signal

This is still a partial suite, not the final result.

But it is already enough to support a stronger interim claim than before:

- `smoothnav-full` is no longer merely “can be repaired on one hard episode”
- on the current partial explicit intact-suite, it is the best-performing profile so far

At the same time, the partial data also warns against overclaiming:

- `full` still has elevated override ratio in the current partial aggregate
- `no-monitor` is not universally superior; its `episode 296` run failed hard in this suite
- `fixed-interval` is not obviously collapsing either, and is currently competitive with the same partial sample count

So the right reading is:

- the project has progressed from “single-case rescue” to “partial suite-level competition”
- but the suite is not yet complete enough to freeze the final profile ranking

## 27. Deduplicated Partial Aggregate

Because `smoothnav-full` was launched from more than one lane, I updated the explicit-suite aggregator so that it keeps only the latest run for each `(profile, episode_id)` pair.

This matters because otherwise repeated reruns of the same episode would pollute the suite-level comparison.

After that fix, the current partial aggregate is:

### `baseline-periodic` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2071`

### `smoothnav-no-monitor` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2338`

### `smoothnav-fixed-interval` (4/5 complete)

- `SR = 0.5`
- `SPL = 0.2650`

### `smoothnav-full` (4/5 complete, deduplicated)

- `SR = 0.5`
- `SPL = 0.2765`
- `executor_override_ratio = 0.1179`

## 28. Updated Interim Reading

This is still not the final ranking because:

- `smoothnav-fixed-interval` is missing one episode
- `smoothnav-full` is missing one episode

But the deduplicated partial aggregate is already much more informative than the earlier raw run list:

- `full` is now not only stable on the repaired hard episode
- it is also competitive at the partial suite level
- its partial aggregate currently beats:
  - `baseline-periodic` on `SPL`
  - `smoothnav-no-monitor` on both `SR` parity and `SPL`

So the current work is no longer about “can full be rescued at all?”

The current work is about:

- whether that lead survives once the remaining episodes land
- whether `fixed-interval` stays competitive after the suite completes

## 29. Five-Profile Explicit Suite: Stronger Partial Aggregate

I regenerated the explicit-suite aggregate again after:

- `smoothnav-full` reached `5 / 5`
- `smoothnav-no-monitor` remained complete at `5 / 5`
- `smoothnav-no-prefetch` started its own explicit lane

Current aggregate snapshot:

### `baseline-periodic`

- `SR = 0.4`
- `SPL = 0.2071`

### `smoothnav-no-monitor`

- `SR = 0.4`
- `SPL = 0.2338`

### `smoothnav-fixed-interval` (still partial)

- `SR = 0.5`
- `SPL = 0.2650`

### `smoothnav-full` (now 5/5 complete in the deduplicated suite)

- `SR = 0.6`
- `SPL = 0.2636`
- `avg_high_level_calls = 9.6`
- `executor_override_ratio = 0.1146`

### `smoothnav-no-prefetch` (early partial)

- `SR = 0.0`
- `SPL = 0.0`
- only the first explicit episode has landed so far

## 30. Updated Interpretation

This is still not the final project-level conclusion, because:

- `smoothnav-fixed-interval` is still missing one episode
- `smoothnav-no-prefetch` has only just started

But the evidence is now stronger than before:

- `full` is no longer merely competitive
- on the current deduplicated explicit suite, `full` is the strongest profile so far in `SR`
- and it still stays above `baseline-periodic` and `smoothnav-no-monitor` on `SPL`

So the current state of the project is:

- the major `full` stabilization work succeeded
- the controller family is now being compared on a real multi-episode suite
- the partial aggregate currently favors `smoothnav-full`

## 31. Near-Complete Five-Profile Snapshot

The explicit intact-suite aggregate is now substantially more complete.

At this checkpoint:

### `baseline-periodic` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2071`

### `smoothnav-no-monitor` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2338`

### `smoothnav-fixed-interval` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2120`

### `smoothnav-full` (5/5 complete)

- `SR = 0.6`
- `SPL = 0.2636`

### `smoothnav-no-prefetch` (2/5 complete so far)

- `SR = 0.5`
- `SPL = 0.3659`

## 32. Stronger Interim Conclusion

This is now enough to support a materially stronger interim conclusion than before:

- `smoothnav-full` is currently the strongest completed profile on the explicit intact `dev5` suite
- it beats:
  - `baseline-periodic` on both `SR` and `SPL`
  - `smoothnav-no-monitor` on both `SR` and `SPL`
  - `smoothnav-fixed-interval` on `SR`, and remains ahead on `SPL`

This still remains an interim result because `smoothnav-no-prefetch` is not complete yet.

But the center of gravity has clearly shifted:

- the project is no longer trying to prove that `full` can work
- it is now collecting evidence for whether `full` is the best-performing controller family member under the current implementation

## 33. Near-Complete Matrix After `no-prefetch` Catch-up

After another aggregate refresh, the five-profile matrix is now even closer to completion.

Current explicit intact-suite snapshot:

### `baseline-periodic` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2071`

### `smoothnav-fixed-interval` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2120`

### `smoothnav-no-monitor` (5/5 complete)

- `SR = 0.4`
- `SPL = 0.2338`

### `smoothnav-full` (5/5 complete)

- `SR = 0.6`
- `SPL = 0.2636`

### `smoothnav-no-prefetch` (4/5 complete)

- `SR = 0.5`
- `SPL = 0.2922`

## 34. Updated Reading

This is now a more meaningful controller-family comparison than anything the project had earlier in the session.

What stands out:

- `smoothnav-full` remains the strongest completed profile by `SR`
- `smoothnav-full` is also above the completed baseline and ablation lines on `SPL`
- `smoothnav-no-prefetch` is competitive in the current partial view, so the prefetch story is not settled yet

So the next step is no longer more debugging of the same failure class.

The next step is:

1. finish the last `smoothnav-no-prefetch` episode
2. regenerate the aggregate one more time
3. produce the first suite-level synthesis from a nearly complete five-profile matrix

## 35. First Complete Five-Profile Explicit Aggregate

The explicit intact-scene suite is now complete enough to support the first full five-profile aggregate.

Current suite-level result:

### `baseline-periodic`

- `SR = 0.4`
- `SPL = 0.2071`

### `smoothnav-fixed-interval`

- `SR = 0.4`
- `SPL = 0.2120`

### `smoothnav-no-monitor`

- `SR = 0.4`
- `SPL = 0.2338`

### `smoothnav-full`

- `SR = 0.6`
- `SPL = 0.2636`

### `smoothnav-no-prefetch`

- `SR = 0.4`
- `SPL = 0.2338`

## 36. First Completed Suite-Level Conclusion

At this point, the project has crossed a meaningful experimental threshold.

On the current explicit intact `dev5` suite:

- `smoothnav-full` is the strongest profile by `SR`
- `smoothnav-full` is also the strongest profile by `SPL`

Compared with the semantic baseline:

- `smoothnav-full` beats `baseline-periodic` on `SR`
- `smoothnav-full` beats `baseline-periodic` on `SPL`

Compared with the key internal ablations:

- it is above `smoothnav-no-monitor`
- it is above `smoothnav-fixed-interval`
- it is above `smoothnav-no-prefetch`

This is the clearest evidence so far that the repaired `full` controller is not only stable, but currently the best-performing controller-family member on the explicit intact suite.

## 37. What Follows Next

Now that the first complete five-profile explicit suite is available, the next rational step is not more micro-debugging.

The next rational step is scale-up validation:

1. preserve the current explicit-suite results as the first completed comparison set
2. move to a larger explicit intact-scene evaluation set
3. verify whether `smoothnav-full` still stays ahead of `baseline-periodic` and the ablations when the sample count grows

## 38. Dev15 Scale-Up Started

The next scale-up stage has now started on the intact-scene episode block:

- `286 287 288 289 290 291 292 293 294 295 296 297 298 299 300`

Active scale-up profiles:

- `baseline-periodic`
- `smoothnav-full`
- `smoothnav-no-monitor`

These are being run in parallel on separate GPUs so that the next comparison set is not bottlenecked on a single controller lane.

## 39. First Dev15 Fresh Evidence

The first fresh `dev15` aggregate is still very early, but it already provides a useful sanity check.

Current early snapshot:

### `baseline-periodic`

- completed: `2 / 15`
- `SR = 0.5`
- `SPL = 0.3659`

### `smoothnav-full`

- completed: `1 / 15`
- `SR = 0.0`
- `SPL = 0.0`

### `smoothnav-no-monitor`

- completed: `1 / 15`
- `SR = 0.0`
- `SPL = 0.0`

## 40. How To Read This Early Dev15 Snapshot

This is far too early to use as a ranking signal.

The only justified takeaway right now is:

- the larger explicit intact-scene evaluation set is live
- runs are writing correctly under the new `s3_dev15_intact_scene` root
- all three main lanes are progressing with trace/summary output

So the work remains in evidence-collection mode, not conclusion mode.

## 41. Dev15 Trend Shift

After additional `dev15` episodes landed, the early ranking changed in an important way.

Current partial `dev15` aggregate:

### `baseline-periodic` (7/15 complete)

- `SR = 0.8571`
- `SPL = 0.4662`

### `smoothnav-no-monitor` (7/15 complete)

- `SR = 0.8571`
- `SPL = 0.4881`

### `smoothnav-full` (6/15 complete)

- `SR = 0.8333`
- `SPL = 0.4052`

## 42. Why This Matters

This is the first strong sign that the profile ranking on the small explicit `dev5` suite may not hold unchanged as the sample count grows.

In particular:

- `full` was strongest on the completed explicit `dev5`
- but on the current `dev15` partial, `no-monitor` has pulled ahead on both `SR` parity and `SPL`

This does **not** yet prove that `full` is worse overall, because:

- `full` and `no-monitor` still have different partial sample counts
- several later episodes have not landed yet

But it *does* mean the project has entered a new phase:

- we are no longer only validating the repaired `full`
- we are now testing whether `full` remains best under scale-up

And the first answer is:

- not obviously

## 43. Immediate Implication

I should keep doing two things in parallel:

1. continue the current `dev15` evidence collection until the ranking is more stable
2. preserve the matched `full` vs `no-monitor` divergences (for example `episode 288`) as the likely starting point for a second `full`-specific improvement round if the scale-up ranking continues to favor `no-monitor`

## 44. Dev15 Scale-Up: Completed Aggregate

The `dev15` scale-up comparison for the three main lines is now complete.

Profiles:

- `baseline-periodic`
- `smoothnav-full`
- `smoothnav-no-monitor`

Episodes:

- `286 287 288 289 290 291 292 293 294 295 296 297 298 299 300`

Completed aggregate:

### `baseline-periodic`

- `SR = 0.5333`
- `SPL = 0.2353`

### `smoothnav-no-monitor`

- `SR = 0.4667`
- `SPL = 0.2416`

### `smoothnav-full`

- `SR = 0.6`
- `SPL = 0.2678`

## 45. Updated Scale-Up Reading

This completed `dev15` result is the strongest evidence in the project so far.

Compared with the semantic baseline:

- `smoothnav-full` beats `baseline-periodic` on `SR`
- `smoothnav-full` beats `baseline-periodic` on `SPL`

Compared with the closest internal reference:

- `smoothnav-full` beats `smoothnav-no-monitor` on `SR`
- `smoothnav-full` beats `smoothnav-no-monitor` on `SPL`

This matters because the earlier `dev15` partial had temporarily suggested that `no-monitor` might overtake `full` at larger scale.

Now that the `dev15` aggregate is complete, the evidence points the other way:

- the repaired `full` controller does not just win on the explicit `dev5`
- it still remains best among the three main lines on the explicit intact `dev15`

## 46. Current Project State

The project has now crossed three distinct milestones:

1. the original `full` stability blockers were diagnosed and repaired
2. `full` surpassed the semantic baseline on the completed explicit `dev5`
3. `full` remained ahead on the completed explicit `dev15` scale-up

At this point, the project is no longer in a “can the architecture work at all?” stage.

It is now in a “how far does the current advantage hold under larger and broader validation?” stage.

## 47. Cross-Scene Transition Began

After the explicit intact-scene `dev15` aggregate completed, I started the next validation step:

- a cross-scene canary on a new scene block
- using episode `229` from scene `LT9Jq6dN3Ea`

Profiles launched:

- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

This step matters because the project has already shown:

- repaired `full` stability
- completed explicit `dev5`
- completed explicit `dev15`

So the next useful question is no longer “does the current scene still work?”

The next useful question is:

- does the current ranking survive when we leave the original scene block?

## 48. Current Highest-Confidence Summary

At the moment, the highest-confidence project-level status is:

1. the original method-level blockers are fixed
2. `smoothnav-full` beats `baseline-periodic` on the completed explicit `dev5`
3. `smoothnav-full` also stays ahead of both `baseline-periodic` and `smoothnav-no-monitor` on the completed explicit `dev15`
4. cross-scene validation has begun

That is now a meaningfully stronger state than any earlier point in the project.

## 49. Cross-Scene Data Blocker Confirmed

The next-stage blocker is now much clearer than before.

I launched a cross-scene canary on `episode 229` from scene `LT9Jq6dN3Ea` and all three profiles failed before rollout began:

- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

The error is not controller-specific. All of them fail in Habitat scene loading with:

- `AssertionError: ESP_CHECK failed: Error loading general mesh data ...`
- `Trade::GltfImporter::openData(): binary glTF size mismatch`

I then probed a wider set of candidate scenes directly through the remote Habitat loader. Result:

- `MHPLjHsuG27` remains the only known-good scene block in the current working set
- the next 35 high-coverage candidate scenes all failed loader checks

I also checked the local archive that looked like the natural fallback:

- [hm3d-val-habitat-v0.2.tar](/Users/xin/Research/Code/research/SmoothNav/base_UniGoal/data/hm3d-val-habitat-v0.2.tar)

That archive is itself truncated:

- `tar: Truncated input file`
- Python `tarfile` also raises `ReadError: unexpected end of data`

So the current cross-scene blocker is not experimental logic. It is a dataset integrity problem.

## 50. Updated Project Status

Right now the state of the project is:

1. method-level `full` stabilization is successful
2. `full` beats the semantic baseline on completed explicit `dev5`
3. `full` remains ahead on completed explicit `dev15`
4. broader cross-scene validation is blocked by corrupted HM3D `val` scene assets and a truncated local fallback archive

This is a much more specific and actionable blocker than before:

- not “cross-scene might be flaky”
- but “the current HM3D `val` asset bundle is not usable for multi-scene validation”

## 51. Official Recovery Path Confirmed

I also verified that the official Habitat download utility is available in the remote environment and already exposes the expected HM3D dataset groups:

- `hm3d_val_habitat_v0.2`
- `hm3d_val_full`
- `hm3d_full`

That matters because it means the data repair path is no longer ambiguous.

The current blocker is **not**:

- missing tooling
- unknown package support
- unclear dataset naming

The current blocker is specifically:

- missing valid Matterport access credentials for a clean redownload of the HM3D `val` assets

## 52. Practical Next Step

So from here, the most direct recovery path is:

1. obtain valid Matterport download credentials
2. redownload `hm3d_val_habitat_v0.2` (or `hm3d_val_full`) into a clean location
3. swap or relink the broken `val` scene bundle on the experiment server
4. rerun the cross-scene canary

At this point, that is the cleanest next gate for the project.

## 53. Data Repair Completed From A Fresh Local Archive

After the earlier blocker was reduced to “missing valid HM3D `val` assets”, a new local archive became available:

- `/Users/xin/Downloads/hm3d-val-habitat-v0.2.tar`

Fresh verification on that file showed:

- size ≈ `3.3G`
- `100` `.basis.glb` members
- member sizes match the expected large per-scene sizes rather than the truncated `18612224` byte pattern

I copied that archive to the experiment server and extracted it into:

- `/mnt/sdd/xxy/SmoothNav/base_UniGoal/data/scene_datasets/hm3d_v0.2/val_repaired`

Then I switched both runtime scene roots to point at the repaired bundle:

- `base_UniGoal/data/scene_datasets/hm3d_v0.2/val`
- `data/scene_datasets/hm3d_v0.2/val`

Both are now symlinked to the repaired directory.

## 54. Loader Audit After Repair

After the repaired bundle was installed, I reran the HM3D loadability audit.

Result:

- multiple previously failing scenes now load successfully
- examples include:
  - `LT9Jq6dN3Ea`
  - `yr17PDCnDDW`
  - `q5QZSEeHe5g`
  - `eF36g7L6Z9M`
  - `zt1RVoi7PcG`

This is a major state change:

- the cross-scene blocker is no longer “dataset is unusable”
- it has moved to “cross-scene controller ranking now needs to be measured on the repaired bundle”

## 55. Repaired Cross-Scene Canary Status

### Scene: `LT9Jq6dN3Ea`, episode `229`

Fresh repaired canary results:

- `baseline-periodic`: fail
- `smoothnav-full`: fail
- `smoothnav-no-monitor`: one failed run and one still-running / duplicate lane during the recovery phase

Current reading:

- this scene no longer dies in loader init
- it now reaches rollout, so this is finally a controller/runtime comparison rather than a data-corruption failure

### Scene: `yr17PDCnDDW`, episode `859`

Fresh repaired canary results:

- `baseline-periodic`: success
- `smoothnav-no-monitor`: success
- `smoothnav-full`: at least one success and one failed duplicate rerun during the recovery phase

Current reading:

- repaired multi-scene validation is now truly live
- the next challenge is no longer asset integrity but de-duplicating and stabilizing the repaired cross-scene comparison protocol

## 56. Updated State Of The Project

The project is now beyond the previous hard blocker.

What has changed:

1. intact-scene `dev5` and `dev15` evidence already showed `smoothnav-full` ahead of the baseline
2. HM3D `val` assets have now been repaired from a valid local archive
3. repaired cross-scene canaries are running and producing real rollout results

So the next center of gravity is:

- not data repair anymore
- but building a clean, deduplicated, cross-scene comparison set on the repaired bundle

## 57. Clean Cross-scene Suite Started

After repairing the HM3D `val` assets, I stopped relying on the old noisy launcher history and started a **clean cross-scene suite** under a fresh results root:

- `results/phase2_revalidation/s4_cross_scene_suite`

Matched episode set:

- `229` (`LT9Jq6dN3Ea`)
- `527` (`eF36g7L6Z9M`)
- `859` (`yr17PDCnDDW`)

Profiles:

- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

## 58. First Clean Cross-scene Results

Current aggregate on the clean repaired suite:

### `baseline-periodic`

- completed: `2 / 3`
- `SR = 0.0`
- `SPL = 0.0`

### `smoothnav-no-monitor`

- completed: `2 / 3`
- `SR = 0.0`
- `SPL = 0.0`

### `smoothnav-full`

- completed: `1 / 3`
- `SR = 0.0`
- `SPL = 0.0`

This is the first repaired cross-scene signal, and it is very different from the intact-scene story:

- on these new scenes, all currently completed runs are failing
- so the intact-scene advantage is **not yet** translating cleanly to these repaired cross-scene canaries

## 59. Current Interpretation

This does **not** invalidate the intact-scene result.

What it does show is:

- once we leave `MHPLjHsuG27`, the task becomes much harder
- the current controller advantage has not yet been demonstrated robustly across repaired new scenes

So the project has entered the next genuine research phase:

- from “does the repaired method beat baseline on a clean scene block?”
- to “why does the current ranking weaken or disappear on repaired cross-scene canaries?”

## 60. Repaired Cross-scene Failure Mode Tightened Again (2026-04-23)

After the repaired multi-scene suites became runnable, I did another round of
matched trace inspection and code-path auditing focused on why
`smoothnav-full` regressed on the repaired cross-scene suite while remaining
strong on the intact-scene block.

### New code-level observations

1. **Planner empty-response handling contained a real implementation bug.**
   In `smoothnav/planner.py`, the `HighLevelPlanner.plan()` loop previously
   treated an empty LLM response as an immediate hard stop:
   - set `error_message = "empty_response"`
   - log a warning
   - `break`
   - then fall through to the hardcoded parse-failure fallback
     `direction = north`

   This meant the planner did not actually use its retry budget when the model
   returned an empty string, and the failure collapsed to a fixed northward
   exploration bias.

2. **That bug affected `smoothnav-full` disproportionately on repaired
   cross-scene runs.**
   Remote inspection of the repaired explicit suite
   `results/phase2_revalidation/s5_cross_scene_dev6` showed:
   - `baseline-periodic`: no empty-response planner fallbacks
   - `smoothnav-no-monitor`: no empty-response planner fallbacks
   - `smoothnav-full`: `16` empty-response planner fallbacks

   So this was not just a theoretical cleanliness issue; it was a concrete
   asymmetry that could bias the repaired cross-scene comparison against
   `full`.

3. **Room-choice formatting was also too permissive.**
   `build_choices_text()` used room lines like:
   - `[room] bedroom: painting picture, windows`

   On repaired cross-scene runs, the LLM could echo the full string back as
   `choice_id`, which then made room parsing / resolution brittle. This was not
   a principled design choice; it was prompt formatting leaking scene text into
   the resolver input.

4. **The most important repaired cross-scene divergence is currently a
   recovery-direction problem, not a monitor problem.**
   In matched repaired cross-scene comparisons on `ep661`, the crucial
   divergence between the failing `full` run and the successful `no-monitor`
   run occurred after object-target stagnation:
   - old `full`: fell back to `unexplored north`
   - successful `no-monitor`: switched to `unexplored east`

   That difference alone was sufficient to produce a successful route in the
   repaired scene.

### Updated interpretation

At this stage, the repaired cross-scene problem looks less like a pure
method-level collapse and more like a stack of:

- implementation-level brittleness in planner fallback,
- prompt formatting ambiguity,
- and a still-open design question around how aggressively semantic recovery
  should be allowed to steer local exploration.

In other words:

- there **is** still a deeper method question,
- but before claiming the method fails cross-scene, we have to remove the very
  concrete planner/runtime bugs that are currently over-penalizing `full`.

## 61. New Cross-scene Planner Patch Loop

### Patch A — empty-response retry fix

I changed `HighLevelPlanner.plan()` so that:

- empty responses no longer immediately terminate the retry loop,
- the planner continues to use the configured retry budget,
- and only after retries are exhausted does it fall back.

### Patch B — deterministic directional fallback

I added a deterministic geometric fallback in `smoothnav/planner.py`:

- `deterministic_direction_fallback(...)`
- `maybe_semantic_direction_fallback(...)`

The goal is:

- when local semantic anchors are clearly weak or stagnant,
- avoid collapsing to a generic `north`,
- and instead choose the strongest frontier sector that is *not* simply the
  current exhausted direction.

This is intentionally closer to the mature cross-scene pattern used by
frontier-/map-anchored systems:

- preserve a map-valid exploration anchor,
- do not let semantic uncertainty collapse into arbitrary local direction drift.

### Patch C — room choice normalization

I changed planner room-choice handling so that:

- room menu lines use room ids only (`[room] bedroom`)
- parser normalization strips any accidental `room: object list` suffix if it
  still appears in model output

This removes prompt-format leakage from the resolver path.

## 62. Immediate Verification Evidence

### Local / remote regression tests

The planner-focused regression suite passed locally and remotely after the
patches.

### Targeted repaired cross-scene rerun (`ep661`)

I ran a targeted repaired rerun for the strongest matched divergence case:

- root:
  `results/phase2_revalidation/s5_cross_scene_ep661_retryfix_v3`
- profile:
  `smoothnav-full`
- episode:
  `661`

Result:

- `SR = 1.0`
- `SPL ≈ 0.8529`
- `executor_override_ratio = 0.0`

This is a real repaired cross-scene positive result, and it is stronger than
the older `full` outcomes on the same episode.

### Trace-level confirmation

The new planner trace for this repaired rerun showed that the key stagnation
replans no longer collapse back to `north`:

- at the earlier critical branch, the new patched run selected `east`
  through `semantic_stagnation_direction_fallback`

This is exactly the failure branch that previously separated the repaired
cross-scene `full` failure from the successful `no-monitor` behavior.

## 63. Current Status After The Patch

The project is now in a more promising state:

1. repaired multi-scene execution is healthy;
2. a concrete cross-scene implementation bug has been fixed;
3. a targeted repaired rerun now shows clear positive numeric improvement;
4. the broader repaired suite still needs to finish before we can claim that
   the cross-scene advantage is established.

So the next gate is no longer:

- “can `full` ever recover cross-scene?”

but rather:

- “does this repaired planner/fallback fix survive a broader repaired
  explicit suite?”

## 64. Live Monitoring Update: `ep228` After Room/Object Sanitization

I continued monitoring the next repaired cross-scene loop after adding a second
planner patch:

- treat room-like captions such as `living room` as room choices instead of
  object choices
- keep them out of the object list shown to the planner
- normalize any returned `object: living room` back into a room decision

### Why this mattered

The previous repaired `ep228` failure had a pathological planner branch in
which `living room` appeared inside the object list and the planner later chose:

- `choice_type = object`
- `choice_id = living room`

That was a prompt/parse problem, not a principled object decision.

### Fresh runtime evidence

I launched:

- targeted rerun:
  `results/phase2_revalidation/s5_cross_scene_ep228_retryfix_v3`
- broader full-only repaired suite:
  `results/phase2_revalidation/s5_cross_scene_dev6_v5_fullonly`

Both are still running, but their partial traces already show a healthier
trajectory than the older repaired `ep228` runs:

- old repaired `ep228` runs drifted back into repeated `north` exploration and
  later nonsensical `object: living room`-style branches
- the new targeted and suite runs are both still executing under
  `unexplored east` after `400+` steps
- no terminal failure has been recorded yet in the current partial traces

So, even before final summaries are written, the latest patched branch is
showing a more coherent repaired cross-scene control path for `ep228`.

## 65. Fresh Verification: `ep228` Still Fails, But Failure Mode Improved

The targeted repaired rerun for `ep228` has now completed:

- root:
  `results/phase2_revalidation/s5_cross_scene_ep228_retryfix_v3`
- run:
  `smoothnav_text_123727_31a4d93d`

Result:

- `SR = 0.0`
- `SPL = 0.0`
- terminal outcome:
  `FAILURE_NO_PROGRESS_TIMEOUT`

So this patch does **not** yet convert `ep228` into a success.

However, the failure shape is materially cleaner than earlier repaired runs:

- `executor_override_ratio = 0.0`
- `temp/stuck/global/visible` override ratios all remain `0.0`
- `grounding_noop_reason_counts` shrinks to:
  - `out_of_local_window = 4`
  - `same_frontier_as_prev = 2`

This means the patch did **not** solve the final task-level metric, but it did
remove the earlier hijack-style behavior and leave a cleaner late-stage failure:

- long-horizon search keeps running,
- semantic/object hijack does not recur,
- but the planner still experiences repeated empty-response fallback and slow
  late-stage goal updates.

### Updated interpretation

`ep228` is now a cleaner negative control:

- before: cross-scene failure mixed together room/object prompt pollution,
  override issues, and bad fallback branches
- now: the main remaining problem is closer to **slow planner-driven search
  under sparse/weak semantic evidence**, not execution hijack

This is a useful state change, because it narrows the next patch loop to:

1. planner empty-response robustness
2. late-stage exploration efficiency
3. possible overuse of auto-prefetch / delayed goal refresh under weak evidence

## 66. Fresh Verification After Planner Empty-response Cooldown

I then added another narrow runtime change:

- when a planner call fails with `empty_response`,
- and that call came from `Auto-PREFETCH`,
- the controller now suppresses promotion of the new pending strategy and enters
  a cooldown window before the next prefetch planner attempt

This was intended to stop the repaired cross-scene loop from repeatedly
re-triggering weak prefetch replans after empty responses.

### Fresh partial evidence (`v5` / `v7`)

New roots:

- `results/phase2_revalidation/s5_cross_scene_ep228_retryfix_v5`
- `results/phase2_revalidation/s5_cross_scene_dev6_v7_fullonly`

Both are still running, but the partial traces now show a healthier planner mix
than the previous `v4` / `v6` branch.

#### Targeted `ep228` partial

- planner calls observed so far: `8`
- planner fallbacks: `5`
- non-fallback planner decisions: `3`

In the previous branch, the planner had effectively collapsed into
all-fallback/empty-response behavior much earlier. In the new branch, some
later replans are now real planner decisions:

- `Direction reuse limit reached` -> non-fallback direction choice

The partial step trace also stays in a clean control regime:

- `steps = 196`
- adoption sources dominated by `exp_goal`
- no override re-explosion
- no grounding-noop accumulation in the targeted partial window

#### Suite partial

- planner calls observed so far: `10`
- planner fallbacks: `5`
- non-fallback planner decisions: `5`

Again, this is a healthier split than the earlier all-fallback pattern. The
system is no longer simply thrashing on empty responses.

The partial state remains:

- `steps = 233`
- adoption sources dominated by `exp_goal`
- only `out_of_local_window = 2` as a residual grounding issue so far

### Updated interpretation

The cooldown patch does not yet prove a final metric win, but it does produce a
new positive control-layer effect:

- empty-response planner failures no longer dominate every subsequent replan
- the controller now allows some genuine late-stage planner decisions to
  re-enter the loop
- the repaired cross-scene run remains in a clean non-hijacked regime while
  continuing to search

So the next open question is no longer:

- “are we still trapped in planner empty-response churn?”

but rather:

- “does this reduced churn eventually convert into a successful repaired
  episode / better repaired aggregate?”

## 67. Completed Cooldown Verification: Cleaner But Still Negative On `ep228`

After the partial `v5` / `v7` monitoring window, the targeted repaired rerun
for `ep228` completed.

Root:

- `results/phase2_revalidation/s5_cross_scene_ep228_retryfix_v5`

Run:

- `smoothnav_text_131556_c512ba3c`

Result:

- `SR = 0.0`
- `SPL = 0.0`
- terminal outcome: `FAILURE_NO_PROGRESS_TIMEOUT`
- `executor_override_ratio = 0.0`
- `grounding_noop_reason_counts = {'same_frontier_as_prev': 1, 'out_of_local_window': 4}`
- `pending_created_count = 0`
- `goal_update_delay_steps = 1.625`

Planner trace summary:

- planner calls: `10`
- empty-response fallback calls: `5`
- non-fallback direction decisions: `5`
- the run cycles through multiple map-valid directions (`east`, `north`,
  `west`, `south`) instead of collapsing permanently to `north`

Interpretation:

- the cooldown / deterministic fallback work improved the control shape;
- executor hijack is gone in this run;
- goal-update delay is no longer the dominant signal;
- but task-level success is still not recovered on `ep228`.

So `ep228` is now a cleaner hard negative: it is not primarily an executor
recovery failure anymore, but it still exposes insufficient long-horizon search
quality under weak semantic evidence.

## 68. Additional Hard-case Check: `ep527` Still Fails Under Formal `full`

I also checked the latest formal `smoothnav-full` rerun for `ep527`.

Root:

- `results/phase2_revalidation/s5_cross_scene_ep527_formal_v4`

Run:

- `smoothnav_text_143731_2c7d7774`

Result:

- `SR = 0.0`
- `SPL = 0.0`
- terminal outcome: `FAILURE_NO_PROGRESS_TIMEOUT`
- `executor_override_ratio = 0.05714285714285714`
- `grounding_noop_rate = 0.0`
- `pending_created_count = 4`
- `pending_promoted_count = 3`
- `goal_update_delay_steps = 22.25`

Trace-level reading:

- the run begins with a direction / room sequence (`unexplored north` ->
  `bedroom` -> `unexplored north`);
- stale or redundant `unexplored north` pending proposals persist for a long
  window;
- later the planner selects `object: chair`, which is semantically plausible;
- the episode still terminates by no-progress timeout.

Interpretation:

- `ep527` is not dominated by `out_of_local_window` or empty grounding no-ops;
- it is closer to a pending / target commitment and late-stage progress issue;
- unlike `ep228`, it still shows a small executor recovery signal.

This means the repaired cross-scene blocker is not a single uniform bug. The
same `full` stack now has at least two cleaned hard negatives:

1. `ep228`: direction-level search quality remains insufficient after planner
   fallback cleanup.
2. `ep527`: semantically plausible target selection and pending promotion do not
   translate reliably into progress.

## 69. Next Matched Repaired Mini-matrix Launched

Because the latest full-only hard-case checks are still negative, I launched a
fresh matched repaired mini-matrix under a new clean root:

- `results/phase2_revalidation/s6_cross_scene_minimatrix_20260423`

Episodes:

- `228`
- `527`
- `661`

Profiles:

- `baseline-periodic` on GPU `0`
- `smoothnav-no-monitor` on GPU `1`
- `smoothnav-full` on GPU `2`

Launcher logs:

- `s6_baseline_20260423_164644.launcher.log`
- `s6_no_monitor_20260423_164644.launcher.log`
- `s6_full_20260423_164645.launcher.log`

Purpose:

- keep `ep661` as the positive anchor;
- keep `ep228` and `ep527` as hard negatives;
- avoid drawing conclusions from full-only reruns without current matched
  controls;
- determine whether the latest `full` behavior is at least non-inferior to the
  current-code `baseline-periodic` and `smoothnav-no-monitor` on the same three
  repaired episodes.

Decision gate for the next research turn:

1. Aggregate the `s6_cross_scene_minimatrix_20260423` root after all three lanes
   finish.
2. If `smoothnav-full` still fails `ep228` and `ep527` while controls also fail,
   treat the issue as a broader cross-scene search-target quality problem.
3. If a control succeeds where `full` fails, inspect the matched divergence and
   patch the specific `full` policy path.
4. If `full` improves only on `ep661`, do not expand to a larger suite yet;
   move to the frontier/value-level semantic target anchoring experiment.

## 70. Completed `s6` Matched Repaired Mini-matrix

The matched repaired mini-matrix launched in section 69 has completed and the
combined aggregate has been written at:

- `results/phase2_revalidation/s6_cross_scene_minimatrix_20260423/suite_summary.json`

Episodes:

- `228`
- `527`
- `661`

Profiles:

- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

### Aggregate result

| Profile | SR | SPL | Terminal outcomes |
| --- | ---: | ---: | --- |
| `baseline-periodic` | `0.3333` | `0.1006` | `2` no-progress failures, `1` success |
| `smoothnav-no-monitor` | `0.3333` | `0.2843` | `2` no-progress failures, `1` success |
| `smoothnav-full` | `0.3333` | `0.2843` | `2` no-progress failures, `1` success |

### Per-episode outcome

#### `ep228`

All three profiles failed:

- `baseline-periodic`: `SR = 0.0`, `SPL = 0.0`
- `smoothnav-no-monitor`: `SR = 0.0`, `SPL = 0.0`
- `smoothnav-full`: `SR = 0.0`, `SPL = 0.0`

This confirms that `ep228` is not a `full`-only regression under the current
code. It is a broader repaired cross-scene hard case.

Important profile differences remain:

- `baseline-periodic`: `goal_update_delay_steps = 8.11`, no pending proposals
- `smoothnav-no-monitor`: `goal_update_delay_steps = 24.67`, `pending = 5 / 3`
- `smoothnav-full`: `goal_update_delay_steps = 34.33`, `pending = 8 / 7`

So `full` is not uniquely failing the episode, but it still pays a larger
controller-latency / pending-churn cost on this hard case.

#### `ep527`

All three profiles failed:

- `baseline-periodic`: `SR = 0.0`, `SPL = 0.0`
- `smoothnav-no-monitor`: `SR = 0.0`, `SPL = 0.0`
- `smoothnav-full`: `SR = 0.0`, `SPL = 0.0`

`no-monitor` and `full` are nearly identical here:

- `executor_override_ratio = 0.0571`
- `goal_update_delay_steps = 22.25`
- `pending = 4 / 3`
- no grounding noop accumulation

This strengthens the interpretation that `ep527` exposes a shared target
commitment / late progress problem rather than a monitor-specific failure.

#### `ep661`

All three profiles succeeded:

- `baseline-periodic`: `SR = 1.0`, `SPL = 0.3017`
- `smoothnav-no-monitor`: `SR = 1.0`, `SPL = 0.8529`
- `smoothnav-full`: `SR = 1.0`, `SPL = 0.8529`

This is the clean positive anchor. The SmoothNav event-driven lines are much
more efficient than the periodic semantic baseline on this episode, while
`full` and `no-monitor` are tied.

### Updated interpretation

The `s6` matrix changes the cross-scene picture in an important way:

1. `smoothnav-full` is not uniquely broken cross-scene.
2. `ep228` and `ep527` are shared hard cases across the current controller
   family.
3. `ep661` shows that the current SmoothNav event-driven stack can produce a
   strong repaired cross-scene route and large SPL gain over the periodic
   baseline.
4. The monitor itself is not yet showing a separate measurable gain over
   `no-monitor` on this mini-matrix.
5. The remaining blocker is therefore best framed as **cross-scene semantic
   search-target quality / frontier-value anchoring**, not executor hijack and
   not a `full`-specific monitor regression.

### Next decision

Do not expand directly to a larger repaired suite yet.

The next high-leverage experiment should be a narrow frontier/value-level target
anchoring pass evaluated first on:

- `ep228`
- `ep527`
- `ep661`

Success should be judged by:

- preserving the `ep661` positive anchor;
- improving `ep228` or `ep527` from failure to success, or at least producing a
  non-trivial SPL / progress improvement;
- reducing `goal_update_delay_steps` and pending churn on the hard cases;
- maintaining `executor_override_ratio` near zero.

## 71. Ralph Narrow Patch: Frontier/value Semantic Target Anchoring

Date: 2026-04-23.

### Observation

The `s6` matched repaired mini-matrix showed that `ep228` and `ep527` are shared
hard cases across `baseline-periodic`, `smoothnav-no-monitor`, and
`smoothnav-full`, while `ep661` is a clean positive anchor for the SmoothNav
event-driven stack. The code path analysis for this pass found a concrete target
anchoring boundary:

- `HighLevelPlanner` produces a symbolic region/object/direction plus a bias,
  but `Graph.get_goal()` previously ranked frontiers mostly by agent distance
  plus distance to that semantic bias.
- Direct object grounding for text goals could bypass frontier selection and
  turn an observed object center into the executable navigation target.
- The frontier ranking trace did not expose information-gain/actionability/repeat
  terms, making `same_frontier_as_prev`, pending churn, and out-of-window failures
  hard to diagnose at value level.

### Hypothesis

For repaired cross-scene hard cases, the next useful intervention is not a wider
suite or another monitor change. The narrow intervention is to keep text-goal
semantic evidence as an **anchor** and let the executable target remain a
map-valid frontier selected by a lightweight value score:

`frontier_value = distance/base + semantic_bias + unknown_novelty + local_actionability - repeat_penalties`

This matches the common pattern in mature semantic exploration systems: semantic
evidence shapes a frontier/value map, while the low-level goal remains a
reachable exploration target rather than a raw symbolic object center.

### Implemented patch

Code paths updated:

- `smoothnav/frontier_scoring.py`
  - added unknown-neighborhood novelty scoring;
  - added local-window actionability scoring and bias-subset relaxation to a
    locally actionable frontier;
  - added exact last/recent frontier repeat penalties;
  - extended frontier ranking diagnostics with novelty/actionability/repeat
    breakdowns.
- `base_UniGoal/src/graph/graph.py`
  - composes the new frontier value score inside `Graph.get_goal()`;
  - stores a bounded recent selected-frontier history;
  - emits `local_actionable_candidate_count`, `actionability_filter_mode`, and
    full value-term diagnostics in `last_goal_debug`.
- `smoothnav/strategy_grounding.py`
  - text-goal object matches are now frontier anchors by default:
    `object center -> bias -> graph.get_goal(goal=bias)`;
  - direct object centers remain available only behind
    `graph_enable_direct_object_goal` for legacy/non-text modes;
  - the BEV local planning window is passed into `Graph` before `get_goal()` so
    actionability filtering can reject/relax out-of-window candidates.
- `smoothnav/planner.py`
  - explored/stagnant object anchors are not re-offered in the LLM choice menu;
  - sparse stuck fallback now requires object relevance, otherwise deterministic
    semantic direction fallback owns recovery.
  - empty object sections are omitted when filtering removes all object choices.
- `base_UniGoal/configs/config_habitat*.yaml`
  - added knobs for novelty/actionability/repeat weights and text-goal frontier
    anchoring defaults.

### Verification evidence

Fresh local regression evidence:

- `/tmp/smoothnav-test-venv/bin/python -m unittest discover -s tests`
  - `Ran 136 tests in 0.009s`
  - `OK`
- `/tmp/smoothnav-test-venv/bin/python -m py_compile ...`
  - affected code and test files compile successfully.

Focused tests now cover:

- actionability can beat a bias-only frontier when the bias subset is not locally
  projectable;
- bias filtering relaxes to locally actionable frontiers;
- object text targets become semantic anchors rather than direct executable goals
  by default;
- the legacy direct-object path is still explicitly available;
- sparse stuck fallback ignores irrelevant objects and falls back to frontier
  direction search.

### Current experiment status

Observation: this local machine cannot run the full Habitat/torch experiment
entrypoint yet:

- `scripts/launch_profile_suite_background.sh` expects
  `/mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh`, which is absent locally;
- `PYTHONPATH=$PWD /tmp/smoothnav-test-venv/bin/python -m smoothnav.main --help`
  reaches `ModuleNotFoundError: No module named 'torch'`.

This is an environment/runtime blocker for executing the actual `ep228` / `ep527`
/ `ep661` rerun in the current local session, not evidence against the patch.

### Next runnable experiment

Once the Habitat/torch `unigoal` runtime is available, run the same narrow
three-episode anchor suite first, without expanding the benchmark:

```bash
RESULTS_ROOT=results/phase2_revalidation/s7_frontier_value_anchor_20260423 \
EPISODES="228 527 661" \
CONFIG_FILE=base_UniGoal/configs/config_habitat.yaml \
./scripts/run_b3_intact_dev5.sh smoothnav-no-monitor smoothnav-full

python scripts/summarize_explicit_suite.py \
  --results-root results/phase2_revalidation/s7_frontier_value_anchor_20260423 \
  --episodes 228 527 661 \
  --profiles smoothnav-no-monitor smoothnav-full
```

Primary success criteria remain:

- preserve the `ep661` positive anchor;
- improve `ep228` or `ep527` to success, or at least improve SPL/progress;
- reduce `goal_update_delay_steps`, pending churn, `same_frontier_as_prev`, and
  `out_of_local_window` on hard cases;
- keep `executor_override_ratio` near zero.

### Claim

The code now implements the intended narrow frontier/value-level semantic target
anchoring mechanism and has local regression coverage. The task-level empirical
claim for `ep228` / `ep527` / `ep661` is still pending the Habitat/torch runtime
rerun.

## 72. Completed `s7` / `s7b` Frontier-value Anchoring Reruns

Date: 2026-04-23.

### Observation: `s7` pure frontier/value anchoring over-constrained the positive anchor

Remote server: `10.176.56.73`.

Result root:

- `results/phase2_revalidation/s7_frontier_value_anchor_20260423/suite_summary.json`

Episodes:

- `228`
- `527`
- `661`

Aggregate result:

| Profile | SR | SPL | Terminal outcomes |
| --- | ---: | ---: | --- |
| `baseline-periodic` | `0.0000` | `0.0000` | `3` no-progress failures |
| `smoothnav-no-monitor` | `0.0000` | `0.0000` | `3` no-progress failures |
| `smoothnav-full` | `0.0000` | `0.0000` | `3` no-progress failures |

`s7` intentionally made text-goal object evidence act as frontier bias instead
of direct executable object goals. That preserved local regression tests but
broke the `ep661` positive anchor: both SmoothNav event-driven profiles failed
`ep661`, whereas `s6` had both succeeding with `SPL = 0.8529`.

Hypothesis: pure frontier/value anchoring is too conservative. When the target
object is already observed and the object center is locally projectable, the
controller should be allowed to exploit that local target instead of always
projecting back through frontier choice.

### Observation: `s7b` restores the positive anchor with a local direct-object gate

Result root:

- `results/phase2_revalidation/s7b_local_direct_anchor_20260423/suite_summary.json`

Patch delta from `s7`:

- keep frontier/value anchoring as the default for text goals;
- allow direct object execution only when the text-goal object center is already
  projectable inside the active local map;
- add config knob `graph_text_goal_allow_local_direct_object_goal: true`.

Aggregate result:

| Profile | SR | SPL | Terminal outcomes |
| --- | ---: | ---: | --- |
| `baseline-periodic` | `0.0000` | `0.0000` | `3` no-progress failures |
| `smoothnav-no-monitor` | `0.3333` | `0.2843` | `2` no-progress failures, `1` success |
| `smoothnav-full` | `0.3333` | `0.2843` | `2` no-progress failures, `1` success |

Per-episode result:

| Profile | ep228 | ep527 | ep661 |
| --- | --- | --- | --- |
| `baseline-periodic` | fail | fail | fail |
| `smoothnav-no-monitor` | fail | fail | success, `SPL = 0.8529` |
| `smoothnav-full` | fail | fail | success, `SPL = 0.8529` |

Key control metrics:

| Profile | goal_update_delay_steps | pending created / promoted | executor_override_ratio |
| --- | ---: | ---: | ---: |
| `baseline-periodic` | `0.0000` | `0.0000 / 0.0000` | `0.0000` |
| `smoothnav-no-monitor` | `6.5370` | `5.0000 / 3.3333` | `0.0296` |
| `smoothnav-full` | `1.4444` | `5.0000 / 3.3333` | `0.0313` |

### Interpretation

Claim: `s7b` restores the `ep661` positive anchor while preserving the narrow
frontier/value diagnostics added in `s7`.

Observation: `s7b` does not recover `ep228` or `ep527`. Both remain shared hard
cases for `smoothnav-no-monitor` and `smoothnav-full`.

Observation: `smoothnav-full` and `smoothnav-no-monitor` tie on aggregate SR/SPL
in `s7b`, but `full` has much lower average goal update delay on this mini-matrix
(`1.44` vs `6.54`). This is a controller-latency improvement, not a task-success
improvement.

Hypothesis: the remaining hard-case blocker is no longer the coarse direct-object
vs frontier-anchor switch. It is more likely in target-candidate semantics and
late-stage target commitment: alias/category ambiguity, object plausibility vs
true target identity, and frontier sector choice around those candidates.

### Stop condition for this Ralph pass

The user requested stopping after the current experiment round completed. The
`s7b` suite completed, `suite_summary.json` was written, and no `s7` / `s7b`
experiment processes remained active on the remote server at completion check.
No further experiments are launched in this pass.

## 73. Completed `s8` Target-evidence Uptake Smoke Check

Date: 2026-04-23.

Remote server: `10.176.56.73`.

Result root:

- `results/phase2_revalidation/s8_target_evidence_uptake_20260423/suite_summary.json`

Scope:

- profile: `smoothnav-full`
- episodes: `228`, `661`
- purpose: compare the target-evidence uptake patch against already-tested hard / positive anchors without expanding the suite.

### Observation: aggregate smoke result

| Profile | Episodes | SR | SPL | Terminal outcomes |
| --- | --- | ---: | ---: | --- |
| `smoothnav-full` | `228`, `661` | `0.5000` | `0.4265` | `1` no-progress failure, `1` success |

Per episode:

| Episode | Outcome | SPL |
| ---: | --- | ---: |
| `228` | fail, `FAILURE_NO_PROGRESS_TIMEOUT` | `0.0` |
| `661` | success | `0.8529` |

### Observation: `ep228` target evidence uptake is fixed but task success is not

Compared with `s7b` full `ep228`, where `tv` first appeared at step `564` and did
not trigger planner uptake, the `s8` run first detected `tv` as a target candidate
at step `490`:

- `first_target_step = 490`
- `first_target_captions = ["tv"]`
- `first_planner_after_target = 490`
- `first_object_commit_after_target = 490`
- `evidence_to_planner_latency = 0`
- `unhandled_target_candidate_steps = 0`

The controller switched to `object: tv` at step `490`, so the specific previously
observed failure mode—target-like object evidence being ignored during direction
search—has been repaired.

However, `ep228` still failed. The direct object path was not used; the object was
used as a frontier anchor:

- object anchor: `[180, 299]`
- selected frontier at first commit: `[447, 297]`
- projected local goal: `[147, 234]`
- `direct_object_goal = None`
- terminal: `FAILURE_NO_PROGRESS_TIMEOUT`

### Observation: `ep661` positive anchor is preserved

`ep661` remained successful:

- `SR = 1.0`
- `SPL = 0.8529184432440698`
- `first_target_step = 44`
- `first_planner_after_target = 44`
- `first_object_commit_after_target = 44`
- `first_direct_goal_after_target = 44`

This confirms the target-evidence patch did not break the local direct-object
positive anchor.

### Hypothesis: remaining `ep228` blocker moved to non-local target actionability

The new boundary is no longer semantic evidence uptake. It is the transition from
confirmed but non-local target evidence to an executable goal. In `ep228`, the
controller now correctly identifies `tv`, but because the object center is not
locally direct-projectable, it commits to an `object: tv` strategy while grounding
through frontier anchoring. The selected frontier remains far from the object
anchor, and the episode times out.

Next narrow intervention should therefore focus on the non-local target commit
contract:

- treat non-local target object evidence as a **target hypothesis / support
  region**, not as a stable object-stage commitment;
- keep searching via frontier/value until the target center becomes locally
  exploitable or visibility/approach progress improves;
- add explicit monitoring for object-anchor approach progress after target
  evidence uptake.

### Claim

The `s8` smoke check establishes a narrower boundary:

1. target-like evidence detection and immediate planner uptake now work on the
   known `ep228` hard case;
2. the `ep661` direct-object positive anchor is preserved;
3. `ep228` task success is still not recovered, so the remaining problem is
   non-local target actionability / target-anchor approach, not missing semantic
   uptake.

## 74. Completed `s9` / `s10` Non-local Target-anchor Follow-up

Date: 2026-04-23.

Remote server: `10.176.56.73`.

Result roots:

- `results/phase2_revalidation/s9_nonlocal_target_search_anchor_20260423/suite_summary.json`
- `results/phase2_revalidation/s10_target_anchor_local_map_refresh_20260423/suite_summary.json`

Scope:

- profile: `smoothnav-full`
- episode: `228` only
- purpose: keep the experiment minimal and test whether the `s8` non-local target
  blocker can be repaired without widening the suite.

### Patch under test

The follow-up patch changed the non-local text-target object contract:

1. if a text-goal object choice is locally projectable, keep the `object:<name>`
   stage so the clean direct-object path remains available;
2. if a text-goal object choice is **not** locally projectable, convert it to
   `unexplored target:<name>` and keep the object name as `anchor_object`;
3. pass controller-owned `explored_regions` / current target-region overrides
   through `plan_stage_goal`, so stuck/stagnant annotations are not lost behind a
   stale `WorldState` snapshot;
4. after `s9`, add a target-anchor local-map refresh hook: when a local map
   recenter happens while following `unexplored target:<name>`, re-ground the
   current target anchor so the local goal is not held in stale local-map
   coordinates.

Local and remote validation before smoke runs:

- local: `python -m unittest discover -s tests` passed (`148` tests after the
  refresh-hook test was added);
- remote: targeted `test_target_evidence_uptake.py`, `test_planner_gen2.py`,
  `test_controller_logic.py`, plus `py_compile` for changed runtime files passed.

### Observation: `s9` converts the target to search-anchor form but still fails

Result:

| Episode | Profile | SR | SPL | Terminal outcome |
| ---: | --- | ---: | ---: | --- |
| `228` | `smoothnav-full` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Target-evidence trace:

- `first_target_step = 490`
- `first_target_captions = ["tv"]`
- `first_planner_after_target = 490`
- `first_object_commit_after_target = null`
- `first_direct_goal_after_target = null`
- `evidence_to_planner_latency = 0`
- `target_event_count = 169`
- `unhandled_target_candidate_steps = 0`

Planner / grounding details around target uptake:

| Step | Planner choice | Strategy | Bias | Grounded frontier / local goal | Notes |
| ---: | --- | --- | --- | --- | --- |
| `490` | `object tv` | `unexplored target:tv` | `[180,299]` | `[447,297]` / `[147,234]` | target event handled immediately |
| `494` | `object tv` | `unexplored target:tv` | `[180,299]` | `[448,298]` / `[148,235]` | stuck replan reselected same anchor |
| `498` | `object tv` | `unexplored target:tv` | `[180,299]` | unchanged | auto-prefetch still same anchor |
| `658` | `object tv` | `unexplored target:tv` | `[180,299]` | `[490,131]` / `[82,17]` | final late target-like update, still fail |

Observation: the original `s8` object-stage lock is removed (`object: tv` no
longer appears after non-local target uptake), but the executable behavior is
still weak. The controller keeps reselecting the same non-local target anchor;
frontier grounding updates only a few times and does not convert the anchor into
successful approach or stop evidence.

Observation: the local goal became stale after local-map movement in the `s9`
trace. The same local goal `[148,235]` was held across local-map recentering,
while `dist_to_goal` jumped (for example around step `520`). This motivated the
`s10` local-map refresh hook.

### Observation: `s10` did not form a valid target-anchor test

Result:

| Episode | Profile | SR | SPL | Terminal outcome |
| ---: | --- | ---: | ---: | --- |
| `228` | `smoothnav-full` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Target-evidence trace:

- `first_target_step = null`
- `target_event_count = 0`
- `first_planner_after_target = null`

Observation: `s10` took a different stochastic high-level path and never detected
`tv` / `tv mirror` as a target candidate. Therefore it does **not** validate or
falsify the local-map refresh hook; it is a path-coverage miss, not evidence that
the target-anchor hook failed after activation.

### Hypothesis: concrete remaining blocker

The strongest current blocker is now a three-part target-anchor approach gap:

1. **Anchor persistence without progress semantics.** Once `tv` is detected, the
   planner can repeatedly choose the same non-local anchor after stuck/prefetch
   events because target evidence remains the only high-relevance object.
2. **Frontier value is still not target-approach aware enough.** The selected
   frontier is locally actionable, but not proven to decrease semantic distance
   to the target anchor or improve target visibility.
3. **Local-map goal coordinates can become stale.** A target frontier selected in
   one local map may be held after local-map recentering unless an explicit
   refresh/re-grounding event fires.

### Claim: narrowed boundary after `s9` / `s10`

Claim: converting non-local target objects from `object:<name>` to
`unexplored target:<name>` fixes the **stage-type** error but is insufficient for
`ep228` success. The next repair should not broaden the suite; it should add
value-level target-approach monitoring / decommit semantics around a single
non-local target anchor.

Recommended next minimal intervention:

- persist a `target_anchor_attempt` state for `unexplored target:<name>`;
- on stuck or repeated target-anchor replan, mark the current anchor attempt as
  `stalled` instead of immediately reselecting it;
- score frontiers by expected target-approach progress, not just generic
  local actionability + distance-to-anchor;
- keep the `s8`/`s9` trace metrics (`first_target_step`, planner latency,
  object-commit latency, selected frontier, target-anchor refresh events) as the
  required monitor set before another single-episode smoke run.

### Stop state

Observation: both remote runs completed and wrote aggregate summaries. A process
check after `s10` found no remaining `s9` / `s10` / `episode_id 228` experiment
processes on the remote server.

## 75. Completed `s12`–`s14` Temperature-controlled `ep228` Follow-up

Date: 2026-04-24.

Remote server: `10.176.56.73`.

Result roots:

- `results/phase2_revalidation/s12_temperature0_target_anchor_20260424/suite_summary.json`
- `results/phase2_revalidation/s13_temp0_pending_fix_20260424/suite_summary.json`
- `results/phase2_revalidation/s14_temp0_pending_fix_rerun_20260424/suite_summary.json`

### Patch / setup under test

This pass introduced two narrow controls:

1. a runtime env switch `SMOOTHNAV_LLM_TEMPERATURE` in `base_UniGoal/src/utils/llm.py`;
2. a controller fix so `unexplored target:<name>` is treated as more specific than
   generic direction search, and a stale pending direction is not allowed to
   overwrite the active target-anchor stage at `frontier_reached`.

The temperature switch does **not** change default behavior; it only affects runs
that explicitly export `SMOOTHNAV_LLM_TEMPERATURE=0`.

### Observation: `s12` (`temperature=0`) reached the target branch and validated the refresh hook

Result:

| Run | SR | SPL | Outcome |
| --- | ---: | ---: | --- |
| `s12 smoothnav-full ep228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Target-evidence trace:

- `first_target_step = 564`
- `first_target_captions = ["tv"]`
- `first_planner_after_target = 564`
- `evidence_to_planner_latency = 0`
- `first_object_commit_after_target = null`
- `first_direct_goal_after_target = null`
- `target_event_count = 371`

Refresh-hook evidence:

`target_anchor_local_map_refresh` grounding events fired repeatedly after the
first target uptake, for example at steps:

- `599`
- `639`
- `679`
- `719`
- `759`
- `799`
- `839`
- `879`
- `919`

Observation: this is the first direct evidence that the local-map refresh hook is
actually live on the non-local target-anchor branch.

### Observation: `s12` surfaced a concrete pending-promotion bug

At `s12` step `564`, the controller correctly switched to `unexplored target:tv`.
However, a stale prefetched direction target still existed in `pending_strategy`.
Later:

- step `599`: target-anchor refresh updated the local goal;
- step `610`: `frontier_reached` promoted the stale pending direction and the
  current strategy became `unexplored east`;
- step `612`: a new pending `unexplored target:tv` was created again.

Observation: this means the controller had already entered the correct target
branch, but `frontier_reached_pending` could still demote it back to generic
exploration. This is a controller bug, not a semantic-detection failure.

### Observation: `s13` / `s14` did not validate the pending-fix because they never re-entered the target branch

After applying the pending-fix, two more `temperature=0` single-episode reruns
were executed:

| Run | SR | SPL | first_target_step | target_event_count |
| --- | ---: | ---: | ---: | ---: |
| `s13` | `0.0` | `0.0` | `null` | `0` |
| `s14` | `0.0` | `0.0` | `null` | `0` |

Observation: both reruns failed before ever observing a `target_candidate_detected`
event. Therefore they do **not** validate or invalidate the pending-fix itself.
They only show that `temperature=0` is still insufficient to make the target
branch reproducible on demand.

### Claim: current highest-leverage blocker has changed again

Claim: after `s12`, the strongest newly verified blocker is no longer “does the
refresh hook run?” — it does. The stronger blocker is **branch reproducibility**:
we still cannot reliably force `ep228` into the same target-anchor branch, even
with fixed Habitat seed and `temperature=0`.

### Hypothesis: next step should stop blind reruns and move to controlled replay / branch capture

Hypothesis: the next minimal experiment should not be another blind `ep228`
rerun. It should use one of these tighter control surfaces:

1. replay / offline inspection starting from the `s12` target-anchor trace;
2. a captured-branch harness that reuses the `s12` planner / graph-delta state
   from step `564` onward;
3. stronger deterministic controls over non-Anthropic randomness before another
   online rerun.

### Stop condition for this pass

Observation: `s12`, `s13`, and `s14` all completed and wrote `suite_summary.json`.
A remote process check after `s14` showed no remaining `ep228` experiment
processes.

## 76. Entered `s12` Experimental Round: Branch Capture Interface

Date: 2026-04-24.

Purpose:

- stop using blind online reruns as the main validation surface for `ep228`;
- convert the already-observed `s12` target branch into a reusable analysis / replay input.

### Added tooling

New script:

- `scripts/capture_target_branch.py`

New test:

- `tests/test_capture_target_branch.py`

The script reads a completed SmoothNav run directory and writes:

- `target_branch_capture.json`

inside that run directory.

### Current captured artifact

Captured from:

- `results/phase2_revalidation/s12_temperature0_target_anchor_20260424/smoothnav-full/20260424/smoothnav_text_112731_317b2c0e`

Written artifact:

- `results/phase2_revalidation/s12_temperature0_target_anchor_20260424/smoothnav-full/20260424/smoothnav_text_112731_317b2c0e/target_branch_capture.json`

### Observation: the capture artifact preserves the key `s12` branch facts

The captured summary records:

- `first_target_step = 564`
- `first_target_anchor_step = 564`
- capture window: `564 -> 934`
- target-anchor segments:
  - `564 -> 609`
  - `632 -> 934`
- refresh count: `9`
- branch drop event:
  - step `610`
  - `from_region = unexplored target:tv`
  - `to_region = unexplored east`
  - `planner_reasons = [frontier_reached]`
  - `pending_region_previous_step = unexplored east`

### Claim

This is the first reusable branch-capture interface for the current `ep228`
problem. It gives the next experiment round a fixed, inspectable target branch
without requiring the full online episode to re-enter that branch every time.

### Immediate next use

The next narrow experiment should consume `target_branch_capture.json` to do one
of the following before any new online rerun:

1. verify the pending-promotion fix on the exact `s12` branch boundary around
   steps `599 -> 610 -> 632`;
2. add `target_anchor_attempt` lifecycle state and inspect whether the branch
   would remain stable under the captured controller context;
3. define a replay/counterfactual harness from the captured planner / monitor /
   grounding slices.

## 77. Completed `s12` Pending-fix Counterfactual Replay

Date: 2026-04-24.

Purpose:

- validate the stale pending-promotion fix on the exact captured `s12` branch,
  without depending on a fresh online rerun to re-enter that branch.

### Added tooling

New script:

- `scripts/replay_pending_fix_counterfactual.py`

New test:

- `tests/test_replay_pending_fix_counterfactual.py`

Generated artifact:

- `results/phase2_revalidation/s12_temperature0_target_anchor_20260424/smoothnav-full/20260424/smoothnav_text_112731_317b2c0e/pending_fix_counterfactual.json`

### Observation: the historical `s12` drop is correctly reconstructed

The counterfactual artifact records the exact historical boundary:

- step `609`: `current_target_region = unexplored target:tv`
- step `609`: `pending_target_region = unexplored east`
- step `610`: historical `current_target_region = unexplored east`

This matches the earlier diagnosis: the branch did enter the target-anchor state,
but a stale pending direction was promoted at `frontier_reached`.

### Observation: current code fixes the captured drop event

Counterfactual replay on the captured context produced:

- `applied_target_regions = ["unexplored target:tv"]`
- `current_target_region_after = unexplored target:tv`
- `pending_target_region_after = null`
- `pending_promoted = false`
- `planner_call_count = 0`

Verdict flags in the artifact:

- `kept_target_anchor = true`
- `cleared_stale_pending = true`
- `avoided_historical_drop = true`
- `bug_fixed_on_captured_context = true`

### Claim

This is the first controlled validation that the stale pending-promotion bug has
been repaired for the exact `s12` target-branch boundary that originally failed.

Importantly, this claim is **not** based on a new online run. It is based on a
counterfactual replay of the historical captured context.

### Remaining gap

The pending-fix is now validated on the captured branch boundary, but the full
`ep228` problem is not yet solved. The remaining open question is:

> once the target-anchor branch is preserved, does the branch itself make enough
> target-approach progress, or does it still need `target_anchor_attempt` /
> stalled-decommit / target-approach frontier value changes?

That should be the next controlled experiment, again using the captured `s12`
branch before returning to new online reruns.

## 78. Completed `s12` Target-anchor Attempt Stall Analysis

Date: 2026-04-24.

Purpose:

- move from “does the branch survive stale pending promotion?” to
  “once the branch survives, how aggressively should we mark it stalled?”
- use the captured `s12` branch to choose a safer `target_anchor_attempt`
  patience rule before editing runtime behavior.

### Added tooling

New script:

- `scripts/analyze_target_anchor_attempt.py`

New test:

- `tests/test_analyze_target_anchor_attempt.py`

Generated artifact:

- `results/phase2_revalidation/s12_temperature0_target_anchor_20260424/smoothnav-full/20260424/smoothnav_text_112731_317b2c0e/target_anchor_attempt_analysis.json`

### Observation: `s12` naturally decomposes into two target-anchor attempts

Attempt summary extracted from the artifact:

| Attempt | Window | Updates | Initial frontier-anchor dist | Best dist | Net improvement |
| ---: | --- | ---: | ---: | ---: | ---: |
| `1` | `564 -> 609` | `2` | `331.22` | `296.11` | `35.11` |
| `2` | `632 -> 934` | `11` | `361.00` | `265.17` | `95.83` |

Observation: attempt 2 is the real hard case. It does make large total progress,
but that progress is bursty and punctuated by long refresh sequences.

### Observation: `patience=3` is too aggressive on the captured branch

For attempt 2:

- policy: `patience_updates = 3`, `min_improvement = 1.0`
- simulated decommit step: `799`
- verdict: `would_false_early_decommit = true`
- later better evidence: step `824`, trigger `target_candidate_detected`,
  distance `265.89`

Observation: a `3`-update stall patience would have dropped the branch **before**
a later target-evidence refresh improved the frontier-anchor distance again.

### Observation: `patience=4` is materially safer on this trace

For attempt 2:

- policy: `patience_updates = 4`, `min_improvement = 1.0`
- simulated decommit step: `null`
- verdict: no false-early decommit on this trace

Observation: on the captured `s12` branch, increasing patience from `3` to `4`
is enough to avoid killing the branch before the later step-`824` target-evidence
refresh.

### Hypothesis: the first runtime attempt rule should be conservative

A reasonable first runtime rule is:

- introduce `target_anchor_attempt` state;
- count only target-branch updates (`target_candidate_detected`,
  `target_anchor_local_map_refresh`, `frontier_reached`) toward stall patience;
- do **not** decommit before at least `4` non-improving target-branch updates;
- use a small improvement threshold (around `1` full-map cell in current trace
  units) so the branch is not reset by tiny noise, but also not forced to show
  unrealistically large jumps.

### Claim

The captured `s12` branch now supports two separate controlled conclusions:

1. the stale pending-promotion bug is fixed on the historical captured context;
2. a naive `patience=3` target-anchor stall rule would be too aggressive for the
   captured hard-case branch.

That means the next runtime edit should **not** use an eager decommit threshold.
It should start with a more conservative `target_anchor_attempt` patience policy,
then return to controlled replay before another online rerun.

## 79. Completed `s15` Online Probe After Target-anchor Attempt Runtime Patch

Date: 2026-04-24.

Result root:

- `results/phase2_revalidation/s15_target_anchor_attempt_online_probe_20260424/suite_summary.json`

Scope:

- profile: `smoothnav-full`
- episode: `228`
- setting: `SMOOTHNAV_LLM_TEMPERATURE=0`
- code state: includes `target_anchor_attempt` runtime state and conservative
  stall policy (`patience≈4`, `min_improvement≈1` via runtime defaults)

### Observation: online probe still failed before reaching the target branch

Result:

| Run | SR | SPL | Outcome |
| --- | ---: | ---: | --- |
| `s15 smoothnav-full ep228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Target-evidence trace:

- `first_target_step = null`
- `target_event_count = 0`
- `first_planner_after_target = null`
- `interesting_count(target-anchor trace fields) = 0`

Observation: this run never entered `unexplored target:tv`, so it does not test
whether the new runtime `target_anchor_attempt` state improves the branch once
entered.

### Observation: the runtime patch did not introduce visible regressions

Although `s15` did not reach the target branch, it also showed:

- `executor_override_ratio = 0.0`
- `goal_update_delay_steps ≈ 3.78`
- no new grounding-noop failure mode
- no target-anchor stall-replan events fired (because no target-anchor branch was entered)

This is weak but useful negative evidence: the patch did not obviously destabilize
non-target-branch execution.

### Claim

After `s15`, the decision is now clear:

> do **not** return to online reruns as the primary validation surface yet.

The replay/capture validation is still the higher-leverage path, because the main
remaining blocker is branch reproducibility, not target-anchor logic correctness
on an already-captured branch.

### Recommended next step

Before another online rerun, use the captured `s12` branch to implement and test:

1. `target_anchor_attempt`-aware stalled replan counterfactual on the branch;
2. target-approach utility changes at the frontier/value level;
3. stronger deterministic branch-capture / partial replay tooling.

Only after one of those produces a stronger captured-branch improvement should a
new online `ep228` probe be considered worthwhile.

## 80. Full Experiment Review And Coverage-oriented Detection Plan

Date: 2026-04-26.

A full review report has been written to:

- `analysis/flywheel_runs/20260426-smoothnav-complete-experiment-review-and-detection-plan.md`

### Observation: current evidence boundary

The completed experiments now separate the problem into two high-confidence layers:

1. `ep228` / `ep527` are shared repaired cross-scene hard cases, not a
   `smoothnav-full`-only monitor regression. `ep661` remains the positive anchor
   that must keep the local direct-object path.
2. `s8` / `s9` / `s12` show that target evidence can be detected and consumed by
   the planner, but non-local `unexplored target:<name>` still does not reliably
   become successful target approach.
3. `s10` / `s11` / `s13` / `s14` / `s15` show that online `ep228` reruns often do
   not enter the target branch at all, even with `SMOOTHNAV_LLM_TEMPERATURE=0`.
   Therefore online rerun is still a poor primary validation surface.

### Observation: new highest-leverage diagnostic from the review

The `s9` / `s12` target-frontier audit found that the target branch can lose its
semantic value signal after the target anchor becomes far in FMM space:

- `s9`: `1 / 3` target grounding events had `bias_score = 0`.
- `s12`: `11 / 13` target grounding events had `bias_score = 0`.
- In late `s12`, both `base_score` and `bias_score` are frequently `0`, so the
  selected frontier is effectively chosen by `novelty_score + actionability_score`
  rather than by target approach.

Concrete late-branch examples:

| Step | Trigger | bias distance | bias score | base score | Reading |
| ---: | --- | ---: | ---: | ---: | --- |
| `632` | `frontier_reached` | `11.69` | `0.0` | `4.03` | semantic term saturated |
| `799` | `target_anchor_local_map_refresh` | `24.49` | `0.0` | `0.0` | novelty/actionability only |
| `824` | `target_candidate_detected` | `24.94` | `0.0` | `0.0` | target evidence updates, but value remains flat |
| `919` | `target_anchor_local_map_refresh` | `24.99` | `0.0` | `0.0` | branch retained, approach utility still weak |

### Claim

The current strongest diagnosis is no longer just “target branch not held”. The
captured replay evidence shows the stale pending drop is fixed and the conservative
attempt policy is safe on `s12`. The next primary blocker is that the frontier
value function loses target-discriminative power when target-anchor FMM distances
saturate, causing the branch to behave like generic exploration.

### Required detection coverage before the next online run

The review proposes a layered detector that classifies failures across:

1. suite/artifact/config validity;
2. dataset/scene/episode integrity;
3. determinism and branch reproducibility;
4. perception / graph target evidence;
5. planner choice and menu contract;
6. strategy contract (`object:<name>` vs `unexplored target:<name>` vs direct goal);
7. grounding / frontier-value diagnostics;
8. local-map coordinate lifecycle;
9. controller pending / prefetch / target-attempt state;
10. monitor trigger coverage;
11. executor / tactical / low-level control;
12. terminal arbitration and metric correctness;
13. replay / counterfactual coverage.

### Next implementation order

Do not return to broad online reruns yet. The smallest useful next tools are:

1. offline `target_frontier_value` audit / counterfactual for the captured `s12`
   branch;
2. branch reproducibility comparator for `s10`–`s15`;
3. run-level trace contract auditor that assigns one dominant first-failing layer
   to each failed run.

## 81. Monitoring System Implemented And `s16` Single-episode Audit Completed

Date: 2026-04-26.

### Observation: layered monitoring has been implemented

New monitoring tool:

- `scripts/audit_trace_contracts.py`

New tests:

- `tests/test_audit_trace_contracts.py`

The auditor now checks completed run directories across:

1. suite/artifact/config validity;
2. dataset/scene/episode metadata;
3. determinism / branch reproducibility;
4. perception / graph target evidence;
5. planner menu / choice;
6. strategy contract;
7. grounding / frontier value;
8. local-map lifecycle;
9. controller pending / prefetch;
10. target-anchor attempt;
11. monitor trigger coverage;
12. executor / low-level control;
13. terminal / metric arbitration;
14. replay / counterfactual coverage.

Verification evidence:

- local targeted: `python3 -m unittest tests/test_audit_trace_contracts.py` -> `5` tests OK;
- local compile: `python3 -m py_compile scripts/audit_trace_contracts.py tests/test_audit_trace_contracts.py` -> OK;
- remote targeted on server 73: `5` tests OK;
- remote full suite on server 73: `156` tests OK.

### Observation: `s16` failed, but the first failing layer is now explicit

Result root:

- `results/phase2_revalidation/s16_contract_audit_ep228_20260426/suite_summary.json`

Run:

- `results/phase2_revalidation/s16_contract_audit_ep228_20260426/smoothnav-full/20260426/smoothnav_text_195315_7c1397e9`

Outcome:

| Profile | Episode | SR | SPL | Terminal outcome |
| --- | ---: | ---: | ---: | --- |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Audit artifact:

- `results/phase2_revalidation/s16_contract_audit_ep228_20260426/contract_audit.json`

Dominant audit classification:

- `dominant_failure_layer = planner_menu_choice`
- `dominant_failure_reason = target evidence was not consumed by planner`

### Evidence: target evidence appeared, but empty LLM responses prevented target commitment

Target-evidence metrics:

- `first_target_step = 668`
- `target_event_count = 2`
- `first_planner_after_target = 668`
- `evidence_to_planner_latency = 0`
- `first_object_commit_after_target = null`
- `first_direct_goal_after_target = null`

Planner audit metrics:

- `planner_call_count = 16`
- `empty_response_count = 13`
- `fallback_triggered_count = 13`
- `target_in_prompt_after_target = true`
- `selected_target_after_target = false`

At step `668`, the prompt did contain `[object] tv` and the replan reason was
`Target-like object detected, decide whether to commit`; however, the LLM response
was empty, causing deterministic parse-failure fallback to stay on generic
direction search instead of entering `unexplored target:tv`. The same pattern
reappeared at step `768`.

### Cross-run audit: `s10`–`s16`

Aggregate audit artifact:

- `results/phase2_revalidation/s10_s16_contract_audit_20260426.json`

Across seven `ep228/smoothnav-full` online runs:

- `determinism_branch_reproducibility`: `5` runs;
- `controller_pending_prefetch`: `1` run (`s12` historical stale pending drop);
- `planner_menu_choice`: `1` run (`s16` target evidence seen but not consumed);
- target branch coverage rate: `1 / 7 = 0.1429`.

### Claim

The complete monitor confirms that online `ep228` currently has two gates before
target-approach can be evaluated:

1. the run must actually expose target evidence;
2. once evidence appears, planner output / fallback must preserve the target
   choice instead of reverting to generic direction search.

`s16` passed gate 1 weakly (`2` target events) but failed gate 2 because empty LLM
responses triggered deterministic direction fallback.

### Next code decision

The next minimal code intervention should be a target-aware planner fallback:

- if an LLM response is empty or unparsable;
- and the current structured choices contain a high-relevance target object;
- then return the target object choice instead of deterministic direction fallback.

This should preserve the existing non-local object conversion to
`unexplored target:<name>` and should be verified with unit tests before another
online `ep228` audit run.

## 82. Empty LLM Response Root Cause, Target-aware Fallback Fix, And `s18` Monitored Replay

Date: 2026-04-26.

### Observation: the "empty LLM response" was an API failure being masked as blank text

Remote `s16` logs show the planner was not receiving genuine empty model text.
The underlying wrapper exhausted retries on Clauddy HTTP `403` responses:

- endpoint: `https://clauddy.com/v1/messages`;
- error message: `分组 default 已被弃用`;
- wrapper behavior: `base_UniGoal/src/utils/llm.py::_retry_api_call` returned `""` after retry exhaustion;
- planner behavior: `smoothnav/planner.py::HighLevelPlanner.plan` recorded `error_message = "empty_response"` and fell back to deterministic direction search.

At `s16` step `668`, the planner prompt already contained `[object] tv`, but the
empty-response direction fallback selected `unexplored west`, so target evidence
was not consumed.

### Observation: implemented fixes

Code changes:

1. `base_UniGoal/src/utils/llm.py`
   - records `LLM.last_error` / `VLM.last_error` when retry exhaustion returns
     the legacy empty string;
   - keeps the existing return contract (`""`) so callers do not crash, but the
     planner trace can now expose the real API error.
2. `smoothnav/planner.py`
   - adds `maybe_target_object_parse_failure_fallback(...)`;
   - after empty/unparseable LLM responses, if the structured choices contain a
     high-relevance unsearched primary target object, returns that object instead
     of a generic direction fallback;
   - preserves the existing non-local conversion to `unexplored target:<name>`;
   - records `llm_error_message` in planner traces and `last_plan_meta`.
3. `smoothnav/main.py` + `smoothnav/controller_logic.py`
   - fixes a second gate found by `s17`: `plan_strategy_budgeted(...)` previously
     discarded every planner result whose `error_message == "empty_response"` and
     restored the old strategy, even when the result was now a target-aware
     semantic fallback;
   - generic empty-response direction fallbacks are still suppressed when a
     current plan exists, but strictly more specific fallbacks such as
     `unexplored target:tv` are now allowed through.
4. `scripts/analyze_target_evidence_trace.py`
   - reports `first_target_anchor_after_target` and
     `evidence_to_target_anchor_latency`, so target-anchor uptake is no longer
     misreported as no commitment.

### Verification: code-level tests

Local verification:

- `python3 -m unittest tests.test_analyze_target_evidence_trace tests.test_controller_logic tests.test_planner_gen2 tests.test_llm_protocols` -> `60` tests OK;
- `python3 -m py_compile scripts/analyze_target_evidence_trace.py smoothnav/main.py smoothnav/controller_logic.py smoothnav/planner.py base_UniGoal/src/utils/llm.py` -> OK.

Remote verification on server `73`:

- targeted planner + LLM tests -> `38` tests OK;
- targeted controller test -> `21` tests OK;
- target-evidence trace test -> `1` test OK;
- full per-file test sweep with `PYTHONPATH=.` -> `26` test files OK, `0` failed.

### Observation: `s17` proved the planner fallback worked but exposed a controller adoption gate

Result root:

- `results/phase2_revalidation/s17_target_aware_llm_empty_fix_ep228_20260426`

Outcome:

| Profile | Episode | SR | SPL | Terminal outcome |
| --- | ---: | ---: | ---: | --- |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

At step `668`, planner trace showed:

- `error_message = empty_response`;
- `llm_error_message = HTTP 403 ... 分组 default 已被弃用`;
- `choice_type = object`, `choice_id = tv`;
- `strategy.target_region = unexplored target:tv`;
- `parsed_result.reasoning = target_aware_parse_failure_fallback...`.

However, the step trace still showed `current_strategy = unexplored south`. The
reason was `plan_strategy_budgeted(...)` discarding any empty-response-derived
plan when `fallback_strategy` existed. This was fixed before `s18`.

### Observation: `s18` confirmed target branch adoption under monitoring

Result root:

- `results/phase2_revalidation/s18_empty_response_specificity_fix_ep228_20260426`

Run:

- `results/phase2_revalidation/s18_empty_response_specificity_fix_ep228_20260426/smoothnav-full/20260426/smoothnav_text_204449_5ef821c8`

Outcome:

| Profile | Episode | SR | SPL | Terminal outcome | Steps |
| --- | ---: | ---: | ---: | --- | ---: |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` | `764` |

Artifacts:

- `results/phase2_revalidation/s18_empty_response_specificity_fix_ep228_20260426/suite_summary.json`
- `results/phase2_revalidation/s18_empty_response_specificity_fix_ep228_20260426/contract_audit.json`
- `results/phase2_revalidation/s18_empty_response_specificity_fix_ep228_20260426/target_evidence_trace.jsonl`

Key audit metrics:

- `first_target_step = 668`;
- `first_planner_after_target = 668`;
- `selected_target_after_target = true`;
- `first_target_anchor_after_target = 668`;
- `evidence_to_target_anchor_latency = 0`;
- `target_branch_covered = true`;
- target-anchor attempt observations: `96` steps;
- target-anchor updates: `6`;
- `best_distance = 305.235974288746`;
- `last_distance = 376.06648348397124`;
- `stall_updates = 3`;
- dominant failure layer moved to `executor_low_level_control`.

### Observation: the remaining blocker is no longer planner target uptake

`s18` moved the failure boundary downstream:

1. target evidence appeared once (`tv` at step `668`);
2. target-aware planner fallback selected `tv` immediately despite the broken LLM
   API path;
3. controller adopted `unexplored target:tv` from step `668` onward;
4. grounding produced multiple target-anchor frontier updates;
5. low-level execution still failed before success.

Target-anchor grounding samples show the search did not monotonically approach the
anchor:

- first selected-frontier anchor distance: `313.60`;
- best observed distance: `305.24` at step `699`;
- final observed distance: `376.07` at step `759`;
- `target_selected_frontier_anchor_distance_delta = -62.47`;
- target semantic term share stayed low-to-moderate (`median ≈ 0.2425`);
- several top-1/top-2 frontier gaps were very small, e.g. `0.0024` at step `727`
  and `0.0034` at step `759`.

### Hypothesis

The current primary blocker is now target-anchor execution/grounding quality, not
planner target recognition. The frontier value function can keep selecting
frontiers near the target bias in map distance, but the selected frontiers do not
translate into reliable physical approach or visual confirmation. The low-level
controller then spends many steps oscillating / pausing around changing local
frontier goals without direct target lock.

### Claim

The empty-LLM-response failure is fixed at the SmoothNav controller level for the
critical target-evidence case: with the Clauddy API still returning HTTP `403`,
`s18` nevertheless converts `[object] tv` into an adopted `unexplored target:tv`
strategy at the same step target evidence appears.

The episode still fails, so `ep228` is now a downstream execution/grounding hard
case. The next minimal experiment should not add more LLM planning. It should
replay or instrument the `s18` target-anchor segment (`668`-`763`) and test one
change at a time in the frontier/value/executor layer, especially:

1. make target-anchor decommit/replan happen when selected-frontier anchor distance
   worsens sharply before timeout;
2. compare target-anchor frontier scoring with a direct local projection / visible
   target lock when the target object center is known;
3. add an executor-level detector for high pause ratio / repeated local-goal churn
   during an active target anchor.

## 83. LLM API Smoke Test And Cheap Claude Probe

Date: 2026-04-26.

### Observation: current remote API failure is credential/group-level, not model-level

Added a reusable smoke script:

- `scripts/smoke_llm_api.py`

The script uses the exact SmoothNav `base_UniGoal/src/utils/llm.py::LLM` call path,
prints only sanitized context, and probes one or more model IDs with a tiny prompt.

Remote smoke command on server `73`:

```bash
cd /mnt/sdd/xxy/SmoothNav
. .local/clauddy.env.sh
SMOOTHNAV_LLM_MAX_RETRIES=1 SMOOTHNAV_LLM_RETRY_DELAYS=0 \
  /mnt/sdd/xxy/miniconda3/envs/unigoal/bin/python scripts/smoke_llm_api.py \
  --model claude-haiku-4-5-20251001 \
  --model claude-haiku-4-5 \
  --model claude-3-5-haiku-20241022 \
  --model claude-sonnet-4-6
```

Sanitized context:

- `base_url_host = clauddy.com`
- `SMOOTHNAV_API_KEY` is present, length `51`
- provider/protocol: `anthropic` / `anthropic-messages`

All tested Claude-family models failed with the same error:

- `HTTP 403 from https://clauddy.com/v1/messages`
- `分组 default 已被弃用`

Tested cheap / cheaper Claude candidates:

- `claude-haiku-4-5-20251001`
- `claude-haiku-4-5`
- `claude-3-5-haiku-20241022`

Also tested a stronger current Sonnet candidate:

- `claude-sonnet-4-6`

### Claim

Switching the model name from Sonnet to Haiku does not fix the current remote API
issue. The proxy accepts the request path far enough to return its own structured
error, and the error is identical across models. The active blocker is the
Clauddy account/key routing group (`default`) being deprecated, not SmoothNav's
model name or JSON request schema.

### Next operational fix

To restore real LLM calls, update `/mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh`
with a key or base URL bound to an active Clauddy group. After updating, rerun:

```bash
cd /mnt/sdd/xxy/SmoothNav
. .local/clauddy.env.sh
SMOOTHNAV_LLM_MAX_RETRIES=1 SMOOTHNAV_LLM_RETRY_DELAYS=0 \
  /mnt/sdd/xxy/miniconda3/envs/unigoal/bin/python scripts/smoke_llm_api.py \
  --model claude-haiku-4-5-20251001
```

The preferred low-cost first model remains `claude-haiku-4-5-20251001` (or alias
`claude-haiku-4-5` if the proxy supports aliases). If quality is insufficient,
use `claude-sonnet-4-6` only for planner calls and keep monitor/VLM on Haiku.

### Observation: temporary-key two-model Claude-only smoke

A follow-up minimal smoke used a temporary user-provided key override only in the
shell environment; the key was not written to `.local/clauddy.env.sh` and is not
recorded here.

Command shape on server `73`:

```bash
cd /mnt/sdd/xxy/SmoothNav
. .local/clauddy.env.sh
export SMOOTHNAV_API_KEY=<temporary-user-key>
SMOOTHNAV_LLM_MAX_RETRIES=1 SMOOTHNAV_LLM_RETRY_DELAYS=0 \
  /mnt/sdd/xxy/miniconda3/envs/unigoal/bin/python scripts/smoke_llm_api.py \
  --model claude-3-5-haiku-20241022 \
  --model claude-3-haiku-20240307
```

Sanitized context remained:

- `base_url_host = clauddy.com`
- `SMOOTHNAV_API_KEY` is present, length `51`
- provider/protocol: `anthropic` / `anthropic-messages`

Both Claude-family probes failed before returning text:

- `claude-3-5-haiku-20241022`: `HTTP 503`, `model_not_found`,
  `No available channel for model ... under group Claude (distributor)`
- `claude-3-haiku-20240307`: `HTTP 503`, `model_not_found`,
  `No available channel for model ... under group Claude (distributor)`

This is different from the previous `default`-group deprecation error: this key
routes into a `Claude (distributor)` group, but that group has no available
channel for the two exact Anthropic Haiku model IDs tested.

### Observation: Opus 4.5 Clauddy smoke succeeded with Anthropic env shape

A second temporary-key smoke used the user-provided Clauddy/Anthropic environment
shape without persisting the key:

```bash
export ANTHROPIC_BASE_URL=https://clauddy.com
export ANTHROPIC_AUTH_TOKEN=<temporary-user-key>
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export SMOOTHNAV_API_BASE=$ANTHROPIC_BASE_URL
export SMOOTHNAV_API_KEY=$ANTHROPIC_AUTH_TOKEN
SMOOTHNAV_LLM_MAX_RETRIES=1 SMOOTHNAV_LLM_RETRY_DELAYS=0 \
  /mnt/sdd/xxy/miniconda3/envs/unigoal/bin/python scripts/smoke_llm_api.py \
  --model claude-opus-4-5-20251101
```

Result on server `73`:

- `base_url_host = clauddy.com`
- `SMOOTHNAV_API_KEY` present, length `51`
- provider/protocol: `anthropic` / `anthropic-messages`
- model: `claude-opus-4-5-20251101`
- `ok = true`
- response text: `OK`

Claim: the current temporary key has at least one working Anthropic Messages
channel when using `claude-opus-4-5-20251101`. The earlier failures were model / channel
availability issues for Haiku-family IDs under the key's group, not a universal
credential failure.

## 84. `s19` Sonnet 4.5 Main-model Online Probe

Date: 2026-04-26.

### Observation: `claude-sonnet-4-5-20250929` API channel is usable

A smoke test on server `73` used the private, gitignored Clauddy environment file
and the Anthropic Messages path:

- base host: `clauddy.com`
- provider/protocol: `anthropic` / `anthropic-messages`
- model: `claude-sonnet-4-5-20250929`
- result: `ok = true`, response text `OK`

The raw key is intentionally stored only in `.local/clauddy.env.sh` on the remote
machine and is not recorded in this document.

### Experiment setup

Result root:

- `results/phase2_revalidation/s19_sonnet45_ep228_20260426`

Run:

- `results/phase2_revalidation/s19_sonnet45_ep228_20260426/smoothnav-full/20260426/smoothnav_text_212734_c46b851b`

Config:

- `base_UniGoal/configs/config_habitat_sonnet45.yaml`
- `llm_model = claude-sonnet-4-5-20250929`
- `llm_model_fast = claude-sonnet-4-5-20250929`
- `vlm_model = claude-sonnet-4-5-20250929`

Using Sonnet for the fast/VLM slots was deliberate for this narrow probe: the
current key previously had no available channel for the tested Haiku model IDs,
so leaving Haiku in the auxiliary slots would confound the main-model test with
known API-channel failures.

### Outcome

| Profile | Episode | SR | SPL | Terminal outcome | Steps |
| --- | ---: | ---: | ---: | --- | ---: |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` | `999` |

Aggregate summary:

- `results/phase2_revalidation/s19_sonnet45_ep228_20260426/suite_summary.json`

Key suite metrics:

- `avg_high_level_calls = 20.0`
- `avg_low_level_calls = 6.0`
- `goal_update_delay_steps = 21.733333333333334`
- `pending_created_count = 9`
- `pending_promoted_count = 5`
- `grounding_noop_reason_counts = {"out_of_local_window": 8}`
- `executor_override_ratio = 0.0`
- no `HTTP 4xx/5xx`, `LLM failed`, or `empty_response` messages were found in the launcher log.

### Contract audit

Artifacts:

- `results/phase2_revalidation/s19_sonnet45_ep228_20260426/contract_audit.json`
- `results/phase2_revalidation/s19_sonnet45_ep228_20260426/target_evidence_trace.jsonl`

Dominant failure:

- `dominant_failure_layer = grounding_frontier_value`
- `dominant_failure_reason = target frontier value lost semantic signal`

Important layer statuses:

- `perception_graph_target_evidence = pass`
- `planner_menu_choice = pass`
- `target_anchor_attempt = pass`
- `executor_low_level_control = pass`
- `grounding_frontier_value = fail`
- `local_map_lifecycle = fail`

Target-evidence trace:

- `first_target_step = 330`
- `first_planner_after_target = 330`
- `first_target_anchor_after_target = 330`
- `evidence_to_planner_latency = 0`
- `evidence_to_target_anchor_latency = 0`
- `target_event_count = 669`
- `unhandled_target_candidate_steps = 0`

Planner/API behavior improved relative to `s18`:

- `empty_response_count` changed from `11` in `s18` to `0` in `s19`;
- target evidence appeared much earlier (`330` vs `668`);
- planner selected `unexplored target:tv` immediately at the first post-evidence planner call.

But the online episode still failed because target-anchor grounding degraded:

- `target_grounding_event_count = 30`
- `target_semantic_term_share_median = 0.0`
- `target_base_score_zero_rate = 0.9`
- `target_bias_score_zero_rate = 0.9`
- `target_best_seen_anchor_distance_delta = 150.65249296345007`
- `target_selected_frontier_anchor_distance_delta = 85.64869311605929`
- `grounding_noop_reason_counts = {"out_of_local_window": 8}`
- `local_map_lifecycle.stale_local_goal_risk_count = 3`
- stale local-goal risk steps: `[334, 355, 435]`
- target-anchor refresh steps: `[359, 399, 439, 479, 519, 559, 599, 639, 679, 719, 759, 799, 839, 879, 919, 959]`

### Claim

Switching the main model to `claude-sonnet-4-5-20250929` fixes the API/empty-LLM
blocker for this probe, but does **not** solve `ep228`. The failure boundary is
now cleaner and lower-level: target evidence is detected, consumed by the
planner, and converted into a target anchor; the remaining blocker is that the
frontier-value / local-map grounding loop repeatedly loses semantic pull for the
non-local `tv` target and sometimes keeps stale local goals after local-map
movement.

### Next narrow repair target

Do not broaden the suite yet. The next code change should target the
`grounding_frontier_value` / `local_map_lifecycle` boundary around a single
`ep228` replay or online smoke:

1. keep semantic target value non-zero when the target anchor is outside the
   current local map but graph evidence is persistent;
2. invalidate or reproject local goals immediately after local-map movement for
   active target anchors;
3. make target-anchor frontier scoring report whether it is optimizing distance
   to the latest semantic target center or drifting to generic frontier novelty.

## 85. Implemented Minimal Target-progress Repair And `s20` Online Smoke

### Patch scope

Observation:

The minimal repair was implemented as a deterministic target-progress value term
plus target-anchor lifecycle diagnostics, without adding new dependencies or
changing the planner model:

- `smoothnav/frontier_scoring.py`
  - added `compute_target_progress_scores`;
  - threaded `target_progress_score`, `target_progress_weight`, and
    `target_progress_mode` into frontier score composition and trace summaries.
- `base_UniGoal/src/graph/graph.py`
  - enables target-progress scoring only for active
    `unexplored target:<label>` strategies;
  - uses target FMM distance when sane and Euclidean fallback when FMM distances
    are missing or sentinel-like;
  - records `target_progress_active`, `target_progress_mode`, and related config
    in `last_goal_debug`.
- `smoothnav/strategy_grounding.py`
  - writes the active target region / anchor object onto the graph before
    grounding, so graph scoring can distinguish target anchors from generic
    directional frontiers.
- `smoothnav/controller_state.py`, `smoothnav/controller_logic.py`,
  `smoothnav/main.py`
  - persist target-anchor label/full-map coord/last projected local coord/value
    source;
  - invalidate stale local goals when an active target anchor moves outside the
    current local window;
  - trace whether the target anchor was reprojected or reprojection failed.
- `scripts/audit_trace_contracts.py`
  - counts `target_progress_score_zero_rate`;
  - includes target-progress contribution in the target semantic-term share;
  - treats explicit target-anchor reprojection as a local-map refresh signal.
- Config:
  - `graph_target_progress_weight: 4.0`
  - `graph_target_progress_distance_scale: 20.0`

### Verification

Observation:

Local targeted verification passed:

```text
python3 -m py_compile smoothnav/frontier_scoring.py \
  smoothnav/strategy_grounding.py smoothnav/controller_logic.py \
  smoothnav/controller_state.py smoothnav/main.py \
  scripts/audit_trace_contracts.py base_UniGoal/src/graph/graph.py

python3 -m unittest \
  tests/test_graph_goal_scoring_gen2.py \
  tests/test_strategy_grounding_gen2.py \
  tests/test_controller_logic.py \
  tests/test_audit_trace_contracts.py

Ran 47 tests in 0.014s
OK (skipped=11)
```

Observation:

Remote `73` verification passed after syncing the patch to
`/mnt/sdd/xxy/SmoothNav`:

- targeted tests: `47` tests passed, `11` skipped;
- full repository test-file sweep: `26` `tests/test_*.py` files, `0` failures.

### Counterfactual check on the old `s19` target branch

Observation:

The updated audit on the old `s19` run still identifies the original failure,
because the old trace has no target-progress fields:

- source:
  `results/phase2_revalidation/s19_sonnet45_ep228_20260426`
- refreshed audit:
  `results/phase2_revalidation/s19_sonnet45_ep228_20260426/contract_audit_after_target_progress_patch.json`
- `dominant_failure_layer = grounding_frontier_value`
- `dominant_failure_reason = target frontier value lost semantic signal`
- `target_grounding_event_count = 30`
- `target_progress_score_zero_rate = 1.0`
- `target_semantic_term_share_median = 0.0`
- `stale_local_goal_risk_count = 3`

Observation:

A lightweight counterfactual was then computed on the selected target-frontier
events from `s19`:

- artifact:
  `results/phase2_revalidation/s19_sonnet45_ep228_20260426/target_progress_counterfactual_from_s19_selected_frontiers.json`
- target selected-frontier events evaluated: `26`
- `target_progress_min = 0.5193265371794679`
- `target_progress_median = 0.5915337638446547`
- `target_progress_zero_count = 0`
- old zero semantic-term events: `23`
- counterfactual zero semantic-term events after adding
  `4.0 * target_progress_score`: `0`

Hypothesis:

This is strong unit/counterfactual evidence that the implemented value term
directly addresses the previously observed semantic-value collapse: even when
`base_score == 0` and `bias_score == 0`, the target anchor now retains a
non-zero value term instead of falling back to pure novelty.

It is not yet an online success claim, because the next online smoke did not
re-enter the same target-anchor branch.

### `s20` online smoke on the previously failed `ep228`

Observation:

The patched code was run as the smallest online smoke on the same failed
episode/profile:

- result root:
  `results/phase2_revalidation/s20_target_progress_ep228_20260426`
- run:
  `results/phase2_revalidation/s20_target_progress_ep228_20260426/smoothnav-full/20260426/smoothnav_text_221311_bb5cb465`
- config:
  `base_UniGoal/configs/config_habitat_sonnet45.yaml`
- profile / episode:
  `smoothnav-full`, `ep228`

Suite result:

| Profile | Episode | SR | SPL | Terminal outcome | Steps |
| --- | ---: | ---: | ---: | --- | ---: |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` | `722` |

Key metrics:

- `avg_high_level_calls = 6.0`
- `avg_low_level_calls = 0.0`
- `grounding_attempt_count = 5`
- `grounding_noop_count = 0`
- `pending_created_count = 4`
- `pending_promoted_count = 3`
- `strategy_switch_count = 2.0`

Target-evidence trace:

- `target_event_count = 0`
- `first_target_step = null`
- `first_target_anchor_after_target = null`
- `planner_calls = 6`

Contract audit:

- artifact:
  `results/phase2_revalidation/s20_target_progress_ep228_20260426/contract_audit.json`
- `dominant_failure_layer = determinism_branch_reproducibility`
- `dominant_failure_reason = target branch was not covered in this run`
- `perception_graph_target_evidence = fail`
- `grounding_frontier_value = pass`
- `local_map_lifecycle = pass`
- `target_anchor_attempt = pass`
- `target_branch_covered = false`

Observation:

The `s20` online smoke did **not** reproduce the `s19` target-anchor branch.
The planner trajectory was:

1. `unexplored north`
2. `unexplored south`

and it never detected a target candidate or selected
`unexplored target:tv`.

### Current interpretation

Claim:

The minimal code repair is implemented and passes local + remote regression
verification.

Claim:

The old `s19` failure boundary remains correctly identified:
`grounding_frontier_value` collapsed because target-anchor value became
semantically zero; the new target-progress term is non-zero on the same selected
target-frontier samples and removes the zero semantic-term condition in
counterfactual scoring.

Observation:

The first patched online smoke (`s20`) is not a valid online proof that the
target-progress repair fixes `s19`, because it failed earlier/differently:
target evidence was never detected and the target-anchor branch was not covered.

Hypothesis:

The next smallest useful validation is not a broader suite. It should be a
branch-covered replay/online probe that forces or reproduces the `s19` branch
condition (`target_event_count > 0` and `first_target_anchor_after_target !=
null`) and then checks:

1. `target_progress_score_zero_rate < 1.0`;
2. `target_semantic_term_share_median > 0`;
3. `grounding_noop_reason_counts["out_of_local_window"]` decreases or is paired
   with `grounding_patch_target_anchor_reprojected`;
4. `stale_local_goal_risk_count` decreases from the old `s19` value `3`.

## 86. Branch-level Grounding Snapshot Monitor

### Motivation

Observation:

Full online reruns of `ep228` are too expensive and not always branch
reproducible. The `s20` run did not reach the `s19` target-anchor branch, so it
could not validate whether the target-progress repair works at the failing
grounding boundary.

Claim:

The next validation should monitor the exact failing stage boundary and replay
that boundary offline, instead of requiring every attempt to rediscover the same
target branch from episode start.

### Implemented staged monitor

Observation:

A branch-level grounding snapshot hook has been added:

- online switch:
  - CLI: `--grounding-snapshot-policy {off,target,failures,all}`
  - env: `SMOOTHNAV_GROUNDING_SNAPSHOT_POLICY=target`
- output:
  - `<run_dir>/grounding_snapshots/episode_000000.jsonl`
- schema:
  - `smoothnav.grounding_stage_monitor.v1`
- replay snapshot schema:
  - `smoothnav.frontier_value_replay.v1`

Each target snapshot records the stage boundary:

1. semantic strategy before grounding:
   - `target_region`
   - `anchor_object`
   - `trigger`
2. frontier-value inputs:
   - frontier candidates
   - base / bias / novelty / actionability / repeat / recent terms
   - target-progress terms
   - candidate indices after filtering
   - all value weights
3. grounding result after the boundary:
   - selected frontier
   - projected local goal
   - local projection validity
   - target-anchor attempt state
4. immediate replay verdict:
   - `target_progress_nonzero`
   - `semantic_term_nonzero`
   - `selected_frontier_matches_snapshot`

### Offline replay

Observation:

The no-Habitat replay entrypoint is:

```bash
python scripts/replay_grounding_snapshot.py \
  <run_dir>/grounding_snapshots \
  --output <run_dir>/grounding_snapshot_replay.json
```

It recomputes the frontier-value ranking from the captured snapshot with the
current code and reports:

- `record_count`
- `target_record_count`
- `fail_count`
- `target_progress_zero_count`
- `target_semantic_term_zero_count`

### Audit integration

Observation:

`scripts/audit_trace_contracts.py` now has a separate stage:

- `grounding_stage_snapshot_replay`

Current old-run audit behavior:

- `s19`: `grounding_stage_snapshot_replay = warn`
  - reason: target branch existed but was produced before this hook, so no
    replay snapshot exists.
- `s20`: `grounding_stage_snapshot_replay = pass`
  - reason: no target branch was covered, so no target snapshot was expected.

This separates two different issues:

1. `grounding_frontier_value` — did the target-anchor scoring collapse?
2. `grounding_stage_snapshot_replay` — do we have a replayable hook for the
   precise boundary?

### Verification

Observation:

Remote `73` verification after adding the monitor:

- targeted tests:
  - `tests/test_grounding_snapshot_replay.py`
  - `tests/test_graph_goal_scoring_gen2.py`
  - `tests/test_audit_trace_contracts.py`
- all passed.

Full remote test-file sweep:

```text
SUMMARY test_files=27 failures=0
```

### Next validation shape

Hypothesis:

The next useful online run should be launched with:

```bash
SMOOTHNAV_GROUNDING_SNAPSHOT_POLICY=target \
CONFIG_FILE=base_UniGoal/configs/config_habitat_sonnet45.yaml \
./scripts/launch_profile_suite_background.sh \
  smoothnav-full \
  results/phase2_revalidation/s21_target_snapshot_ep228_20260426 \
  "228" 0 s21_target_snapshot_ep228
```

Then the decision should be made from the staged monitor:

1. if `target_event_count == 0`, the run did not cover the target branch and is
   a branch-reproducibility/perception-path probe, not a repair validation;
2. if target snapshots exist, run `scripts/replay_grounding_snapshot.py` and
   check whether the target branch passes the grounding boundary without a full
   second online rerun;
3. only if the replay passes should another online branch-covered smoke be used
   to test downstream execution effects.

## 87. Plan: BEV/MLLM Frontier-ID Planner-Grounding Contract

Date: 2026-04-27
Owner: Ralph / experiment lead
Scope: planning artifact only; no online experiment launch in this step.

### 1. Requirements Summary

Observation: The current SmoothNav high-level planner is a text-only planner. Its prompt lists the target, a serialized scene graph, already searched regions, a replan reason, and a textual menu of objects / rooms / cardinal directions (`smoothnav/planner.py:45-67`). `plan_stage_goal()` can access `world_state.bev_map`, but only extracts map size before calling the text planner (`smoothnav/planner.py:739-790`). The BEV map is already maintained by the mapping stack (`base_UniGoal/src/map/bev_mapping.py:202-220`, `base_UniGoal/src/map/bev_mapping.py:246-265`) and carried in `WorldState` (`smoothnav/types.py:558-596`; populated in `smoothnav/world_state.py:99-132`).

Observation: The current planner-grounding handoff is loose. The LLM returns `{choice_type, choice_id, reasoning}` (`smoothnav/planner.py:66-67`), which is resolved into a semantic `Strategy` and optional bias coordinate (`smoothnav/planner.py:648-686`). Grounding then decides whether to use a direct object goal or call `graph.get_goal(goal=bias)` (`smoothnav/strategy_grounding.py:266-303`). The frontier scorer already has replayable scalar terms and top-k diagnostics (`smoothnav/frontier_scoring.py:120-233`) and a snapshot schema (`smoothnav/frontier_scoring.py:532-612`), but it does not yet accept a planner-provided frontier prior.

Hypothesis: For hard cases where the target is not currently visible, the key missing information is not another text-only room/direction label, but a spatially grounded choice among executable frontier candidates. A BEV-aware MLLM should reason over the known layout, room/object evidence, and candidate frontiers, but the executable output should be a **frontier ID** selected from annotated candidates rather than a free-form BEV coordinate.

Claim: The next implementation should not start with a larger online suite. It should first build a replayable BEV/frontier prompt and strict ID-based planner-grounding contract, validate it offline on stored snapshots, then integrate it behind an off-by-default scoring prior and run one previously failed episode only after replay passes.

### 2. Design Principles

1. **Executable candidate set first**: The MLLM may reason freely, but it must choose from frontier IDs that are already valid for the current map and low-level planner.
2. **No free coordinates from the model**: Coordinates are generated by SmoothNav and stored in an ID-to-coordinate table. The MLLM returns IDs and scores, not raw map points.
3. **Replay before online**: Every new decision boundary must be testable without Habitat, detection, or a live API call.
4. **Planner and grounding must share one contract**: The planner output should become a frontier prior consumed by frontier scoring, not an unrelated text decision that grounding has to reinterpret.
5. **Smallest experiment wins**: Validate parser, renderer, replay, scoring, and one hard episode before any suite expansion.
6. **Monitor every stage**: If the run fails, we must know whether the failure is rendering, MLLM response, ID validation, stale candidates, scoring integration, projection, low-level execution, or perception/branch coverage.

### 3. Current Code Boundary

#### 3.1 Planner input today

Observation:

- Prompt template: target + scene text + searched regions + replan reason + text choices (`smoothnav/planner.py:45-67`).
- Scene serialization: room/object list plus up to 20 graph edges (`smoothnav/planner.py:70-103`).
- Choice menu: object choices, room choices, and four cardinal direction choices (`smoothnav/planner.py:106-183`).
- LLM call and parse: `HighLevelPlanner.plan()` constructs one text prompt and calls `self.llm(prompt=prompt)` (`smoothnav/planner.py:554-624`).
- Output mapping: parsed `choice_type` / `choice_id` becomes `Strategy.target_region`, optional `anchor_object`, and optional `bias_position` (`smoothnav/planner.py:648-686`).

Hypothesis:

This is enough for simple semantic choices but weak for long-horizon search because the model cannot see map topology, candidate frontier geometry, bottlenecks, unexplored room-like regions, or whether a direction is actually executable.

#### 3.2 BEV availability today

Observation:

- `BEV_Map` stores `full_map`, `local_map`, poses, local map boundary, and planner pose inputs (`base_UniGoal/src/map/bev_mapping.py:202-220`).
- `move_local_map()` writes local map updates into the global map and updates local boundaries / origins (`base_UniGoal/src/map/bev_mapping.py:246-265`).
- `WorldState` already carries `bev_map`, frontier count/summary, room summary, object summary, visible targets, and pose (`smoothnav/types.py:558-596`).
- `build_world_state()` fills `bev_map`, frontier summaries, rooms, objects, and visible target summaries (`smoothnav/world_state.py:99-132`).

Hypothesis:

The smallest architectural change is not to replace the map stack, but to add a BEV snapshot renderer and frontier-candidate annotation layer at the existing `WorldState -> planner` boundary.

#### 3.3 Grounding / frontier-value boundary today

Observation:

- Grounding writes the active target region / anchor into the graph, then calls `graph.get_goal(goal=bias)` unless a direct object goal is valid (`smoothnav/strategy_grounding.py:266-303`).
- `Graph.get_goal()` computes frontier candidates, bias, novelty, actionability, repeat/recent penalties, and target-progress terms (`base_UniGoal/src/graph/graph.py:868-940`, `base_UniGoal/src/graph/graph.py:1206-1220`).
- `frontier_locations_16` is retained as the current candidate set (`base_UniGoal/src/graph/graph.py:1024-1026`).
- `summarize_frontier_selection()` records top-k terms and selected frontier details (`smoothnav/frontier_scoring.py:120-233`).
- `build_frontier_value_replay_snapshot()` already stores replayable frontier-value inputs and selected frontier (`smoothnav/frontier_scoring.py:532-612`).
- `apply_strategy_with_trace()` already records grounding snapshots and immediate replay verdicts when enabled (`smoothnav/main.py:492-604`).

Hypothesis:

The right insertion point is an **optional planner prior score vector aligned to `frontier_locations_16`**, then included in `compose_frontier_value_scores()`, top-k summaries, selected frontier breakdowns, and replay snapshots.

### 4. Mature-work Anchors And SmoothNav Inference

Observation:

- SemExp / Goal-Oriented Semantic Exploration builds an episodic semantic map and uses goal-conditioned semantic exploration instead of relying only on local visual actions: https://arxiv.org/abs/2007.00643
- PONI separates “where to look?” from “how to navigate to (x, y)?” and predicts potential functions over a top-down semantic map: https://arxiv.org/abs/2201.10029 and https://vision.cs.utexas.edu/projects/poni/
- VLFM builds occupancy/frontier maps and uses vision-language value maps to select promising frontiers for zero-shot semantic navigation: https://arxiv.org/abs/2312.03275 and https://github.com/bdaiinstitute/vlfm
- ESC turns commonsense room/object priors into soft constraints for exploration rather than hard navigation goals: https://proceedings.mlr.press/v202/zhou23r.html
- OpenFrontier frames navigation as discrete sub-goal identification and uses set-of-mark frontier prompting with VLMs: https://openreview.net/forum?id=ER9ebVLq29
- FOM-Nav combines frontier-object maps with a VLM for high-level goal prediction and low-level planning: https://arxiv.org/abs/2512.01009

Inference for SmoothNav:

These works point toward the same interface pattern: use map/semantic context to estimate **where to search**, then let the classical map/low-level planner execute a constrained subgoal. For SmoothNav, this supports an ID-based frontier-prior design rather than asking the MLLM to output raw map coordinates.

### 5. Core Design Decision: MLLM Outputs Frontier IDs, Not Points

Decision:

When the target is not currently visible, the MLLM should output a ranked list of **frontier IDs** from an annotated BEV image and accompanying JSON candidate table.

Rejected alternative: MLLM outputs a raw BEV point.

Reason:

- A raw point can land in obstacle, explored free space, unknown space without an actionable boundary, or a different coordinate convention.
- Raw points are hard to replay because coordinate projection bugs and MLLM semantic reasoning bugs become entangled.
- A raw point bypasses the existing actionability, distance, repeat, recent, and target-progress filters.
- A point output makes stale-map validation harder: the system cannot know whether the point belonged to the candidate set seen by the model.

Accepted contract:

- SmoothNav computes candidate frontiers.
- SmoothNav renders each candidate as `F1`, `F2`, ..., `Fk` on a BEV image.
- SmoothNav emits a JSON table with each frontier ID, map coordinate, current score terms, nearby semantic hints, and validity metadata.
- The MLLM returns IDs and normalized preference scores.
- Grounding maps selected IDs back to SmoothNav-owned coordinates and treats them as a soft prior unless a direct visible target branch is valid.

### 6. Target Planner Contract

#### 6.1 Inputs to BEV/MLLM planner

The BEV-aware planner should receive three synchronized inputs:

1. **Annotated BEV image**
   - Occupied / free / unknown / explored channels rendered with a fixed legend.
   - Agent pose and heading marked clearly.
   - Current local-map boundary and global context marked if useful.
   - Frontier candidates labeled `F1..Fk`.
   - Optional room/object labels only if they are already observed and reliable.

2. **Frontier candidate JSON**
   - `snapshot_id`
   - `episode_id`, `step_idx`, `world_epoch`, `planner_epoch`
   - `target_text`
   - `frontiers`: each with `id`, `coord`, `agent_distance`, `base_score`, `bias_score`, `novelty_score`, `actionability_score`, `repeat_penalty`, `recent_penalty`, `target_progress_score`, and any nearby room/object hints
   - `candidate_indices` aligned to `frontier_locations_16`
   - `id_to_coord` table controlled by SmoothNav

3. **Semantic context text**
   - Mission target.
   - Searched / failed regions.
   - Visible target evidence if any.
   - Current strategy / reason for replanning.
   - Compact room-object summary from the existing graph.

#### 6.2 Output schema

The first implementation should accept only this schema:

```json
{
  "schema_version": "smoothnav.mllm_frontier_plan.v1",
  "decision_type": "explore_frontier",
  "target_visible": false,
  "selected_frontier_id": "F3",
  "ranked_frontiers": [
    {"id": "F3", "score": 0.78, "reason": "most likely path toward living-room-like open area"},
    {"id": "F1", "score": 0.51, "reason": "secondary unexplored branch"}
  ],
  "room_priors": [
    {"label": "living room", "score": 0.70},
    {"label": "bedroom", "score": 0.35}
  ],
  "avoid_frontier_ids": ["F5"],
  "grounding_contract": {
    "output_space": "frontier_id",
    "id_set": ["F1", "F2", "F3", "F4", "F5"],
    "must_not_output_raw_coordinate": true
  },
  "replan_if": [
    "selected frontier becomes invalid or unreachable",
    "new direct target evidence appears",
    "no map progress after N local steps"
  ],
  "confidence": 0.62
}
```

Allowed `decision_type` values:

- `direct_target`: only if target is visible / directly grounded by existing perception.
- `target_anchor`: if an observed object is probably the target but not locally projectable; still must produce frontier IDs for execution.
- `explore_frontier`: normal not-visible-target case; must include `selected_frontier_id` and ranked IDs.
- `hold_or_recover`: only for no valid frontiers / map inconsistency / severe uncertainty.

Invalid outputs:

- Any raw coordinate from the MLLM as the primary target.
- Any ID not in the current `id_set`.
- Any selected ID whose current SmoothNav candidate is stale or not actionable.
- Any response without a parseable schema version.

### 7. Implementation Phases

#### Phase A — Offline BEV/frontier snapshot renderer only

Goal: Generate the exact artifact that a future MLLM planner would see, without changing planner behavior.

Files likely touched:

- Add `smoothnav/bev_frontier_prompt.py` or `smoothnav/map_snapshot.py`.
- Add tests under `tests/test_bev_frontier_snapshot.py`.
- Extend docs in `docs/implementation/episode296_grounding_and_graph_growth_diagnosis_20260421.md`.

Steps:

1. Build a pure function that receives `WorldState`, graph frontier candidates, and optional current strategy.
2. Extract `frontier_locations_16` after `Graph.get_goal()` or from a captured replay snapshot.
3. Assign stable per-snapshot IDs `F1..Fk` in candidate order and store `id -> local_idx -> coord -> candidate_index`.
4. Render an annotated BEV PNG with agent pose, frontier IDs, obstacle/free/unknown legend, and optional observed semantic labels.
5. Save JSON sidecar next to the PNG.
6. Make this callable from replay snapshots first; online hooks can be added later.

Artifacts:

- `<run_dir>/mllm_frontier_snapshots/episode_000228_step_XXXXX.png`
- `<run_dir>/mllm_frontier_snapshots/episode_000228_step_XXXXX.json`

Acceptance criteria:

- Snapshot generation does not call any LLM.
- A synthetic map test verifies labels and ID table are deterministic.
- The JSON includes enough information to map every ID back to a SmoothNav frontier coordinate.
- If no frontiers exist, the artifact explicitly records `frontier_count=0` and a fallback reason.

#### Phase B — Strict parser and no-Habitat MLLM replay probe

Goal: Validate the MLLM contract without online navigation.

Files likely touched:

- Add `smoothnav/mllm_frontier_contract.py`.
- Add `scripts/replay_mllm_frontier_planner.py`.
- Add `tests/test_mllm_frontier_contract.py`.

Steps:

1. Implement a strict parser that accepts only `smoothnav.mllm_frontier_plan.v1`.
2. Validate `selected_frontier_id` membership in `id_set`.
3. Validate all ranked IDs, score ranges, and no free-coordinate primary goal.
4. Add a deterministic stub mode so replay tests never require an API.
5. Add optional real-API mode for a tiny manually selected snapshot set after parser tests pass.
6. Record raw response, parse status, invalid reason, selected ID, and ID-to-coordinate mapping.

Artifacts:

- `<snapshot_dir>/mllm_frontier_replay.json`
- `<snapshot_dir>/mllm_frontier_replay_failures.jsonl` if any parse/validation failures exist.

Acceptance criteria:

- Unknown ID is rejected.
- Raw coordinate primary target is rejected.
- Missing `ranked_frontiers` is rejected for `explore_frontier`.
- Stub replay passes on 3-5 stored snapshots.
- Real MLLM replay, when run, is never allowed to mutate online behavior.

#### Phase C — Optional frontier-prior integration behind default-off flag

Goal: Allow an ID-based planner decision to influence existing frontier scoring without bypassing existing safety/value terms.

Files likely touched:

- `smoothnav/frontier_scoring.py`
- `base_UniGoal/src/graph/graph.py`
- `smoothnav/strategy_grounding.py`
- related tests: `tests/test_graph_goal_scoring_gen2.py`, `tests/test_grounding_snapshot_replay.py`, possibly `tests/test_layered_contracts.py`

Steps:

1. Add optional planner prior vector aligned to `frontier_locations_16`.
2. Add config flag `graph_mllm_frontier_prior_weight`, default `0.0`.
3. Extend `compose_frontier_value_scores()` with `+ prior_weight * planner_prior_scores`.
4. Extend `summarize_frontier_selection()` to include `planner_prior_score` and `planner_prior_weight` in top-k and selected breakdown.
5. Extend replay snapshot schema with `planner_prior_scores`, `planner_prior_weight`, `planner_selected_frontier_id`, and `planner_prior_source`.
6. Ensure `weight=0.0` keeps legacy behavior equivalent in tests.
7. Reject stale prior if `world_epoch`, `frontier_id_set`, or candidate count mismatches.

Acceptance criteria:

- With prior weight `0.0`, current tests remain unchanged.
- With synthetic prior weight >0, replay can deterministically flip a close frontier ranking and record why.
- Invalid/stale ID falls back to current scoring and logs a structured fallback reason.
- Target-progress, actionability, repeat/recent penalties remain visible in the final breakdown.

#### Phase D — Online hook, still default-off

Goal: Make the BEV/MLLM planner callable online only when explicitly enabled.

Files likely touched:

- `smoothnav/planner.py`
- `smoothnav/main.py`
- config file for a one-off smoke profile only, e.g. `base_UniGoal/configs/config_habitat_sonnet45.yaml` or a new local experiment config.

Steps:

1. Add a planner mode flag such as `smoothnav_mllm_frontier_planner={off,snapshot,replay,online}`.
2. In `snapshot` mode, record BEV/frontier prompt artifacts but do not call MLLM.
3. In `replay` mode, consume a stored MLLM response / stub response and apply prior offline.
4. In `online` mode, call the configured MLLM only after snapshot and parser are verified.
5. Add budget controls: max calls per episode, timeout, retry count, and fallback to current planner.
6. Never log API keys; log provider/model name, prompt hash, image path, response hash, and validation state.

Acceptance criteria:

- Running with mode `off` preserves current code path.
- Running with mode `snapshot` writes artifacts and changes no action.
- Running with mode `online` writes parse/validation/selection traces before grounding.
- Empty API response is handled as a structured fallback, not a silent no-op.

#### Phase E — Minimal experiment sequence

Goal: Use the fewest experiments that can falsify the new contract.

Experiment labels proposed:

1. `s21a_bev_frontier_snapshot_replay`
   - Input: existing failed-episode grounding snapshots if available.
   - Action: render BEV/frontier prompt artifacts only.
   - Pass condition: all target-branch snapshots produce valid image/JSON and ID table.

2. `s21b_mllm_frontier_contract_replay`
   - Input: 3-5 stored snapshots, prioritizing ep228 target/search branch if captured.
   - Action: parser + stub response first; optional one real Sonnet 4.5 call after stub passes.
   - Pass condition: model chooses valid frontier IDs; parser rejects no output; fallback is visible if API fails.

3. `s21c_frontier_prior_counterfactual_replay`
   - Input: same snapshots.
   - Action: apply synthetic and/or MLLM prior in replay scoring only.
   - Pass condition: final scores show exactly how the prior changes or does not change selected frontier.

4. `s21d_ep228_single_online_smoke`
   - Input: previous failed ep228 only.
   - Action: one online run with `snapshot` first; only if replay passes, one online run with prior weight >0.
   - Pass condition: branch coverage, valid planner prior, valid grounding, and no regression in local projection / execution monitors.

Decision rules:

- If no target/search branch is captured, do not call it a planner repair failure; classify as branch coverage / perception path issue.
- If parser invalid rate is nonzero on controlled snapshots, fix contract/rendering before any online run.
- If MLLM selects a frontier that actionability strongly rejects, cap prior weight or require top-N actionability membership.
- If replay shows a good prior but online still fails after projection, inspect low-level execution, not planner semantics.
- If replay and online both choose the same failing frontier as current SmoothNav, the issue likely sits in perception/map/frontier candidate generation rather than high-level semantic reasoning.

### 8. Monitoring Plan

Every BEV/MLLM planner decision should record:

1. Snapshot metadata
   - `episode_id`, `step_idx`, `world_epoch`, `planner_epoch`, `snapshot_id`
   - `bev_image_path`, `json_path`, `prompt_hash`
2. Model call metadata
   - provider, model, timeout, retry count, response status
   - raw response path or hash; no API key
3. Parser/validator metadata
   - parse status, schema version, selected ID, ranked IDs
   - invalid reason if rejected
4. Frontier ID mapping
   - `id_to_coord`, `id_to_local_idx`, `id_to_candidate_index`
   - stale-candidate check result
5. Scoring integration
   - prior vector aligned to `frontier_locations_16`
   - prior weight
   - top-k before prior and after prior
   - selected frontier score breakdown
6. Grounding result
   - selected frontier coordinate
   - projected local goal
   - local projection validity
   - target-anchor attempt state
7. Downstream execution
   - movement progress after selecting the frontier
   - repeated-frontier reuse count
   - no-progress / stuck signals

The goal is that one failed run can be classified into one of these buckets:

- BEV render/label ambiguity
- invalid or empty MLLM response
- stale frontier ID
- prior too weak / too strong
- actionability conflict
- projection/local-goal failure
- low-level execution failure
- perception or target-branch coverage failure
- true hard-case layout ambiguity

### 9. Verification Steps

Unit tests:

- `tests/test_bev_frontier_snapshot.py`
  - synthetic map renders deterministic IDs and JSON table.
  - no-frontier case records explicit fallback.
- `tests/test_mllm_frontier_contract.py`
  - valid ID schema accepted.
  - unknown ID rejected.
  - raw coordinate output rejected.
  - `explore_frontier` without ranked IDs rejected.
- `tests/test_graph_goal_scoring_gen2.py`
  - prior weight `0.0` preserves existing behavior.
  - synthetic prior changes ranking only when configured.
- `tests/test_grounding_snapshot_replay.py`
  - replay snapshot includes prior fields when present.
  - stale prior falls back with structured reason.

Offline replay:

```bash
python scripts/replay_grounding_snapshot.py \
  <run_dir>/grounding_snapshots \
  --output <run_dir>/grounding_snapshot_replay.json

python scripts/replay_mllm_frontier_planner.py \
  <run_dir>/mllm_frontier_snapshots \
  --mode stub \
  --output <run_dir>/mllm_frontier_replay.json
```

Online smoke only after offline pass:

```bash
SMOOTHNAV_GROUNDING_SNAPSHOT_POLICY=target \
SMOOTHNAV_MLLM_FRONTIER_PLANNER=snapshot \
CONFIG_FILE=base_UniGoal/configs/config_habitat_sonnet45.yaml \
./scripts/launch_profile_suite_background.sh \
  smoothnav-full \
  results/phase2_revalidation/s21d_ep228_mllm_frontier_snapshot_20260427 \
  "228" 0 s21d_ep228_mllm_frontier_snapshot
```

Only after snapshot/replay passes should an online prior-enabled smoke be run:

```bash
SMOOTHNAV_GROUNDING_SNAPSHOT_POLICY=target \
SMOOTHNAV_MLLM_FRONTIER_PLANNER=online \
SMOOTHNAV_MLLM_FRONTIER_PRIOR_WEIGHT=<small_weight> \
CONFIG_FILE=base_UniGoal/configs/config_habitat_sonnet45.yaml \
./scripts/launch_profile_suite_background.sh \
  smoothnav-full \
  results/phase2_revalidation/s21d_ep228_mllm_frontier_prior_20260427 \
  "228" 0 s21d_ep228_mllm_frontier_prior
```

### 10. Risks And Mitigations

Risk: The MLLM reads the BEV labels incorrectly.
Mitigation: use large high-contrast labels, JSON ID table, parser validation, and snapshot visual inspection before online.

Risk: Frontier IDs become stale after map update.
Mitigation: bind each response to `snapshot_id`, `world_epoch`, `frontier_count`, and ID set; reject if mismatched.

Risk: The prior overwhelms actionability and causes unreachable goals.
Mitigation: default prior weight `0.0`; start with small weight; optionally require selected ID to be in top-N actionability or not below an actionability floor.

Risk: API empty responses return again.
Mitigation: treat empty response as a first-class decision failure with provider/model/error metadata and deterministic fallback.

Risk: The target branch is not covered in online ep228.
Mitigation: classify as branch coverage/perception-path issue; do not call it a failed planner repair.

Risk: The BEV image omits room semantics needed for commonsense.
Mitigation: include compact scene graph text and optional observed object/room labels; do not over-render unreliable inferred room names.

Risk: Implementation becomes too large before evidence.
Mitigation: Phase A/B are artifact/replay only; Phase C is behind a default-off flag; Phase D runs one episode only.

### 11. Success Criteria For This Round

This round is successful when all of the following are true:

1. A replayable BEV/frontier prompt artifact exists for at least one stored target/search snapshot.
2. The MLLM planner contract parser accepts only ID-based outputs and rejects unsafe outputs.
3. Frontier scoring can record planner prior terms without changing legacy behavior when the prior weight is zero.
4. A replay result can answer: which ID did the model choose, what coordinate did SmoothNav map it to, how did the frontier ranking change, and why.
5. One ep228 online smoke is launched only after replay evidence says the planner-grounding boundary is valid.

### 12. ADR

Decision: Build a BEV/MLLM frontier-ID planner contract and replay pipeline before online experimentation.

Drivers:

- Current text-only planner lacks spatial layout input despite BEV already existing.
- Current planner output is too loosely coupled to frontier grounding.
- Prior successful mature systems use map/frontier/value interfaces for “where to search?” decisions.
- The project needs smallest replayable experiments, not broader suites.

Alternatives considered:

1. Keep text-only planner and tune prompts.
   - Rejected: does not address missing spatial topology and weak grounding contract.
2. Ask MLLM for raw BEV coordinates.
   - Rejected: unsafe, hard to validate, bypasses existing frontier/actionability logic.
3. Replace frontier scorer with an end-to-end MLLM decision.
   - Rejected: too large, hard to replay, and discards existing diagnostics.
4. Use MLLM frontier IDs as a soft prior.
   - Chosen: aligns with existing scoring/replay boundary and preserves fallbacks.

Consequences:

- More monitoring artifacts will be produced per decision.
- Initial progress is slower than launching online experiments, but failures become much easier to localize.
- The online experiment surface stays small and reversible.

Follow-ups:

- If ID-based replay works, compare raw SmoothNav scoring vs MLLM-prior scoring on ep228 snapshots.
- If ep228 improves, run one matched positive anchor such as ep661 to check for regression.
- Only then consider a small mini-matrix; do not return directly to broad suites.

## 88. Addendum: Frontier Determinism, Branch IDs, And Frozen Task-frame Evaluation

Date: 2026-04-27
Scope: refinement of Section 87; no online experiment launched.

### 1. Frontier determinism after the BEV/MLLM contract

Observation:

The current frontier computation is mostly deterministic once the following inputs are fixed:

- `bev_map.full_map`
- `bev_map.full_pose`
- `bev_map.local_map_boundary`
- map / graph args such as `graph_frontier_min_size`, `graph_goal_distance_threshold`, `graph_filter_local_actionable_frontiers`, and frontier-value weights
- optional semantic bias / target anchor passed into `Graph.get_goal(goal=bias)`

The code path is deterministic in structure:

1. build `fbe_map` from free / obstacle / unknown channels (`base_UniGoal/src/graph/graph.py:933-948`);
2. compute raw frontier pixels at free/unknown boundaries (`base_UniGoal/src/graph/graph.py:947-951`);
3. remove small connected frontier components (`base_UniGoal/src/utils/map.py:4-18` and `base_UniGoal/src/graph/graph.py:952-957`);
4. choose filtered or raw fallback frontiers (`smoothnav/frontier_scoring.py:8-45`);
5. compute FMM distance from agent to candidates (`base_UniGoal/src/graph/graph.py:998-1010`);
6. apply distance / bias / actionability gates (`smoothnav/frontier_scoring.py:48-117`, `smoothnav/frontier_scoring.py:390-416`);
7. compose final value scores (`smoothnav/frontier_scoring.py:352-387`).

Hypothesis:

A BEV/MLLM frontier-ID interface makes the frontier side **more able to carry the MLLM intent**, because the model no longer emits a vague direction or unsafe free coordinate. It selects among map-derived, executable hypotheses. However, this only works if the IDs correspond to semantically meaningful frontier **branches/components**, not unstable individual frontier pixels.

Claim:

The Section 87 plan should be refined from “MLLM outputs frontier ID” to “MLLM outputs frontier-branch ID, backed by SmoothNav-owned candidate points.” This preserves deterministic execution while giving the model a choice unit that matches human/map semantics: doorway, corridor opening, left room entrance, right unexplored branch, etc.

### 2. Current weakness: frontier pixels are not yet branch-level choices

Observation:

`remove_small_frontiers()` already labels connected components internally (`base_UniGoal/src/utils/map.py:8-17`), but returns only a binary frontier map. `Graph.get_goal()` then turns the remaining map into many frontier pixel coordinates (`base_UniGoal/src/graph/graph.py:955-957`) and scores them as candidate points.

Hypothesis:

If we simply label the top frontier pixels as `F1..Fk`, the ID set may still be noisy:

- several IDs may belong to the same doorway / same frontier component;
- adjacent pixels may swap rank after tiny map changes;
- the MLLM may intend “the right-side room opening,” but SmoothNav may attach that intention to a single pixel rather than the whole branch;
- a valid semantic branch may be underrepresented if one pixel loses on local actionability or distance.

Refinement:

The MLLM-facing abstraction should be:

```text
frontier_branch_id -> component/branch metadata -> one or more executable candidate points
```

not directly:

```text
frontier_pixel_id -> one coordinate
```

### 3. Proposed frontier-branch decomposition

For each frontier component / branch, store:

- `branch_id`: `B1`, `B2`, ...
- `component_label`: connected-component label from the frontier map
- `pixel_count`
- `centroid`
- `representative_coord`: best executable point in the component
- `candidate_coords`: top-N candidate pixels in this branch
- `agent_distance_min / mean`
- `unknown_novelty_mean / max`
- `actionability_max`
- `repeat_recent_overlap`
- `nearby_observed_objects`
- `nearby_room_labels` if reliable
- `direction_from_agent`: coarse left/right/front/back or N/E/S/W in map frame
- `score_terms_aggregate`

The BEV image should label branch IDs (`B1..Bk`) at component centroids or representative points. The sidecar JSON maps each branch to its candidate points. The MLLM chooses a branch, while grounding/scoring chooses the best executable point inside that branch.

Hypothesis:

This is a better carrier for MLLM intent because the MLLM reasons at the level of visible map openings, while the navigation stack still executes at coordinate/frontier-point level.

### 4. How to carry MLLM intent into scoring

If the MLLM selects branch `B3`, do not hard-force one coordinate. Instead:

1. build `planner_branch_prior_scores` over all frontier candidate points;
2. assign high prior to candidate points inside `B3`;
3. assign smaller prior to second/third ranked branches;
4. keep actionability, distance, repeat/recent, novelty, and target-progress terms active;
5. let `Graph.get_goal()` select the final coordinate inside the chosen branch.

This gives the following behavior:

- MLLM says: “search the likely living-room branch on the right.”
- SmoothNav says: “within that branch, choose the most reachable frontier point that still maximizes value.”

Claim:

This is a stronger contract than both current text-only strategy and raw point output. It lets MLLM express semantic search intent while preserving deterministic local execution and replayability.

### 5. Complete intermediate-data capture: task-frame capsules

Observation:

The current monitor already records grounding snapshots and replayable frontier-value terms (`smoothnav/main.py:540-594`, `smoothnav/frontier_scoring.py:532-612`). This is sufficient for the existing target-progress replay, but not enough to evaluate planner reasonableness, branch decomposition quality, BEV label quality, or MLLM contract validity.

Claim:

We can and should save complete stage-specific intermediate data for selected episodes / selected steps. The unit should be a **task-frame capsule**: one frozen decision frame at a specific episode and step.

Suggested directory:

```text
<run_dir>/task_frame_capsules/
  episode_000228_step_000330_tv_unknown/
    manifest.json
    world_state.json
    graph_snapshot.json
    maps.npz
    bev_raw.png
    bev_annotated_branches.png
    frontier_raw.json
    frontier_branches.json
    planner_prompt.json
    planner_raw_response.json
    planner_parsed_response.json
    planner_contract_verdict.json
    scoring_input.json
    scoring_replay.json
    grounding_result.json
    executor_followup.json
```

Recommended content:

1. `manifest.json`
   - episode id, scene id, step, target, trigger, config hash, git hash, model/provider, capsule schema version.
2. `world_state.json`
   - pose, world epoch, explored regions, visible targets, object summary, room summary, frontier summary.
3. `graph_snapshot.json`
   - compact graph nodes/edges/room-object relations relevant to the planner.
4. `maps.npz`
   - `full_map`, `local_map`, `fbe_map`, raw frontier map, filtered frontier map, traversible map, local boundary, pose.
5. `bev_annotated_branches.png`
   - the exact image shown to the MLLM.
6. `frontier_branches.json`
   - branch IDs, component labels, representative points, candidate points, score terms, semantic hints.
7. `planner_prompt.json`
   - target text, text context, branch candidate table, prompt hash, image path.
8. `planner_raw_response.json`
   - raw model response or empty-response error; no API key.
9. `planner_contract_verdict.json`
   - parse status, selected branch ID, ranked IDs, invalid reason if any.
10. `scoring_replay.json`
    - before/after planner prior scores, final selected coordinate, score breakdown.
11. `grounding_result.json`
    - projected local goal, validity, selected frontier, no-op reason if any.
12. `executor_followup.json`
    - progress after N steps, repeated frontier count, stuck/no-progress signals.

### 6. Localized evaluation for “TV unknown”

For a specific frame such as “target = TV, TV not visible,” the capsule allows separate checks:

#### 6.1 Planner reasonableness check

Question:

Given the BEV map, observed objects/rooms, searched regions, and branch IDs, did the planner choose a plausible search branch for TV?

Possible verdict fields:

- `target_visible`: should be false.
- `direct_target_allowed`: should be false unless perception has a credible TV detection.
- `selected_branch_id`: must be in the branch ID set.
- `semantic_prior`: e.g. living-room / bedroom / lounge-like region should be above kitchen/bathroom for TV unless evidence says otherwise.
- `reasoning_consistent_with_map`: true/false.
- `avoid_revisit_ok`: true/false.

This can be evaluated by:

- deterministic schema checks;
- a human-readable capsule viewer;
- optional judge prompt over the same BEV image and JSON, but only as analysis, not as ground truth.

#### 6.2 Frontier decomposition check

Question:

Does the frontier branch set contain the meaningful choices that a planner needs?

Metrics:

- `raw_frontier_component_count`
- `kept_branch_count`
- `dropped_small_component_count`
- `candidate_points_per_branch`
- `branch_actionability_max`
- `branch_unknown_novelty_max`
- `branch_distance_min`
- `coverage_of_unexplored_openings`: manual or heuristic label for whether major openings are represented.

Failure examples:

- TV search direction is visible as an unexplored room opening, but no branch ID covers it.
- The same doorway becomes 5 IDs, confusing the MLLM.
- A branch exists but all candidate points inside it are filtered out by local actionability.
- A branch exists in BEV image but the JSON ID mapping is stale.

#### 6.3 Scoring integration check

Question:

After planner chooses `B3`, does frontier scoring preserve that intent unless there is a clear map-level reason to override it?

Checks:

- selected branch prior nonzero;
- final selected coordinate belongs to selected or top-ranked branch;
- if not, record override reason: low actionability, too close, repeated, unreachable, no candidate after filter;
- top-k before/after planner prior visible in replay.

#### 6.4 Grounding and execution check

Question:

Was the chosen branch converted into a valid local goal and did the robot make progress?

Checks:

- local projection valid;
- selected frontier not same as recent failed frontier unless explicitly justified;
- movement progress after N steps;
- no immediate stuck/no-op loop.

### 7. Resume/replay feasibility

Observation:

Full online resume from an arbitrary Habitat step is harder than replaying a stage because it requires simulator state, action history, maps, graph, controller state, and perception state. However, most planner/frontier bugs do not require full simulator resume.

Recommended levels:

1. **Stage replay**: save maps/graph/frontiers/prompt and rerun planner, branch decomposition, scoring, and grounding offline. This should be the default for debugging.
2. **Prefix replay**: save the action prefix to reach the frame, then rerun the episode up to that step before testing a branch online. This is useful when low-level execution effects must be checked.
3. **Full state capsule**: save observations, map tensors, graph, controller state, and task belief. Use this for near-online harness tests if exact Habitat resume is impractical.

Claim:

For the immediate TV-unknown planner/frontier question, Stage replay is sufficient and much cheaper than full online reruns. Prefix replay should be reserved for cases where the offline-selected branch looks correct but execution still fails.

### 8. Updated next implementation implication

The next implementation should add two layers before real MLLM online calls:

1. `frontier_branch_decomposition`
   - expose branch/component IDs, representative points, candidate points, and aggregate score terms.
2. `task_frame_capsule`
   - save all inputs/outputs around planner -> frontier branch -> scoring -> grounding for selected frames.

Then the first target-specific inspection can be:

```text
TV unknown frame:
  Is TV visible? no.
  Does planner choose a plausible branch ID? yes/no.
  Does frontier decomposition expose that branch? yes/no.
  Does scoring select a coordinate inside that branch? yes/no, with reason.
  Does grounding project it locally? yes/no.
```

This creates a clean boundary between:

- bad semantic planner choice;
- missing/over-fragmented frontier branches;
- planner intent lost during scoring;
- grounding/projection failure;
- downstream execution failure.

## 89. Implemented `s21` Frontier-branch Capsule Infrastructure

Date: 2026-04-27
Scope: code implementation for Sections 87-88; default-off instrumentation only; no online experiment launched.

### Observation

The first landing pass implemented the replayable infrastructure needed before online MLLM frontier planning:

- `smoothnav/frontier_branching.py`
  - deterministic frontier point grouping into branch IDs (`B1..Bk`);
  - branch metadata: centroid, representative point, candidate points, aggregate score terms, direction from agent;
  - branch-intent expansion into per-frontier prior scores;
  - optional annotated BEV PNG renderer for capsule inspection.
- `smoothnav/mllm_frontier_contract.py`
  - strict parser/validator for `smoothnav.mllm_frontier_plan.v1`;
  - accepts branch-ID choices and ranked branches;
  - rejects unknown IDs and raw-coordinate primary outputs.
- `smoothnav/task_frame_capsule.py`
  - writes frozen task-frame capsule directories with JSON/NPZ/optional PNG artifacts.
- `smoothnav/frontier_scoring.py`
  - adds optional `planner_prior_scores` and `planner_prior_weight` to frontier-value scoring;
  - records planner-prior and branch ID terms in top-k summaries and selected score breakdown;
  - stores branch/prior fields in replay snapshots and replays them offline.
- `base_UniGoal/src/graph/graph.py`
  - builds `frontier_branches` after frontier-value scoring;
  - includes branch metadata, planner-prior fields, and selected branch ID in `last_goal_debug` and replay snapshots;
  - preserves default behavior because `graph_mllm_frontier_prior_weight` defaults to `0.0` when absent.
- `smoothnav/tracing.py` and `smoothnav/main.py`
  - add `RunTracer.record_task_frame_capsule()`;
  - add CLI/env policy `--task-frame-capsule-policy` / `SMOOTHNAV_TASK_FRAME_CAPSULE_POLICY` with `{off,target,failures,all}`;
  - default remains `off`;
  - when enabled, target/failure/all frames can save frontier branches, scoring replay, grounding result, world-state summary, and map tensors under `task_frame_capsules/`.

### Hypothesis

This makes the frontier side more capable of carrying MLLM intent because the planner can choose a stable branch-level exploration unit, while SmoothNav still chooses the final executable frontier point with distance/actionability/novelty/repeat/target-progress scoring.

### Claim

This pass does not yet enable online MLLM control. It completes the safe substrate needed for the next offline checks:

1. branch decomposition can be inspected independently;
2. MLLM output can be schema-validated before influencing navigation;
3. branch intent can be replayed as a soft frontier prior;
4. specific frames such as `TV unknown` can be frozen and evaluated without a full online rerun.

### Verification

Local verification on this machine:

```text
python3 -m py_compile smoothnav/frontier_scoring.py smoothnav/frontier_branching.py \
  smoothnav/mllm_frontier_contract.py smoothnav/task_frame_capsule.py \
  smoothnav/tracing.py smoothnav/main.py scripts/replay_grounding_snapshot.py

python3 -m py_compile base_UniGoal/src/graph/graph.py

python3 -m unittest tests.test_tracing
# Ran 4 tests: OK

python3 -m unittest tests.test_frontier_branching tests.test_mllm_frontier_contract \
  tests.test_task_frame_capsule tests.test_graph_goal_scoring_gen2 \
  tests.test_grounding_snapshot_replay
# Ran 24 tests: OK (skipped=24)
```

Observation: local `/usr/bin/python3` lacks `numpy`, `Pillow`, and `pytest`, so numpy-dependent tests were collected but skipped locally. The runtime/remote environment should run the new tests with the project dependencies installed.

### Next validation command shape

Offline/unit validation on the experiment environment should run:

```bash
python -m pytest \
  tests/test_frontier_branching.py \
  tests/test_mllm_frontier_contract.py \
  tests/test_task_frame_capsule.py \
  tests/test_graph_goal_scoring_gen2.py \
  tests/test_grounding_snapshot_replay.py \
  tests/test_tracing.py -q
```

First online capture should remain non-invasive:

```bash
SMOOTHNAV_GROUNDING_SNAPSHOT_POLICY=target \
SMOOTHNAV_TASK_FRAME_CAPSULE_POLICY=target \
CONFIG_FILE=base_UniGoal/configs/config_habitat_sonnet45.yaml \
./scripts/launch_profile_suite_background.sh \
  smoothnav-full \
  results/phase2_revalidation/s21a_task_frame_capsule_ep228_20260427 \
  "228" 0 s21a_task_frame_capsule_ep228
```

Decision rule: inspect the produced `task_frame_capsules/` first. Do not enable a positive `graph_mllm_frontier_prior_weight` until branch decomposition, parser validation, and replayed scoring all pass on the frozen frame.

## 90. DeepSeek V4 DashScope API Path And Remote Smoke

Date: 2026-04-27
Scope: configure the MLLM/API path for future BEV/frontier-branch planner tests; no Habitat online episode launched in this step.

### Observation

DeepSeek V4 Pro is now available through the existing OpenAI-compatible chat-completions path:

- model: `deepseek-v4-pro`
- provider/protocol: `openai` / `openai-chat-completions`
- base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- key source: `DASHSCOPE_API_KEY` only; the key is not written into repository config or logs.

Implemented changes:

- `base_UniGoal/src/utils/llm.py`
  - `openai-chat-completions` payloads now include `max_tokens`;
  - optional extra body can be injected via `SMOOTHNAV_OPENAI_CHAT_EXTRA_BODY_JSON`;
  - thinking mode can be enabled via `SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING=1`, which adds `enable_thinking: true` for DashScope-style compatible calls.
- `scripts/smoke_deepseek_v4_dashscope.py`
  - minimal DeepSeek V4 smoke test using the same SmoothNav LLM path;
  - does not print credentials.
- `base_UniGoal/configs/config_habitat_deepseek_v4_dashscope.yaml`
  - DeepSeek/DashScope experiment config;
  - `graph_mllm_frontier_prior_weight` remains `0.0`.
- `smoothnav/experiment_io.py`
  - API base URL can now fall back to the non-secret config value if the env var is absent;
  - API key still comes from env unless explicitly configured, and effective config redaction still applies.

### Remote verification on server `10.176.56.73`

Observation:

Targeted unit tests passed on the remote `unigoal` env using `unittest` because `pytest` is not installed there:

```text
test_frontier_branching.py              3 tests OK
test_mllm_frontier_contract.py          3 tests OK
test_task_frame_capsule.py              2 tests OK
test_graph_goal_scoring_gen2.py        12 tests OK
test_grounding_snapshot_replay.py       4 tests OK
test_llm_protocols.py                  18 tests OK
test_experiment_io.py                   6 tests OK
```

DeepSeek V4 DashScope smoke also passed remotely:

```text
model=deepseek-v4-pro
provider=openai
protocol=openai-chat-completions
enable_thinking=true
ok=true
response_excerpt="OKOK"
```

The key was provided at runtime through `DASHSCOPE_API_KEY`; only key presence/length was printed by the smoke script.

### Next safe experiment shape

Hypothesis:

Before any online DeepSeek/MLLM-driven control, the next useful command should be a non-invasive capsule capture using the DeepSeek config but with the MLLM frontier prior still disabled:

```bash
DASHSCOPE_API_KEY=<runtime secret> \
SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING=1 \
SMOOTHNAV_GROUNDING_SNAPSHOT_POLICY=target \
SMOOTHNAV_TASK_FRAME_CAPSULE_POLICY=target \
CONFIG_FILE=base_UniGoal/configs/config_habitat_deepseek_v4_dashscope.yaml \
./scripts/launch_profile_suite_background.sh \
  smoothnav-full \
  results/phase2_revalidation/s21a_deepseek_v4_capsule_ep228_20260427 \
  "228" 0 s21a_deepseek_v4_capsule_ep228
```

Claim:

DeepSeek V4 is now verified as callable through the SmoothNav API path. The project is ready for a first remote capsule-only run, but not yet for a positive `graph_mllm_frontier_prior_weight` online control run.

### Remote capsule-only online probe launched

Observation:

After the API smoke, the first non-invasive DeepSeek V4 capsule run was launched on `10.176.56.73`:

```text
results/phase2_revalidation/s21a_deepseek_v4_capsule_ep228_20260427
profile=smoothnav-full
episode=228
pid=200196
log=results/phase2_revalidation/s21a_deepseek_v4_capsule_ep228_20260427/s21a_deepseek_v4_capsule_ep228_retry_20260427_180329.launcher.log
run_dir=results/phase2_revalidation/s21a_deepseek_v4_capsule_ep228_20260427/smoothnav-full/20260427/smoothnav_text_180334_1224f67a
```

Initial status:

- process still running at elapsed `~5m`;
- `step_traces/episode_000000.jsonl`: 124 records at the last check;
- `planner_calls/episode_000000.jsonl`: 7 records at the last check;
- no API error / empty-response line observed in the eval-log tail;
- no `task_frame_capsules/` directory yet at the last check, which means the target-branch capsule policy has not yet captured a target frame.

Hypothesis:

The run has verified that DeepSeek V4 is usable inside the online SmoothNav planner path. The remaining question for this run is whether it reaches a target branch and writes a capsule.

## 91. Completed `s21a` DeepSeek V4 Capsule-only `ep228` Probe

Date: 2026-04-27
Remote: `10.176.56.73:/mnt/sdd/xxy/SmoothNav`

### Observation: suite-level result

Result path:

- `results/phase2_revalidation/s21a_deepseek_v4_capsule_ep228_20260427/suite_summary.json`

Suite aggregate:

| Profile | Episode | SR | SPL | Outcome |
| --- | ---: | ---: | ---: | --- |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Per-run summary:

- run dir: `results/phase2_revalidation/s21a_deepseek_v4_capsule_ep228_20260427/smoothnav-full/20260427/smoothnav_text_180334_1224f67a`
- total steps: `435`
- high-level planner calls: `14`
- total LLM calls: `22`
- grounding attempts: `14`
- grounding no-op count/rate: `0 / 0.0`
- executor override ratio: `0.0115`

### Observation: DeepSeek V4 API path worked online

The online run used:

- `llm_model = deepseek-v4-pro`
- `api_provider = openai`
- `api_protocol = openai-chat-completions`

No API failure / empty-response evidence was found in the eval-log tail. Planner calls were recorded normally in:

- `planner_calls/episode_000000.jsonl`

The planner produced valid JSON choices on all inspected calls. The strategy sequence remained generic exploration:

- `unexplored north`
- `unexplored east`
- back to `unexplored north`
- back to `unexplored east`
- back to `unexplored north`

### Observation: target branch was not covered

Target evidence analysis:

```json
{
  "first_target_step": null,
  "target_event_count": 0,
  "first_target_anchor_after_target": null,
  "terminal_outcome": "FAILURE_NO_PROGRESS_TIMEOUT"
}
```

Contract audit dominant failure layer:

- `determinism_branch_reproducibility`
- reason: `target branch was not covered in this run`

Relevant audit layer statuses:

- `perception_graph_target_evidence = fail`
- `determinism_branch_reproducibility = warn`
- `planner_menu_choice = pass`
- `strategy_contract = pass`
- `grounding_frontier_value = pass`
- `grounding_stage_snapshot_replay = pass` because no target branch existed, so no target snapshot was expected.

### Observation: no task-frame capsules were generated

No `task_frame_capsules/` directory was produced.

Reason:

- the run was launched with `SMOOTHNAV_TASK_FRAME_CAPSULE_POLICY=target`;
- target branch was never covered;
- therefore the target-only capsule hook had no qualifying frame to save.

### Claim

This run validates the **DeepSeek V4/DashScope online API path**, but it does **not** validate the new target-branch capsule mechanism or solve `ep228`.

The failure should be classified as:

- not an API-call failure;
- not an empty-LLM-response failure;
- not a grounding-frontier replay failure;
- primarily a branch/perception coverage failure: the agent never observed or entered the `TV` target branch.

### Next smallest useful probe

Hypothesis:

Because `target` capsule policy is too narrow when the target branch is not reproducible, the next diagnostic run should use capsule policy `all` for a single episode, still with frontier prior disabled:

```bash
DASHSCOPE_API_KEY=<runtime secret> \
SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING=1 \
SMOOTHNAV_GROUNDING_SNAPSHOT_POLICY=all \
SMOOTHNAV_TASK_FRAME_CAPSULE_POLICY=all \
CONFIG_FILE=base_UniGoal/configs/config_habitat_deepseek_v4_dashscope.yaml \
./scripts/launch_profile_suite_background.sh \
  smoothnav-full \
  results/phase2_revalidation/s21b_deepseek_v4_capsule_all_ep228_20260427 \
  "228" 0 s21b_deepseek_v4_capsule_all_ep228
```

This would save generic `TV unknown` planner/frontier frames even if no target detection happens, allowing direct inspection of whether the planner's search choice and frontier branch decomposition are reasonable before target visibility.

## 92. Implemented Online BEV Frontier-branch MLLM Planner And `s22` Ep228 Probes

Date: 2026-04-27
Scope: implement the previously planned BEV/frontier-branch intervention path and run the smallest online `ep228` probes on `10.176.56.73`.

### Observation: code path implemented

The online intervention is now wired rather than only specified as a contract:

- `smoothnav/bev_frontier_planner.py`
  - builds a BEV+branch prompt from target text, current strategy, scene graph text, and branch summaries;
  - renders the annotated BEV image via `render_branch_bev_image()`;
  - calls the configured VLM path and validates only `smoothnav.mllm_frontier_plan.v1` branch-ID outputs.
- `smoothnav/frontier_branching.py`
  - exposes `render_branch_bev_image()` for online VLM calls while preserving `save_branch_bev_image()` for capsules.
- `smoothnav/strategy_grounding.py`
  - performs a dry `Graph.get_goal(..., record_selection_history=False)` pass to freeze current frontier branches;
  - calls the BEV MLLM planner when `mllm_frontier_planner_mode=online`;
  - temporarily injects valid branch priors into `graph.planner_frontier_prior`;
  - recomputes the final executable frontier with selection history recorded only on the final pass;
  - logs `mllm_frontier_calls/` records and stores MLLM status in grounding debug.
- `base_UniGoal/src/graph/graph.py`
  - adds `record_selection_history` to `get_goal()`;
  - unions positive MLLM prior indices back into the candidate set before final selection, so the BEV planner can override the text-planner bias filter when a valid branch prior exists.
- `smoothnav/main.py`
  - adds CLI/env config for `--mllm-frontier-planner-mode {off,online}`;
  - instantiates `VLM` + `BEVFrontierMLLMPlanner` only when enabled.
- `base_UniGoal/configs/config_habitat_deepseek_v4_bev_mllm.yaml`
  - enables the online BEV planner with `deepseek-v4-pro` and `graph_mllm_frontier_prior_weight: 4.0`.

### Observation: verification

Local lightweight checks:

```bash
python3 -m py_compile smoothnav/bev_frontier_planner.py \
  smoothnav/frontier_branching.py smoothnav/strategy_grounding.py \
  smoothnav/geometric_grounder.py smoothnav/main.py \
  base_UniGoal/src/graph/graph.py

python3 -m unittest tests.test_bev_frontier_mllm_planner \
  tests.test_planner_gen2 tests.test_strategy_grounding_gen2 \
  tests.test_mllm_frontier_contract tests.test_frontier_branching \
  tests.test_graph_goal_scoring_gen2 tests.test_grounding_snapshot_replay \
  tests.test_llm_protocols tests.test_experiment_io
# OK locally; numpy/Pillow-dependent tests skipped in the local system Python.
```

Remote `unigoal` env checks:

```bash
python -m unittest discover -s tests -p "test_bev_frontier_mllm_planner.py"
python -m unittest discover -s tests -p "test_graph_goal_scoring_gen2.py"
python -m unittest discover -s tests -p "test_mllm_frontier_contract.py"
python -m unittest discover -s tests -p "test_strategy_grounding_gen2.py"
# all OK
```

A remote synthetic BEV/VLM smoke returned a valid branch-ID JSON verdict through `deepseek-v4-pro`, but a separate label-reading smoke was not reliable. Treat the current DashScope DeepSeek path as **schema-callable**, not yet proven as visually grounded OCR/map perception.

### Observation: `s22a` before candidate-union fix

Result root:

- `results/phase2_revalidation/s22a_bev_mllm_ep228_20260427`
- run dir: `results/phase2_revalidation/s22a_bev_mllm_ep228_20260427/smoothnav-full/20260427/smoothnav_text_192608_ae66d1ac`

Suite result:

| Profile | Episode | SR | SPL | Outcome |
| --- | ---: | ---: | ---: | --- |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Key observations:

- total recorded steps: `407`
- high-level planner calls: `13`
- MLLM frontier calls: `13`
- MLLM valid/applied: `11 / 13`
- task-frame capsules: `13`
- target candidate events: `0`
- observed new captions: `windows x9`, `mirror x2`, `cabinet x1`, `curtain x1`
- strategies: mostly `unexplored north` (`280` steps) and `unexplored east` (`127` steps)

The MLLM selected the same branch as the dry frontier scorer on every valid call. This exposed an implementation boundary issue: the BEV planner was being constrained by `candidate_branch_ids` that had already passed through the text-planner bias filter, so it could not meaningfully override the textual `north/east` intent.

### Observation: `s22b` after exposing all summarized BEV branches and unioning MLLM-prior candidates

Result root:

- `results/phase2_revalidation/s22b_bev_mllm_union_ep228_20260427`
- run dir: `results/phase2_revalidation/s22b_bev_mllm_union_ep228_20260427/smoothnav-full/20260427/smoothnav_text_194639_3b40f51d`

Suite result:

| Profile | Episode | SR | SPL | Outcome |
| --- | ---: | ---: | ---: | --- |
| `smoothnav-full` | `228` | `0.0` | `0.0` | `FAILURE_NO_PROGRESS_TIMEOUT` |

Key observations:

- total recorded steps: `146`
- high-level planner calls: `7`
- MLLM frontier calls: `9`
- MLLM valid/applied: `8 / 9`
- task-frame capsules: `9`
- target candidate events: `0`
- observed new captions: `windows x1`, `mirror x1`, `cabinet x1`
- strategies: `unexplored north` (`133` steps), `unexplored east` (`7` steps), `bedroom` (`6` steps)
- branch intervention occurred at least twice:
  - step `43`: MLLM selected `B1`, dry scorer selected `B2`, final selected `B1`;
  - step `98`: MLLM selected `B3`, dry scorer selected `B1`, final selected `B3`.

`s22b` proves that the BEV branch prior can now change final frontier selection, but the changed path was worse for `ep228`: it timed out earlier and generated less useful semantic evidence than `s22a` / prior runs.

### Claim

The BEV intervention is now **implemented and active**, but it is not yet a successful navigation improvement.

The new evidence separates the failure into two clearer layers:

1. **Implementation layer fixed:** the online MLLM branch prior can be called, validated, traced, and can alter final frontier selection.
2. **Policy/model layer still weak:** DeepSeek V4's branch choices do not yet improve semantic search for `ep228`; no TV/living-room evidence appears, and the model sometimes emits noisy token artifacts or chooses semantically dubious room/branch commitments.

### Hypothesis

The current BEV planner is too easily anchored by the text strategy and branch score summaries. It does not yet perform robust map-level reasoning such as “leave bedroom-like area, prefer doorway/corridor/open-space branch likely leading to living room.” The next minimal offline check should use the saved `s22b` capsules to score human/heuristic branch choices against the MLLM choices before any more online runs.

Recommended next non-online probes:

1. Inspect `s22b` capsules at steps `31`, `43`, `98`, and `145`.
2. For each, compare:
   - dry selected branch;
   - MLLM selected branch;
   - final selected branch;
   - BEV image topology;
   - whether the branch actually moved toward new semantic evidence.
3. Add a replay evaluator that labels each branch decision as `new_area_gain`, `same_room_loop`, or `dead_end_revisit` from subsequent trace deltas.
4. Do not run another full online episode until the replay/capsule review shows a branch policy that beats the current dry scorer on these frames.

## 93. Completed s22b Offline Capsule Branch Review

### Observation: local replay/evaluator artifacts

I did **not** start another online episode for this pass. The `s22b` saved capsules were copied into a local offline review packet:

- local review root: `analysis/s22b_capsule_review_20260427/`
- BEV images for manual screening: `analysis/s22b_capsule_review_20260427/bev_images/`
- per-step MLLM prompt/call inputs: `analysis/s22b_capsule_review_20260427/planner_inputs/`
- structured evaluator output: `analysis/s22b_capsule_review_20260427/capsule_branch_review.json`
- readable report: `analysis/s22b_capsule_review_20260427/capsule_branch_review.md`
- replay script: `analysis/s22b_capsule_review_20260427/evaluate_s22b_capsules.py`

The evaluator covers the requested steps `31`, `43`, `98`, and `145`, extracting the dry branch, parsed MLLM branch, raw MLLM branch for malformed responses, final branch, branch summaries, prompt inputs, subsequent graph deltas, and a conservative automatic outcome label.

### Observation: branch review summary

| step | dry | MLLM parsed | MLLM raw | final | immediate interpretation |
| ---: | --- | --- | --- | --- | --- |
| `31` | `B2` | invalid / blank | `B2` | `B2` | local same-room branch; next 12 steps produced no objects/rooms and 9 `no_progress` events |
| `43` | `B2` | `B1` | `B1` | `B1` | MLLM changed final selection; produced one new `cabinet`, but no room/TV-relevant evidence |
| `98` | `B1` | `B3` | `B3` | `B3` | MLLM avoided a repeated north branch and selected the west exit; reached frontier but produced no new semantic evidence |
| `145` | `B4` | `B4` | `B4` | `B4` | geometrically plausible in the newly opened south/southwest area, but the episode ended immediately with `FAILURE_NO_PROGRESS_TIMEOUT` |

### Observation: BEV-level screening notes

- Step `31`: `B2` is a nearby small local frontier inside/near the same main room. BEV suggests `B1` or `B3` are more plausible high-value exits than `B2`; the trace confirms `B2` behaved like a same-room loop/dead-end revisit.
- Step `43`: `B1` is visually explainable under the stale `unexplored north` text strategy, but it likely continues checking an adjacent small bedroom-like area. It yielded only a `cabinet`, not a living-room/TV anchor.
- Step `98`: `B3` looks like the best BEV correction because `B1` is already repeated and `B3` opens the west-side area. The failure after this step is less a branch-ID override failure and more a downstream semantic/room discovery failure.
- Step `145`: the map has finally opened a larger south/southwest region, but the text strategy still says `unexplored north`. This is a strategy-memory mismatch: branch selection can become reasonable locally while the semantic intent remains stale.

### Hypothesis

The active BEV/MLLM branch prior is no longer blocked at the plumbing layer, but the policy is still too weak because:

1. the prompt exposes branch IDs and scores, but not a persistent room-transition state such as “this branch already failed to leave a bedroom-like area”;
2. the text strategy remains sticky (`unexplored north`) even after the BEV topology changes enough that the best frontier is west/south;
3. the evaluator still measures semantic gain only after full execution windows, so the next useful tool should be a stage-local replay that can test branch ranking and branch suppression without rerunning the whole online episode.

### Next minimal non-online step

Use the local review packet to define a branch-decision evaluator target:

- suppress same-room micro-frontiers like step `31` `B2` when larger exit-like branches are available;
- penalize previously explored/repeated branches like step `98` `B1` more strongly at the branch level;
- require the planner intent to be rewritten when the selected BEV branch direction contradicts the stale text target region for multiple capsule events;
- only return to online replay after the offline evaluator chooses a better branch than the current final policy on at least the step `31` and step `43` failure frames.

## 94. BEV Image Information Gap and Renderer Repair

### Observation

Manual review of the `s22b` capsule BEV images shows that the original MLLM image input is too sparse: most pixels are unknown gray background, the known map occupies a small region, and the image only marks `A` plus branch dots. It does not show object positions, room labels, trajectory/history, branch masks, branch actionability, or dead-end/revisit state.

This means the online MLLM was not really receiving a rich map-grounded planning view. It could see branch IDs, but most useful information still came from the text branch table and stale strategy text.

### Observation: implemented minimal renderer repair

`/Users/xin/Research/Code/research/SmoothNav/smoothnav/frontier_branching.py` now renders richer BEV branch images by default:

- crops to known map + branch/agent content instead of showing a mostly-empty global canvas;
- overlays branch candidate samples;
- labels each branch with direction, candidate count, and max final score;
- marks non-actionable branches with a red X;
- adds a legend and north arrow;
- clamps labels inside the image bounds.

The updated renderer was synced to the remote 73 repo and py-compiled there. I regenerated enriched offline images for the four requested `s22b` frames and copied them back locally:

- `analysis/s22b_capsule_review_20260427/bev_images_enriched/step_000031_bev_enriched.png`
- `analysis/s22b_capsule_review_20260427/bev_images_enriched/step_000043_bev_enriched.png`
- `analysis/s22b_capsule_review_20260427/bev_images_enriched/step_000098_bev_enriched.png`
- `analysis/s22b_capsule_review_20260427/bev_images_enriched/step_000145_bev_enriched.png`

### Hypothesis

This repair improves geometric readability but does **not** fully solve the planning-input problem. The enriched image is still mostly geometry-only. Before trusting another online MLLM frontier experiment, the BEV/capsule view should add at least:

1. recent robot trajectory / visited frontier history;
2. repeated/dead-end branch labels;
3. object/node labels and approximate positions where available;
4. room/region hypotheses from the scene graph;
5. explicit dry-vs-MLLM-vs-final branch markers for replay review.

The immediate conclusion is therefore stronger: `s22b` did not only fail because of branch policy; the visual input given to the branch policy was under-specified.

### Observation: robot marker coordinate bug found after BEV review

The suspicious `A robot` marker outside the room in the enriched step `145` image was a real coordinate-frame bug, not just a rendering artifact.

Root cause in `base_UniGoal/src/graph/graph.py`: branch/frontier coordinates are stored in the unflipped full-map frame, but `agent_coord` was copied from the FMM/traversible `start` state computed after using `full_map[..., ::-1]`. The row coordinate therefore needed to be converted back with `map_height - row - 1` before being attached to branch summaries and replay snapshots.

Fix applied:

- `base_UniGoal/src/graph/graph.py` now computes `agent_frontier_coord` in the same full-map frame as frontier branches before calling `build_frontier_branches()` and `build_frontier_value_replay_snapshot()`.
- The s22b enriched offline images were regenerated using the true current-location channel from `maps.npz`, so the local images now show `A robot` inside the free-space map.

Verification:

```bash
python3 -m py_compile base_UniGoal/src/graph/graph.py smoothnav/frontier_branching.py
# remote 73 / unigoal:
python -m unittest discover -s tests -p "test_frontier_branching.py"
python -m unittest discover -s tests -p "test_graph_goal_scoring_gen2.py"
# both OK
```

## 95. Completed s23 Semantic Annotated BEV Monitoring Implementation

### Observation: implementation completed

I implemented the planned semantic annotated BEV monitoring path without launching a new online episode. The new implementation separates the semantic BEV data contract from rendering, so the same replayable JSON frame can produce both planner-facing and human-debug images.

Changed / added local files:

- `smoothnav/semantic_bev.py`
  - new `smoothnav.semantic_bev_frame.v1` contract;
  - map coordinate metadata for `full_map_rc_unflipped`;
  - agent/object/room/branch/decision layers;
  - quality checks such as `agent_on_known_or_near_known`;
  - two render modes: `planner` and `debug`;
  - compact `semantic_bev_summary_for_prompt()` for MLLM text prompts.
- `smoothnav/world_state.py`
  - object summaries now include stable `O###` ids, `center_rc`, source and confidence fields;
  - room summaries now include stable `R###` ids plus `member_object_hypothesis` center/bbox when graph member objects have map centers.
- `smoothnav/task_frame_capsule.py`
  - capsules now write `semantic_bev_frame.json`;
  - capsules now write `semantic_annotated_bev_planner.png` and `semantic_annotated_bev_debug.png` alongside legacy `bev_annotated_branches.png`.
- `smoothnav/bev_frontier_planner.py`
  - online BEV frontier planner now prefers the semantic BEV planner image;
  - prompt now explains `A`, `B#`, `O#`, and `R#` markers and includes a compact semantic BEV JSON summary;
  - fallback to the legacy branch-only renderer remains available.
- `smoothnav/strategy_grounding.py`
  - MLLM frontier trace metadata now records `image_source` and semantic BEV quality checks.
- `scripts/replay_semantic_bev_capsules.py`
  - offline capsule replay tool that regenerates semantic BEV frames/images with no Habitat and no LLM API call.
- `tests/test_semantic_bev_renderer.py`
  - unit tests for object/room/branch layers, image rendering, and agent row-flip quality guard.
- Existing tests updated:
  - `tests/test_task_frame_capsule.py`
  - `tests/test_bev_frontier_mllm_planner.py`

### Observation: verification completed on remote 73

Remote environment:

```bash
cd /mnt/sdd/xxy/SmoothNav
source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
```

Verification commands completed successfully:

```bash
python -m py_compile \
  smoothnav/semantic_bev.py \
  smoothnav/world_state.py \
  smoothnav/task_frame_capsule.py \
  smoothnav/bev_frontier_planner.py \
  smoothnav/strategy_grounding.py \
  scripts/replay_semantic_bev_capsules.py

python -m unittest discover -s tests -p "test_semantic_bev_renderer.py" -v
python -m unittest discover -s tests -p "test_task_frame_capsule.py" -v
python -m unittest discover -s tests -p "test_bev_frontier_mllm_planner.py" -v
python -m unittest discover -s tests -p "test_frontier_branching.py" -v
python -m unittest discover -s tests -p "test_mllm_frontier_contract.py" -v
```

All listed remote tests passed.

Local macOS verification is limited by missing local `numpy` / `Pillow`, so the same tests skip locally. I still py-compiled the edited Python files locally and validated the generated JSON with `python3 -m json.tool`.

### Observation: s22b capsule semantic BEV replay completed

Offline replay artifacts were generated locally from remote execution and copied back to:

- `analysis/s23_semantic_bev_review_20260427/summary.json`
- `analysis/s23_semantic_bev_review_20260427/review.md`
- `analysis/s23_semantic_bev_review_20260427/step_000031/`
- `analysis/s23_semantic_bev_review_20260427/step_000043/`
- `analysis/s23_semantic_bev_review_20260427/step_000098/`
- `analysis/s23_semantic_bev_review_20260427/step_000145/`

Each step directory contains:

- `semantic_bev_frame.json`
- `semantic_bev_prompt_summary.json`
- `semantic_annotated_bev_planner.png`
- `semantic_annotated_bev_debug.png`
- `legacy_bev_annotated_branches.png`
- `review_result.json`

Replay command:

```bash
python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s23_semantic_bev_review_20260427
```

Replay summary:

| step | dry | MLLM | final | agent alignment ok | localized objects | localized room hypotheses | branch semantic hints | new objects next capsule | warnings |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `31` | `B2` | blank/invalid | `B2` | true | 2 | 1 | 3 | 0 | none |
| `43` | `B2` | `B1` | `B1` | true | 2 | 1 | 2 | 1 (`cabinet`) | none |
| `98` | `B1` | `B3` | `B3` | false | 3 | 1 | 3 | 0 | `agent_not_on_known_or_near_known` |
| `145` | `B4` | `B4` | `B4` | false | 3 | 1 | 2 | 0 | `agent_not_on_known_or_near_known` |

### Observation: what the new monitor reveals

The semantic debug images now show object labels (`windows`, `mirror`, `cabinet`), a room hypothesis (`bedroom`), branch IDs, and dry/MLLM/final decision rings. This directly addresses the previous “BEV 图像信息太少” issue: the image is no longer only branch geometry.

The old saved s22b capsules still carry suspicious agent coordinates in steps `98` and `145` when interpreted through `scoring_input.agent_coord`; the new quality check flags them as `agent_not_on_known_or_near_known`. This is useful: it means the semantic BEV monitor can catch exactly the class of coordinate-frame bug that previously made `A robot` appear outside the room. Future online capsules after the `Graph.get_goal()` coordinate fix should not show this warning; if they do, it becomes a direct regression signal.

### Hypothesis: current branch policy issue after semantic BEV replay

The s23 monitor suggests the remaining s22b failure is not a single missing renderer feature. It decomposes into at least three observable issues:

1. **Semantic source limitation**: by step `145`, the graph still only has bedroom-like evidence (`windows`, `mirror`, `cabinet`) and no target-relevant TV/living-room anchor. The semantic BEV can display this absence clearly, but cannot invent missing semantic observations.
2. **Branch-intent mismatch**: step `43` MLLM changes `B2 -> B1` and later discovers `cabinet`, but this does not move the task toward TV-relevant evidence. The branch looked plausible under stale `unexplored north`, but semantic gain remained weak.
3. **Coordinate/monitor sensitivity**: old capsules at steps `98`/`145` expose agent alignment warnings, so future online validation should treat `agent_on_known_or_near_known` as a hard monitor gate before trusting MLLM image decisions.

### Next minimal step

Do **not** run a broad suite yet. The next online smoke should be only one known-failed episode after confirming that new capsules contain:

1. `semantic_bev_frame.json`;
2. both semantic planner/debug images;
3. `agent_on_known_or_near_known == true` for MLLM frames;
4. object/room overlays present when `world_state.object_summary` has localized objects;
5. no raw secrets in prompt/capsule artifacts.

If the online semantic BEV still shows only bedroom objects and no target-relevant semantic gain, the next research problem is no longer BEV rendering; it is upstream semantic coverage / room-transition memory / search policy.

Additional post-integration regression tests also passed on remote 73:

```bash
python -m unittest discover -s tests -p "test_strategy_grounding_gen2.py" -v
python -m unittest discover -s tests -p "test_graph_goal_scoring_gen2.py" -v
```

Both completed with all tests OK.

### Observation: online smoke gate

I did not launch a new online episode in this pass. The remote 73 runtime currently does not expose `DASHSCOPE_API_KEY` after sourcing the available local env file, and the project rule is to avoid writing raw secrets into docs/logs/artifacts. Therefore the completion evidence for this pass is the implemented instrumentation plus no-API offline capsule replay. The next online smoke should be launched only from an environment where the DashScope key is already provided securely as `DASHSCOPE_API_KEY`.

## 96. s24 Semantic BEV Repair: Dense Channels, Target Heatmap, and Agent-Frame Diagnostics

### Observation: user-facing problem re-checked

The s23 replay images were still too weak to be called mature semantic BEV. They showed occupancy/free-space plus branch IDs, graph object points, and weak room hypotheses, but did not expose the dense BEV semantic channels, target/query likelihood, visited trajectory, or coordinate-frame disagreement.

The repaired replay output is now stored at:

- `analysis/s24_semantic_bev_repair_review_20260427/summary.json`
- `analysis/s24_semantic_bev_repair_review_20260427/review.md`
- `analysis/s24_semantic_bev_repair_review_20260427/step_000031/`
- `analysis/s24_semantic_bev_repair_review_20260427/step_000043/`
- `analysis/s24_semantic_bev_repair_review_20260427/step_000098/`
- `analysis/s24_semantic_bev_repair_review_20260427/step_000145/`

Replay command:

```bash
uv run --with numpy --with pillow python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s24_semantic_bev_repair_review_20260427
```

### Observation: mature-work delta applied to SmoothNav

The repair was guided by local mature-code inspection:

- SemExp/ObjectNav-style map contract in `base_UniGoal/src/map/bev_mapping.py` and `analysis/external_repos/Object-Goal-Navigation/main.py`: channels `0/1/2/3/4+` encode obstacle, explored, current agent, past agent, and semantic categories.
- SemExp-style visualization in `analysis/external_repos/Object-Goal-Navigation/agents/sem_exp.py`: semantic map, visited trajectory, goal map, and agent marker must be shown together.
- VLMaps-style language map / heatmap pattern in `analysis/external_repos/vlmaps/vlmaps/utils/visualize_utils.py` and `analysis/external_repos/vlmaps/vlmaps/robot/habitat_lang_robot.py`: target language should produce a spatial heatmap, not only text.
- iPPD-sem-style region/object records in `analysis/external_repos/iPPD-sem/traj_sampling/map_tools_3D.py`: object/room/semantic entities need explicit spatial records for downstream evaluation.

Implemented code changes:

- `smoothnav/semantic_bev.py`
  - decodes BEV_Map channels: `0=obstacle`, `1=free/explored`, `2=current agent`, `3=past agent`, `4+=semantic categories`;
  - renders semantic category colors from channel `4+`;
  - extracts `semantic_regions` (`S001`, `S002`, ...) from active semantic-channel connected components;
  - infers target category hints from goal text and stored `target_primary_categories`;
  - creates a target/query heatmap from direct target channels when present, or explicitly weak co-occurrence priors when direct target is absent;
  - overlays current-agent and visited-agent channels;
  - detects and repairs display-only agent coordinate mismatch using channel-2 current-agent center, while preserving raw coordinate and warning metadata;
  - adds branch-level `target_heatmap_score`, `target_heatmap_rank`, nearby semantic region IDs/labels, and map-frame direction.
- `scripts/replay_semantic_bev_capsules.py`
  - adds replay summary fields for semantic channels, semantic regions, target heatmap, agent repair, and heatmap branch ranking.
- `tests/test_semantic_bev_renderer.py`
  - adds regression coverage for semantic-channel extraction, TV prior heatmap, semantic region output, and agent-coordinate repair.

### Observation: s24 replay evidence

| step | active semantic categories | semantic regions | target-anchor regions | target heatmap | direct TV pixels | branch heat ranking | warnings |
| ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| `31` | 0 | 0 | 0 | none | 0 | all `H=0` | raw agent differs from map-current by 25 px; semantic channels empty; direct TV empty |
| `43` | 0 | 0 | 0 | none | 0 | all `H=0`; weak graph affinity only | raw agent differs from map-current by 25 px; semantic channels empty; direct TV empty |
| `98` | 1 | 1 | 0 | none | 0 | all `H=0`; refrigerator only | raw agent differs from map-current by 113 px; direct TV empty |
| `145` | 4 | 10 | 7 | semantic-prior-only | 0 | `B5 H=0.90`, `B6 H=0.72`, `B4 H=0.72`, others `H=0` | raw agent differs from map-current by 93 px; direct TV empty; prior-only heatmap |

Step `145` now exposes the information that the previous image hid:

- active semantic channels: `refrigerator`, `dining-table`, `couch`, `chair`;
- TV direct channel is absent (`direct_target_pixel_count=0`);
- TV search heatmap is therefore explicitly prior-only, sourced from couch/chair/dining-table co-occurrence;
- branch heatmap ranking prefers `B5` first, then `B6`, then `B4`;
- old dry/MLLM/final selected `B4`, so the repaired BEV surfaces a concrete branch-value disagreement instead of only showing geometry.

### Hypothesis: remaining failure boundary after s24

The BEV renderer is no longer the only blocker. The repaired BEV now separates three issues:

1. **Upstream semantic coverage is weak before step 145.** Steps `31` and `43` have no active semantic channels; step `98` only has refrigerator. MLLM cannot make a strong TV-search decision from dense semantics because dense semantics are missing.
2. **Direct target evidence is absent in all reviewed frames.** The TV channel is empty in every frame. Any TV-directed branch choice is therefore a prior-based exploration decision, not target grounding.
3. **Old capsules contain agent-frame disagreement.** The raw `scoring_input.agent_coord` is vertically inconsistent with the map-current channel in all reviewed frames. The repaired BEV display uses channel-2 current-agent for image correctness, but the discrepancy must remain a hard warning for replay/online validation.
4. **Branch policy can now be audited.** At step `145`, the semantic-prior heatmap ranks `B5/B6` above or equal to `B4`, while the recorded MLLM/final choice is `B4`. This is the first concrete value-level disagreement produced by the repaired monitor.

### Claim: what is fixed vs not fixed

Claim: The BEV monitor now has enough structure to tell the MLLM and the human reviewer when a branch decision is evidence-backed, prior-only, or unsupported. This claim is supported by completed s24 offline replay on the same s22b capsules and by new unit tests.

Claim: The reviewed s22b frames still do **not** contain direct TV semantic evidence. Therefore a successful planner must either choose a prior-backed exploration branch with uncertainty, or report insufficient target grounding; it should not claim target visibility.

### Next minimal experiment plan

Do not run a broad online suite yet. The next smallest loop should be:

1. **Offline branch decision evaluator** on `analysis/s24_semantic_bev_repair_review_20260427`:
   - verify whether recorded MLLM/final branches agree with `target_heatmap_branch_ranking`;
   - flag `direct_target_pixel_count=0` as `target_not_visible`;
   - require MLLM rationale to say `prior-only` when heatmap source is `semantic_prior_only`.
2. **One online capsule smoke only after secure API env is present**:
   - known failed episode `228`;
   - save the same semantic BEV frame/images;
   - hard gate: `agent_display_repaired_from_map_current_channel` should be false in fresh online capsules after current graph coordinate fix;
   - hard gate: MLLM output must not claim target visible when `direct_target_pixel_count=0`.
3. **If online still fails with good BEV**:
   - shift from BEV rendering to upstream semantic source repair: why semantic channels are empty until late, why graph object summary misses couch/chair/table channel evidence, and why room localization stays unlocalized.

### Verification

Local verification command:

```bash
uv run --with numpy --with pillow python -m unittest tests/test_semantic_bev_renderer.py -v
```

Result: `Ran 4 tests ... OK`.

## 97. s24 Branch Decision Evaluator on Repaired Semantic BEV

### Observation: evaluator added

A replay-only evaluator was added at:

- `scripts/evaluate_semantic_bev_branch_decisions.py`

It reads saved `semantic_bev_frame.json` files and checks whether recorded dry/MLLM/final branch decisions are supported by the repaired BEV evidence. It does not call Habitat or any LLM.

Evaluator output for s24:

- `analysis/s24_semantic_bev_repair_review_20260427/branch_decision_eval/summary.json`
- `analysis/s24_semantic_bev_repair_review_20260427/branch_decision_eval/review.md`

Command:

```bash
python3 scripts/evaluate_semantic_bev_branch_decisions.py \
  --replay-dir analysis/s24_semantic_bev_repair_review_20260427 \
  --out analysis/s24_semantic_bev_repair_review_20260427/branch_decision_eval
```

### Observation: evaluator results

| step | final | top heat | final rank | sem cats | sem regions | heat source | support | issues |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `31` | `B2` | `B1` | 2 | 0 | 0 | none | geometry-only / graph-only | no active semantic channels; direct target empty; no target heatmap |
| `43` | `B1` | `B1` | 1 | 0 | 0 | none | geometry-only / graph-only | no active semantic channels; direct target empty; no target heatmap |
| `98` | `B3` | `B3` | 1 | 1 | 1 | none | semantic context only | direct target empty; no target heatmap |
| `145` | `B4` | `B5` | 3 | 4 | 10 | semantic-prior-only | prior-supported exploration | direct target empty; prior-only heatmap; final/MLLM not top heatmap branch |

All four reviewed frames still show `agent_display_repaired` because these are old s22b capsules with raw agent coordinate disagreement. Fresh online capsules should eliminate this warning if the current `Graph.get_goal()` coordinate-frame fix is active.

### Hypothesis: corrected planner behavior after s24

Given the repaired BEV contract, a correct MLLM planner should obey these rules:

1. If `direct_target_pixel_count == 0`, output `target_visible=false`.
2. If `target_heatmap_source == semantic_prior_only`, explain that the choice is prior-guided exploration, not target grounding.
3. If a branch has clearly higher `target_heatmap_score` and is actionable, it should normally outrank lower-heat branches unless there is an explicit geometric/actionability reason.
4. If semantic channels and target heatmap are both absent, the planner should avoid confident semantic rationales and use geometry/exploration wording only.

This prompt behavior is now reflected in `smoothnav/bev_frontier_planner.py`: the BEV prompt explicitly explains `S#` semantic regions, target/query heat source, direct-target pixel absence, and the requirement not to claim visibility when the direct target channel is empty.

Additional s24 regression verification after prompt/evaluator updates:

```bash
uv run --with numpy --with pillow python -m py_compile \
  smoothnav/semantic_bev.py \
  smoothnav/bev_frontier_planner.py \
  scripts/replay_semantic_bev_capsules.py \
  scripts/evaluate_semantic_bev_branch_decisions.py

uv run --with numpy --with pillow python -m unittest \
  tests/test_semantic_bev_renderer.py \
  tests/test_task_frame_capsule.py \
  tests/test_bev_frontier_mllm_planner.py \
  tests/test_frontier_branching.py \
  tests/test_mllm_frontier_contract.py \
  tests/test_strategy_grounding_gen2.py \
  tests/test_graph_goal_scoring_gen2.py -v
```

Result: `Ran 35 tests ... OK`.

## 98. MLLM BEV/Prompt Input Design Brief For Scientist Discussion

### Observation

Added a dedicated discussion brief for the current MLLM input-construction blocker:

- `docs/implementation/mllm_bev_prompt_input_design_brief_20260428.md`

The brief summarizes the current SmoothNav implementation path from BEV map construction to high-level planner, strategy grounding, frontier branch decomposition, semantic BEV rendering, MLLM branch prompt/output contract, and task-frame capsule replay.

### Observation

The brief records the latest replay evidence from `analysis/latest_semantic_bev_manual_check_20260428` and `analysis/s24_semantic_bev_repair_review_20260427/branch_decision_eval`, especially the early-frame issue where step31/43 expose `sem=0/16` and `target_heat=none`, making the MLLM input effectively geometry/debug-only rather than a mature semantic BEV.

### Hypothesis

The current blocker is primarily an input-state design problem, not only a prompt wording problem: SmoothNav needs a richer map-conditioned decision state, likely combining dense semantic map, target-conditioned value/potential map, localized room/function hypotheses, branch value table, and coordinate-frame cleanup before the MLLM can make robust branch decisions.

## 99. Completed s25 Planner Decision-state BEV Optimization Pass

### Observation

Implemented the first decision-state BEV optimization pass from the 2026-04-28 plan. The planner-side BEV is no longer only a geometry/debug overlay; replay now writes a structured decision state with these new or upgraded artifacts:

- `smoothnav/target_value_map.py`
  - builds a target-conditioned value field summary;
  - produces per-branch evidence levels: `direct`, `anchor`, `room_prior`, `geometry_only`;
  - separates direct target evidence from semantic anchors, localized room priors, and geometry-only exploration.
- `smoothnav/semantic_bev.py`
  - adds `decision_state`, `room_state`, `target_value_state`, and `branch_decision_table` to the replayable frame;
  - adds canonical branch coordinate/direction fields;
  - splits localized room regions from unlocalized room priors;
  - adds a multi-panel decision-state renderer: geometry / semantic / target-value / branch table.
- `smoothnav/bev_frontier_planner.py`
  - switches online BEV planner image source to the decision-state renderer;
  - updates prompt text to consume the branch decision table and evidence-level contract.
- `smoothnav/mllm_frontier_contract.py`
  - requires `evidence_level`;
  - rejects `target_visible=true` unless evidence is `direct`.
- `smoothnav/strategy_grounding.py`
  - gates MLLM prior strength by evidence level;
  - blocks `geometry_only` MLLM output from becoming a hard planner prior.
- `scripts/replay_semantic_bev_capsules.py`
  - writes `planner_decision_state.json` and `decision_state_bev_planner.png` per replay step.
- `scripts/evaluate_semantic_bev_branch_decisions.py`
  - audits branch decision table completeness, evidence levels, target-value ranking, and geometry-only hard-gate mismatches.

### Observation: replay outputs

Replay output path:

- `analysis/s25_decision_state_bev_review_20260428/summary.json`
- `analysis/s25_decision_state_bev_review_20260428/review.md`
- `analysis/s25_decision_state_bev_review_20260428/branch_decision_eval/summary.json`
- Per-step images: `analysis/s25_decision_state_bev_review_20260428/step_*/decision_state_bev_planner.png`

Command:

```bash
uv run --with numpy --with pillow python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s25_decision_state_bev_review_20260428

uv run --with numpy --with pillow python scripts/evaluate_semantic_bev_branch_decisions.py \
  --replay-dir analysis/s25_decision_state_bev_review_20260428 \
  --out analysis/s25_decision_state_bev_review_20260428/branch_decision_eval
```

### Observation: s25 decision-state evaluator summary

| step | final | top target-value | final target-value rank | final evidence | sem cats | direct target px | heat source | interpretation |
| ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| `31` | `B2` | `B1` | 2 | `geometry_only` | 0 | 0 | `none` | no dense semantics or target heat; planner must not claim semantic grounding |
| `43` | `B1` | `B1` | 1 | `geometry_only` | 0 | 0 | `none` | no dense semantics/heat; old capsule shows MLLM hard-gated geometry-only branch |
| `98` | `B3` | `B2` | 2 | `geometry_only` | 1 | 0 | `none` | one semantic category exists, but no target/anchor value; old capsule hard-gated geometry-only branch |
| `145` | `B4` | `B5` | 3 | `anchor` | 4 | 0 | `semantic_prior_only` | prior-only semantic anchors favor B5/B6/B4; recorded B4 is not top value branch |

### Observation: verification

Static and unit verification completed locally with ephemeral `uv` dependencies because system `python3` lacks NumPy/Pillow:

```bash
python3 -m py_compile \
  smoothnav/target_value_map.py \
  smoothnav/semantic_bev.py \
  smoothnav/bev_frontier_planner.py \
  smoothnav/mllm_frontier_contract.py \
  smoothnav/strategy_grounding.py \
  scripts/replay_semantic_bev_capsules.py \
  scripts/evaluate_semantic_bev_branch_decisions.py

uv run --with numpy --with pillow python -m unittest \
  tests.test_target_value_map \
  tests.test_semantic_bev_renderer \
  tests.test_mllm_frontier_contract \
  tests.test_bev_frontier_mllm_planner \
  tests.test_frontier_branching \
  tests.test_grounding_snapshot_replay \
  tests.test_task_frame_capsule \
  tests.test_strategy_grounding_gen2 \
  tests.test_graph_goal_scoring_gen2
```

Result: targeted suite `Ran 44 tests ... OK`. Full local discovery also passed with required ephemeral deps:

```bash
uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests -p 'test_*.py'
```

Result: `Ran 198 tests ... OK`.

### Hypothesis

The new monitoring makes the current failure boundary clearer: the remaining blocker is not merely prompt formatting. For the hard frames, the system often still lacks direct target pixels and sometimes lacks dense semantics entirely. The MLLM can now be constrained to say `geometry_only` or `anchor`, and fresh online code blocks `geometry_only` output from becoming a hard prior. Online success will still depend on improving semantic evidence availability and on whether the value-ranked branch actually leads to new target-relevant observations.

### Next smallest validation

Run one online scene only after manually inspecting the new decision-state images. The first online check should verify that:

1. fresh capsules no longer need `agent_display_coord_repaired_from_map_current_channel`;
2. MLLM responses include `evidence_level`;
3. `geometry_only` responses do not apply hard planner priors;
4. prior-only anchor choices are logged as exploration, not target visibility;
5. if step-like state resembles s25 step145, the planner either chooses top value branch (`B5`/equivalent) or explains a concrete risk reason for selecting a lower-value branch.

## 100. Completed s26 Dedicated BEV DecisionState Module

### Observation

Implemented the requested dedicated planner decision-state module:

- `smoothnav/bev_decision_state.py`

This module now constructs a structured `DecisionState` from replay/runtime BEV inputs rather than treating the planner BEV as a debug overlay. It builds and separates:

1. `geometry_field`
   - obstacle / free / explored / unknown / visited / frontier mask summaries;
   - corresponding dense arrays saved to replay NPZ.
2. `semantic_field`
   - true semantic channel masks from `full_map[4:]` when present;
   - localized object support blobs with `source="object_center_prior"` and low confidence;
   - localized room/function-area supports only when the room is spatialized;
   - unlocalized room hypotheses kept as text priors only.
3. `target_value_field`
   - deterministic PONI-lite / VLFM-lite components:
     - `direct_target_value`,
     - `anchor_value`,
     - `room_prior_value`,
     - `geometry_fallback_value`,
     - `combined_value`.
4. `branch_table`
   - branch-level target value mean/max;
   - component values;
   - canonical direction and representative coordinate;
   - evidence level;
   - risk flags.
5. `canonical_frame`
   - planner-critical coordinate contract uses only `full_map_rc_unflipped` / full-map row-col frame.

### Observation: replay artifacts

New replay path:

- `analysis/s26_bev_decision_state_review_20260428/`

Each replay step now writes:

- `planner_decision_state.json`
- `planner_decision_state_fields.npz`
- `semantic_field_panel.png`
- `target_value_field_panel.png`
- `decision_state_bev_planner.png`
- legacy/debug images for comparison

Command:

```bash
uv run --with numpy --with pillow --with httpx python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s26_bev_decision_state_review_20260428

uv run --with numpy --with pillow --with httpx python scripts/evaluate_semantic_bev_branch_decisions.py \
  --replay-dir analysis/s26_bev_decision_state_review_20260428 \
  --out analysis/s26_bev_decision_state_review_20260428/branch_decision_eval
```

### Observation: s26 replay evaluator summary

| step | final | top target-value | final evidence | semantic reliability | target value role | key issue |
| ---: | --- | --- | --- | --- | --- | --- |
| `31` | `B2` | `B1` | `geometry_only` | `empty` | geometry fallback only | no dense semantic channels / no target heat |
| `43` | `B1` | `B1` | `geometry_only` | `empty` | geometry fallback only | old capsule shows geometry-only branch was hard-gated |
| `98` | `B3` | `B2` | `geometry_only` | sparse true semantic context, no target anchor | geometry fallback dominates | old capsule shows geometry-only branch was hard-gated |
| `145` | `B4` | `B6` | `anchor` | sparse/dense mixed semantic anchors | anchor value dominates | recorded B4 is below top target-value branch |

### Observation: semantic honesty checks

- Step31/43 now produce legal `DecisionState` with `semantic_reliability="empty"` for true semantic channels.
- Localized objects such as `windows` and `mirror` appear only as low-confidence `object_center_prior` support blobs, not as fake dense semantic masks.
- Unlocalized rooms are listed under `unlocalized_room_hypotheses` and are not drawn as spatial masks.
- `planner_decision_state_fields.npz` contains dense component arrays such as `target_combined_value`, `target_direct_target_value`, `target_anchor_value`, `target_room_prior_value`, and `target_geometry_fallback_value`.

### Observation: verification

```bash
python3 -m py_compile \
  smoothnav/bev_decision_state.py \
  smoothnav/semantic_bev.py \
  smoothnav/bev_frontier_planner.py \
  smoothnav/mllm_frontier_contract.py \
  smoothnav/strategy_grounding.py \
  scripts/replay_semantic_bev_capsules.py \
  scripts/evaluate_semantic_bev_branch_decisions.py

uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests -p 'test_*.py'
```

Result: `Ran 201 tests ... OK`.

### Hypothesis

The planner input has now crossed the intended boundary from debug visualization to decision-state BEV. Remaining failures in these old capsules are now explicitly interpretable: early steps are geometry-only because true semantic channels are empty, while step145 is prior/anchor-driven and exposes disagreement between recorded MLLM/final branch and the deterministic target-value ranking.

### Next smallest validation

The next online run should be a single previously failed scene. It should verify fresh behavior, not old-capsule behavior:

1. fresh MLLM responses must include `evidence_level`;
2. `geometry_only` responses must not apply hard branch priors;
3. semantic panel must remain honest when true semantic channels are empty;
4. if anchor value ranks one branch above the MLLM choice, the prompt/rationale should explicitly justify the disagreement.

## 101. Completed s27 Dense Semantic BEV Raster Output

### Observation

Implemented the narrowed BEV visualization repair requested for this pass: planner BEV export now has a separate dense semantic raster artifact instead of relying on the old debug overlay.

Changed outputs are generated by replay at:

- `analysis/s27_dense_semantic_bev_review_20260428/`

Each inspected step now writes:

- `dense_semantic_bev.png` — dense occupancy / semantic raster only;
- `dense_semantic_bev.json` — semantic channel coverage and renderer guarantees;
- `debug_overlay_full.png` — the old label-heavy debug image, kept separate;
- optional `pseudo_semantic_support_bev.png/json` — object-center prior support, explicitly not counted as true semantic projection.

The dense renderer contract is intentionally strict:

- no `B#` branch labels on the dense map;
- no `O#` object labels on the dense map;
- no `R#` room boxes on the dense map;
- unlocalized room hypotheses are recorded in JSON only;
- when `full_map[4:]` has no active semantic cells, the artifact reports `semantic_source="empty_map_channels"` rather than fabricating object masks.

### Observation: replay semantic channel coverage

Command:

```bash
uv run --with numpy --with pillow --with httpx python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s27_dense_semantic_bev_review_20260428
```

| step | semantic source | active dense categories | true semantic pixels | coverage ratio | hard label checks |
| ---: | --- | ---: | ---: | ---: | --- |
| `31` | `empty_map_channels` | 0 | 0 | 0.000000 | rooms/object/branch labels drawn = 0/0/0 |
| `43` | `empty_map_channels` | 0 | 0 | 0.000000 | rooms/object/branch labels drawn = 0/0/0 |
| `98` | `map_channels` | 1 | 23 | 0.000044 | rooms/object/branch labels drawn = 0/0/0 |
| `145` | `map_channels` | 3 | 50 | 0.000096 | rooms/object/branch labels drawn = 0/0/0 |

### Observation

The new dense BEV renderer is doing the correct thing, but the old capsules reveal that the true semantic projection remains extremely sparse:

- Step31/43: no active semantic cells at all, so the dense raster is honestly geometry-only.
- Step98: only 23 true semantic pixels, all from the refrigerator channel.
- Step145: only 50 true semantic pixels from couch / dining-table / refrigerator channels.

Pseudo support is saved separately from true semantic raster output and does not change `semantic_coverage_ratio`.

### Observation: verification

```bash
python3 -m py_compile \
  smoothnav/semantic_bev.py \
  scripts/replay_semantic_bev_capsules.py \
  scripts/evaluate_semantic_bev_branch_decisions.py

uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests -p 'test_*.py'
```

Result: `Ran 203 tests ... OK`.

### Hypothesis

The current visual weakness is now localized mostly upstream of the renderer: `full_map[4:]` semantic channels are empty or near-empty in the replay capsules. A mature SemExp/PONI-like dense semantic BEV needs sustained semantic projection into these channels; the renderer can now expose that absence without confusing it with graph labels, room text, or branch annotations.

## 102. Completed s28 Graph Object Footprint Path For Dense BEV

### Observation

The dense semantic BEV weakness was narrowed further: the graph already stores object-level spatial extent, but SmoothNav was compressing each object to a center point before BEV export.

Relevant code path before this repair:

- `base_UniGoal/src/graph/graph.py` and `base_UniGoal/src/graph/utils/utils.py` keep detection masks, image `xyxy`, point clouds, and 3D bbox-like geometry on each graph object.
- `smoothnav/world_state.py::summarize_objects()` exported only object center / caption / detection count.
- `smoothnav/semantic_bev.py` therefore had no area-level object footprint available and could only draw center-based support.

### Observation: repair implemented

The current pass adds a graph-object footprint path without changing planner prompt, scorer, branch decision logic, or executor:

- `smoothnav/world_state.py`
  - projects each graph object's point cloud / bbox into the existing BEV rc frame;
  - exports `bbox_rc`, `footprint_source`, `footprint_pixel_count`, and detection area diagnostics in `object_summary`.
- `smoothnav/semantic_bev.py`
  - reads `bbox_rc` from object summaries;
  - fills dense class cells for object footprints using category-specific colors;
  - keeps true semantic channel metrics separate from graph-object footprint metrics.
- `scripts/replay_semantic_bev_capsules.py`
  - passes frame objects into `build_dense_semantic_raster()`;
  - adds footprint/display metrics to replay summary.

### Observation: old capsule limit

Replay on the existing s22b capsules still shows zero object-footprint pixels because those frozen `world_state.json` files only contain object centers and no bbox/footprint fields:

| step | true semantic pixels | graph object footprint pixels | display semantic pixels |
| ---: | ---: | ---: | ---: |
| `31` | 0 | 0 | 0 |
| `43` | 0 | 0 | 0 |
| `98` | 23 | 0 | 23 |
| `145` | 50 | 0 | 50 |

This is an artifact-capture limitation of the old capsules, not evidence that the new bbox path is inactive online.

### Observation: synthetic bbox verification

A synthetic bbox-footprint artifact verifies the renderer now fills object extents as colored cells even when true semantic channels are empty:

- `analysis/s28_dense_semantic_bbox_footprint_review_20260428/synthetic_bbox_footprint/dense_semantic_bev.png`
- `analysis/s28_dense_semantic_bbox_footprint_review_20260428/synthetic_bbox_footprint/dense_semantic_bev.json`

Synthetic metrics:

```json
{
  "semantic_source": "empty_map_channels",
  "true_semantic_pixel_count": 0,
  "object_footprint_pixel_count": 707,
  "display_semantic_pixel_count": 707,
  "display_semantic_source": "graph_object_footprints",
  "object_footprint_per_category_pixel_count": {
    "cabinet": 272,
    "couch": 435
  }
}
```

### Observation: verification

```bash
python3 -m py_compile \
  smoothnav/world_state.py \
  smoothnav/semantic_bev.py \
  scripts/replay_semantic_bev_capsules.py

uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests -p 'test_*.py'
```

Result: `Ran 205 tests ... OK`.

### Hypothesis

The next online capsule should show nonzero `object_footprint_pixel_count` if graph nodes have valid `node.object['pcd']` or `node.object['bbox']` at capture time. If it remains zero online, the remaining issue is either object graph nodes lack persisted pcd/bbox at `world_state` build time or coordinate projection does not match the active graph frame.

## 103. Corrected s29 Visible Dense BBox BEV Review

### Observation

The previous s28 artifact was not acceptable as a manual-check result: although the code path supported object bbox footprints, the replay image for old capsules still looked visually unchanged because those capsules lacked graph bbox fields and the renderer showed the whole 720x720 canvas at low visual scale.

This pass corrected the review artifact itself:

- semantic-channel sparse pixels are now converted into explicit category-level bbox-fill footprints for visualization, with metrics kept separate from true semantic-channel pixels;
- dense BEV replay images are cropped to content and rendered at higher scale so the occupancy boxes are actually visible;
- true semantic metrics remain unchanged and are not confused with bbox-fill display metrics.

### Observation: new artifact

New checked path:

- `analysis/s29_visible_dense_bbox_bev_review_20260428/`

Key image for manual inspection:

- `analysis/s29_visible_dense_bbox_bev_review_20260428/step_000145/dense_semantic_bev.png`

This image now visibly contains multiple category occupancy boxes, including refrigerator / couch / dining-table regions.

### Observation: replay metrics

| step | true semantic pixels | semantic bbox-fill pixels | graph-object bbox pixels | display semantic pixels | display source |
| ---: | ---: | ---: | ---: | ---: | --- |
| `31` | 0 | 0 | 0 | 0 | `empty_map_channels` |
| `43` | 0 | 0 | 0 | 0 | `empty_map_channels` |
| `98` | 23 | 110 | 0 | 110 | `map_channels+semantic_channel_bbox_fill` |
| `145` | 50 | 4278 | 0 | 4278 | `map_channels+semantic_channel_bbox_fill` |

Step145 category bbox-fill counts:

```json
{
  "chair": 63,
  "couch": 1890,
  "dining-table": 288,
  "refrigerator": 2190
}
```

### Observation: interpretation

The old capsules still have `graph-object bbox pixels = 0` because their frozen `world_state.json` files do not contain graph bbox/pcd fields. However, they do contain sparse semantic-channel evidence in step98/145, so the new semantic-channel bbox-fill path can produce visible object-area occupancy boxes for those steps.

### Observation: verification

```bash
uv run --with numpy --with pillow --with httpx python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s29_visible_dense_bbox_bev_review_20260428

uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests -p 'test_*.py'
```

Result: `Ran 206 tests ... OK`.

### Hypothesis

For future online capsules, the best dense BEV should combine both sources:

1. graph-object bbox/pcd footprints from current `world_state` export;
2. semantic-channel bbox-fill from projected `full_map[4:]` evidence.

If online graph-object bbox remains zero, the next blocker is in graph object geometry persistence or coordinate projection, not the dense BEV renderer.

## 104. Superseding BEV Projection Contract Repair

### Observation

The s29/s30 bbox-fill review exposed a methodological error: sparse semantic-channel seed pixels should not be inflated into primary planner BEV object boxes. Mature SemExp/PONI-style BEV maps are produced by semantic mask / depth / point-cloud projection into map cells, not by guessing an object rectangle from sparse raster seeds.

This pass therefore changes the primary dense BEV contract:

- `smoothnav/semantic_bev.py::build_dense_semantic_raster()` now defaults `semantic_bbox_fill=False`.
- `scripts/replay_semantic_bev_capsules.py` explicitly builds primary `dense_semantic_bev.png` with `semantic_bbox_fill=False`.
- `smoothnav/world_state.py` now serializes projected object BEV footprint cells from graph pcd / 3D bbox support when available.
- `base_UniGoal/src/map/bev_mapping.py` now exposes `semantic_projection_summary()` to diagnose whether semantic information is missing at `obs[:,4:]`, local map, or full map.
- `smoothnav/task_frame_capsule.py` now writes `dense_semantic_bev.png/json`, `object_footprints.json`, and `semantic_projection_summary.json` into new capsules when available.

### Observation: replay artifact

New checked path:

- `analysis/s31_dense_bev_projection_contract_review_20260429/`

Key image for manual inspection:

- `analysis/s31_dense_bev_projection_contract_review_20260429/step_000145/dense_semantic_bev.png`

### Observation: replay metrics on old s22b capsules

| step | true semantic pixels | semantic bbox-fill pixels | graph-object footprint pixels | display semantic pixels | display source | warning |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| `31` | 0 | 0 | 0 | 0 | `empty_map_channels` | `object_summaries_missing_projected_footprints` |
| `43` | 0 | 0 | 0 | 0 | `empty_map_channels` | `object_summaries_missing_projected_footprints` |
| `98` | 23 | 0 | 0 | 23 | `map_channels` | `object_summaries_missing_projected_footprints` |
| `145` | 50 | 0 | 0 | 50 | `map_channels` | `object_summaries_missing_projected_footprints` |

### Interpretation

Observation: the old s22b capsules cannot recover mature dense object extents because their frozen `world_state.json` files contain object centers but not graph pcd/bbox-derived BEV footprints.

Hypothesis: a new online capsule after this patch should show nonzero `object_footprint_pixel_count` if graph nodes still retain `node.object['pcd']` or `node.object['bbox']` at `world_state` build time. If that count remains zero, the next blocker is graph geometry persistence or coordinate projection, not the renderer.

### Verification

```bash
python3 -m py_compile \
  smoothnav/semantic_bev.py \
  smoothnav/world_state.py \
  smoothnav/types.py \
  smoothnav/task_frame_capsule.py \
  scripts/replay_semantic_bev_capsules.py \
  base_UniGoal/src/map/bev_mapping.py

uv run --with numpy --with pillow python -m unittest \
  tests.test_semantic_bev_renderer \
  tests.test_task_frame_capsule \
  tests.test_bev_frontier_mllm_planner \
  tests.test_bev_decision_state

uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests

uv run --with numpy --with pillow python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s31_dense_bev_projection_contract_review_20260429
```

Result: targeted BEV tests `Ran 17 tests ... OK`; full suite `Ran 206 tests ... OK`; old-capsule replay completed and now reports honest sparse/empty semantic projection instead of false bbox-fill.

### Observation: renderer sanity with real footprint payload

A synthetic footprint sanity artifact confirms that the primary dense renderer now draws object-area cells when `footprint_rc_indices` are provided by upstream graph projection:

- `analysis/s31_dense_bev_projection_contract_review_20260429/synthetic_object_footprint_sanity/dense_semantic_bev.png`
- `object_footprint_pixel_count=511`

This separates renderer correctness from the old-capsule data gap: the renderer can draw dense category boxes, but old s22b capsules do not contain the real projected object footprints needed to do so honestly.

## 105. Semantic Footprint BEV Follow-up After Manual Visual Check

### Observation

Manual comparison against PONI/SemExp showed that `dense_semantic_bev.png` is still too close to raw sparse surface projection. Mature planner maps are better understood as semantic footprint/state maps: object categories occupy spatial cells/patches, not only the few observed high-confidence surface pixels.

### Repair

This pass keeps two separate artifacts:

1. `dense_semantic_bev.png/json`: faithful true semantic-channel raster, no sparse bbox inflation.
2. `semantic_footprint_bev.png/json`: planner-facing semantic footprint raster. It fills connected semantic-channel component bboxes with a low threshold (`0.001`) when graph pcd footprints are unavailable in old capsules. The source is explicitly recorded as `semantic_channel_bbox_fill` and does not overwrite true semantic pixel metrics.

Additionally, graph object coordinate handling was corrected:

- `WorldState.object_summary.center_rc` now uses canonical `full_map_rc_unflipped`.
- The original graph/FMM coordinate is retained as `graph_center_rc`.
- Old replay capsules with only `center` and no coordinate-frame tag are converted from graph frame to full-map row/col before BEV use.
- Object pcd projection now targets full-map row/col (`row=world_y/res`, `col=world_x/res`) instead of graph/FMM frame.

### Observation: new artifact

New checked path:

- `analysis/s33_semantic_footprint_frame_repair_review_20260429/`

Manual-check image:

- `analysis/s33_semantic_footprint_frame_repair_review_20260429/step_000145/semantic_footprint_bev.png`

Step145 footprint metrics:

```json
{
  "true_semantic_pixel_count": 50,
  "semantic_bbox_footprint_pixel_count": 705,
  "semantic_bbox_footprint_source": "semantic_channel_bbox_fill",
  "semantic_bbox_footprint_per_category_pixel_count": {
    "chair": 90,
    "couch": 287,
    "dining-table": 312,
    "refrigerator": 204
  },
  "object_footprint_pixel_count": 0,
  "display_semantic_source": "map_channels+semantic_channel_bbox_fill"
}
```

### Interpretation

Observation: the planner-facing footprint image now has dense category patches rather than only sparse dots, but old s22b capsules still have `object_footprint_pixel_count=0`, so these patches are component-bbox approximations from semantic map channels, not graph pcd footprints.

Hypothesis: the next online capsule should be judged by `object_footprint_pixel_count` and `footprint_source=graph_pcd_bbox_cells` / `graph_3d_bbox_cells`. If those remain zero after this coordinate repair, the remaining blocker is graph geometry persistence, not BEV rendering.

### Verification

```bash
python3 -m py_compile smoothnav/semantic_bev.py smoothnav/world_state.py smoothnav/task_frame_capsule.py scripts/replay_semantic_bev_capsules.py
uv run --with numpy --with pillow python -m unittest tests.test_semantic_bev_renderer tests.test_task_frame_capsule
uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests
uv run --with numpy --with pillow python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s33_semantic_footprint_frame_repair_review_20260429
```

Result: targeted tests `Ran 12 tests ... OK`; full suite `Ran 206 tests ... OK`; replay completed.

## 106. BEV Footprint Repair: Instance-Depth Footprint Contract (2026-04-29)

### Observation: exact repair target

The previous `semantic_footprint_bev` still depended on sparse semantic-channel surface pixels. In s22b step145, the raw map had only 50 true semantic pixels at threshold 0.5, and all positive semantic pixels overlapped obstacle/surface cells. This made bbox inflation a diagnostic fallback, not a mature SemExp/PONI-style object footprint.

### Implemented changes

1. `base_UniGoal/src/agent/unigoal/agent.py`
   - Preserve reset-frame detector boxes instead of clearing `pred_box` immediately after preprocessing.
   - Export compact per-instance records: category id/name, detector bbox in mapping-frame pixels, confidence, and semantic seed counts.

2. `smoothnav/main.py` + `base_UniGoal/src/map/bev_mapping.py`
   - Pass detector instance records into `BEV_Map.mapping()` through `infos["semantic_instances"]`.
   - Project instance mask/bbox support through the same depth/pose BEV transform path.
   - Serialize planner-facing footprint boxes as `semantic_instance_depth_bbox_cells` with explicit `footprint_rc_indices`, `bbox_rc`, `raw_surface_bbox_rc`, confidence, and seed-source metadata.
   - Keep these footprints separate from `full_map[4:]`; they do not mutate navigation channels.

3. `smoothnav/world_state.py`, `smoothnav/types.py`, `smoothnav/semantic_bev.py`, `smoothnav/task_frame_capsule.py`
   - Add `semantic_instance_footprints` to `WorldState.summary()` and task-frame capsules.
   - Merge these footprints into semantic BEV objects only as localized, source-tagged footprint evidence.
   - Fix room-summary center aggregation to convert graph/FMM centers into canonical `full_map_rc_unflipped` before deriving room bboxes.

4. `smoothnav/semantic_bev.py`
   - Improve the old-capsule fallback by expanding sparse semantic component bboxes to category-minimum footprint sizes before interior shifting.
   - This remains labeled `semantic_channel_bbox_fill`; it is not counted as true semantic coverage or graph/object footprint evidence.

### Observation: replay evidence on old s22b capsule

New replay path:

- `analysis/s35_bev_footprint_repair_review_20260429/step_000145/semantic_footprint_bev.png`
- `analysis/s35_bev_footprint_repair_review_20260429/step_000145/semantic_footprint_bev.json`

Step145 metrics on the old capsule:

```json
{
  "true_semantic_pixel_count": 50,
  "object_footprint_pixel_count": 0,
  "semantic_bbox_footprint_pixel_count": 1213,
  "semantic_bbox_footprint_source": "semantic_channel_bbox_fill",
  "display_semantic_source": "map_channels+semantic_channel_bbox_fill"
}
```

Observation: old capsules still cannot show the new online `semantic_instance_depth_bbox_cells` path because they never saved detector instance records. They only validate the improved fallback renderer. Future online capsules must be judged by nonzero `semantic_instance_footprints` / `object_footprint_pixel_count`.

### Verification

```bash
python3 -m py_compile base_UniGoal/src/map/bev_mapping.py base_UniGoal/src/agent/unigoal/agent.py smoothnav/world_state.py smoothnav/types.py smoothnav/semantic_bev.py smoothnav/main.py smoothnav/task_frame_capsule.py
uv run --with numpy --with pillow python -m unittest tests.test_semantic_bev_renderer tests.test_task_frame_capsule
uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests
# Synthetic BEV_Map projection smoke was run separately; result: one chair footprint, footprint_pixel_count=136.
uv run --with numpy --with pillow python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 145 \
  --out analysis/s35_bev_footprint_repair_review_20260429
```

Result: targeted tests `Ran 13 tests ... OK`; full suite `Ran 207 tests ... OK`; synthetic BEV_Map projection artifact `analysis/s35_bev_footprint_repair_review_20260429/synthetic_instance_projection.json` produced one `chair` footprint (`footprint_pixel_count=136`); old-capsule replay completed.

## 107. Semantic BEV Mature-work Construction Review (2026-04-29)

Observation: a dedicated comparison note now records how SemExp and PONI construct dense semantic BEV maps, what elements their BEV images contain, and how SmoothNav currently builds `dense_semantic_bev` / `semantic_footprint_bev` artifacts.

Document:

- `docs/implementation/semantic_bev_mature_work_comparison_20260429.md`

Claim: the main remaining BEV gap is upstream cell-level semantic / footprint evidence, not prompt wording or debug-label rendering. SemExp consumes projected `obs[:,4:]` semantic channels as map state, while PONI uses scene-level `map_semantic` cells and potential fields; SmoothNav now separates dense raster artifacts from debug overlays but still needs online evidence that `semantic_instance_depth_bbox_cells` produces correctly placed object-sized footprints.

Evidence: the note cites SemExp projection/visualization code, PONI semantic-map generation/potential-field code, and SmoothNav current mapping/rendering/capsule paths.

## 108. BEV Correction: Remove Misleading BBox Inflation from Planner State (2026-04-30)

### Observation: corrected failure mode

The previous repair made `semantic_footprint_bev` visually denser by inflating sparse semantic-channel surface pixels into object-sized rectangles. On the old s22b step145 capsule this produced `semantic_bbox_footprint_pixel_count=1213` from only 50 true semantic pixels. That looked like object boxes, but it was not mature SemExp/PONI-style semantic projection because the boxes were derived from sparse category surfaces and then shifted toward nearby interior cells.

### Implemented changes

1. `base_UniGoal/src/map/bev_mapping.py`
   - The online semantic-instance footprint path no longer expands or shifts projected instance support to a nearby "nice" interior patch.
   - `semantic_instance_depth_bbox_cells` now uses the tight bbox around actual depth-projected instance support, with `bbox_expansion_applied=false` and `bbox_shift_applied=false` metadata.

2. `smoothnav/semantic_bev.py`
   - `dense_semantic_bev` and `semantic_footprint_bev` no longer use `semantic_channel_bbox_fill` in replay/capsule outputs.
   - The decision-state semantic panel now renders the same dense raster contract: true map channels plus trusted projected footprints only, without O#/R#/B# labels.
   - Semantic channels, semantic regions, and target heatmap contributions now use a confident threshold (`>0.5`) instead of any positive residual.
   - Branch semantic affinity is downweighted for one-pixel semantic blips so a single sparse anchor pixel cannot dominate a branch.
   - Dense object footprints now record overlap diagnostics and reject footprints that are mostly outside explored map area.

3. `smoothnav/bev_decision_state.py`
   - Target value fields no longer fill category-wide bboxes. Direct/anchor value is built from actual thresholded semantic masks and compact component-centered potential, preventing sparse disconnected pixels from becoming a large rectangular target/anchor field.
   - Trusted object footprints are consumed as footprint masks when available; object centers remain low-confidence support and are not treated as dense semantics.

4. `scripts/replay_semantic_bev_capsules.py` and `smoothnav/task_frame_capsule.py`
   - Replay and online capsule artifacts keep `semantic_footprint_bev` honest: no semantic-channel bbox-fill fallback is used as planner-facing dense semantics.

### Observation: old-capsule replay after correction

Replay path:

```bash
uv run --with numpy --with pillow python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s36_bev_correction_review_20260430
```

Key step145 metrics:

```json
{
  "true_semantic_pixel_count": 50,
  "semantic_channel_active_count": 3,
  "object_footprint_pixel_count": 0,
  "semantic_bbox_footprint_pixel_count": 0,
  "display_semantic_source": "map_channels",
  "dense_semantic_warnings": ["object_summaries_missing_projected_footprints"]
}
```

Observation: the corrected old-capsule BEV is still sparse because the old s22b capsule never stored detector instance footprints; this is now represented honestly rather than hidden by fake bbox fill.

Hypothesis: future online capsules should show object-sized colored regions only if `semantic_instance_footprints` / trusted graph PCD footprints become nonzero. If they remain zero, the remaining blocker is upstream semantic-instance projection / graph footprint generation, not the renderer.

### Verification

```bash
python3 -m py_compile smoothnav/semantic_bev.py smoothnav/bev_decision_state.py base_UniGoal/src/map/bev_mapping.py scripts/replay_semantic_bev_capsules.py
uv run --with numpy --with pillow python -m unittest tests.test_semantic_bev_renderer tests.test_bev_decision_state
uv run --with numpy --with pillow --with httpx python -m unittest discover -s tests
```

Result: targeted tests `Ran 16 tests ... OK`; full suite `Ran 209 tests ... OK`; replay completed at `analysis/s36_bev_correction_review_20260430/summary.json`.
