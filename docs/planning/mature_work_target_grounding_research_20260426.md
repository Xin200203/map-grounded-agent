# Mature Work Research: Persistent Target Grounding For SmoothNav `ep228`

Date: 2026-04-26

Current SmoothNav failure boundary from `s19_sonnet45_ep228_20260426`:

- API / LLM response is no longer the blocker.
- `claude-sonnet-4-5-20250929` works and produced no empty responses.
- Target evidence is detected early: `first_target_step = 330`.
- Planner consumes it immediately: `first_target_anchor_after_target = 330`.
- Episode still fails with `FAILURE_NO_PROGRESS_TIMEOUT`.
- Contract audit dominant failure:
  - `grounding_frontier_value`
  - `target frontier value lost semantic signal`
- Key symptoms:
  - `target_semantic_term_share_median = 0.0`
  - `target_base_score_zero_rate = 0.9`
  - `target_bias_score_zero_rate = 0.9`
  - `grounding_noop_reason_counts = {"out_of_local_window": 8}`
  - stale local-goal risk after local-map movement.

This document reviews mature navigation systems for the same class of problem:
**how to keep a non-local semantic target alive while executing through local maps and frontiers**.

---

## Sources and code inspected

### Papers / project pages

1. SemExp / Goal-Oriented Semantic Exploration
   - Paper / arXiv: https://arxiv.org/abs/2007.00643
   - GitHub: https://github.com/devendrachaplot/Object-Goal-Navigation
   - Project claim: episodic semantic map + goal-oriented semantic policy + deterministic local policy.

2. PONI
   - Project: https://vision.cs.utexas.edu/projects/poni/
   - GitHub: https://github.com/srama2512/PONI
   - Paper / arXiv: https://arxiv.org/abs/2201.10029
   - Core claim: predict potential functions over a semantic map, then select where to look.

3. VLMaps
   - Project: https://vlmaps.github.io/
   - GitHub: https://github.com/vlmaps/vlmaps
   - Core claim: fuse visual-language features into a spatial map so language queries directly localize map coordinates.

4. CoW / CLIP on Wheels
   - Paper / arXiv: https://arxiv.org/abs/2203.10421
   - GitHub: https://github.com/real-stanford/cow
   - Core claim: zero-shot object navigation can combine object localization with classical frontier exploration.

5. Open Scene Graphs / OSG Navigator
   - Project: https://open-scene-graphs.github.io/
   - Paper / arXiv: https://arxiv.org/abs/2508.04678
   - Core claim: foundation models need an explicit spatial memory, implemented as a hierarchical open scene graph.

### Local code checkouts

Representative repositories were cloned locally under:

- `analysis/external_repos/semexp`
- `analysis/external_repos/poni`
- `analysis/external_repos/vlmaps`
- `analysis/external_repos/cow`

Important inspected files:

- SemExp:
  - `analysis/external_repos/semexp/agents/sem_exp.py`
  - `analysis/external_repos/semexp/envs/utils/fmm_planner.py`
- PONI:
  - `analysis/external_repos/poni/hlab/global_agent.py`
  - `analysis/external_repos/poni/semexp/model_pf.py`
  - `analysis/external_repos/poni/poni/fmm_planner.py`
  - `analysis/external_repos/poni/poni/dataset.py`
- VLMaps:
  - `analysis/external_repos/vlmaps/vlmaps/map/vlmap.py`
  - `analysis/external_repos/vlmaps/vlmaps/map/map.py`
  - `analysis/external_repos/vlmaps/vlmaps/robot/habitat_lang_robot.py`
- CoW:
  - `analysis/external_repos/cow/src/models/agent_fbe.py`
  - `analysis/external_repos/cow/src/models/exploration/frontier_based_exploration.py`

---

## Closest parallels

### Parallel 1: SemExp separates semantic search from local motion

#### What is directly supported by sources

The SemExp repository describes three modules:

1. Semantic Mapping Module.
2. Goal-Oriented Semantic Policy.
3. Deterministic Local Policy.

The README says the semantic map is built over time, the semantic policy selects a long-term goal, and an analytical local planner reaches that long-term goal.

Relevant code pattern:

- `agents/sem_exp.py::_get_stg(...)`
  - takes a goal map;
  - dilates it;
  - calls `FMMPlanner.set_multi_goal(goal)`;
  - obtains a short-term goal every local-planning step.
- `envs/utils/fmm_planner.py::set_multi_goal(...)`
  - creates an FMM distance field from every active goal cell.

#### What is only inferred for SmoothNav

SmoothNav currently does part of this, but in a weaker way:

- it has a semantic graph and frontier scoring;
- it creates a local `global_goals` point;
- but the target semantic signal can collapse when the target is outside the local map.

SemExp suggests a stricter separation:

- maintain a long-term semantic goal map / target distribution;
- recompute local short-term goal from that map;
- never treat the local goal as the only copy of the semantic target.

For SmoothNav, this means the `unexplored target:tv` anchor should be a persistent **global semantic search objective**, not just a local frontier bias that may become zero after map recentering.

---

### Parallel 2: PONI turns “where to look?” into a persistent map value problem

#### What is directly supported by sources

PONI defines a potential function as a value over map locations/frontiers: high value means visiting that location is likely to help find the target. The project page states that PONI predicts complementary potential functions conditioned on a semantic map and uses them to decide where to look.

Relevant code pattern:

- `semexp/model_pf.py`
  - predicts potential fields;
  - selects the max-valued location for the target category;
  - can mask to unexplored regions;
  - can combine object potential, area potential, and distance-to-location terms.
- `poni/dataset.py`
  - explicitly computes frontier masks and weights potential-function loss around frontiers.
- `hlab/global_agent.py`
  - keeps a `full_map` and a `local_map`;
  - writes local-map updates into full map;
  - recenters the local map from the full map;
  - then samples a long-term goal and converts it into a local goal map.

#### What is only inferred for SmoothNav

SmoothNav's `s19` target branch fails exactly where a PONI-like value map would be useful:

- the target is known (`tv`);
- but many target-grounding events have `bias_score = 0` and `base_score = 0`;
- frontier choice becomes dominated by novelty/actionability instead of target progress.

PONI does **not** prove SmoothNav needs a trained neural potential function. But it strongly suggests the missing abstraction:

> Keep a target-conditioned value field over the map/frontiers, rather than allowing the target contribution to become zero whenever the target is non-local.

For SmoothNav, a minimal non-learned “PONI-lite” version is more appropriate than adopting PONI wholesale:

- no training;
- no new dependency;
- compute a deterministic target-progress potential over current frontier candidates;
- keep a non-zero semantic floor for persistent target anchors.

---

### Parallel 3: VLMaps anchors language directly in full-map coordinates

#### What is directly supported by sources

VLMaps fuses visual-language features into a spatial map. The project page states that language instructions can be localized directly in the map. The code exposes map-level language query functions:

- `vlmaps/map/vlmap.py::get_pos(name)`
  - indexes the full map with a language/object name;
  - returns object contours, centers, and bounding boxes.
- `vlmaps/map/map.py::get_nearest_pos(curr_pos, name)`
  - selects the nearest map position for the queried object.
- `vlmaps/robot/habitat_lang_robot.py::move_to(pos)`
  - plans to a position on the full map.

#### What is only inferred for SmoothNav

VLMaps is not an online frontier-search solution by itself. It often assumes a built map or a map-building phase. However, its design is important:

- language grounding returns a full-map location/mask;
- local navigation receives a projected path to that full-map target;
- the target does not disappear when the local map crop changes.

For SmoothNav, this supports the conclusion that `target:tv` should store and update a full-map target hypothesis. The local map should be a view into that hypothesis, not the source of truth.

---

### Parallel 4: CoW uses explicit modes and persistent ROI targets

#### What is directly supported by sources

CoW combines object localization with frontier-based exploration. Its paper reports a simple framework with CLIP-based object localization and classical exploration. In code:

- `agent_fbe.py`
  - modes include `SPIN`, `EXPLORE`, and `EXPLOIT`;
  - if an ROI exists, it switches to `EXPLOIT`.
- `frontier_based_exploration.py`
  - stores `roi_targets`;
  - updates ROI targets from repeated observations;
  - plans directly to ROI when possible;
  - if direct target path fails, it falls back to frontiers sorted relative to the ROI.

#### What is only inferred for SmoothNav

SmoothNav already has a controller state, but target-anchor behavior is still too soft:

- after target detection, generic frontier logic can still dominate;
- target value can go to zero;
- stale local goals can survive local-map movement.

CoW suggests a simple control principle:

> Once a credible ROI/target exists, enter a target-exploit/search mode. Generic exploration becomes subordinate to progress toward that target.

For SmoothNav, `unexplored target:tv` should activate a stricter target-anchor mode:

- target-oriented frontier fallback first;
- generic exploration only after explicit decommit;
- ROI/target history retained across map updates.

---

### Parallel 5: Open Scene Graphs emphasize spatial memory for foundation models

#### What is directly supported by sources

The OSG Navigator project states that foundation models provide semantic knowledge but are insufficient at organizing and maintaining spatial information at scale. Its Open Scene Graph acts as spatial memory and organizes open-set objects and spatial regions hierarchically.

#### What is only inferred for SmoothNav

SmoothNav already has a graph, so the lesson is not “add a graph”. The lesson is:

- the graph should own persistent target hypotheses;
- target hypotheses should have geometry, confidence, recency, and lifecycle;
- LLM output should update this memory, not replace it each planning step.

For `ep228`, the graph knows about `tv`, but the grounding loop does not preserve a stable target-value signal from graph memory into frontier selection.

---

## Cross-work pattern summary

Across SemExp, PONI, VLMaps, CoW, and OSG-like systems, the mature pattern is consistent:

1. **Keep semantic target memory separate from local action goals.**
   - Mature systems maintain a full/episodic map, potential map, ROI list, or scene graph.
   - Local goals are temporary projections of this memory.

2. **Use a target-conditioned value map, not just a target point.**
   - PONI is the clearest example: “where to look” is a potential function over locations/frontiers.
   - SmoothNav currently uses a target point/bias whose contribution can become zero.

3. **Recompute local goals after local-map movement.**
   - PONI explicitly copies local map into full map, recenters local boundaries, and recomputes local pose and goal maps.
   - SmoothNav `s19` still reports stale local-goal risk after local-map movement.

4. **Once a target candidate exists, exploration should become target-conditioned.**
   - CoW switches from exploration to exploitation when ROI exists.
   - SmoothNav should not let generic novelty dominate while an active `target:tv` anchor is still valid.

5. **Direct object navigation and target search are different states.**
   - If the target is local and actionable, navigate directly.
   - If it is non-local, choose frontiers that improve target reachability / visibility.
   - Do not collapse non-local target search into generic frontier exploration.

---

## Route review for SmoothNav

### What our current route got right

The current SmoothNav route is broadly aligned with mature work:

- explicit map/graph memory exists;
- planner now consumes target evidence;
- target-anchor attempts are monitored;
- local-map movement and out-of-window failures are explicit;
- frontier scoring already logs score breakdowns.

The `s19` result is therefore useful: it shows the system has reached the correct failure layer.

### What is missing

The missing abstraction is:

> a persistent target-conditioned frontier value / potential layer over full-map coordinates.

Current behavior:

1. Planner says `unexplored target:tv`.
2. Graph uses detected `tv` center as bias.
3. Frontier scoring computes distance-based scores.
4. When the target is non-local or poorly connected, semantic/bias scores become zero.
5. The controller keeps refreshing, but scoring becomes weakly semantic or generic.
6. Local-map movement can leave stale local goals.

Mature-work-inspired desired behavior:

1. Planner says `unexplored target:tv`.
2. Graph creates/updates a persistent target hypothesis in full-map coordinates.
3. Grounder builds a target-conditioned value field for frontiers:
   - object center evidence;
   - graph node confidence;
   - geodesic / FMM progress toward target or target region;
   - unknown/frontier information gain near likely target path;
   - local actionability.
4. Local goal is reprojected every time the local map changes.
5. If the target is outside the local map, semantic value remains non-zero through the target-progress field.
6. Generic exploration only takes over after explicit target-anchor decommit.

---

## Recommended implementation plan

### Phase A: deterministic target-potential layer, no new dependency

Implement a “PONI-lite” target potential in existing SmoothNav code.

Likely files:

- `smoothnav/frontier_scoring.py`
- `base_UniGoal/src/graph/graph.py`
- `smoothnav/strategy_grounding.py`
- `smoothnav/controller_logic.py`
- tests under `tests/`

Core idea:

For active `target:*` strategies, frontier scoring should include a non-zero target-progress term:

```text
final_score =
    base_distance_score
  + target_progress_weight * target_progress_score
  + semantic_bias_weight * semantic_bias_score
  + novelty_weight * novelty_score
  + actionability_weight * local_actionability_score
  - repeat_penalties
```

Where `target_progress_score` should answer:

```text
Does selecting this frontier move us closer to the persistent target hypothesis,
or to the most plausible unknown boundary toward that target?
```

Important: this term should not become zero merely because the target is outside the current local map.

### Phase B: explicit target-anchor state

Add or extend target-anchor state with:

- `target_label`: e.g. `tv`;
- `full_map_coord`: best known target center or target-region center;
- `confidence`: from caption/graph evidence;
- `last_seen_step`;
- `last_projected_local_coord`;
- `last_value_source`: direct object / graph node / inferred frontier / fallback;
- `decommit_reason` when abandoned.

The graph or controller should own this state. The executable local goal should be derived from it, not treated as the source of truth.

### Phase C: local-map lifecycle fix

After `bev_map.move_local_map()`:

1. mark current local `global_goals` invalid;
2. force `apply_strategy` for active target anchors;
3. require trace field:
   - `local_map_moved = true`;
   - `stale_local_goal_invalidated = true`;
   - `target_anchor_reprojected = true/false`;
   - `reprojection_failure_reason`.

Success criterion for this layer:

- `local_map_lifecycle.stale_local_goal_risk_count` should go to `0` on the replay / rerun.

### Phase D: target-mode arbitration

For active target anchors:

- prefer direct object goal if target is locally projectable and relevance passes;
- otherwise prefer target-progress frontiers;
- only use generic novelty frontiers if:
  - target confidence decays below threshold;
  - target-progress frontiers are exhausted;
  - or target-anchor stall patience triggers explicit decommit.

This mirrors CoW's explore/exploit split without importing its whole architecture.

---

## Minimal validation plan

Do not broaden the benchmark yet.

### Test 1: pure unit tests

Add frontier-scoring tests for:

1. target full-map coordinate outside local window;
2. several locally actionable frontiers;
3. generic novelty frontier has higher novelty;
4. target-progress frontier should still win if it reduces global target distance;
5. local-map movement invalidates stale local goal.

Expected result:

- target semantic/progress term is non-zero;
- selected frontier is locally actionable;
- selected frontier improves target distance or points to nearest unknown boundary toward target.

### Test 2: replay/counterfactual on `s19`

Use existing `s19` trace, especially steps:

- `334`
- `355`
- `435`

These are stale local-goal risk steps.

Expected replay/counterfactual improvements:

- `target_semantic_term_share_median > 0`;
- `target_base_score_zero_rate` and `target_bias_score_zero_rate` no longer near `0.9`;
- selected target frontier distance delta should be non-positive or substantially smaller;
- stale local-goal risk should be eliminated.

### Test 3: one online rerun

Only after unit + replay pass:

- `episode 228`
- `smoothnav-full`
- `claude-sonnet-4-5-20250929`
- same monitoring as `s19`

Primary success:

- SR/SPL improvement if possible.

Secondary success:

- even if SR remains `0`, failure layer should move past `grounding_frontier_value`;
- `target_semantic_term_share_median > 0`;
- no stale local-goal risk;
- target-anchor best distance should not drift badly after first target commitment.

---

## What not to do yet

1. Do not expand the suite.
2. Do not switch models again as the main intervention.
3. Do not import PONI as a dependency.
4. Do not add a learned model or pretrained weights.
5. Do not make LLM choose low-level frontier coordinates directly.

The mature-work lesson is not “use a bigger model”. It is:

> preserve semantic target value in map space, then project it into local action space repeatedly and safely.

---

## Bottom-line recommendation

Current SmoothNav is on the right research route, but it is missing the standard mature-system boundary between:

- **semantic target memory**: persistent, full-map, graph/potential-based;
- **local executable goal**: temporary, local-map-projected, invalidated after map movement.

The next implementation should be a deterministic, testable target-potential / target-progress layer, plus local-map goal invalidation. This is the smallest change that directly addresses the `s19` failure without turning the project into a new learned ObjectNav system.
