# SmoothNav 完整实验复盘与覆盖式检测方案（2026-04-26）

- project: `/Users/xin/Research/Code/research/SmoothNav`
- remote evidence: `10.176.56.73:/mnt/sdd/xxy/SmoothNav/results/phase2_revalidation`
- local collected inputs:
  - `analysis/flywheel_runs/20260426_smoothnav_all_experiments_suite_rows.json`
  - `analysis/flywheel_runs/20260426_smoothnav_target_evidence_s8_s15.jsonl`
  - `analysis/flywheel_runs/20260426_smoothnav_target_frontier_stats_s9_s12.jsonl`

## Observation

### 1. Intact-scene 阶段证明 SmoothNav 栈不是整体无效

| Suite | Scope | 关键观察 |
| --- | --- | --- |
| `s2_dev5_intact_scene` | 5 episodes, 5 profiles | `smoothnav-full` SR `0.6` / SPL `0.2636`，高于 baseline-periodic SR `0.4` / SPL `0.2071`。 |
| `s3_dev15_intact_scene` | 15 episodes, 3 profiles | `smoothnav-full` SR `0.6` / SPL `0.2678`，高于 baseline SR `0.5333` / SPL `0.2353`，也高于 no-monitor SR `0.4667` / SPL `0.2416`。 |

Observation: 早期 intact-scene 结果支持“event-driven SmoothNav 栈可以工作”，问题不是全局执行器或 evaluator 完全坏掉。

### 2. Cross-scene 后，优势不稳定，并出现 shared hard cases

| Suite | Scope | 关键观察 |
| --- | --- | --- |
| `s4_cross_scene_suite` | ep229/527/859 | 三条 profile 均 SR `0.3333`，full/no-monitor SPL 低于 baseline。 |
| `s5_cross_scene_dev6` | ep228/527/661/717/859/955 | baseline 和 no-monitor SR `0.3333`，full SR `0.1667`；full 的 cross-scene 表现不稳定。 |
| `s6_cross_scene_minimatrix_20260423` | ep228/527/661 × baseline/no-monitor/full | 三条 profile 均 SR `0.3333`；ep228 和 ep527 三者全 fail，ep661 三者成功；SmoothNav 两条在 ep661 SPL `0.8529` 明显高于 baseline `0.3017`。 |

Claim: `ep228` / `ep527` 是当前 controller family 的 shared hard cases，不是 `smoothnav-full` 独有坏掉。`ep661` 是必须保护的 positive anchor。

### 3. Frontier/value anchoring 的第一次大改方向正确但过强

| Suite | 关键观察 |
| --- | --- |
| `s7_frontier_value_anchor_20260423` | 纯 frontier/value anchoring 让 baseline、no-monitor、full 全部 SR `0.0`，破坏 ep661 positive anchor。 |
| `s7b_local_direct_anchor_20260423` | 加回“本地可投影 object 允许 direct goal”后，SmoothNav 恢复 ep661 成功，SR 回到 `0.3333` / SPL `0.2843`，但 ep228/527 仍 fail。 |

Claim: “证据不应直接等于行动目标”这个方向对 non-local target 是对的；但 local direct object path 必须保留，否则会错杀 clean positive anchor。

### 4. Target evidence uptake 已修好，但 non-local target approach 未修好

| Suite | ep | 关键 trace |
| --- | ---: | --- |
| `s8_target_evidence_uptake_20260423` | 228 | first_target_step `490`，first_planner_after_target `490`，first_object_commit_after_target `490`，但仍 fail。 |
| `s8_target_evidence_uptake_20260423` | 661 | first_target_step `44`，direct goal latency `0`，成功，SPL `0.8529`。 |
| `s9_nonlocal_target_search_anchor_20260423` | 228 | 非本地 object 不再锁成 `object: tv`，而是 `unexplored target:tv`；first_target_step `490`，仍 fail。 |

Claim: `target_candidate_detected -> planner uptake` 链路已成立。剩余问题不是“看不到 tv / planner 不响应”，而是“非本地 target-anchor 不能稳定转成有效 approach”。

### 5. s10/s11/s13/s14/s15 多次 online probe 没进 target branch，暴露 branch reproducibility

| Suite | first_target_step | target_event_count | 解释 |
| --- | ---: | ---: | --- |
| `s10` | null | 0 | local-map refresh hook 未被覆盖，不能判断 hook 好坏。 |
| `s11` | null | 0 | 同上。 |
| `s12` | 564 | 371 | 进入 target branch，成为关键可复用 trace。 |
| `s13` | null | 0 | pending-fix online 未覆盖 target branch。 |
| `s14` | null | 0 | 同上。 |
| `s15` | null | 0 | target_anchor_attempt patch online probe 仍未覆盖 branch。 |

Claim: 即使 `SMOOTHNAV_LLM_TEMPERATURE=0`，online ep228 仍不能稳定进入同一个 target branch。当前 online rerun 不能作为主验证面。

### 6. s12 captured branch 已经把两个 controller bug 边界固定住

Observation: `target_branch_capture.json` 固化了：

- first_target_step `564`
- target-anchor segments: `564 -> 609`, `632 -> 934`
- refresh_count `9`
- historical branch drop: step `610`, stale pending `unexplored east` 覆盖 `unexplored target:tv`

Claim: `pending_fix_counterfactual.json` 已验证当前代码在同一 captured context 上会保持 `unexplored target:tv`、清空 stale pending、避免历史 drop。

Claim: `target_anchor_attempt_analysis.json` 已验证 naive `patience=3` 会在 step `799` 错杀 attempt 2；`patience=4, min_improvement=1` 在 s12 capture 上更安全。`target_anchor_attempt_replay.json` 显示当前 runtime policy 不会误触发 stall replan。

### 7. 新增最关键 Observation：s12 target branch 后半段 value function 发生语义塌缩

从 `s9/s12` target-frontier 统计看：

- `s9` target-branch grounding events: `3`，其中 bias_score 为 `0` 的事件 `1/3`。
- `s12` target-branch grounding events: `13`，其中 bias_score 为 `0` 的事件 `11/13`。
- `s12` 后半段多步 `base_score = 0` 且 `bias_score = 0`，例如 step `799/824/839/879/919`。
- 此时 final score 几乎只由 `novelty_score + local_actionability_score` 决定，top1/top2 gap 多次接近或等于 `0`。

Concrete examples from s12:

| Step | Trigger | Bias | Frontier | bias_distance | bias_score | base_score | final driver |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 564 | target_candidate_detected | `[166,296]` | `[497,284]` | `2.87` | `0.833` | `4.03` | semantic term active |
| 599 | target_anchor_local_map_refresh | `[166,296]` | `[462,288]` | `4.57` | `0.663` | `9.99` | semantic + base active |
| 632 | frontier_reached | `[166,296]` | `[485,127]` | `11.69` | `0.0` | `4.03` | semantic saturated |
| 799 | target_anchor_local_map_refresh | `[166,296]` | `[366,109]` | `24.49` | `0.0` | `0.0` | novelty/actionability only |
| 824 | target_candidate_detected | `[179,300]` | `[365,110]` | `24.94` | `0.0` | `0.0` | novelty/actionability only |

Claim: 这是当前最高杠杆问题点。系统“保持了 target-anchor branch”，但 frontier/value 层在目标 FMM 距离过大或不可达时把 semantic bias 压成 0，于是执行目标重新退化成 generic exploration。

## Evidence

- Suite aggregate artifacts from remote `results/phase2_revalidation/*/suite_summary.json` were copied and summarized locally.
- Target-evidence traces for `s8`–`s15` were regenerated with `scripts/analyze_target_evidence_trace.py`.
- s12 branch artifacts were read from:
  - `target_branch_capture.json`
  - `pending_fix_counterfactual.json`
  - `target_anchor_attempt_analysis.json`
  - `target_anchor_attempt_replay.json`
- Current code inspection covered:
  - `smoothnav/planner.py`
  - `smoothnav/strategy_grounding.py`
  - `smoothnav/controller_logic.py`
  - `smoothnav/controller_state.py`
  - `smoothnav/main.py`
  - `smoothnav/frontier_scoring.py`
  - `base_UniGoal/src/graph/graph.py`

## Hypotheses (<=3)

1. **Frontier/value target-approach utility is the current primary blocker.**  
   In late s12, both target bias score and agent-distance base score saturate to zero; selected frontiers are then ranked by novelty/actionability, not by expected target approach. This explains why preserving `unexplored target:tv` still times out.

2. **Branch reproducibility is the current primary validation blocker.**  
   Multiple temp0 online reruns never entered target branch. So online SR/SPL is currently mixing at least two problems: whether target evidence appears at all, and whether target branch succeeds once entered.

3. **Controller state contract remains a residual risk, but not the top unresolved bug after replay.**  
   Stale pending promotion and over-aggressive stall policy have controlled fixes/diagnostics now. Residual risks are pending age, same-target pending duplication, stale local-map goal, executor adoption lag, and low-level no-progress after a valid goal.

## Claims disallowed / not yet supported

- Do not claim ep228 is solved: all completed online ep228 probes still fail.
- Do not claim target_anchor_attempt improves SR: s15 did not enter target branch.
- Do not claim s10/s11/s13/s14/s15 invalidate target-anchor logic: they are target-branch coverage misses.
- Do not claim monitor has independent cross-scene gain: s6/s7b full and no-monitor tie on SR/SPL.
- Do not claim pure frontier anchoring is safe: s7 broke ep661.
- Do not claim broad cross-scene advantage until data integrity and multi-scene validation are fully restored.

## Complete detection plan

The detection system should classify every failure into one of these layers. The goal is that every failed run has exactly one dominant first-failing contract plus downstream symptoms.

### A. Suite / artifact / config validity

Detect whether the experiment itself is comparable before interpreting behavior.

Required checks:

- requested profiles × episodes all have exactly one completed run, unless marked partial;
- `summary.json`, `episode_results.json`, `step_traces`, `planner_calls`, `monitor_calls`, `effective_config.json` exist for every run;
- profile comparison has matched episode sets and matching code/config hash except controlled knobs;
- terminal outcome source agrees between suite aggregate and per-run summary;
- no active experiment process remains after completion.

Output fields:

- `suite_complete`, `matched_control`, `missing_runs`, `config_diff_keys`, `artifact_missing`, `terminal_mismatch`.

### B. Dataset / scene / episode integrity

Detect data blockers separately from controller failures.

Required checks:

- HM3D scene loadability and navmesh availability;
- target category/text goal present in episode metadata;
- start pose valid and reachable region non-empty;
- evaluator success distance and episode budget recorded;
- for cross-scene claims, full validation asset repair status recorded.

Output fields:

- `scene_loadable`, `episode_metadata_valid`, `navmesh_valid`, `target_metadata_present`, `dataset_blocker`.

### C. Determinism / branch reproducibility

Detect whether a patch was actually tested on the intended branch.

Required checks:

- repeated same-seed runs aligned by first divergence step;
- hash of LLM prompts/responses, planner parsed outputs, target candidate captions, first strategy sequence;
- `first_target_step`, `target_event_count`, `first_planner_after_target`, `first_target_anchor_step` compared across reruns;
- random-control flags captured: Habitat seed, torch/numpy/python seeds, LLM temperature, model name, API response id if available.

Output fields:

- `branch_id`, `first_divergence_step`, `target_branch_covered`, `planner_response_hashes`, `perception_caption_hashes`, `determinism_config`.

Decision gate:

- If `target_branch_covered=false`, do not use the run to validate target-anchor runtime patches.

### D. Perception / graph target evidence

Detect whether target-like evidence exists and whether it is represented correctly.

Required checks:

- target candidate details per step: caption, score, reason, object center, detection count, room, object id;
- target candidate persistence / flicker rate;
- alias/category matching correctness (`tv`, `television`, `screen`, etc.);
- whether a target candidate exists in world summary but not in `graph_delta.target_candidate_details`.

Output fields:

- `first_target_step`, `target_event_count`, `target_candidate_flicker_rate`, `target_relevance_score`, `target_alias_match_reason`, `target_center_history`.

### E. Planner choice / menu contract

Detect whether the planner saw the right choices and made a valid stage-level decision.

Required checks:

- choices_text snapshot and parsed choice for every planner call;
- target candidate appears in choices when available;
- parsed choice type/id is in menu;
- fallback/empty-response status;
- explored regions and current target region injected into prompt match controller state.

Output fields:

- `target_in_choices`, `planner_selected_target_when_available`, `parse_valid`, `used_fallback`, `empty_response`, `prompt_state_matches_controller`.

### F. Strategy contract: direct object vs non-local target-anchor

Detect whether semantic evidence becomes the right strategy type.

Required checks:

- local projectability of object center;
- if local projectable text target: `object:<name>` and direct-object path allowed;
- if non-local text target: `unexplored target:<name>`, not stable `object:<name>`;
- anchor object preserved in `anchor_object`;
- object relevance threshold not wrongly rejecting true target.

Output fields:

- `strategy_target_region`, `anchor_object`, `direct_object_goal_used`, `direct_object_blocked_reason`, `local_projectable`, `strategy_contract_violation`.

### G. Grounding / frontier-value diagnostics

This is the most important new detector for ep228.

Required checks per grounding event:

- selected frontier, projected local goal, local projection validity;
- full score breakdown: base, semantic bias, novelty, actionability, repeat/recent penalties;
- top-k frontier scores and margin;
- target-anchor distance in both Euclidean and FMM units;
- whether semantic bias/base score saturated to zero;
- whether selected frontier improves target-anchor distance over the attempt;
- whether all candidate scores are flat/tied.

New summary metrics:

- `target_bias_score_zero_rate`
- `target_base_score_zero_rate`
- `target_value_flatline_rate`
- `target_selected_frontier_anchor_distance_delta`
- `target_best_seen_anchor_distance_delta`
- `target_top1_top2_gap_min/median`
- `target_candidate_count_after_bias_filter`
- `target_semantic_term_share = semantic_bias_weight*bias_score/final_score`

Decision gate:

- If `target_bias_score_zero_rate > 0.5` and `target_semantic_term_share≈0` during a target branch, classify the failure as `frontier_value_semantic_collapse`, not as planner failure.

### H. Local-map / coordinate lifecycle

Detect stale goals after local-map recentering.

Required checks:

- local map boundary every step;
- recenter step and reason;
- full-map selected frontier -> local projection before and after recenter;
- refresh hook latency while current strategy is `unexplored target:*`;
- adopted goal still inside local window after recenter.

Output fields:

- `local_map_recentered`, `target_anchor_refresh_latency`, `stale_local_goal_age`, `projected_goal_valid_after_recenter`.

Decision gate:

- If local map moved and no target-anchor refresh occurs in the same step / next trace event, classify as `stale_local_goal_risk`.

### I. Controller state / pending / prefetch contract

Detect stale pending proposals and harmful promotions.

Required checks:

- current vs pending target region and specificity each step;
- pending age and creation reason;
- promotion reason;
- promotion from lower/equal specificity over target anchor forbidden;
- same-target pending duplication tracked separately from stale direction pending;
- target_anchor_attempt state at every target grounding event.

Output fields:

- `pending_age`, `pending_specificity_delta`, `stale_pending_discarded`, `harmful_pending_promotion`, `target_anchor_attempt_summary`, `stall_replan_triggered`.

### J. Monitor / trigger coverage

Detect whether monitor/planner events are consumed.

Required checks:

- graph_delta event types per step;
- monitor `should_evaluate` skip reason;
- target event with neither planner nor monitor call;
- stuck/no-progress trigger counts;
- budget denials vs actual calls.

Output fields:

- `unhandled_target_candidate_steps`, `monitor_skip_reason`, `trigger_to_planner_latency`, `trigger_to_monitor_latency`, `budget_denial_count`.

### K. Executor / tactical / low-level control

Detect whether a good high-level goal fails at the execution layer.

Required checks:

- goal_epoch advanced vs executor adopted goal epoch;
- adopted goal source and adoption latency;
- temp/stuck/global/visible override ratios;
- distance-to-local-goal slope after goal update;
- repeated pauses, collisions, reversals, persistent stuck;
- local goal reached but high-level frontier not refreshed.

Output fields:

- `goal_adoption_latency`, `executor_override_ratio`, `dist_to_goal_slope`, `pause_ratio`, `frontier_reached_without_valid_transition`.

### L. Terminal arbitration / metric correctness

Detect missed success or wrong failure attribution.

Required checks:

- final distance-to-goal and success threshold;
- `terminal_decision` reason and confidence;
- simulator `done` vs SmoothNav terminal reason;
- per-run summary and suite aggregate consistency.

Output fields:

- `terminal_reason`, `final_distance_to_goal`, `success_threshold`, `metric_mismatch`, `terminal_source`.

### M. Replay / counterfactual coverage

Make online runs verification-only, not the first diagnostic surface.

Required replay surfaces:

1. `target_branch_capture`: fixed branch window from step `first_target_step` onward.
2. `pending_fix_counterfactual`: verify stale pending cannot demote target anchor.
3. `target_anchor_attempt_policy_replay`: verify stall/decommit policy on captured branch.
4. `frontier_value_counterfactual`: replay target branch score terms and ask whether an alternate utility would select frontiers with better target-approach progress.
5. `branch_repro_compare`: align multiple online runs and report first divergence.

## Minimal next ablations (<=3)

1. **Offline target-value audit / counterfactual on s12.**  
   Build `analyze_target_frontier_value_trace.py` or `replay_target_frontier_value_counterfactual.py` to quantify value flatline and test a candidate target-approach utility before changing runtime.

2. **Branch reproducibility comparator.**  
   Build `compare_branch_repro.py` over s10–s15 traces to align first divergence and classify why target evidence appears in s12 but not in other temp0 runs.

3. **Trace contract auditor.**  
   Build one run-level auditor that emits the A–L layer statuses above and marks a single dominant first-failing layer for each failed run.

## Literature parallels

- SemExp/PONI support the principle that semantic evidence should shape a map-level long-term goal / potential, not become a raw label commit.
- How To Not Train Your Dragon and VLFM support frontier/value-level semantic conditioning rather than pure event-rule switching.
- SEEK supports belief over support regions rather than one-shot object commitment.

Inference for SmoothNav: the next implementation should turn target evidence into a value/potential/belief term that remains informative when raw FMM distance saturates, while preserving direct-object exploitation for local positive anchors like ep661.

## Decision

Claim: 当前最可能的问题点在 `frontier/value-level target-approach utility` 和 `branch reproducibility` 两层。Controller pending/stall/local-refresh 已经有受控证据说明部分修复或至少安全，不能再作为第一优先盲改方向。

Next action: 先实现离线 target frontier-value audit/counterfactual，再实现 branch reproducibility comparator；online ep228 只作为最后 smoke，不再作为主调试面。
