# Generic Rendered Flywheel Report - SmoothNav Experiment Review

- mode: `mixed`
- project_root: `/Users/xin/Research/Code/research/SmoothNav`
- team_mode: `team-ready`

## Observation
- 当前输入模式：`mixed`。
- summary 输入：`/Users/xin/Research/Code/research/SmoothNav/docs/implementation/episode296_grounding_and_graph_growth_diagnosis_20260421.md`。
- raw artifact 数量：`38`。

## Evidence
识别到 8 个主指标条目。
识别到 4 个诊断指标条目。
发现 38 个原始 artifact 文件。
发现 22 个指标名存在多值冲突。
- `graph_node_count = 31` 来源：`/Users/xin/Research/Code/research/SmoothNav/docs/implementation/episode296_grounding_and_graph_growth_diagnosis_20260421.md`
- `success_dist = 1.0` 来源：`/Users/xin/Research/Code/research/SmoothNav/analysis/flywheel_inputs/20260426_smoothnav_all_experiments/s12_diagnostics/effective_config.json`
- `map_pred_threshold = 1.0` 来源：`/Users/xin/Research/Code/research/SmoothNav/analysis/flywheel_inputs/20260426_smoothnav_all_experiments/s12_diagnostics/effective_config.json`
- `map_size_cm = 3600` 来源：`/Users/xin/Research/Code/research/SmoothNav/analysis/flywheel_inputs/20260426_smoothnav_all_experiments/s12_diagnostics/effective_config.json`
- `map_resolution = 5` 来源：`/Users/xin/Research/Code/research/SmoothNav/analysis/flywheel_inputs/20260426_smoothnav_all_experiments/s12_diagnostics/effective_config.json`
- `graph_frontier_min_size = 20` 来源：`/Users/xin/Research/Code/research/SmoothNav/analysis/flywheel_inputs/20260426_smoothnav_all_experiments/s12_diagnostics/effective_config.json`
- 指标冲突：`H` -> `10, 11`
- 指标冲突：`SR` -> `0.0, 0.4, 0.4667, 0.5, 0.5333, 0.6, 0.6667, 0.8333, 0.8571, 1.0`
- 指标冲突：`SPL` -> `0.0, 0.07728753292092762, 0.15034257918122163, 0.2071, 0.2120, 0.2157720952950592, 0.2338, 0.2353, 0.2416, 0.2636, 0.2650, 0.2678, 0.2765, 0.2922, 0.2927, 0.3017, 0.3659, 0.3687, 0.4052, 0.4662, 0.4881, 0.7406789288874713, 0.8529, 0.8529184432440698`
- 指标冲突：`avg_high_level_calls` -> `10.0, 11.0, 11.4, 12.0, 14.0, 18.0, 22.0, 6.0, 6.5, 8.0, 9.0, 9.6, 9.8`
- 指标冲突：`strategy_switch_count` -> `13.0, 14.0, 6.0, 7.0, 8.0`
- 指标冲突：`goal_update_delay_steps` -> `1.625, 10.272727272727273, 17.846153846153847, 19.470588235294116, 22.25, 24.67, 3.75, 34.33, 6.285714285714286, 7.333333333333333, 8.11`
- 指标冲突：`grounding_noop_rate` -> `0.0, 0.6, 0.6521739130434783, 0.6744186046511628, 0.7058823529411765`
- 指标冲突：`pending_created_count` -> `0, 11, 2, 3, 4, 8`
- 指标冲突：`pending_promoted_count` -> `1, 10, 2, 3, 5`
- 指标冲突：`executor_override_ratio` -> `0.0, 0.0042780748663101605, 0.019455252918287938, 0.02526315789473684, 0.0571, 0.05714285714285714, 0.1146, 0.1179, 0.13043478260869565, 0.1314, 0.1334, 0.1409, 0.1446, 0.1847, 0.2031`
- 指标冲突：`avg_low_level_calls` -> `0.0, 17.0, 2.0`
- 指标冲突：`override_duration` -> `12, 2, 7`
- 指标冲突：`num_eval` -> `1, 5`
- 指标冲突：`out_of_local_window` -> `2, 4`
- 指标冲突：`steps` -> `196, 233`
- 指标冲突：`pending` -> `4, 5, 8`
- 指标冲突：`first_target_step` -> `44, 490, 564`
- 指标冲突：`first_planner_after_target` -> `44, 490, 564`
- 指标冲突：`first_object_commit_after_target` -> `44, 490`
- 指标冲突：`target_event_count` -> `0, 169, 371`
- 指标冲突：`patience` -> `3, 4`
- 指标冲突：`patience_updates` -> `3, 4`

## Hypotheses (<=3)
1. 当前负例更像是模块接口/分支触发/配置路径问题，而不是核心想法完全错误。
2. 当前异常可能主要来自数据或 split 条件变化，而不是算法本身退化。
3. 当前差异可能被 evaluator、threshold 或汇总脚本放大，需要先校验指标口径。

## Claims disallowed / not yet supported
- 当前存在同名指标多值冲突，不能直接把单一数值当作最终结论。
- 当前未附带 literature 搜索结果，不能声称已有工作已直接证明本轮解释。

## Minimal next ablations (<=3)
1. 增加一组最小 runtime instrumentation，验证目标 branch 是否真实触发且输出被消费。
2. 对同一 evaluator 下的两个数据切分做最小对照，检查异常是否随 split 消失。
3. 在同一 checkpoint 上重跑一个最小 evaluator smoke test，核对 threshold 与 report path。

## Literature parallels
- 待在交互式运行中通过网页检索补齐 literature parallels。

## Decision
- 优先按以下判断推进：当前负例更像是模块接口/分支触发/配置路径问题，而不是核心想法完全错误。
