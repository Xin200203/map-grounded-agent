# SmoothNav 当前实验计划（2026-04-16）

本文档用于把当前项目从“代码已完成 MEP / 模块重构、本地测试通过”推进到“远端 same-backbone 因果实验可稳定主持”的状态。

它是对 [smoothnav_implementation_master_plan_20260412.md](/Users/xin/Code/research/SmoothNav/docs/planning/smoothnav_implementation_master_plan_20260412.md) 在当前时点的实验化细化，不替代总计划书，而是把“现在应该怎么跑、先跑什么、什么情况下停止放大”写成可执行清单。

## 0. 当前定位

当前实验的真实目标不是立刻冲 benchmark，而是完成一次 **MEP / 模块重构后的 Phase 2 重新验证**：

- 在当前 same-backbone 任务上验证控制器 profile 是否已经稳定分化。
- 确认新的 trace / metrics / terminal arbitration 能否可靠支持结论。
- 判断当前 family 的主参考线究竟是 `smoothnav-full` 还是 `smoothnav-no-monitor`。
- 在进入更大样本或 benchmark 前，先把 shared-path regression、grounding failure、monitor 价值函数这三类问题分开看清。

当前阶段的主实验赛道：

- 主赛道：`goal_type=text`
- 次赛道：`goal_type=ins-image`
- 主机：73 服务器
- 远端路径：`/mnt/sdd/xxy/SmoothNav`
- 环境：`unigoal`

## 1. 代码真相与当前 profile 语义

以当前代码为准，可直接使用的 controller profile 如下：

- `baseline-periodic`
  - 语义：固定间隔 refresh 的 semantic baseline
  - 配置：`fixed_interval`、无 monitor、无 prefetch、无 controller stuck replan
  - 用途：所有批次的主 baseline / canary

- `smoothnav-fixed-interval`
  - 语义：保留 SmoothNav 其余机制，只把 replan 改成固定间隔
  - 配置：`fixed_interval`、无 monitor、无 prefetch、保留 stuck replan
  - 用途：回答 “event-driven 是否优于 fixed-interval”

- `smoothnav-no-monitor`
  - 语义：event-driven + prefetch + stuck replan，但 monitor 完全关闭
  - 用途：当前 family 的健康参考线

- `smoothnav-full`
  - 语义：event-driven + prefetch + stuck replan + `llm_escalation`
  - 说明：当前代码里的 `full` 已不是旧版“monitor 高频主控”，而是 heuristic-first / LLM-only-for-escalation 候选线
  - 用途：主假设检验线

- `smoothnav-no-prefetch`
  - 语义：保留 monitor / event-driven，但关闭 prefetch
  - 用途：回答 “prefetch 是否带来净收益”

- `smoothnav-rules-only`
  - 语义：event-driven + rules monitor + prefetch
  - 用途：压力测试 / 诊断线，不默认当主候选方法

- `baseline-explore`
  - 语义：旧式 frontier explore baseline
  - 用途：仅限 smoke / 回归核查，不进入当前主对照表

以上语义由 [controller_config.py](/Users/xin/Code/research/SmoothNav/smoothnav/controller_config.py) 决定，正式实验优先使用 `--controller-profile` 选择，不建议在正式对照中手动混配 `monitor_policy` / `replan_policy`。

## 2. 当前实验必须遵守的护栏

### 2.1 代码冻结护栏

每一批次实验必须只对应一个 git hash。

执行规则：

1. 本地开发完成后先提交并推送。
2. 73 上 `git checkout <branch> && git pull --ff-only`。
3. 记录本批次 git hash，不混跑不同代码版本的结果。

### 2.2 结果隔离护栏

每个 stage 使用独立 `results_root`。

建议约定：

- `results/phase2_revalidation/s0_preflight`
- `results/phase2_revalidation/s1_canary`
- `results/phase2_revalidation/s2_dev5`
- `results/phase2_revalidation/s3_dev20`
- `results/phase2_revalidation/s4_insimage`
- `results/phase2_revalidation/s5_scaleup`

代码会在该目录下继续创建：

- `<date>/<run_id>/manifest.json`
- `<date>/<run_id>/effective_config.json`
- `<date>/<run_id>/summary.json`
- `<date>/<run_id>/episode_results.json`
- `<date>/<run_id>/step_traces/`
- `<date>/<run_id>/planner_calls/`
- `<date>/<run_id>/monitor_calls/`

### 2.3 基线护栏

`baseline-periodic` 必须出现在每一批对照里。

如果某一批中：

- `baseline-periodic` 明显回退，
- 或 `baseline-periodic` 的 trace / summary 不完整，

则该批结果不进入方法解释，先按 shared-path regression 处理。

### 2.4 分析优先级护栏

每批结果先看：

1. 数据完整性
2. baseline 是否健康
3. 控制指标是否可解释
4. 最后才看 `SR / SPL`

顺序不能反过来。

### 2.5 并行护栏

当前阶段不建议一上来用满 8 张卡。

建议：

- `S0-S1`：单卡顺序跑
- `S2`：最多 2 路并行
- `S3`：最多 3-4 路并行
- `S4-S5`：根据 API 速率和 GPU 占用再扩

原因：

- 当前 monitor / planner 仍依赖同一套 API 网关
- Habitat + LLM 调用一起并发时，更容易把问题混成 “模型失败 / 网络失败 / profile 问题”

## 3. 当前应重点读取的 summary / trace 字段

每一批比较时，优先读 `summary.json` 里的这些键：

- `SR`
- `SPL`
- `avg_high_level_calls`
- `avg_low_level_calls`
- `strategy_switch_count`
- `decision_delay_steps`
- `goal_update_delay_steps`
- `executor_adoption_delay_steps`
- `pending_created_count`
- `pending_promoted_count`
- `pending_created_and_promoted_count`
- `pending_promotion_rate`
- `grounding_noop_rate`
- `grounding_noop_reason_counts`
- `grounding_no_goal_reason_counts`
- `selected_frontier_same_as_prev_rate`
- `temp_goal_override_ratio`
- `stuck_goal_override_ratio`
- `global_goal_override_ratio`
- `visible_target_override_ratio`
- `executor_override_ratio`
- `direction_reuse_count`
- `terminal_outcome_counts`
- `missing_epoch_trace_steps`

对 `step_traces` 的重点字段：

- `current_strategy`
- `pending_strategy`
- `planner_called`
- `monitor_called`
- `monitor_decision`
- `goal_before`
- `goal_after`
- `goal_updated`
- `grounding_events`
- `grounding_noop_reason`
- `executor_adoption_changed`
- `executor_adopted_goal_source`
- `temp_goal_override`
- `stuck_goal_override`
- `terminal_outcome`

## 4. 远端执行模板

推荐统一用下面的命令模板：

```bash
ssh -b "$(ipconfig getifaddr en0)" 10.176.56.73
source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
cd /mnt/sdd/xxy/SmoothNav
git pull --ff-only

export CUDA_VISIBLE_DEVICES=<free_gpu>
export SMOOTHNAV_API_KEY=...
export SMOOTHNAV_BASE_URL=...

python smoothnav/main.py \
  --config-file base_UniGoal/configs/config_habitat.yaml \
  --goal_type text \
  --controller-profile smoothnav-no-monitor \
  --num_eval 5 \
  --api-provider anthropic \
  --api-protocol anthropic-messages \
  --results-root results/phase2_revalidation/s2_dev5
```

补充规则：

- 结果对照时，除 `controller_profile` 和 `results_root` 外，尽量不改其他 CLI 参数。
- 需要做 matched trace 时，优先使用 `--episode_id <k> --num_eval 1`，不要只靠 `num_eval` 的前缀切片。
- 正式对照统一用 `python smoothnav/main.py`，`run.sh` 只保留作快速 smoke 入口。

## 5. 分阶段实验计划

下面的阶段编号是“实验阶段”，不是总计划书里的实现 phase。

### S0：预飞检查与代码冻结

目标：

- 确认 73 上的代码、环境、Git、API、结果输出都处于可主持状态。
- 给后续所有结果建立一个干净起点。

执行内容：

1. 本地确认当前代码已经提交并推送到目标分支。
2. 73 上：
   - `git checkout <branch>`
   - `git pull --ff-only`
   - `git rev-parse --short HEAD`
3. 执行：
   - `python smoothnav/main.py --help`
   - `python base_UniGoal/main.py --help`
4. 跑 2 个单 episode smoke：
   - `baseline-periodic`, `goal_type=text`, `episode_id=0`
   - `smoothnav-full`, `goal_type=text`, `episode_id=0`
5. 检查结果包：
   - `manifest.json`
   - `effective_config.json`
   - `summary.json`
   - `step_traces/episode_000000.jsonl`
   - `planner_calls/`
   - `monitor_calls/`

S0 通过标准：

- 无 import / dataset / API / write-permission 错误
- `manifest.json` 中 git hash 与远端分支一致
- `missing_epoch_trace_steps == 0`
- `summary.json` 中 `terminal_outcomes`、`grounding_noop_reason_counts`、`controller_profile` 正常写出

S0 失败时的动作：

- 不进入 S1
- 先修基础设施，不解释方法表现

### S1：episode-locked canary

目标：

- 用极小样本确认当前 profile 差异是否已经真正穿透到 trace 与 summary。
- 给后面的 `dev5` 提供一个明确的“起点健康度”。

执行内容：

- 选定 `episode_id = 0, 1`
- 对每个 episode 跑 3 组：
  - `baseline-periodic`
  - `smoothnav-no-monitor`
  - `smoothnav-full`

共 6 个 run，全部 `--num_eval 1`。

本阶段要回答的 4 个问题：

1. `baseline-periodic` 是否健康且稳定？
2. `smoothnav-no-monitor` 与 `smoothnav-full` 是否已经在 trace 层分化？
3. `smoothnav-full` 的 `monitor_calls` 是否已经稀疏且有后果，而不是重新变成高频噪声？
4. `grounding_noop_reason_counts` 是否仍以 `get_goal_none.no_frontiers` 为主？

S1 通过标准：

- `baseline-periodic` 两个 canary episode 没有明显 shared-path 异常
- `smoothnav-no-monitor` 与 `smoothnav-full` 至少在以下字段之一上稳定分化：
  - `avg_high_level_calls`
  - `avg_low_level_calls`
  - `grounding_noop_reason_counts`
  - `terminal_outcome_counts`
- `pending_promoted_count <= pending_created_count + pending_created_and_promoted_count`
- `monitor_calls` 和 `monitor_decision` 与 `llm_escalation` 设计一致

S1 失败时的动作：

- 如果 baseline 异常：回到 shared-path regression 审计
- 如果 only full 异常：优先查 monitor / grounding / no-frontiers

### S2：dev5 结构矩阵

目标：

- 在最小可比较样本上完成一次完整结构因果实验。
- 回答 “event-driven / monitor / prefetch / rules” 各自的净作用。

默认 profile 矩阵：

- `baseline-periodic`
- `smoothnav-fixed-interval`
- `smoothnav-no-monitor`
- `smoothnav-full`
- `smoothnav-no-prefetch`
- `smoothnav-rules-only`

统一设置：

- `goal_type=text`
- `num_eval=5`
- 同一 git hash
- 同一 config file
- 同一 API provider/protocol

本阶段对应的问题：

- `baseline-periodic` vs `smoothnav-no-monitor`
  - event-driven semantic replanning 是否至少有正向信号
- `smoothnav-fixed-interval` vs `smoothnav-no-monitor`
  - event-driven 是否优于 fixed-interval
- `smoothnav-no-monitor` vs `smoothnav-full`
  - monitor 当前是增益还是负担
- `smoothnav-full` vs `smoothnav-no-prefetch`
  - prefetch 是否已经活到能影响结果
- `smoothnav-rules-only`
  - 当前 rules 路径是候选方法还是单纯压力测试

S2 通过标准：

- 所有 run 的 `missing_epoch_trace_steps == 0`
- 没有统计口径异常
- `baseline-periodic` 未出现明显异常回退
- 至少一条 SmoothNav 线在 `SR` 或 `SPL` 上不弱于 `baseline-periodic`
- `smoothnav-full`、`smoothnav-no-monitor`、`smoothnav-no-prefetch` 在控制指标上出现可解释差异

S2 后的决策规则：

- 如果 `smoothnav-no-monitor` 明显最好：
  - 暂定它为主参考线
  - `full` 继续作为 monitor 价值函数定位线
- 如果 `smoothnav-full` 已追平或超过 `no-monitor`：
  - `full` 可以进入下一阶段主线
- 如果 `rules-only` 只表现出高 churn 和差性能：
  - 后续 stage 默认移除，不再消耗主实验预算

### S3：dev20 核心矩阵

目标：

- 把 `dev5` 里出现的趋势拉到更可靠的样本量。
- 决定当前 family 的真正主线。

默认 profile 集合：

- `baseline-periodic`
- `smoothnav-fixed-interval`
- `smoothnav-no-monitor`
- `smoothnav-full`
- `smoothnav-no-prefetch`

条件保留：

- `smoothnav-rules-only` 只有在 S2 中显示出独立解释价值时才保留

统一设置：

- `goal_type=text`
- `num_eval=20`

S3 重点关注：

- `terminal_outcome_counts` 是否开始稳定
- `grounding_noop_rate` 是否下降到可解释水平
- `grounding_no_goal_reason_counts` 中 `no_frontiers / no_candidate_frontiers` 的占比
- `executor_override_ratio` 是否仍明显过高
- `pending_created_count / pending_promoted_count` 是否已经显示 prefetch 活性

S3 通过标准：

- `baseline-periodic` 健康
- `smoothnav-no-monitor` 或 `smoothnav-full` 中至少一条相对 baseline 有稳定正信号
- `smoothnav-full` 不再显著稳定劣于 `smoothnav-no-monitor`
- `grounding_noop_reason_counts` 的主因已清楚，不再只有“总比例高”这种模糊结论

S3 失败时的动作：

- 不进入大样本
- 先 patch，再重跑 `dev5` 或 `dev20`

### S4：次赛道迁移 sanity（ins-image）

目标：

- 验证当前控制器不是只在 text-goal trace 上“会说话”，而是至少能迁移到当前第二个 same-backbone 任务。

建议 profile：

- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

统一设置：

- `goal_type=ins-image`
- `num_eval=5`

注意：

- S4 是 sanity，不是主表
- 只有在 S3 通过后才进入

S4 通过标准：

- 结果包完整
- 不出现新型基础设施错误
- 控制指标仍然可解释

### S5：主线放大验证（dev50 / dev100）

目标：

- 在当前任务上给主线 profile 做更大样本确认
- 为下一阶段论文叙事准备更稳的 same-backbone 证据

进入条件：

- S3 已通过
- 已明确主参考线

推荐 profile 集：

- 必带：`baseline-periodic`
- 主线 1：`smoothnav-no-monitor` 或 `smoothnav-full`
- 主线 2：另一条最接近的 ablation
- 可选：`smoothnav-fixed-interval`

统一设置：

- `goal_type=text`
- `num_eval=50`，必要时扩到 `100`

S5 的目标不是继续找 bug，而是回答：

- 当前最优线是否相对 baseline 有稳定优势或至少稳定的机制收益
- `full` 是否值得继续作为主方法
- 如果 `no-monitor` 仍最佳，论文主线是否要重构为 “event-driven planner + heuristic tactical layer”

## 6. 每个阶段结束后必须写出的结论

每一阶段结束后，都应至少产出一段简短结论，内容固定为：

1. 本阶段跑了哪些 profile、多少 episode
2. baseline 是否健康
3. 当前 best profile 是谁
4. 当前 main failure mode 是谁
5. 是否进入下一阶段

不要跳过这一步，否则后面很容易重新陷入“代码在变、实验在跑、但主持结论在漂移”的状态。

## 7. 当前最重要的 go / no-go gate

在进入 benchmark adapter 之前，至少要满足下面 3 个 gate：

- Gate A：`baseline-periodic` 不再表现出 shared-path regression
- Gate B：`smoothnav-full` 或 `smoothnav-no-monitor` 至少有一条在当前任务上呈现稳定正信号
- Gate C：`grounding_noop / pending / terminal_outcome / executor adoption` 这四类控制指标已经足够可信，不再出现明显统计语义错误

只有这 3 个 gate 同时过了，才值得继续推进 benchmark 级实验。

## 8. 当前推荐执行顺序

如果现在立刻开始实验，推荐顺序如下：

1. S0：预飞检查与代码冻结
2. S1：episode-locked canary
3. S2：text-goal dev5 结构矩阵
4. S3：text-goal dev20 核心矩阵
5. S4：ins-image dev5 sanity
6. S5：主线 profile 的 dev50 / dev100 放大验证

当前不推荐的顺序：

- 不先上 benchmark
- 不先跑大样本
- 不先让 `rules-only` 吃大量预算
- 不在 baseline 未健康时解释 `full`
- 不在 `full` 仍明显输给 `no-monitor` 时硬把 `full` 继续当主结果线

一句话收束：

当前最合理的实验主持方式，不是“尽快多跑”，而是“先用小批次把 gate 跑实，再把主线 profile 放大”。这会比反复大规模试错更快得到可信结论。
