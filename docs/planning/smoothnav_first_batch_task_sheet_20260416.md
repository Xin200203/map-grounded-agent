# SmoothNav 第一批实验任务单（2026-04-16）

本文档把 [smoothnav_experiment_plan_20260416.md](/Users/xin/Code/research/SmoothNav/docs/planning/smoothnav_experiment_plan_20260416.md) 进一步收成一份可直接执行的“第一批实验任务单”。

目标不是覆盖所有后续实验，而是把当前最该做的第一波工作写清楚：

1. 先把当前代码冻结并推成 73 可拉取的实验分支。
2. 先跑 `S0 preflight`。
3. 再跑 `S1 canary`。
4. 只有在前两步通过后，才进入 `S2 dev5` 结构矩阵。

本文档默认：

- 主机：73 服务器
- 远端路径：`/mnt/sdd/xxy/SmoothNav`
- 环境：`unigoal`
- 主赛道：`goal_type=text`
- API：Clauddy 转发，`provider=anthropic`，`protocol=anthropic-messages`

## 0. 当前分支现实状态

截至写本文档时，本地当前工作分支是：

- `codex/module-refactor-20260415`

但当前 `origin` 上可见的远端分支只有：

- `main`

因此，**73 现在还不能直接拉 `codex/module-refactor-20260415`**。  
第一批实验的第一个动作，不是跑实验，而是先把当前本地代码冻结并推送成 73 可拉的实验分支。

本任务单默认采用下面这条实验分支：

- `codex/module-refactor-20260415`

如果后续你决定改成更明确的实验分支名，例如 `codex/phase2-revalidation-20260416`，只要把本文档里的分支名整体替换即可。

## 1. 本批次总顺序

第一批实验按下面顺序执行，不跳步：

1. `B0`：本地代码冻结与推送
2. `B1`：73 远端预飞检查（S0）
3. `B2`：73 远端 canary（S1）
4. `B3`：73 远端 text-goal `dev5` 结构矩阵（S2）

进入下一批的前提：

- 当前批次所有 run 的结果包完整
- `baseline-periodic` 健康
- 控制指标可解释
- 没有明显统计口径异常

## 2. B0：本地代码冻结与推送

### 2.1 目标

把当前本地代码整理成一个 73 可以 `git pull` 的实验版本。

### 2.2 执行命令

在本地执行：

```bash
cd /Users/xin/Code/research/SmoothNav
git status --short --branch
git branch -vv
```

建议先跑当前这组本地验证，再提交：

```bash
cd /Users/xin/Code/research/SmoothNav
python3 -m unittest \
  tests.test_mep_contracts \
  tests.test_layered_contracts \
  tests.test_controller_logic \
  tests.test_graph_delta_gen2 \
  tests.test_strategy_grounding_gen2 \
  tests.test_apply_strategy \
  tests.test_executor_adoption_gen2 \
  tests.test_tracing \
  tests.test_monitor_gen2 \
  tests.test_planner_gen2 \
  tests.test_controller_config_phase2 \
  tests.test_control_metrics_gen2
```

然后提交并推送：

```bash
cd /Users/xin/Code/research/SmoothNav
git add .
git commit -m "Prepare first batch phase2 revalidation experiments"
git push -u origin codex/module-refactor-20260415
```

### 2.3 B0 通过标准

- 本地测试通过
- `git push -u origin codex/module-refactor-20260415` 成功
- GitHub 上已经存在 `origin/codex/module-refactor-20260415`

### 2.4 B0 不通过时怎么处理

- 如果测试失败：先修代码，不进入 B1
- 如果 push 失败：先修 Git/认证问题，不进入 B1

## 3. B1：73 远端预飞检查（S0）

### 3.1 目标

确认 73 上的代码、分支、环境、API、结果目录、trace 落盘都正常。

### 3.2 推荐使用的 GPU

- `CUDA_VISIBLE_DEVICES=0`

### 3.3 远端准备命令

登录：

```bash
ssh -b "$(ipconfig getifaddr en0)" 10.176.56.73
```

进入环境并拉取目标分支：

```bash
source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
cd /mnt/sdd/xxy/SmoothNav
git fetch origin
git checkout codex/module-refactor-20260415
git pull --ff-only
git rev-parse --short HEAD
```

准备 API 和 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
if [ -f /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh ]; then
  source /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh
fi
echo "$SMOOTHNAV_BASE_URL"
python smoothnav/main.py --help >/tmp/smoothnav_help.txt && tail -n 20 /tmp/smoothnav_help.txt
python base_UniGoal/main.py --help >/tmp/base_unigoal_help.txt && tail -n 20 /tmp/base_unigoal_help.txt
```

### 3.4 预飞 run 命令

先跑 `baseline-periodic`：

```bash
cd /mnt/sdd/xxy/SmoothNav
python smoothnav/main.py \
  --config-file base_UniGoal/configs/config_habitat.yaml \
  --goal_type text \
  --controller-profile baseline-periodic \
  --episode_id 0 \
  --num_eval 1 \
  --api-provider anthropic \
  --api-protocol anthropic-messages \
  --results-root results/phase2_revalidation/s0_preflight
```

再跑 `smoothnav-full`：

```bash
cd /mnt/sdd/xxy/SmoothNav
python smoothnav/main.py \
  --config-file base_UniGoal/configs/config_habitat.yaml \
  --goal_type text \
  --controller-profile smoothnav-full \
  --episode_id 0 \
  --num_eval 1 \
  --api-provider anthropic \
  --api-protocol anthropic-messages \
  --results-root results/phase2_revalidation/s0_preflight
```

### 3.5 B1 结果检查命令

先列出最新结果：

```bash
cd /mnt/sdd/xxy/SmoothNav
find results/phase2_revalidation/s0_preflight -name summary.json | sort
```

再打印关键字段：

```bash
cd /mnt/sdd/xxy/SmoothNav
python - <<'PY'
import glob, json, os
paths = sorted(glob.glob('results/phase2_revalidation/s0_preflight/*/*/summary.json'))
for path in paths[-2:]:
    summary = json.load(open(path))
    print('=' * 80)
    print(path)
    for key in [
        'controller_profile',
        'goal_type',
        'num_episodes',
        'SR',
        'SPL',
        'avg_high_level_calls',
        'avg_low_level_calls',
        'grounding_noop_rate',
        'grounding_noop_reason_counts',
        'grounding_no_goal_reason_counts',
        'terminal_outcomes',
        'missing_epoch_trace_steps',
    ]:
        print(f'{key}: {summary.get(key)}')
PY
```

### 3.6 B1 进入 B2 的 gate

必须同时满足：

- `missing_epoch_trace_steps == 0`
- `controller_profile` 写对
- `terminal_outcomes` 正常写出
- `grounding_noop_reason_counts` 正常写出
- `manifest.json`、`effective_config.json`、`summary.json`、`step_traces/`、`planner_calls/`、`monitor_calls/` 全部存在

如果不满足：

- 停在 B1
- 先修基础设施，不解释 profile 表现

## 4. B2：73 远端 canary（S1）

### 4.1 目标

在极小样本上先确认：

- baseline 是否健康
- `smoothnav-no-monitor` 与 `smoothnav-full` 是否已在行为和控制指标上分化

### 4.2 推荐使用的 GPU

- 单卡顺序跑：`CUDA_VISIBLE_DEVICES=0`

### 4.3 运行命令

在 73 上执行：

```bash
source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
cd /mnt/sdd/xxy/SmoothNav
git pull --ff-only
export CUDA_VISIBLE_DEVICES=0
if [ -f /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh ]; then
  source /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh
fi

for ep in 0 1; do
  for profile in baseline-periodic smoothnav-no-monitor smoothnav-full; do
    python smoothnav/main.py \
      --config-file base_UniGoal/configs/config_habitat.yaml \
      --goal_type text \
      --controller-profile "${profile}" \
      --episode_id "${ep}" \
      --num_eval 1 \
      --api-provider anthropic \
      --api-protocol anthropic-messages \
      --results-root results/phase2_revalidation/s1_canary
  done
done
```

### 4.4 B2 结果检查命令

打印 6 个 run 的关键字段：

```bash
cd /mnt/sdd/xxy/SmoothNav
python - <<'PY'
import glob, json
paths = sorted(glob.glob('results/phase2_revalidation/s1_canary/*/*/summary.json'))
for path in paths:
    summary = json.load(open(path))
    print('=' * 80)
    print(path)
    for key in [
        'controller_profile',
        'SR',
        'SPL',
        'avg_high_level_calls',
        'avg_low_level_calls',
        'grounding_noop_rate',
        'grounding_noop_reason_counts',
        'grounding_no_goal_reason_counts',
        'pending_created_count',
        'pending_promoted_count',
        'pending_created_and_promoted_count',
        'executor_override_ratio',
        'terminal_outcomes',
        'missing_epoch_trace_steps',
    ]:
        print(f'{key}: {summary.get(key)}')
PY
```

如果需要看 matched trace，优先查同一 `episode_id` 的三条线：

- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

重点看：

- `current_strategy`
- `planner_called`
- `monitor_called`
- `monitor_decision`
- `goal_before`
- `goal_after`
- `goal_updated`
- `grounding_events`
- `temp_goal_override`
- `stuck_goal_override`
- `terminal_outcome`

### 4.5 B2 进入 B3 的 gate

必须同时满足：

- `baseline-periodic` 没有明显 shared-path 异常
- 所有 run 的 `missing_epoch_trace_steps == 0`
- `pending_promoted_count <= pending_created_count + pending_created_and_promoted_count`
- `smoothnav-no-monitor` 与 `smoothnav-full` 至少在以下任一项上稳定分化：
  - `avg_high_level_calls`
  - `avg_low_level_calls`
  - `grounding_noop_reason_counts`
  - `terminal_outcomes`

如果不满足：

- 如果 baseline 异常：先审 shared path
- 如果 full 异常而 no-monitor 健康：先审 monitor / grounding
- 不进入 B3

## 5. B3：73 远端 text-goal dev5 结构矩阵（S2）

### 5.1 目标

在 `dev5` 上完成当前第一轮结构因果对照，回答：

- event-driven 是否优于 fixed-interval
- monitor 是否带来净收益
- prefetch 是否已经活起来
- rules-only 是否只是压力测试

### 5.2 profile 集

- `baseline-periodic`
- `smoothnav-fixed-interval`
- `smoothnav-no-monitor`
- `smoothnav-full`
- `smoothnav-no-prefetch`
- `smoothnav-rules-only`

### 5.3 推荐并行方式

第一批 `dev5` 建议 2 路并行，不要超过。

终端 A，GPU 0：

```bash
ssh -b "$(ipconfig getifaddr en0)" 10.176.56.73
source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
cd /mnt/sdd/xxy/SmoothNav
git pull --ff-only
export CUDA_VISIBLE_DEVICES=0
if [ -f /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh ]; then
  source /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh
fi

for profile in baseline-periodic smoothnav-no-monitor smoothnav-full; do
  python smoothnav/main.py \
    --config-file base_UniGoal/configs/config_habitat.yaml \
    --goal_type text \
    --controller-profile "${profile}" \
    --num_eval 5 \
    --api-provider anthropic \
    --api-protocol anthropic-messages \
    --results-root results/phase2_revalidation/s2_dev5
done
```

终端 B，GPU 1：

```bash
ssh -b "$(ipconfig getifaddr en0)" 10.176.56.73
source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh
conda activate unigoal
cd /mnt/sdd/xxy/SmoothNav
git pull --ff-only
export CUDA_VISIBLE_DEVICES=1
if [ -f /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh ]; then
  source /mnt/sdd/xxy/SmoothNav/.local/clauddy.env.sh
fi

for profile in smoothnav-fixed-interval smoothnav-no-prefetch smoothnav-rules-only; do
  python smoothnav/main.py \
    --config-file base_UniGoal/configs/config_habitat.yaml \
    --goal_type text \
    --controller-profile "${profile}" \
    --num_eval 5 \
    --api-provider anthropic \
    --api-protocol anthropic-messages \
    --results-root results/phase2_revalidation/s2_dev5
done
```

### 5.4 B3 结果检查命令

输出各 profile 的关键 summary 字段：

```bash
cd /mnt/sdd/xxy/SmoothNav
python - <<'PY'
import glob, json, os
paths = sorted(glob.glob('results/phase2_revalidation/s2_dev5/*/*/summary.json'))
for path in paths:
    summary = json.load(open(path))
    print('=' * 80)
    print(path)
    for key in [
        'controller_profile',
        'SR',
        'SPL',
        'avg_high_level_calls',
        'avg_low_level_calls',
        'strategy_switch_count',
        'goal_update_delay_steps',
        'executor_adoption_delay_steps',
        'pending_created_count',
        'pending_promoted_count',
        'pending_created_and_promoted_count',
        'grounding_noop_rate',
        'grounding_noop_reason_counts',
        'grounding_no_goal_reason_counts',
        'selected_frontier_same_as_prev_rate',
        'executor_override_ratio',
        'direction_reuse_count',
        'terminal_outcomes',
        'missing_epoch_trace_steps',
    ]:
        print(f'{key}: {summary.get(key)}')
PY
```

### 5.5 B3 结束后决定是否进入下一批

进入下一批 `dev20` 的前提：

- `baseline-periodic` 健康
- 所有 run 的 `missing_epoch_trace_steps == 0`
- 没有统计语义异常
- 至少一条 SmoothNav 线在 `SR` 或 `SPL` 上不弱于 `baseline-periodic`
- `smoothnav-full`、`smoothnav-no-monitor`、`smoothnav-no-prefetch` 的差异已经能通过控制指标解释

重点读取的决策键：

- `SR`
- `SPL`
- `avg_high_level_calls`
- `avg_low_level_calls`
- `grounding_noop_rate`
- `grounding_noop_reason_counts`
- `grounding_no_goal_reason_counts`
- `pending_created_count`
- `pending_promoted_count`
- `pending_created_and_promoted_count`
- `executor_override_ratio`
- `terminal_outcomes`
- `missing_epoch_trace_steps`

### 5.6 B3 后的 go / no-go 规则

- 如果 `smoothnav-no-monitor` 明显最好：
  - 下一批以 `no-monitor` 为主参考线
  - `full` 留作 monitor 价值函数修正线

- 如果 `smoothnav-full` 已追平或超过 `no-monitor`：
  - 下一批 `full` 可以进入主线

- 如果 `smoothnav-rules-only` 仍只表现为高 churn 和差性能：
  - 下一批移除 `rules-only`

- 如果 `baseline-periodic` 明显异常：
  - 暂停扩大实验
  - 回到 shared-path regression 审计

## 6. 本任务单最短执行摘要

如果只看最短版，顺序就是：

1. 本地把 `codex/module-refactor-20260415` 提交并推上 GitHub。
2. 73 上 `git checkout codex/module-refactor-20260415 && git pull --ff-only`。
3. 跑 `B1 preflight`：
   - `baseline-periodic`, episode 0
   - `smoothnav-full`, episode 0
4. 结果完整且 `missing_epoch_trace_steps == 0` 后，跑 `B2 canary`：
   - `baseline-periodic / smoothnav-no-monitor / smoothnav-full`
   - `episode 0, 1`
5. baseline 健康且 profile 已分化后，跑 `B3 dev5` 六组结构矩阵。
6. 根据 `SR/SPL + grounding_noop + pending + override + terminal_outcomes` 决定是否进入 `dev20`。

一句话收束：

第一批实验不是“立刻大跑”，而是“先把当前代码冻结到 73 可拉、再用 preflight 和 canary 把 gate 跑实，然后才让 `dev5` 开始提供结构结论”。
