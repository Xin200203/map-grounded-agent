# SmoothNav Docs

当前 `docs/` 按用途重组为四类：

- `implementation/`
  - 与当前代码实现直接相关的架构、审计和实现细节
- `planning/`
  - 项目状态、阶段计划、执行路线
- `paper/`
  - 论文定位、文献调研、投稿写作材料
- `archive/`
  - 历史分析材料，仅保留对当前方向转向仍有解释价值的文档

推荐阅读顺序：

1. `planning/smoothnav_implementation_master_plan_20260412.md`
   - 看当前项目的 phase gate、实施顺序和验收口径
2. `planning/smoothnav_experiment_plan_20260416.md`
   - 看当前这一轮实验应如何分阶段推进
   - 直接用于安排 73 服务器上的 `preflight / canary / dev5 / dev20 / scale-up`
   - 说明每个阶段的 go/no-go gate 和应优先读取的 summary 字段
3. `planning/smoothnav_first_batch_task_sheet_20260416.md`
   - 看第一批实验如何实际落地执行
   - 精确到先推哪个分支、73 上跑哪些命令、每一批结果出来后看哪些键决定是否进入下一批
4. `implementation/phase2_dev5_failure_taxonomy_20260413.md`
   - 看第一轮 `dev5` 的修复前实验基线，理解问题最初是如何暴露出来的
5. `implementation/phase2_code_level_diagnostic_20260413.md`
   - 看问题如何被收紧到具体函数和条件
6. `implementation/phase2_patch_status_20260413.md`
   - 看当前代码已经完成了哪些修复、哪些还只是部分完成
7. `implementation/phase2_next_patch_decision_sheet_20260415.md`
   - 看 `dev5_patch` 之后下一轮 patch 的最小决策面
   - 直接用于安排 `get_goal_none`、`pending` 统计、monitor 收缩和 baseline 回退检查
8. `implementation/phase2_shared_path_regression_audit_20260415.md`
   - 看 `baseline-periodic` 回退为什么优先要按 shared `graph/grounding` 路径来审
   - 说明这轮已经加入的 raw frontier / relaxed distance fallback
9. `implementation/module_refactor_status_20260415.md`
   - 看模块级重构如何把当前系统迁移到 `WorldState / MissionState / TacticalDecision / GeometricGoal / ExecutorCommand`
   - 说明哪些层已经串进主循环，哪些仍是后续行为改进项

当前目录结构：

```text
docs/
├── README.md
├── implementation/
│   ├── architecture_design.md
│   ├── controller_profile_and_trace_alignment_notes_20260413.md
│   ├── implementation_audit_20260408.md
│   ├── llm_gateway_protocol_notes_20260413.md
│   ├── module_refactor_status_20260415.md
│   ├── phase2_code_level_diagnostic_20260413.md
│   ├── phase2_dev5_failure_taxonomy_20260413.md
│   ├── phase2_next_patch_decision_sheet_20260415.md
│   ├── phase2_shared_path_regression_audit_20260415.md
│   └── phase2_patch_status_20260413.md
├── planning/
│   ├── project_state_briefing_20260408.md
│   ├── smoothnav_experiment_plan_20260416.md
│   ├── smoothnav_first_batch_task_sheet_20260416.md
│   └── smoothnav_implementation_master_plan_20260412.md
├── paper/
│   ├── literature_survey.md
│   └── paper_submission_prep_cn_20260409.md
└── archive/
    └── findings_action_analysis.md
```

已删除的过时文档：

- `static_analysis.md`
- `unigoal_analysis.md`

删除原因：

- 两者都属于第一代方案时期的材料，核心假设依赖 custom executor / override-style control。
- 当前项目已经明确转向 “event-driven semantic replanning + preserved UniGoal executor”。
- 历史结论里仍有价值的部分，已经在现存的 briefing / audit / paper-facing 文档中被吸收和转述。

新增的重要实现说明：

- `implementation/llm_gateway_protocol_notes_20260413.md`
  - 记录 Clauddy 下 Claude / GPT / Codex 的协议差异
  - 说明为什么当前项目新增了 `api_provider` / `api_protocol`
  - 给出后续排障顺序，避免重复掉进 `chat.completions` 兼容性误判
  - 补充当前真实跑通实验所使用的工程细节，包括 `httpx + Anthropic headers`、本地凭据文件和 conda 覆盖顺序
  - 记录 184 服务器上的实验加速经验，包括更快的 retry profile、空响应短路 fallback 和轻量化默认运行配置
- `implementation/controller_profile_and_trace_alignment_notes_20260413.md`
  - 说明 `baseline-explore` 与 `baseline-periodic` 的职责边界
  - 记录 phantom episode / summary 对齐问题的根因与修复口径
  - 用于后续 Phase 2 对照实验时避免基线语义混淆
- `implementation/phase2_dev5_failure_taxonomy_20260413.md`
  - 整理 Phase 2 第一轮 `dev5` 五组对照的失败类型、证据链和当前实现优先级
  - 当前应把它当作“修复前实验基线”，而不是“当前代码状态”
  - 明确当前问题更偏向 “semantic intent 没有及时落到 executor” 而不是 “没有 planner/monitor”
- `implementation/phase2_code_level_diagnostic_20260413.md`
  - 把 Phase 2 的问题进一步收紧到具体函数、具体条件和具体修改点
  - 逐条回答 `main.py / apply_strategy / get_goal / instance_discriminator / control_metrics` 里的关键实现细节
  - 给出下一轮最值得直接开始改的 3 个实现入口
  - 当前应把它当作“修复前函数级定位”，而不是“当前代码状态”
- `implementation/phase2_patch_status_20260413.md`
  - 记录第一轮 P0/P1/P2/P3/P4 改动之后，当前代码已经完成了什么
  - 最新同步里已经补入两阶段 grounding、grounding-failure replan、executor adoption helper、`enable_controller_trace` 开关和 6 个 gen2 专项测试文件
  - 作为“当前代码状态”主文档使用；是否真的通过 gate，仍需看后续远端实验
- `implementation/phase2_next_patch_decision_sheet_20260415.md`
  - 承接 `dev5_patch` 结果后的下一轮短决策单
  - 明确当前最优先的 4 个动作：先修 `get_goal_none`、修 `pending` 统计、把 monitor 收缩到 heuristic-first / escalation-only、专门审 baseline 回退
  - 用于后续 patch 排程和实验主持，而不是替代完整 failure taxonomy
- `implementation/phase2_shared_path_regression_audit_20260415.md`
  - 单独审 baseline 回退对应的 shared-path 风险
  - 当前结论是：优先怀疑共享的 `graph/grounding` 路径，而不是 monitor 本身
  - 记录本轮已经加入的 frontier fallback 与下一轮 baseline sanity 应重点看的字段

本轮整理后移除的重复探针文档：

- `implementation/phase2_get_goal_topk_probe_20260413.md`
- `implementation/phase2_smoothnav_full_ep0_step40_60_trace_20260413.md`
- `implementation/phase2_temp_goal_lifecycle_trace_20260413.md`

移除原因：

- 3 份文档的关键证据已被吸收到 `phase2_patch_status_20260413.md`
- 原文里多处“当前还没有某字段/某机制”的说法已经被后续实现追平
- 继续保留原文件，会混淆“修复前证据”和“当前状态”
