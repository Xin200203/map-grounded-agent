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

当前目录结构：

```text
docs/
├── README.md
├── implementation/
│   ├── architecture_design.md
│   └── implementation_audit_20260408.md
├── planning/
│   ├── project_state_briefing_20260408.md
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
