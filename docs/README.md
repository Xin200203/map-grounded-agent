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
│   ├── implementation_audit_20260408.md
│   └── llm_gateway_protocol_notes_20260413.md
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

新增的重要实现说明：

- `implementation/llm_gateway_protocol_notes_20260413.md`
  - 记录 Clauddy 下 Claude / GPT / Codex 的协议差异
  - 说明为什么当前项目新增了 `api_provider` / `api_protocol`
  - 给出后续排障顺序，避免重复掉进 `chat.completions` 兼容性误判
  - 补充当前真实跑通实验所使用的工程细节，包括 `httpx + Anthropic headers`、本地凭据文件和 conda 覆盖顺序
  - 记录 184 服务器上的实验加速经验，包括更快的 retry profile、空响应短路 fallback 和轻量化默认运行配置
