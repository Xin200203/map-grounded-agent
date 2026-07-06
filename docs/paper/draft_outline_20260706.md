# SmoothNav ICRA 2027 论文骨架 v1（2026-07-06）

> 除强条件（Sonnet 通道）单元格外，全部数字已锁定（来源：73 服务器 suite_summary + 本文档系列分析）。
> 标题候选：**"Lean Event-Driven Semantic Replanning: What Actually Helps in Zero-Shot Object Navigation"**
> 备选："SmoothNav: Evidence-Gated Semantic Replanning with Honest Ablations for Zero-Shot Goal Navigation"

## Abstract（要点序列）

1. LLM 驱动的零样本目标导航系统日益复杂（周期语义推理、monitor、投机预取、多级语义先验），但各机制的真实贡献很少被逐层测量。
2. 我们在冻结的 UniGoal backbone 上构建了一个分层事件驱动重规划控制器（6 层、证据分级语义权限、显式失败码与恢复），并做了迄今最系统的 matched-backbone 消融（7+ 变体 × 48 episode × 12 场景 + 门控绑定事件级归因）。
3. 发现（诚实且反直觉）：(a) 控制器复杂度不改变弱 LLM 下的成功率（n=48 无差异）——成功由语义信号质量决定[强条件结果待入]；(b) 但控制质量大幅改善：出窗退化 14× 减少（归因为构造性避免而非事后修复）、执行器冲突 5× 减少、同 LLM 预算；(c) 事件驱动调度贡献全部效率增益（SPL 2.3×），LLM monitor 零贡献，投机 prefetch 有害（去除后成为唯一全面超越基线的配置：SR 0.167 vs 0.125，SPL +74%）；(d) 语义门控单个可被下游层吸收（纵深防御），联合移除损失效率与锚点利用而非成功率。
4. 结论：精益配置（事件调度+恢复+门控锚定，无 monitor 无 prefetch）+ 可回放归因体系；对"往导航系统里加 LLM 机制"的社区实践给出测量依据。

## 1. Introduction

- 动机：LLM/VLM 语义导航系统的机制堆叠趋势 vs 归因缺失。
- 核心问题：**哪些控制机制真正起作用、在什么条件下起作用？**
- 贡献 bullets：
  1. 分层事件驱动语义重规划控制器（冻结 backbone、证据分级权限、失败码+恢复），全链路可回放；
  2. 迄今最系统的 matched 消融：调度×恢复×monitor×prefetch×四级门控（含全关组合），48 集×12 场景配对 + trace 级绑定归因；
  3. 反直觉发现三连：monitor 无用、prefetch 有害、复杂度不改成功率[条件化于语义质量——待强条件确认]；
  4. 精益推荐配置 + 开源诊断工具链（capsule/replay/14 层审计）。

## 2. Related Work

- 零样本目标导航：UniGoal（backbone，引用其数字为上游参照）、L3MVN、VLFM、ESC、CoW、SemExp/PONI（价值场传统）。
- LLM 规划+导航的系统类工作；消融文化欠缺的批评（引 1-2 篇 benchmark 反思文）。
- 定位：不是新 SOTA 方法，是**机制测量 + 精益系统**论文。

## 3. System（方法）

### 3.1 分层控制器（图 1：架构图）
Layer 0 WorldState → L1 Mission → L2 语义 planner（菜单约束choice）→ L3 TacticalArbiter（事件→模式）→ L4 GeometricGrounder（两段式，失败码）→ L5 ExecutorAdapter（epoch 化 stale 清理）。

### 3.2 证据分级语义权限（图 2：门控层级示意）
- 菜单相关性门（≥0.75）→ 直接执行双重门（相关性+局部可投影）→ 非本地降级为搜索锚点（PONI-lite 目标进度价值项）→ 锚点停滞退委。
- 关键设计句：语义假设**偏置**地图价值，永不**绕过**。

### 3.3 事件驱动重规划与恢复
- 事件类型表；出窗修复路径；grounding noop 阈值重规划；stuck 抑制。

### 3.4 可回放归因体系（简短，工具贡献）
- step trace / planner calls / grounding snapshot / task-frame capsule / 14 层合约审计。

## 4. Experiments

### 4.0 Setup
- HM3D val text-goal，12 个 loadability 审计场景 × 4 集 = cross-48（matched 配对）；intact-15 单场景对照。
- 弱通道 deepseek-chat（声明 text-only 限制、全 profile 公平）；强通道 Sonnet4.5+Haiku4.5[待入]。
- 方法论声明：n≤15 的 SR 差异不可分辨（回溯 April 教训）；默认温度的混沌敏感性；因此以 48 集配对 + 机制指标为主。
- baseline 声明：baseline = UniGoal backbone + 共享基础设施修复（比上游更强，保守比较）。

### 4.1 主表（表 1：cross-48 + intact-15 × 5 profile × 2 通道[强通道待入]）
弱通道行（已锁定）：explore 0.125/0.036、periodic 0.125/0.035、full 0.104/0.042、**no-prefetch 0.167/0.061**；intact：0.333/0.089、0.467/0.128、0.400/0.122、0.400/0.121。

### 4.2 机制分解（表 2：E2 阶梯，cross-12）
periodic→fixed-interval→no-monitor→full 的 oow 43→7→3、SPL 0.042→0.037→0.085→0.087、调用数；结论三条（恢复机制/事件调度/monitor）。

### 4.3 出窗归因（图 3：修复 vs 避免柱状 + 43/20/1/19 数字）
"构造性避免而非事后修复"——periodic 修复 20 次仅 1 成功、16 次永久搁置；full 全程仅 3 次出窗。

### 4.4 门控消融（表 3：E3 五变体 + 绑定计数列）
单门×4 + 全关组合；绑定计数（e3a：73 低相关放行、44 被下层拦截、调用+69%）；e3e：SPL −36%、oow 2.3×、失守 661。结论：纵深防御 + 联合效率贡献。

### 4.5 prefetch 之害（小节或并入 4.2）
胜集并集分析（661/717/859/358）；机制：pending 晋升挤占；调用减半。

### 4.6 定性分析（图 4：ep228 capsule BEV 时间线——painting→tv 锚定切换 + 出窗修复帧）

## 5. Discussion & Limitations

- SR 受感知/语义上限约束（诚实 null）[或：条件化定理——强通道确认后改写]；
- 单一 backbone、单基准、模拟环境；温度/随机性方法论；
- 门控的价值形态取决于 planner 行为质量。

## 6. Conclusion

## 图表清单

- 图1 架构；图2 门控层级；图3 出窗归因；图4 capsule 定性时间线（现成渲染器）。
- 表1 主表；表2 机制阶梯；表3 门控消融+绑定。
- 附录：14 层审计定义、profile 配置表、逐集结果。

## 待强条件结果后的两个改写点

1. Abstract/Intro 的发现 (a)：null → 条件化定理（若分离出现）。
2. 表 1 增加强通道行；4.0 增加双通道声明。

## 写作分工与时间（修订）

- 07/07–07/09：方法节 + 实验节初稿（数字全现成）；图 1/2 绘制、图 3/4 由渲染器出。
- 07/10：强条件定稿并入 → Abstract/Intro 定稿。
- 07/11+：LaTeX（IEEEtran）+ 内审循环。
