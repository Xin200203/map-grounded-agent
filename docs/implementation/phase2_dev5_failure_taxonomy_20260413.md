# Phase 2 Dev5 Failure Taxonomy (v1)

状态说明：

- 本文记录的是 184 服务器第一轮 `dev5` 五组对照的**修复前实验基线**。
- 它仍然是当前最重要的 failure baseline，但不代表当前代码已经停留在这一步。
- 阅读当前实现状态时，应与 [phase2_patch_status_20260413.md](/Users/xin/Code/research/SmoothNav/docs/implementation/phase2_patch_status_20260413.md) 一起看。

本文基于 2026-04-13 在 184 服务器上完成的第一轮 `dev5` 五组对照实验，整理当前可确认的 failure taxonomy、关键现象和后续实现优先级。

对应计划文档：

- [smoothnav_implementation_master_plan_20260412.md](/Users/xin/Code/research/SmoothNav/docs/planning/smoothnav_implementation_master_plan_20260412.md)

相关实现说明：

- [controller_profile_and_trace_alignment_notes_20260413.md](/Users/xin/Code/research/SmoothNav/docs/implementation/controller_profile_and_trace_alignment_notes_20260413.md)
- [llm_gateway_protocol_notes_20260413.md](/Users/xin/Code/research/SmoothNav/docs/implementation/llm_gateway_protocol_notes_20260413.md)

## 1. Scope

本轮分析覆盖 5 组 profile：

- `baseline-periodic`
- `smoothnav-full`
- `smoothnav-no-monitor`
- `smoothnav-rules-only`
- `smoothnav-no-prefetch`

每组均运行 `text` goal、`num_eval=5`，结果位于 184 服务器：

- `/home/nebula/xxy/SmoothNav/results_phase2/dev5/baseline-periodic/20260413/smoothnav_text_162522_f81b8910`
- `/home/nebula/xxy/SmoothNav/results_phase2/dev5/smoothnav-full/20260413/smoothnav_text_163652_bc68c6fd`
- `/home/nebula/xxy/SmoothNav/results_phase2/dev5/smoothnav-no-monitor/20260413/smoothnav_text_165351_9b7d91d6`
- `/home/nebula/xxy/SmoothNav/results_phase2/dev5/smoothnav-rules-only/20260413/smoothnav_text_170837_ce33d6c3`
- `/home/nebula/xxy/SmoothNav/results_phase2/dev5/smoothnav-no-prefetch/20260413/smoothnav_text_172325_e8276ada`

本轮使用的 profile 语义已经核对过 manifest：

- `baseline-periodic = monitor off + prefetch off + fixed_interval`
- `smoothnav-full = monitor(llm) + prefetch on + event-driven`
- `smoothnav-no-monitor = monitor off + prefetch on + event-driven`
- `smoothnav-rules-only = monitor(rules) + prefetch on + event-driven`
- `smoothnav-no-prefetch = monitor(llm) + prefetch off + event-driven`

## 2. Data Source And Caveat

本轮核心结论来自 3 类文件：

- `episode_results.json`
- `planner_calls/*.jsonl` 与 `monitor_calls/*.jsonl`
- `step_traces/*.jsonl`

本轮使用的数据来源是可靠的：

- `summary.json` 中直接写出了大写键名 `SR` 与 `SPL`
- `episode_results.json` 提供了逐 episode 的 success / SPL / steps / calls 细节
- `planner_calls/*.jsonl`、`monitor_calls/*.jsonl` 和 `step_traces/*.jsonl` 提供了控制层证据链

需要注意的不是“结果包缺 SR/SPL”，而是：

- 当前 `decision_delay` 和 `goal_update_delay` 仍然只是控制层与主循环变量层的指标
- 它们还不是 executor adoption 或 trajectory divergence 指标

## 3. Headline Result

第一轮 `dev5` 的最核心事实非常明确：

- 当前最好的是 `baseline-periodic`
- 4 个 `smoothnav` 变体都没有超过 `baseline-periodic`
- 4 个 `smoothnav` 变体之间的表现差异非常小，远小于它们在配置上看起来应有的差异

按 `episode_results.json` 回算后的总体指标如下：

| Profile | SR | SPL | Avg High-Level Calls | Avg Low-Level Calls | Strategy Switch Count | Decision Delay | Goal Update Delay | Pending Promotion Rate | Avg Smoothness | Avg Pause Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline-periodic` | 0.80 | 0.3847 | 4.8 | 0.0 | 1.0 | 25.21 | 10.75 | 0.0 | 0.7733 | 19.4 |
| `smoothnav-full` | 0.60 | 0.1140 | 3.8 | 13.0 | 1.2 | 4.13 | 56.98 | 0.0 | 0.7636 | 40.0 |
| `smoothnav-no-monitor` | 0.60 | 0.1140 | 3.4 | 0.0 | 0.8 | 0.63 | 58.17 | 0.0 | 0.7636 | 40.0 |
| `smoothnav-rules-only` | 0.60 | 0.1140 | 3.2 | 13.0 | 0.4 | 0.00 | 58.46 | 0.0 | 0.7636 | 40.0 |
| `smoothnav-no-prefetch` | 0.60 | 0.1140 | 3.6 | 15.6 | 0.4 | 5.27 | 48.33 | 0.0 | 0.7237 | 44.4 |

这组数据说明：

- `smoothnav` 的控制层额外开销已经真实发生
- 但这些额外控制并没有转化成更好的成功率或路径效率
- 目前最像瓶颈的不是“没有调用 planner/monitor”，而是“调用发生了，但没有有效改变最终执行轨迹”

## 4. Episode-Level Pattern

逐 episode 结果也非常重要，因为它暴露了 profile 间的“下游行为塌缩”。

### 4.1 Success Pattern

`baseline-periodic`：

- `ep0`: success, `spl=0.3667`, `total_steps=207`
- `ep1`: success, `spl=0.3307`, `total_steps=224`
- `ep2`: success, `spl=0.5040`, `total_steps=67`
- `ep3`: success, `spl=0.7223`, `total_steps=145`
- `ep4`: fail, `spl=0.0`, `total_steps=38`

4 个 `smoothnav` 变体共有的主模式：

- `ep0`: fail
- `ep1`: success, `spl=0.0773`, `total_steps=688`
- `ep2`: success, `spl=0.1397`, `total_steps=239`
- `ep3`: success, `spl=0.3529`, `total_steps=214`
- `ep4`: fail

更关键的是：

- `smoothnav-full`
- `smoothnav-no-monitor`
- `smoothnav-rules-only`

这 3 组在 `ep0-ep4` 的 success / SPL / total_steps 几乎完全一致，只在高低层调用次数上有区别。

`smoothnav-no-prefetch` 与它们也高度相似，只是 `ep4` 没有早停，而是拖到了 `155` 步后失败。

### 4.2 Action Totals

动作分布进一步强化了这一点：

- `baseline-periodic`: `total_steps=681`
- `smoothnav-full`: `total_steps=1361`
- `smoothnav-no-monitor`: `total_steps=1361`
- `smoothnav-rules-only`: `total_steps=1361`
- `smoothnav-no-prefetch`: `total_steps=1505`

其中：

- `smoothnav-full`
- `smoothnav-no-monitor`
- `smoothnav-rules-only`

三组 `action_analysis.json` 的 `grand_totals` 完全一致：

```json
{"0": 0, "1": 600, "2": 311, "3": 361, "None": 89}
```

这几乎可以直接视为一个强信号：

- profile 的控制差异没有成功传递到最终 primitive-action 序列
- 或者说，控制差异被更底层的 frontier / temp-goal 执行链“抹平”了

## 5. Raw Signal Summary

下面是最有解释力的 raw signal 汇总。

### 5.1 Planner Call Structure

`baseline-periodic`：

- planner 总调用 `24`
- `direction=19`, `room=2`, `object=3`
- 触发原因以 `Fixed interval refresh` 为主 `17`
- fallback `1`

`smoothnav-full`：

- planner 总调用 `19`
- `direction=16`, `room=3`
- 触发原因中 `Agent stuck, need alternative route = 7`
- `PREFETCH` 只真正触发了 `1` 次
- fallback `0`

`smoothnav-no-monitor`：

- planner 总调用 `17`
- `direction=14`, `room=2`, `object=1`
- `Agent stuck, need alternative route = 7`
- fallback `0`

`smoothnav-rules-only`：

- planner 总调用 `16`
- `direction=14`, `room=1`, `object=1`
- fallback `2`

`smoothnav-no-prefetch`：

- planner 总调用 `18`
- `direction=17`, `room=1`
- fallback `5`

规划器的主导模式非常稳定：

- overwhelmingly `direction`
- 很少 `room`
- 几乎不落到 `object`

这说明当前 high-level semantic grounding 还很弱，绝大多数时候系统仍在“语义包装过的方向探索”。

### 5.2 Monitor Call Structure

`smoothnav-full`：

- monitor 总调用 `65`
- `CONTINUE=64`
- `PREFETCH=1`
- `ADJUST=0`
- `ESCALATE=0`
- fallback `0`

`smoothnav-rules-only`：

- monitor 总调用 `65`
- `CONTINUE=65`
- 全部是 rule-based no-op continuation

`smoothnav-no-prefetch`：

- monitor 总调用 `78`
- `CONTINUE=76`
- `PREFETCH=2`
- fallback `31`

这里的信息量非常大：

- monitor 确实被频繁调用了
- 但它几乎只会输出 `CONTINUE`
- 真正有行为意义的 `PREFETCH / ADJUST / ESCALATE` 极少
- 这也是为什么 `full` 和 `no-monitor` 在结果上几乎一样

### 5.3 Step-Level Flags

`baseline-periodic`：

- `goal_updated=35`
- `strategy_switched=5`
- `temp_goal_override=28`
- `stuck_goal_override=11`

`smoothnav-full`：

- `goal_updated=23`
- `strategy_switched=6`
- `temp_goal_override=52`
- `pending_promoted=1`

`smoothnav-no-monitor`：

- `goal_updated=23`
- `strategy_switched=4`
- `temp_goal_override=52`

`smoothnav-rules-only`：

- `goal_updated=23`
- `strategy_switched=2`
- `temp_goal_override=52`

`smoothnav-no-prefetch`：

- `goal_updated=24`
- `strategy_switched=2`
- `temp_goal_override=92`
- `stuck_goal_override=13`

重要解释：

- `temp_goal_override` 在所有 `smoothnav` 变体里都非常高
- `pending_created` 没有在任何组里出现
- `pending_promoted` 只在 `smoothnav-full` 出现了 `1` 次

这意味着：

- executor override 仍然是很强的主导因素
- prefetch/pending 机制在这组样本里基本没有真实进入工作态

## 6. Failure Taxonomy v1

下面给出当前最值得采用的第一版 taxonomy。每一类都附带结论边界，避免把 `dev5` 的小样本当成最终定论。

### F1. Late Goal Update / Semantic Intent Does Not Land In Time

定义：

- semantic planner 的选择已经发生
- 但目标更新落到执行链中的时机太晚
- 或被 executor 的临时目标逻辑压制，导致高层意图没有及时改变轨迹

主要证据：

- `smoothnav-full` 的 `decision_delay=4.13` 已经不高，但 `goal_update_delay=56.98` 很高
- `smoothnav-no-monitor` 和 `smoothnav-rules-only` 的 `goal_update_delay` 进一步升到 `58+`
- `baseline-periodic` 的 `goal_update_delay=10.75`，显著更低
- `smoothnav` 组里 `temp_goal_override` 长期高于 `baseline-periodic`

解释：

- 现在真正的主问题更像是“planner 想法发出去了，但没有很快落到实际导航目标”
- 这比“planner 没想法”更贴近当前数据

置信度：

- 高

### F2. Monitor Churn Without Leverage

定义：

- low-level monitor 被频繁调用
- 但输出主要是 `CONTINUE`
- 对结果没有形成可辨识的收益

主要证据：

- `smoothnav-full` monitor 调用 `65` 次，其中 `64` 次是 `CONTINUE`
- `smoothnav-rules-only` monitor 调用 `65` 次，`65` 次全是 `CONTINUE`
- `smoothnav-full` 与 `smoothnav-no-monitor` 的 `SR/SPL` 完全一致
- `smoothnav-full` 与 `smoothnav-no-monitor` 的 per-episode `total_steps` 也一致

代表性 raw case：

- 在 `smoothnav-full` 的 `episode_000000` 中，`bed` 与 `cabinet` 出现后 monitor 连续给出：
  - “consistent with bedroom, continue”
  - “still enough distance, continue”
- 这些判断在语义上合理，但在控制上几乎没有增量价值

解释：

- 当前 monitor 不是“错得很离谱”
- 而是“判断过于保守，几乎从不触发有后果的动作”

置信度：

- 高

### F3. Dead Prefetch / Pending Path

定义：

- 代码里已经实现了 pending / prefetch 路径
- 但在真实实验里几乎从未进入有效工作态

主要证据：

- 所有 profile 的 `pending_promotion_rate = 0.0`
- 所有 step trace 中几乎没有 `pending_created`
- `smoothnav-full` 只有 `1` 次 `pending_promoted`
- `smoothnav-no-prefetch` 和 `smoothnav-full` 的结果差异极小

解释：

- 这说明 prefetch 现在并不是“做得不好”
- 更准确地说，它还没有真正被跑起来

置信度：

- 高

### F4. Profile Delta Collapse

定义：

- 设计上不同的 ablation profile
- 在最终行为上却塌缩成几乎同一条轨迹

主要证据：

- `smoothnav-full / no-monitor / rules-only` 的 `SR/SPL` 完全一致
- 三组 `action_analysis.grand_totals` 完全一致
- 逐 episode 的 success / spl / total_steps 基本一致
- 区别主要只剩下“内部调用次数不同”

解释：

- 当前 ablation knobs 已经改变了控制层日志
- 但还没有显著改变系统的最终导航行为

这类 failure 很关键，因为它说明：

- 我们现在还没有拿到一个“可公平比较且能拉开差距”的 controller family

置信度：

- 高

### F5. Planner/Monitor Empty-Response Fragility

定义：

- 当 Clauddy 返回空响应时
- 当前控制层会大量退回 fallback
- fallback 进一步把语义控制退化成方向探索或继续前进

主要证据：

- `smoothnav-rules-only` 的 planner fallback `2`
- `smoothnav-no-prefetch` 的 planner fallback `5`
- `smoothnav-no-prefetch` 的 monitor fallback `31`
- 代表性 raw trace 中多次出现：
  - `parsed_result.reason = parse_failure_fallback`
  - `error_message = empty_response`

特别是在 `smoothnav-no-prefetch` 的失败 episode 中：

- planner 在 `episode_000004` 的起始和“新房间发现”时都发生了 empty-response fallback
- monitor 也连续 fallback 成 `CONTINUE`

解释：

- 这不是纯粹的服务器偶发问题
- 因为 fallback 一旦发生，系统就会退回最保守的默认策略，直接削弱 controller 差异

置信度：

- 中高

### F6. Weak Semantic Grounding, Direction-Dominant Planning

定义：

- planner 虽然有 object / room / direction 三种选择
- 但大多数时候仍然选择 direction exploration

主要证据：

- `baseline-periodic`: `direction=19/24`
- `smoothnav-full`: `direction=16/19`
- `smoothnav-no-monitor`: `direction=14/17`
- `smoothnav-rules-only`: `direction=14/16`
- `smoothnav-no-prefetch`: `direction=17/18`

解释：

- 当前系统的 semantic planner 更像“用语言包装 frontier bias selection”
- 而不是“稳定地把 scene graph 里的 room/object 证据变成高价值的具体目标”

这也解释了为什么：

- high-level reasoning 文本看起来合理
- 但最终收益并不明显

置信度：

- 中高

### F7. Override-Dominated Execution

定义：

- 即使 high-level strategy 已经切换
- 实际执行仍高度受 `temp_goal` / `stuck_goal` override 支配

主要证据：

- `smoothnav-full`: `temp_goal_override=52`
- `smoothnav-no-monitor`: `temp_goal_override=52`
- `smoothnav-rules-only`: `temp_goal_override=52`
- `smoothnav-no-prefetch`: `temp_goal_override=92`, `stuck_goal_override=13`

而且：

- 这些 override 高发的 profile，恰好也是最终动作序列几乎完全一致的 profile

解释：

- 当前 controller 还没有掌握 executor
- 更像是在 executor 已有逻辑的外围加了一层建议器

置信度：

- 中高

## 7. What Is Not Yet Safe To Conclude

下面这些结论目前还不能下得太重：

- 不能说 monitor 完全无价值
  - 现在更准确的说法是：它当前实现几乎没有释放出价值
- 不能说 prefetch 设计本身错误
  - 当前证据更像“实现未真正打通”
- 不能说 `smoothnav` 一定整体弱于 periodic baseline
  - 当前只是 `dev5` 小样本，而且外部 API fallback 仍在影响系统稳定性
- 不能说所有 room/object planning 都没意义
  - 当前只是它们出现得太少、太晚、太弱

## 8. Immediate Priority After This Taxonomy

基于当前证据，我建议后续优先级如下。

### P0. Fix Instrumentation Gap

- 让 `summary.json` 直接写出正确的 `sr` / `spl`
- 避免后续每次都从 `episode_results.json` 回算

### P1. Make Prefetch/Pending Actually Fire

- 明确记录 `pending_created_count`
- 检查 `frontier_near` 与 `dist_to_goal < threshold` 的触发链
- 先让 `PREFETCH -> pending_created -> pending_promoted` 在小样本中稳定出现

### P2. Reduce Semantic-To-Goal Landing Delay

- 优先检查 `goal_updated` 与 executor override 的关系
- 重点看 high-level bias 是否被 temp-goal 覆盖太久
- 如果必要，缩短 strategy change 后 temp-goal 的保留窗口

### P3. Make Monitor Produce Consequential Actions

- 当前 monitor 几乎只输出 `CONTINUE`
- 需要人为提升 `ADJUST / ESCALATE / PREFETCH` 的可触发性
- 否则 full/no-monitor 的对照永远拉不开

### P4. Strengthen Room/Object Grounding

- 当前 planner 过于 direction-heavy
- 需要增加 room/object 证据出现后的优先级
- 否则 semantic replanning 仍会退化成“LLM 选方向”

## 9. Current Gate Interpretation

对照 [smoothnav_implementation_master_plan_20260412.md](/Users/xin/Code/research/SmoothNav/docs/planning/smoothnav_implementation_master_plan_20260412.md)，当前最稳妥的判断是：

- Phase 2 的 profile family 已经能跑
- 但还没有通过 Phase 2 想要的行为验证
- 当前第一版 taxonomy 已经足够支持“不要急着扩大到 dev20/dev100，而是先修 controller 生效性”

更直接地说：

- 现在最不缺的是更多 run
- 现在最缺的是让 profile 差异真正穿透到执行层

## 10. Bottom Line

如果把当前问题压缩成一句话：

**SmoothNav 当前不是“完全不会思考”，而是“会思考、会调用、会记录，但这些语义控制还没有稳定而及时地改变底层执行轨迹”。**

因此第一版 failure taxonomy 的主轴不是“planner 不存在”，而是：

- semantic intent landing too late
- monitor mostly non-consequential
- prefetch path effectively dead
- profile deltas collapsing before they reach the executor

这也是下一轮实现最应该围绕的方向。
