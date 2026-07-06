# C1 交付：转化层根因诊断（commitment churn）

日期：2026-07-06（召回 regime cross-12，smoothnav-full，10 个失败集全走查）
工具：`scripts/trace_conversion_chain.py`（S1–S8 分环节时间线）

## 结论一句话

失败不在"检测不到"（召回已修复）也不在"planner 不承诺"（每集 2–8 次正确承诺 `object:X`/`unexplored target:X`），而在**承诺不粘**：三种驱逐机制把路径级失败误判为目标级失败，34 次承诺全部被驱逐、零转化。

## 三个负例的分环节断点

### ep358（tv）——完整病理链
- S1/S2 step56：'tv' 进图、候选触发 ✅ → S5 planner 选 `object: tv` ✅
- **断点 1（step66）**：object 目标落地到同一局部点 3 次 → `handle_frontier_reached` 的 **object-stagnation 踢出**（hold_limit=3）→ 转向 direction:south，且 `explored_regions.append("object: tv (stagnant)")` → **菜单永久拉黑 tv**（此后 menu_has_target=False，planner 只能靠场景文本自由发挥重选）
- step102/162：重承诺→再驱逐（churn）
- step304：形成正确的 `unexplored target:tv` 搜索锚（目标进度价值项生效），运行 86 步 ✅
- **断点 2（step390）**：物理卡死 → `handle_stuck_replan` **连目标一起抛弃**（"need alternative route"→ east），锚点 decommit_reason=switched_to:unexplored east → 再未回到 tv → 超时

### ep955（bed）——极端翻转
- step4 'bed' 进图即承诺 `object: bed` → **step5 被驱逐**（1 步存活）→ step24 再承诺 → 再驱逐 → 超时

### ep574（plant）——同型
- step64 承诺 → step98 驱逐 → step101 再承诺 → churn → 超时

## 全失败集量化

| ep | 目标承诺次数 | 策略总切换 | 转化 |
|---|---|---|---|
| 228 | 8 | 21 | 0 |
| 859 | 5 | 13 | 0 |
| 358 | 4 | 12 | 0 |
| 486 | 4 | 8 | 0 |
| 159 | 3 | 6 | 0 |
| 527/64/574/778/955 | 各 2 | 4–5 | 0 |
| **合计** | **34** | — | **0** |

## 根因（三个驱逐器 + 一个毒化器）

1. **object-stagnation 踢出**（controller_logic.handle_frontier_reached）：`same_goal_hold_count≥3` → 放弃目标改选方向——把"这个落地点没进展"当成"这个目标错了"。
2. **stuck replan 目标抛弃**（handle_stuck_replan）：物理卡死 → 换目标而非换路径——目标锚正常工作 86 步也照杀。
3. **explored_regions 毒化**：stagnation/stuck 都把 `object: X (stagnant/stuck)` 记入已探索 → **菜单永久拉黑主目标类目**，后续承诺只能靠 LLM 越过菜单自由发挥（不可靠）。
4.（次要）锚点 stall decommit 与上述叠加。

**设计缺陷本质：路径级失败（这个 viewpoint 走不到）与目标级失败（这个目标是错的）未分离。**

## C2 机制设计（由诊断直接导出）

**Target-commitment persistence（承诺粘性）**——与 graded-authority 原则一致（证据未被推翻则权限保留）：
1. 主目标类目高相关承诺（score≥0.75 类目命中）进入 **committed-target 模式**：stagnation/stuck 触发时**只换接近路径**（换 approach frontier / 换 viewpoint，target-progress 项天然支持），不弃目标；
2. 目标解除仅允许两种证据：(a) 到达并实例判别为非目标（verified-absent），(b) 检测证据被撤销（节点消失/相关性坍塌）；
3. explored_regions 毒化对主目标类目免疫（改记 "approach-N failed"，不拉黑类目）；
4. 保留全局安全阀：同一目标的 approach 尝试次数上限（如 4 个不同 frontier 后才允许降级），防真死锁。

实现面：controller_logic 三处驱逐分支 + planner explored 过滤，一个 `controller_target_commitment_mode` 开关（默认 off 保持现行为）。
