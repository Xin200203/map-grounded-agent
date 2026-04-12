# Action Analysis Findings (2026-03-09)

## 实验设置
- Baseline (UniGoal) mode, text goal, 5 episodes
- 在 main.py 中加入 per-step action instrumentation
- 记录每步的 action、位置变化，用相邻步位移判断碰撞

## 核心数据

### Action 分布 (721 总步数)
| Action | Count | % |
|--------|-------|---|
| FORWARD | 272 | 37.7% |
| LEFT | 185 | 25.7% |
| RIGHT | 159 | 22.1% |
| NONE (wait/invalid) | 105 | 14.6% |
| STOP | 0 | 0% |

### Pause 分解 (zero-displacement steps)
| 来源 | Count | 占 pause 比例 |
|------|-------|--------------|
| **Turns (LEFT/RIGHT)** | 344 | **75.3%** |
| NONE/wait | 105 | 23.0% |
| Collision FORWARD (撞墙) | 8 | **1.8%** |

### 连续转弯序列 (≥2 步)
- 数量: 58 次
- 平均长度: 5.0 步 (= 75° 转弯)
- 最大长度: 25 步 (= 375°, 超过一圈)

### 碰撞序列
- 仅 2 次，平均长度 4.0，集中在 Episode 4

### Episode 5 异常
- 39 步中 38 步为 NONE (action=None)
- pause_ratio=1.0, sigma_v=0, SR=0
- 原因: agent 持续处于 wait=True 或 action<0 状态（待排查）

## 关键结论

### 1. "Pause" 指标在离散动作空间下失去意义
Habitat action space = {FORWARD, LEFT_15°, RIGHT_15°, STOP}。
LEFT/RIGHT 只旋转不移动 → speed=0 → 被 metrics 计为 "pause"。
**pause_ratio=0.63 中 75% 来自正常转弯**，不是导航不流畅。

### 2. 碰撞/卡住不是当前瓶颈
5 episode 中仅 8 次碰撞 FORWARD (1.1%)。
collision_map 膨胀、stuck 正反馈循环等问题**在当前样本中不显著**。

### 3. B+C 平滑方法的前提假设错误
- direction_reversals = 0-3 次/ep → heading 抖动几乎不存在
- B+C 试图平滑的 "chattering" 不是真实问题
- 真正的转弯模式是长连续序列 (avg=5步)，是路径几何决定的
- EMA 滞后导致 SR 从 0.8 崩到 0.4（追不上 FMM 的角度跳变）

### 4. 当前 smoothness metrics 的结构性缺陷
| 指标 | 实际测量的内容 | 是否可优化 |
|------|--------------|-----------|
| pause_count | 转弯次数（moving→stopped 转换） | 否（路径几何决定） |
| pause_ratio | 转弯步占比 | 否（离散动作空间固有） |
| σ_v | 速度方差（forward vs turn 交替） | 否（同上） |
| σ_ω | 角速度方差 | 部分（但 baseline 已经很低） |
| direction_reversals | L↔R 振荡 | 是，但 baseline 几乎没有 |
| jerk | 加速度变化 | 部分 |

## 核心问题
> **在离散动作空间下，"平滑导航"到底应该优化什么？**

需要重新定义 contribution 和评价体系。

## 错误经验总结

### ❌ 不要做的事
1. **不要在 FMM local_goal 的角度上加 EMA**：FMM 每步重算最优解，角度跳变是它正确工作的表现。平滑它 = 降低决策质量 → SR 下降。
2. **不要在 turn 阈值上加 hysteresis**：与 EMA 叠加会过度抑制必要转弯 → 撞墙。
3. **不要用 speed=0 来定义 pause**：离散动作空间中 turn 步必然 speed=0，这不是"停顿"。
4. **不要假设 pause_ratio 高 = 导航不流畅**：需要先分解 pause 的来源。

### ✅ 已验证的事实
1. Baseline (UniGoal) 的 FMM 导航在 action 层面工作正常（碰撞率极低）
2. 转弯占总步数 47.7%，是离散动作空间的固有开销
3. Episode 间 SR/SPL 方差大（5 ep 样本不够，需要更多）
4. graph.explore() 每 40 步调用 LLM，baseline 和 smoothnav 对 LLM 的依赖相同
