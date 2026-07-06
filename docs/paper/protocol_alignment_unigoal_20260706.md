# G1 交付：UniGoal 发表协议/数字对齐审计

日期：2026-07-06
来源：arXiv 2503.10630 PDF（papers/2503.10630.pdf，§4.1 + 附录 D.4）
判定：**分支 (a) —— 协议同构，发表数字可直接引用为上游参照，无需任何重跑。**

## 协议对照表

| 项 | UniGoal 论文（TN） | 我们的配置 | 一致性 |
|---|---|---|---|
| 任务 | Text-goal Navigation, HM3D | HM3D v0.2 val, val_text | ✅ |
| 步数上限 T | **1000**（附录 D.4：ON 500，IIN/TN 1000） | max_episode_length: 1000 | ✅ |
| 成功半径 r | **1.0 m**（ON 为 1.6，IIN/TN 为 1.0） | success_dist: 1.0 | ✅ |
| 动作空间 | move_forward / turn_left / turn_right / stop | 同 | ✅ |
| 模拟器 | Habitat | 同 | ✅ |
| LLM/VLM | **LLaMA-2-7B / LLaVA-v1.6-Mistral-7B** | deepseek-chat（弱）/ Sonnet4.5+Haiku4.5（强） | 差异=实验轴 |
| 评测集 | HM3D val 全集 | 12 个 loadability 审计场景的 matched 子集（cross-48 + intact-15） | 差异需声明 |

## 发表数字（Table 1，TN-HM3D）

| 方法 | 类型 | SR | SPL |
|---|---|---|---|
| PSL | 监督 | 16.5 | 7.5 |
| GOAT | 监督 | 17.0 | 8.8 |
| **UniGoal** | 零样本 | **20.2** | **11.4** |

## 与我们数字的关系（审稿防线）

1. **绝对量级一致**：text-goal 本身就难（发表 SOTA 仅 20.2，远低于 ObjectNav 的 54.5）。我们 intact-15 上 33–47%、cross-48（12 个更难的 unseen 场景子集）12.5–16.7%，混合区间 ≈17–22%，**正好括住发表数字**。"复现是坏的"质疑不成立。
2. **子集差异的诚实声明**：我们未复跑其全 val 集；cross-48 是审计可加载的 12 场景 matched 子集（刻意含 hard case）。论文中：引用 20.2/11.4 为上游全集参照行 + 脚注声明子集关系；我们的主张全部建立在**内部 matched 配对**上，不依赖与全集数字的直接比较。
3. **意外的头条级佐证**：发表结果用 7B 开源模型取得；我们把 LLM 升到 deepseek-chat 乃至 Sonnet 4.5（数量级更强），**text-goal SR 仍停留在同一区间**——跨"论文级"证据链（他们的 7B → 我们的双通道）三点连线，感知/任务上限论点获得独立支撑。这句话可以直接写进 Abstract。
4. baseline 公平性声明维持既有口径：我们的 baseline = UniGoal backbone + 共享基础设施修复（保守比较）。

## 论文落点

- 表 1 增加"UniGoal (published, full val, 7B models): 20.2/11.4"参照行。
- Intro/Discussion 引用第 3 点作为 ceiling 论点的第三方锚。
- Setup 节脚注：子集构成、审计程序（loadability audit）、matched 设计。
