# 关键 episode 定性证据档案（供 §4.7 + 图 4）

数据来源：73 服务器逐集聚合（e1 弱通道 / strong Sonnet / recall 修复三 regime，均 cross-12 matched）。

## 关键 episode 卡片

### ep661（chair，scene p53SfW6mjZe）——锚定 showcase，全档案最强正例
- 弱通道：full/no-prefetch **SUCCESS SPL 0.853**，两 baseline 全 FAIL。三个通道（4 月 Clauddy / DeepSeek / Sonnet）精确复现 0.853。
- 强通道：periodic 也解出但 SPL 仅 0.517 → full 0.853 是 **+65% 路径效率**（图 3 的配对点之一）。
- 修复 regime：full/no-prefetch 保持 0.853，**periodic 反而丢失** → 召回噪声上升时无恢复机制的 baseline 更脆弱。
- 论文用途：图 4 主时间线——target-anchor 状态机对可达目标的锁定与高效逼近；跨 regime/通道稳定性佐证控制器行为的确定性。

### ep159（sofa，scene Dd4bFSTQ8gi）——召回修复解锁的新增成功
- 弱通道 + 强通道：**所有 profile × 所有通道 = FAIL**（战役全程从未解出）。
- 修复 regime：no-prefetch **首次 SUCCESS SPL 0.110**（full 仍 fail，periodic 仍 fail）。
- 论文用途：证明召回修复**解锁了此前不可解的 episode**（非胜集重排）；解锁者是事件驱动控制器，periodic 在同 regime 未转化 → "控制器价值随世界模型质量增长"的 existence proof。

### ep717（plant，scene q5QZSEeHe5g）——三 regime 全解、控制器持续更优
- 弱通道：no-prefetch 0.227 vs full 0.188 vs periodic 0.321（弱通道 baseline 反而路径短，但见下）。
- 强通道：全解。
- 修复 regime：no-prefetch **0.545** > full 0.400 > periodic 0.424 → 修复后 no-prefetch 路径效率最高。
- 论文用途：SPL 随 regime 健康度对控制器越发有利的趋势示例。

### ep228（tv_monitor，scene LT9Jq6dN3Ea）+ ep527（chair，eF36g7L6Z9M）——顽固 hard case
- 全部 4 通道/regime × 全部 profile = FAIL。
- ep228 弱通道 capsule 分析（E0b）：painting→tv 锚定切换序列、planner 0% 空响应、出窗修复帧——**过程正确但任务不可达**。
- 论文用途：图 4 副面板——展示控制器"做对了每一步却仍失败"的感知/可达上限案例（Finding 2 的定性锚）。

## 图 4 规格（渲染器现成：scripts/replay_semantic_bev_capsules.py）

- 面板 A（ep661 修复 regime，full）：4 帧 BEV 时间线——初始探索 → target-anchor 锁定 → 局部窗口重投影 → 到达。标注 anchor 状态机与出窗修复。
- 面板 B（ep228 弱通道，full，E0b capsule step 668/679/699/727）：painting→tv 锚定切换 + dense semantic BEV（true 218px / footprint 995px）——过程正确、任务不可达。
- 面板 C（对比条）：ep159 弱 regime（空图，fail）vs 修复 regime（16 caption，no-prefetch success）——图召回修复的可视化。
- 三面板共用图例：obstacle/free/explored/frontier + object footprint + target heatmap + anchor 标记。

## 表 5 候选（逐集矩阵，附录）

12 集 × {弱/强/修复} × {periodic/full/no-prefetch} 的 SR/SPL 全矩阵——供复现与审稿人核验，正文只放聚合。
