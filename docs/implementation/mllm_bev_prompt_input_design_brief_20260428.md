# SmoothNav MLLM 输入构造设计简报：BEV、Prompt、当前卡点与成熟工作对照

Date: 2026-04-28

Audience: 用于和高级科学家讨论 SmoothNav 当前被卡住的核心问题：**MLLM planner 到底应该看到什么 BEV、读到什么 prompt、输出什么决策，以及我们当前实现为什么仍然不够强。**

Scope:

- 只整理当前实现、实验观察、问题边界、成熟工作启发。
- 不在本文中提出大规模实现细节，不包含任何 API 密钥或私密运行配置。
- 结果表述遵循项目约定：实验/结果类表述标注为 `Observation` / `Hypothesis` / `Claim`。

---

## 0. 一句话摘要

**Observation:** 当前 SmoothNav 已经把 MLLM 接入到 frontier branch 选择边界，并能保存 replay capsule / semantic BEV / branch decision evaluator；但当前 MLLM 输入仍然偏弱：早期 BEV 主要是几何 occupancy + frontier overlay，dense semantic channel 为空，target heatmap 为空，room 多数只是未定位文本假设，导致 MLLM 被迫在低信息、弱空间语义、部分坐标不一致的输入上做自由推理。

**Hypothesis:** 下一步真正要讨论的不是“prompt 再写得更长一点”，而是要把 MLLM 输入从 **debug BEV + 文本补丁** 升级为 **可审计的 map-conditioned decision state**：至少包含 dense semantic map、target-conditioned value / potential map、frontier candidate/value map、room/function-area spatial hypotheses、trajectory/dead-end evidence，以及严格的 branch 输出契约。

---

## 1. 当前 SmoothNav 的整体管线

### 1.1 地图构建：SemExp-like BEV_Map

当前地图来自 `base_UniGoal/src/map/bev_mapping.py`。

**Evidence:** `BEV_Map` 初始化 full/local map：

- `full_map`: `(num_scenes, num_sem_categories + 4, global_width, global_height)`
- `local_map`: `(num_scenes, num_sem_categories + 4, local_width, local_height)`
- 通道约定：
  - ch0: obstacle
  - ch1: explored/free
  - ch2: current agent
  - ch3: past/visited agent
  - ch4+: semantic categories

关键代码位置：

- `base_UniGoal/src/map/bev_mapping.py:185-205`
- `base_UniGoal/src/map/bev_mapping.py:234-245`
- `base_UniGoal/src/map/bev_mapping.py:246-265`

**Evidence:** Mapping module 从 RGBD observation 中使用 depth 建几何，同时把 `obs[:, 4:]` 投影到 semantic channels：

- `base_UniGoal/src/map/bev_mapping.py:84-86`：`obs[:, 4:, :, :]` 被池化写入 feature。
- `base_UniGoal/src/map/bev_mapping.py:137-141`：`agent_view[:, 4:]` 写 semantic projection。
- `base_UniGoal/src/map/bev_mapping.py:177-181`：与上一帧 map 做 max fusion。

**Hypothesis:** 当前 step31 / step43 `sem=0/16` 的根因很可能不在 renderer，而在上游 observation semantic logits / segmentation projection / threshold / local-to-full map persistence 之间。renderer 只能显示已有通道，不能凭空生成 semantic mask。

---

### 1.2 高层 planner：文本 scene graph → Strategy

当前 `smoothnav/planner.py` 是高层语义策略 planner。它输出的是 `Strategy`，而不是具体 frontier。

**Evidence:** `Strategy` 数据结构：

```python
@dataclass
class Strategy:
    target_region: str
    bias_position: Optional[Tuple[int, int]]
    reasoning: str
    explored_regions: List[str]
    anchor_object: str
```

代码位置：`smoothnav/planner.py:35-43`。

当前高层 prompt 主要包含：

- target description；
- scene graph text；
- already searched；
- reason for replanning；
- choices menu；
- 输出 JSON：`choice_type` + `choice_id` + `reasoning`。

代码位置：`smoothnav/planner.py:45-67`。

**Evidence:** choices menu 的设计原则是“LLM 只能选择可 resolve 的对象/房间/方向”：

- object: 必须有已知 center，且目标相关性过阈值；
- room: 必须有 member objects 可以算 centroid；
- direction: north/south/east/west 总是可 resolve。

代码位置：`smoothnav/planner.py:106-180`。

**Observation:** 这个高层 planner 目前基本是文本 planner。它还没有直接看 BEV 图，因此它只能通过 scene graph 和 choices menu 形成 `target_region` / `anchor_object` / `bias_position`。

---

### 1.3 Strategy grounding：Strategy → local/global goal 或 frontier prior

`strategy_grounding.py` 把高层 Strategy 落到几何地图。

**Evidence:** grounding 开始时把当前 BEV map 写入 graph：

- `graph.set_full_map(bev_map.full_map)`
- `graph.set_full_pose(bev_map.full_pose)`
- `graph.local_map_boundary = bev_map.local_map_boundary`
- 同时写入 `active_target_region` / `active_anchor_object`，供 lower-level frontier scorer 读取。

代码位置：`smoothnav/strategy_grounding.py:442-453`。

当前 grounding 分三类：

1. **direct object goal**：如果 target object 足够相关且可直接用，就用 object center。
2. **object as semantic evidence/bias**：如果对象相关但不能直接执行，就作为 bias。
3. **frontier selection**：否则调用 `Graph.get_goal`，可选接入 MLLM frontier prior。

代码位置：`smoothnav/strategy_grounding.py:455-494`。

**Observation:** 当前 MLLM frontier planner 不是替代 controller，而是生成一个 branch prior；最终还要被 `Graph.get_goal` / frontier scoring 消化。

---

### 1.4 Frontier scoring 与 replay snapshot

`frontier_scoring.py` 保存了 frontier-value replay snapshot，用来离线复盘某一步为什么选某个 frontier。

**Evidence:** snapshot 包含：

- `frontier_locations_16`
- `distances_16`
- `base_scores`
- `bias_scores`
- `novelty_scores`
- `actionability_scores`
- `repeat_penalties`
- `recent_penalties`
- `target_progress_scores`
- `planner_prior_scores`
- `candidate_indices`
- `final_scores`
- `frontier_branches`
- `weights`
- selected frontier/debug breakdown

代码位置：`smoothnav/frontier_scoring.py:560-638`。

**Evidence:** replay 时会重新 compose final score，并检查：

- candidate 是否存在；
- target branch 是否 active；
- target progress 是否 nonzero；
- semantic term 是否 nonzero；
- replay selected frontier 是否和 snapshot 一致。

代码位置：`smoothnav/frontier_scoring.py:641-780`。

**Observation:** 这套 replay 体系已经能定位“目标语义信号是否在 frontier value 中消失”。但它还没有解决 MLLM 输入本身弱的问题。

---

### 1.5 Frontier branch decomposition

`frontier_branching.py` 把具体 frontier points 聚合成 branch ID，让 MLLM 选 branch 而不是选 raw coordinate。

**Evidence:** branch payload 包含：

- `id`: B1/B2/...
- `pixel_count`
- `candidate_point_count`
- `local_indices`
- `candidate_local_indices`
- `centroid`
- `representative_coord`
- `candidate_points`
- `direction_from_agent`
- `score_terms_aggregate`

代码位置：`smoothnav/frontier_branching.py:124-281`。

**Evidence:** MLLM 输出 branch rank 后，会被转换成 per-frontier `planner_prior_scores`。

代码位置：`smoothnav/frontier_branching.py:288-327`。

**Observation:** branch ID 是一个很好的输出边界：它比 raw coordinate 稳定，也能让最终 scorer 保持行动可执行。但当前 branch 本身缺少足够强的 semantic/value 描述。

---

## 2. 当前 MLLM frontier planner 的输入/输出契约

### 2.1 当前 planner 的位置

`BEVFrontierMLLMPlanner` 位于 `smoothnav/bev_frontier_planner.py`。

**Evidence:** 文件头部明确约定：SmoothNav 拥有 frontier extraction、branch IDs、scoring、executable coordinates；MLLM 只允许从 annotated BEV view 中选择/rank branch IDs。

代码位置：`smoothnav/bev_frontier_planner.py:1-7`。

---

### 2.2 当前 MLLM 输入：图像部分

当前 image 由 `smoothnav/semantic_bev.py` 生成。

图像包含：

- occupancy / known / free / obstacle；
- ch2 current agent；
- ch3 visited trajectory；
- ch4+ semantic regions，如果存在；
- target/query heatmap，如果存在；
- O# object labels；
- R# room hypotheses；
- B# frontier branch labels；
- side panel: semantic count、target heat source、warnings、objects、rooms、branches。

代码位置：

- map channel decode: `smoothnav/semantic_bev.py:218-233`
- semantic channel stats: `smoothnav/semantic_bev.py:314-359`
- semantic regions: `smoothnav/semantic_bev.py:389-456`
- target heatmap: `smoothnav/semantic_bev.py:489-598`
- frame assembly: `smoothnav/semantic_bev.py:1110-1328`
- prompt summary: `smoothnav/semantic_bev.py:1331-1411`
- rendering: `smoothnav/semantic_bev.py:1486-1842`

当前图像 legend：

- `A=robot`
- `B#=frontier branch`
- `O#=object`
- `R#=room hypothesis`
- `semantic colors=map ch4+`
- `orange=target/query heat`
- `cyan=current ch2`
- `yellow=visited ch3`

**Observation:** 这张图目前是“semantic-capable renderer”，不是保证语义丰富的 semantic BEV。它有能力画 semantic channel，但如果 `full_map[4:]` 为空，图像只能退化成几何图。

---

### 2.3 当前 MLLM 输入：文本 prompt 部分

`build_bev_frontier_prompt(...)` 当前包含：

1. 角色说明：SmoothNav 的 BEV frontier-branch planner。
2. 图例说明：A/B/O/S/R 的含义。
3. target description。
4. current text strategy：
   - `target_region`
   - `anchor_object`
   - `trigger`
   - `step_idx`
   - `already_explored`
5. scene graph text。
6. semantic BEV summary JSON。
7. allowed branch IDs。
8. compact branch summary。
9. 任务规则。
10. 输出 JSON schema。

代码位置：`smoothnav/bev_frontier_planner.py:98-170`。

关键 prompt 约束包括：

- 如果 target / strong anchor 可见，选择接近或验证它的 branch。
- 如果 `direct_target_pixel_count=0` 或 heat source 是 `semantic_prior_only`，不要声称 target visible。
- 如果 target 不可见，从目标语义、S# semantic regions、target heat branch ranking、room commonsense、scene graph、BEV topology 推断探索 branch。
- 输出 branch ID，不输出 raw coordinate。

**Observation:** prompt 的语言约束已经比早期更安全，但它仍然依赖图像/summary 中有足够语义信息。若 semantic summary 是空的，prompt 约束不能弥补信息缺失。

---

### 2.4 当前 MLLM 输出

当前输出 schema：

```json
{
  "schema_version": "...",
  "decision_type": "explore_frontier" | "target_anchor" | "hold_or_recover",
  "target_visible": true | false,
  "selected_branch_id": "B#",
  "ranked_branches": [
    {"id": "B#", "score": 0.0, "rationale": "short reason"}
  ],
  "semantic_search_intent": "short sentence"
}
```

代码位置：`smoothnav/bev_frontier_planner.py:159-169`。

**Evidence:** plan 结果会通过 `validate_mllm_frontier_plan(...)` 校验，并转成 planner prior。

代码位置：

- `smoothnav/bev_frontier_planner.py:275-300`
- `smoothnav/strategy_grounding.py:280-354`

**Observation:** 当前输出是 branch-level decision。这个方向是合理的；问题主要是输入层的 semantic/value 信息不足，以及 final scorer 对 MLLM branch prior 的承接仍可能和 heat/value ranking 不一致。

---

## 3. 最新 replay 暴露出的输入质量问题

主要 artifact：

- `analysis/latest_semantic_bev_manual_check_20260428/summary.json`
- `analysis/latest_semantic_bev_manual_check_20260428/step_000031/semantic_annotated_bev_planner.png`
- `analysis/latest_semantic_bev_manual_check_20260428/step_000031/semantic_bev_frame.json`
- `analysis/s24_semantic_bev_repair_review_20260427/branch_decision_eval/summary.json`

### 3.1 Step-level observation summary

| Step | semantic cats | semantic regions | target heat | direct TV pixels | final branch | MLLM branch | 关键观察 |
|---:|---:|---:|---|---:|---|---|---|
| 31 | 0/16 | 0 | none:0 | 0 | B2 | invalid/empty | 几何 + graph only；B1/B2/B3 H 全 0 |
| 43 | 0/16 | 0 | none:0 | 0 | B1 | B1 | 仍无 semantic channel；选择主要靠图形/文本推断 |
| 98 | 1/16 | 1 | none:0 | 0 | B3 | B3 | 只有 refrigerator；对 TV 没有 target heat |
| 145 | 4/16 | 10 | semantic_prior_only:289 | 0 | B4 | B4 | 出现 couch/chair/dining-table prior，但 final/MLLM 选 B4，heat top 是 B5 |

**Observation:** step31/43 是当前 BEV 输入最典型的问题：`sem=0/16` 且 `target heat=none pix=0`，所以 MLLM 只能依据几何 topology、branch label、少量 object labels 和 room commonsense。

**Observation:** step145 开始出现 semantic prior，但仍没有 direct target TV channel；其 heatmap 是 co-occurrence prior，不是目标可见证据。

**Observation:** branch decision evaluator 在 step145 报告 `final_branch_not_top_heatmap_branch` 和 `mllm_branch_not_top_heatmap_branch`，说明即使有 target prior，当前 final decision 与 value ranking 的耦合仍不稳。

---

## 4. 当前卡点的具体拆解

### 卡点 A：BEV 不是成熟 semantic BEV，而是 debug overlay

**Observation:** 当前 renderer 可以画 semantic map，但当前 episode 的早期 semantic map 本身为空。

具体表现：

- step31: `active_semantic_category_count=0`
- step31: `has_language_query_heatmap=false`
- step31: localized objects 只有 windows/mirror，且 target relevance 为 0
- step31: branches 的 `target_heatmap_score` 都是 0

**Hypothesis:** 如果 MLLM 输入图像没有 dense semantic channel / target value / room mask，那么即使 prompt 写得很清楚，模型也只能做“看图猜探索方向”，无法稳定输出高质量 branch decision。

---

### 卡点 B：room 信息目前多数是未定位文本，不是空间 BEV 结构

**Observation:** 当前 semantic BEV frame 中 room hypotheses 有 living room / kitchen / bathroom 等，但多数 `center_rc=null`，`geometry_type=unlocalized_hypothesis`。

影响：

- prompt 里出现“living room”会诱导 MLLM 用 commonsense；
- 但图上没有 living room 的 mask/区域；
- 因此 MLLM 无法把“TV 可能在 living room”落到某个 branch。

**Hypothesis:** room 文本如果没有空间定位，应该作为弱全局 prior，而不应该和 localized room/semantic mask 混在同一决策层使用。

---

### 卡点 C：target heatmap 不是完整 value map

当前 target heatmap 来源只有三类：

1. direct target semantic channel；
2. semantic co-occurrence prior channel；
3. graph object prior/direct target object。

代码位置：`smoothnav/semantic_bev.py:489-598`。

**Observation:** step31/43/98 的 target heatmap 为 none，step145 为 `semantic_prior_only`。

**Hypothesis:** 对于 TV 不可见的探索场景，成熟系统需要的是 **target-conditioned search value/potential map**，而不是只在看到 couch/chair/table 后才局部变亮的 heatmap。

---

### 卡点 D：坐标系还有残余不一致

**Observation:** 当前 frame 已经用 map-current channel 修补机器人显示坐标，但 replay 中仍存在 raw coord mismatch：

- step31: raw agent 与 map-current 差 25 px；
- step98: 差 113 px；
- step145: 差 93 px。

当前 warning：

- `agent_display_coord_repaired_from_map_current_channel`
- `raw_agent_coord_disagrees_with_map_current_channel:...`

代码位置：`smoothnav/semantic_bev.py:1160-1175`。

影响：

- 图上 agent 已经修正；
- 但 branch summary 中可能还混有不同 frame 下的 direction 字段；
- MLLM 可能同时读到 “direction_from_agent=north” 和图上 “map_direction_from_agent=west”。

**Hypothesis:** 对 MLLM，应只暴露一个 canonical branch direction；旧 frame 字段应降级为 debug，不应出现在 planner-critical prompt 中。

---

### 卡点 E：MLLM 输出与 final scorer 承接不够紧

**Observation:** 当前 MLLM 输出 branch prior，final scorer 再做一次 frontier scoring。这样保留了可执行性，但也可能出现：

- MLLM 选择 B4；
- heat/value top 是 B5；
- final 仍选择 B4；
- evaluator 报告 branch-value disagreement。

**Hypothesis:** 如果 MLLM 的输入中包含 target heat ranking，就应明确后续 scorer 如何使用它：

- 是 soft prior？
- 是 hard gate？
- 是 tie-breaker？
- 在 direct target absent 时是否只允许 exploration prior？

当前 contract 还不够强。

---

### 卡点 F：缺少分面 BEV，而不是单图塞所有信息

当前 planner image 把 occupancy、semantic region、object label、room label、branch label、warnings 都塞在同一张图里。

**Hypothesis:** 对 MLLM，更好的输入可能不是单张密集图，而是多 panel / 多层 summary：

1. occupancy + explored + frontier panel；
2. dense semantic class panel；
3. target value / potential panel；
4. trajectory/dead-end/revisit panel；
5. object/room structured JSON table；
6. branch candidate table。

这样 MLLM 才能区分“我看到的几何”“我看到的语义”“我对目标的猜测”“我对 loop/dead-end 的判断”。

---

## 5. 成熟工作对照：它们如何构造地图/输入/决策

### 5.1 SemExp / Object-Goal-Navigation

Public links:

- Paper: https://arxiv.org/abs/2007.00643
- GitHub: https://github.com/devendrachaplot/Object-Goal-Navigation
- Local checkout: `analysis/external_repos/semexp`
- Local reference image: `analysis/SemExp`

核心做法：

1. semantic mapping module 累积语义地图；
2. goal-oriented semantic policy 从 semantic map 选 long-term goal；
3. deterministic local policy / FMM planner 执行到 goal map。

Local evidence:

- `analysis/external_repos/semexp/README.md:15-17`
- `analysis/external_repos/semexp/main.py:317-329`
- `analysis/external_repos/semexp/main.py:495-509`
- `analysis/external_repos/semexp/agents/sem_exp.py:347-381`

对 SmoothNav 的启发：

- MLLM 不应凭文本猜 raw search direction；它应该看到一个稳定 semantic map / goal map。
- 如果 target channel 出现，goal map 直接切到 target semantic mask。
- 如果 target 不可见，semantic policy 仍应输出 long-term goal，而不是让 target signal 在 frontier scorer 中变成 0。

---

### 5.2 PONI

Public links:

- Project: https://vision.cs.utexas.edu/projects/poni/
- GitHub: https://github.com/srama2512/PONI
- Paper: https://arxiv.org/abs/2201.10029
- Local checkout: `analysis/external_repos/poni`
- Local reference image: `analysis/PONI`

核心做法：

1. 输入 partial semantic map；
2. 输出 target-conditioned potential function；
3. 在 potential map 上决定 “where to look”；
4. frontier 是 potential 的采样/约束对象，而不是纯几何对象。

Local evidence:

- `analysis/external_repos/poni/README.md:14`
- `analysis/external_repos/poni/poni/dataset.py:146-150`
- `analysis/external_repos/poni/poni/dataset.py:219-222`
- `analysis/external_repos/poni/semexp/model_pf.py:222-242`

对 SmoothNav 的启发：

- 对 TV 不可见场景，应构造 `target_potential_map`，不只是 `target_heatmap`。
- potential 可以先做非学习版：基于 visible object anchors、room priors、frontier novelty、dead-end penalty、distance-to-frontier、unexplored gain。
- MLLM 输入可以要求模型判断/修正 potential 的语义合理性，而不是从零选择方向。

---

### 5.3 VLFM

Public link:

- GitHub: https://github.com/bdaiinstitute/vlfm
- Local checkout: `analysis/external_repos/vlfm`

核心做法：

1. 用 VLM/ITM 对当前/历史视野计算 target-related value；
2. 融合成 map-level `ValueMap`；
3. 根据 value map 对 frontiers 排序；
4. 使用 acyclic enforcer 抑制循环；
5. policy info 可视化 value map + frontier + selected goal。

Local evidence:

- `analysis/external_repos/vlfm/vlfm/mapping/value_map.py:33-36`
- `analysis/external_repos/vlfm/vlfm/mapping/value_map.py:160-187`
- `analysis/external_repos/vlfm/vlfm/mapping/value_map.py:360-430`
- `analysis/external_repos/vlfm/vlfm/policy/itm_policy.py:39-54`
- `analysis/external_repos/vlfm/vlfm/policy/itm_policy.py:64-153`
- `analysis/external_repos/vlfm/vlfm/policy/itm_policy.py:191-210`

对 SmoothNav 的启发：

- 当前 SmoothNav 的 `target_heatmap` 更像一个静态 overlay；VLFM 的 value map 是随观察持续融合的。
- MLLM 输入应包含“frontier values sorted by target value”，而不只是 branch H/A 简短字段。
- 对 loop/dead-end，应有显式 acyclic / revisit evidence panel，而不只依赖 prompt 说“avoid repeating”。

---

### 5.4 VLMaps

Public links:

- Project: https://vlmaps.github.io/
- GitHub: https://github.com/vlmaps/vlmaps
- Local checkout: `analysis/external_repos/vlmaps`

核心做法：

1. 将视觉语言特征融合到 3D/2D map；
2. language query 直接 index map；
3. `get_pos(name)` 返回 contour/center/bbox；
4. `get_nearest_pos(curr_pos, name)` 将语言目标转成可导航 map position。

Local evidence:

- `analysis/external_repos/vlmaps/vlmaps/map/vlmap.py:134-167`
- `analysis/external_repos/vlmaps/vlmaps/map/vlmap.py:247-288`
- `analysis/external_repos/vlmaps/vlmaps/map/map.py:143-190`

对 SmoothNav 的启发：

- 如果要让 planner 用“TV / living room / bedroom”等语言概念，最好让这些概念先变成 map masks / contours / bbox。
- 语言 grounding 应返回空间对象，而不是只写入 prompt 文本。

---

### 5.5 CoW / CLIP on Wheels

Public links:

- GitHub: https://github.com/real-stanford/cow
- Paper: https://arxiv.org/abs/2203.10421
- Local checkout: `analysis/external_repos/cow`

核心做法：

1. 用 CLIP/localizer 从 observation 中找 target ROI；
2. 如果 ROI 存在，切换到 exploit；
3. 否则 frontier explore；
4. ROI target 与 frontier fallback 都保存在 map/graph 中。

Local evidence:

- `analysis/external_repos/cow/src/models/agent_fbe.py:71-105`
- `analysis/external_repos/cow/src/models/agent_fbe.py:112-151`
- `analysis/external_repos/cow/src/models/exploration/frontier_based_exploration.py:140-141`
- `analysis/external_repos/cow/src/models/exploration/frontier_based_exploration.py:366-430`
- `analysis/external_repos/cow/src/models/exploration/frontier_based_exploration.py:720-760`

对 SmoothNav 的启发：

- target ROI / target branch 一旦出现，controller 应进入更明确的 target-following 或 target-verification mode。
- 如果 direct target path 不可执行，frontier 应围绕 ROI/target value 排序，而不是回到普通 novelty exploration。

---

### 5.6 NavGPT / LLM VLN prompting

Public link:

- GitHub: https://github.com/GengzeZhou/NavGPT
- Local checkout: `analysis/external_repos/NavGPT`

核心做法：

- prompt 把 history、observation、navigable viewpoints 结构化；
- LLM 输出 viewpoint ID，而不是坐标。

Local evidence:

- `analysis/external_repos/NavGPT/nav_src/prompt/planner_prompt.py`

对 SmoothNav 的启发：

- 当前“输出 branch ID”方向与 NavGPT 类似，是合理边界。
- 但 NavGPT 的前提是 candidate viewpoints 有可读 observation 描述；当前 SmoothNav branch candidates 缺少足够语义/价值描述。

---

## 6. 对当前 MLLM 输入构造的建议讨论点

以下不是最终实现计划，而是给讨论用的问题拆解。

### 6.1 BEV 应从单图升级为多层决策状态

建议讨论的最小 BEV 输入：

1. **Geometry panel**
   - obstacle/free/explored/unknown；
   - frontier branches；
   - agent pose；
   - recent trajectory；
   - dead-end/revisit marks。

2. **Semantic class panel**
   - dense semantic category map；
   - object category legend；
   - direct target mask if present；
   - confidence/unknown class。

3. **Target value panel**
   - direct target heat if target visible；
   - prior/potential heat if target not visible；
   - heat source label: direct / learned / heuristic / prior-only；
   - branch value ranking。

4. **Room/function-area panel**
   - only if spatially localized；
   - room hypothesis confidence；
   - unlocalized room should remain text-only and low confidence。

5. **Branch table**
   - branch ID；
   - candidate count/actionability；
   - target value score；
   - semantic support objects/regions；
   - expected information gain；
   - revisit/dead-end penalty；
   - previous attempt status。

---

### 6.2 Prompt 应明确区分四类 evidence

当前 prompt 应进一步强制 MLLM 分辨：

1. direct target evidence：target mask / target object localized；
2. target anchor evidence：couch/chair/bed/table 等；
3. room commonsense：living room / bedroom / kitchen 等；
4. pure topology exploration：only geometry/no semantic support。

建议 output schema 加强：

```json
{
  "selected_branch_id": "B#",
  "decision_type": "direct_target" | "anchor_follow" | "semantic_explore" | "geometry_explore" | "recover",
  "target_visible": false,
  "evidence_level": "direct" | "anchor" | "room_prior" | "geometry_only",
  "used_evidence_ids": ["S003", "O002", "R001"],
  "ranked_branches": [
    {"id": "B#", "score": 0.0, "evidence_level": "...", "rationale": "..."}
  ],
  "risk_flags": ["no_semantic_channels", "prior_only_heatmap", "coordinate_warning"]
}
```

关键原则：

- 当 `sem=0` 且 `target_heat=none` 时，MLLM 必须承认 `evidence_level=geometry_only`。
- 当 `target_heat=semantic_prior_only` 时，MLLM 不能说 target visible。
- 当 branch value top 与选择不一致时，必须解释为什么违反 value ranking。

---

### 6.3 MLLM 输出应更紧密承接 downstream scorer

需要讨论：MLLM 选择 branch 后，下游 scorer 如何使用？

候选设计：

1. **Soft prior**：当前做法，风险是 MLLM 信号被 final scorer 稀释。
2. **Hard branch gate**：只在 selected/ranked branch 内选 frontier，风险是 MLLM 错误会锁死。
3. **Tiered gate**：
   - direct target evidence: hard gate；
   - anchor evidence: strong soft prior；
   - prior-only/geometry-only: weak prior；
   - coordinate warning: no hard gate。
4. **Value agreement guard**：如果 MLLM selected branch 与 target potential top branch 差距过大，要求二次审查或降权。

---

## 7. 本轮讨论最应该聚焦的问题

1. **BEV 的目标是什么？**
   - 给人 debug？
   - 给 MLLM 做 branch decision？
   - 给下游 scorer 提供 value map？
   - 三者需要不同视图。

2. **target 不可见时，MLLM 应该输出什么？**
   - 不是 object point；
   - 更合理是 branch ID + evidence level + semantic search intent + value uncertainty。

3. **是否引入 PONI/VLFM-style target potential map？**
   - 不一定需要训练模型；
   - 先做 deterministic PONI-lite / VLFM-lite value map 可能更符合当前工程约束。

4. **room/function area 如何空间化？**
   - 不定位的 room text 是否应该出现在图上？
   - 是否需要通过 object clusters / layout / MLLM segmentation 生成 room masks？

5. **MLLM 应看 BEV 还是看 BEV + egocentric RGB？**
   - 成熟工作通常不会只给抽象 occupancy BEV；
   - 对 object/room inference，当前 RGB/panorama evidence 可能仍然重要。

6. **坐标系 contract 是否要先收紧？**
   - MLLM prompt 中只允许 canonical full_map rc；
   - 旧 local/FMM frame 字段只保存 debug，不进入 decision prompt。

---

## 8. 当前文档结论

**Observation:** 当前 MLLM frontier planner 的代码边界已经基本搭好：它可以保存 capsule，生成 semantic-capable BEV，构造 prompt，要求 MLLM 输出 branch ID，并把 branch prior 交给 final scorer。

**Observation:** 当前卡点是输入语义不足和 value map 不足，不是单纯 API/模型调用问题。

**Hypothesis:** 如果继续只改 prompt，而不增强 BEV 的 semantic/value 层，MLLM 仍会在 step31/43 这种 `sem=0`、`target_heat=none` 的场景中做不稳定推理。

**Hypothesis:** 最有希望的方向是：

1. 先把 BEV 输入改成多层决策状态；
2. 给 target 不可见场景引入 PONI/VLFM-style target potential / value map；
3. 将 room/text commonsense 空间化或降权；
4. 收紧 canonical coordinate/frame contract；
5. 让 MLLM 输出 evidence-level branch decision，下游 scorer 按 evidence level 选择 hard/soft gate。

---

## 9. Source map

Current SmoothNav code:

- `base_UniGoal/src/map/bev_mapping.py`
- `smoothnav/planner.py`
- `smoothnav/strategy_grounding.py`
- `smoothnav/frontier_branching.py`
- `smoothnav/frontier_scoring.py`
- `smoothnav/bev_frontier_planner.py`
- `smoothnav/semantic_bev.py`
- `smoothnav/task_frame_capsule.py`
- `scripts/replay_semantic_bev_capsules.py`
- `scripts/evaluate_semantic_bev_branch_decisions.py`

Current artifacts:

- `analysis/latest_semantic_bev_manual_check_20260428/summary.json`
- `analysis/latest_semantic_bev_manual_check_20260428/semantic_bev_contact_sheet.png`
- `analysis/latest_semantic_bev_manual_check_20260428/step_000031/semantic_annotated_bev_planner.png`
- `analysis/latest_semantic_bev_manual_check_20260428/step_000031/semantic_bev_frame.json`
- `analysis/s24_semantic_bev_repair_review_20260427/branch_decision_eval/summary.json`

Mature work local references:

- `analysis/SemExp`
- `analysis/PONI`
- `analysis/external_repos/semexp`
- `analysis/external_repos/poni`
- `analysis/external_repos/vlfm`
- `analysis/external_repos/vlmaps`
- `analysis/external_repos/cow`
- `analysis/external_repos/NavGPT`

Public references:

- SemExp / Object-Goal-Navigation: https://github.com/devendrachaplot/Object-Goal-Navigation
- SemExp paper: https://arxiv.org/abs/2007.00643
- PONI project: https://vision.cs.utexas.edu/projects/poni/
- PONI GitHub: https://github.com/srama2512/PONI
- PONI paper: https://arxiv.org/abs/2201.10029
- VLFM GitHub: https://github.com/bdaiinstitute/vlfm
- VLMaps project: https://vlmaps.github.io/
- VLMaps GitHub: https://github.com/vlmaps/vlmaps
- CoW GitHub: https://github.com/real-stanford/cow
- CoW paper: https://arxiv.org/abs/2203.10421
- NavGPT GitHub: https://github.com/GengzeZhou/NavGPT
