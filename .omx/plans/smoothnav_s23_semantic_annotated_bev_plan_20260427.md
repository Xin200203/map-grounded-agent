# SmoothNav s23 Plan: 真正实现 Semantic Annotated BEV

Date: 2026-04-27
Owner: Ralph / experiment lead
Scope: **planning artifact only**. 本计划不启动新的 online episode；目标是把当前“几何 BEV + branch 标注”升级为可复现、可审查、可喂给 MLLM 的 semantic annotated BEV 体系。

## 0. 一句话结论

Observation: 当前 SmoothNav 的 BEV 图不是成熟工作意义上的 semantic BEV，而是 occupancy/free/unknown + frontier branch debug 图。它缺少物体、房间/区域、轨迹、重复访问、branch 后续收益、planner 决策链这些语义层，所以 MLLM 即使拿到图，也只能看见“有哪些洞/门口”，看不见“这个洞通向什么语义空间”。

Claim: 下一轮实现应先建立一个显式的 `SemanticBEVFrame` 数据契约，再用它生成两类图：

1. **planner image**：干净、低噪声、给 MLLM 选择 branch ID。
2. **debug/replay image**：信息更全，给人类和离线 evaluator 检查 dry / MLLM / final branch、死循环、后续 new area/new object。

这样可以在不完整 online 重跑的前提下，对 s22b 已保存 capsule 的 step 31 / 43 / 98 / 145 逐帧复盘。

---

## 1. 当前代码事实与问题边界

### 1.1 当前 BEV renderer 只画几何与 branch

Observation:

- `smoothnav/frontier_branching.py:457-612` 的 `render_branch_bev_image()` 只把 `full_map` 渲染成 unknown/free/obstacle，再叠加 branch candidate point、representative marker、branch ID、actionability、score、robot marker 和 legend。
- `smoothnav/frontier_branching.py:481-500` 明确只读取 map channels：channel 0 obstacle，channel 1 free/known；没有读取 object / room / semantic class / text heatmap。
- `smoothnav/frontier_branching.py:530-572` 叠加的是 frontier branch；语义来自 branch score terms，而不是视觉/物体/房间层。

Implication:

当前图片更准确的名字应是 `branch_annotated_occupancy_bev`，不是 semantic annotated BEV。

### 1.2 Capsule 保存了语义 JSON，但没有把语义画进 BEV

Observation:

- `smoothnav/task_frame_capsule.py:107-123` 会保存 `world_state.json`、`frontier_branches.json`、`planner_*`、`scoring_*`、`grounding_result.json` 等。
- `smoothnav/task_frame_capsule.py:134-148` 只调用 `save_branch_bev_image(...)` 写 `bev_annotated_branches.png`。
- 当前 capsule 没有 `semantic_bev_frame.json`，也没有 `semantic_annotated_bev_planner.png` / `semantic_annotated_bev_debug.png`。

Implication:

已有数据其实足够生成第一版 object/room/branch semantic overlay，但生成链路没有把这些信息绑定成同一坐标系下的可视化与 planner 输入。

### 1.3 WorldState 已有一部分语义，但房间几何不足

Observation:

- `smoothnav/types.py:558-596` 的 `WorldState.summary()` 已包含 `room_summary`、`object_summary`、`visible_targets`、`frontier_summary`。
- `smoothnav/world_state.py:51-81` 的 `summarize_objects()` 已提取 `caption`、`center`、`num_detections`、`target_relevance`、`target_match_reason`。
- `smoothnav/world_state.py:33-48` 的 `summarize_rooms()` 只有 room caption、object_count、object captions，没有 room center / bbox / polygon / confidence。

Implication:

第一版 semantic BEV 可以可靠画 object 点和 label；room 只能先做 **room hypothesis**，不能假装有精确房间边界。成熟工作若有 room polygon，是来自语义分割/GT map/3D map 后处理；SmoothNav 当前 graph room 不等价于真实房间 mask。

### 1.4 MLLM prompt 说“annotated map”，但图片实际上语义稀疏

Observation:

- `smoothnav/bev_frontier_planner.py:116-147` 的 prompt 告诉模型会看到 annotated BEV map，并要求结合 target semantics、room commonsense、scene graph、BEV topology 选 branch。
- `smoothnav/bev_frontier_planner.py:201-215` 实际传给 MLLM 的 image 来自 `render_branch_bev_image()`，因此只是几何 branch 图。
- 语义主要在文本侧：`smoothnav/bev_frontier_planner.py:105-130` 从 `serialize_for_planner()` 和 `compact_branch_list()` 构造 text prompt。

Implication:

这解释了用户观察到的现象：图上几乎没有物体/房间信息；模型的“语义判断”仍然主要依赖文本，而不是图像上的 spatial semantic layout。

### 1.5 High-level planner 仍是纯文本，BEV 只被间接使用

Observation:

- `smoothnav/planner.py:45-67` 的 `HIGH_LEVEL_PROMPT` 是纯文本：target、scene text、searched regions、choices。
- `smoothnav/planner.py:554-624` 的 `HighLevelPlanner.plan()` 调用的是 `self.llm(prompt=prompt)`，没有图像。
- `smoothnav/planner.py:739-790` 的 `plan_stage_goal()` 能访问 `world_state.bev_map`，但只提取 `map_size` 供方向 fallback 使用。

Implication:

当前系统存在两个 planner 层：旧 high-level text planner 和新 BEV frontier planner。真正的 semantic annotated BEV 应服务于 frontier-branch 选择边界，而不是直接替换所有 high-level planner 逻辑；否则会把“任务语义规划”和“可执行 branch grounding”再次耦合在一起。

### 1.6 坐标问题已经暴露，必须成为一等契约

Observation:

- 之前图中 `A robot` 出现在房间外，根因是 `Graph.get_goal()` 中 FMM `start` 使用 flipped traversible map，而 frontier/render 使用 unflipped full-map frame。
- 当前修复点在 `base_UniGoal/src/graph/graph.py:1002-1024`：把 `agent_state` 转换为 `agent_frontier_coord`，并做 row flip 回 full-map frame。
- `base_UniGoal/src/graph/graph.py:1296-1310` 把 `agent_frontier_coord` 传给 `build_frontier_branches()`。
- `base_UniGoal/src/graph/graph.py:1350-1367` 把同一坐标写入 replay snapshot。

Claim:

Semantic BEV 的第一项工程要求不是“多画几个 label”，而是建立 **单一坐标系契约 + 自动质量检查**。否则 object、room、branch、agent 只要一个层错位，MLLM 的判断会被系统性误导。

---

## 2. 成熟工作给出的直接启发

### 2.1 SemExp / Goal-Oriented Semantic Exploration

Direct support from source / code:

- 项目网页说明系统由 Semantic Mapping 和 Goal-Oriented Semantic Policy 组成，semantic policy 基于 semantic map 选 long-term goal，本地 deterministic planner 执行；见项目页与论文说明：https://devendrachaplot.github.io/projects/semantic-exploration.html ，https://arxiv.org/abs/2007.00643 。
- 本地代码 `analysis/external_repos/Object-Goal-Navigation/main.py:88-95` 定义 full map channels：obstacle、explored、current agent location、past agent locations、semantic categories。
- `analysis/external_repos/Object-Goal-Navigation/main.py:318-329` 和 `:516-526` 把 `map_pred`、`exp_pred`、`pose_pred`、`goal`、`sem_map_pred` 传给 planner / visualization。
- `analysis/external_repos/Object-Goal-Navigation/agents/sem_exp.py:347-385` 把 obstacle、explored、visited、goal、semantic class 合成一张 top-down semantic visualization。
- `analysis/external_repos/Object-Goal-Navigation/agents/utils/visualization.py:19-24` 有 visited trajectory drawing。

Inference for SmoothNav:

SemExp 的关键不是“图更漂亮”，而是 map tensor 本身有语义通道，且 visited/current/goal 是 map 的正式 channel。SmoothNav 若只在 occupancy 上画 branch，就缺少 SemExp 类系统用于长程目标选择的核心信息。

### 2.2 VLMaps

Direct support from source / code:

- VLMaps README 描述其把 pretrained visual-language features 融入 3D reconstruction，使地图可被自然语言索引；见 https://github.com/vlmaps/vlmaps 和论文 https://arxiv.org/abs/2210.05714 。
- 本地代码 `analysis/external_repos/vlmaps/vlmaps/map/vlmap_builder.py:81-132` 初始化 LSeg 并从 RGB-D/pose 构建 feature map。
- `analysis/external_repos/vlmaps/vlmaps/map/vlmap.py:134-167` 用 language description / categories 计算 map mask。
- `analysis/external_repos/vlmaps/application/index_map.py:43-48` 把 text-query mask 投到 2D 并生成 heatmap。
- `analysis/external_repos/vlmaps/vlmaps/robot/habitat_lang_robot.py:283-295` 对 region name 生成 distribution map。

Inference for SmoothNav:

VLMaps 说明“semantic BEV”的上限不是 object label overlay，而是 language-query heatmap。SmoothNav 当前最小可行版本不应立刻引入 LSeg/CLIP embedding，因为这会扩大依赖与数据链路；但应把 `SemanticBEVFrame` schema 设计成以后可接 `query_heatmap` / `semantic_class_channels`。

### 2.3 iPPD-sem / explicit object-room maps

Direct support from local code:

- `analysis/external_repos/iPPD-sem/traj_sampling/map_tools_3D.py:60-72` 处理后的 map 直接返回 `nav_map`、`room_list`、`obj_list`、bounds。
- `analysis/external_repos/iPPD-sem/traj_sampling/post_processing.py:81-97` 把 object bbox center 转换到 map coordinates。
- `analysis/external_repos/iPPD-sem/traj_sampling/post_processing.py:106-121` 把 room bbox center 转换到 map coordinates。
- `analysis/external_repos/iPPD-sem/scoring_trainer/dataset/mlnv3_dataset.py:20-33` 加载 semantic point map + labels + resolution。

Inference for SmoothNav:

如果没有 dense semantic segmentation，成熟路线也可以先走 object/room instance lists：把 graph node / room node 变成 map-space 的点、bbox、label、confidence。这正适合 SmoothNav 当前 graph 结构。

---

## 3. RALPLAN-DR Summary

### 3.1 Principles

1. **数据契约先于渲染**：先定义 `SemanticBEVFrame` JSON，再由同一 frame 渲染 planner/debug 图。
2. **坐标单一真源**：agent、object、room、branch、trajectory 都必须声明同一 `full_map_rc_unflipped` 坐标系，且有自动检查。
3. **语义来源可追溯**：每个 label 都必须带 source/confidence；不把 room hypothesis 画成 ground-truth room boundary。
4. **离线 replay 优先**：先用已保存 s22b capsule 验证 step 31/43/98/145，不再完整 online 重跑。
5. **MLLM 输出仍是 branch ID**：semantic BEV 只是增强 branch 选择，不让模型输出任意坐标。

### 3.2 Decision Drivers

1. **可诊断性**：失败时能区分 perception/graph semantic 缺失、BEV 渲染错误、MLLM branch 判断错误、frontier grounding 错误、low-level 执行错误。
2. **最小改动**：先复用当前 `WorldState.object_summary` / graph nodes / frontier snapshots，不引入 LSeg、Detectron 新链路或新依赖。
3. **可扩展性**：schema 预留 dense semantic channels / query heatmap / room mask，但第一版不强依赖它们。

### 3.3 Viable Options

#### Option A — Graph-instance overlay first（推荐第一阶段）

做法：从 SmoothNav 已有 graph / world_state 提取 object centers、room hypotheses、branch semantics，画到 BEV 上。

Pros:

- 不需要新模型和新数据依赖。
- 可以直接对 s22b capsule 离线重放。
- 最容易定位坐标错位、object 是否缺失、branch 是否语义上合理。

Cons:

- 不是 dense semantic map；未知区域没有语义预测。
- room boundary 只能先做 hypothesis，不是精确房间轮廓。

#### Option B — Semantic channel BEV via segmentation projection（第二阶段）

做法：像 SemExp 一样维护 obstacle/explored/current/past/semantic category channels，并把目标类别/相关类别作为 heatmap 渲染。

Pros:

- 更接近成熟 ObjectNav semantic map。
- 对 “TV 应该在 living room/bedroom” 这类目标更有可计算空间支持。

Cons:

- 需要梳理当前 detection/segmentation 输出，确认 class IDs、投影、去噪、时间融合。
- 容易引入新 failure modes，不适合作为下一步最小修复。

#### Option C — VLMaps-like open-vocabulary feature map（长期研究项）

做法：融合 CLIP/LSeg/OpenScene features，支持 “tv/living room/sofa area” text query heatmap。

Pros:

- 对开放词汇目标和 room commonsense 更强。
- 上限最高。

Cons:

- 依赖重、运行慢、调试成本高。
- 会把当前 SmoothNav 的主要问题从 planner-grounding 变成 semantic mapping benchmark，不适合当前 s22/s23 问题收敛。

### 3.4 Decision

选择 **Option A as Phase 1**，但 schema 按 Option B/C 预留字段。也就是说：先真正实现“可审查的 semantic annotated BEV”，不是立刻实现完整 dense semantic BEV。

---

## 4. 目标数据契约：`SemanticBEVFrame v1`

新增文件建议：`smoothnav/semantic_bev.py`。

### 4.1 顶层 schema

```json
{
  "schema_version": "smoothnav.semantic_bev_frame.v1",
  "episode_id": 228,
  "step_idx": 98,
  "coordinate_frame": {
    "name": "full_map_rc_unflipped",
    "row_axis": "south_down_image_y",
    "col_axis": "east_right_image_x",
    "origin": "full_map[0,0]",
    "resolution_cm_per_cell": 5,
    "crop": {"r0": 229, "r1": 497, "c0": 112, "c1": 439}
  },
  "base_layers": {
    "has_obstacle": true,
    "has_free": true,
    "has_explored": false,
    "has_semantic_channels": false,
    "has_language_query_heatmap": false
  },
  "agent": {...},
  "trajectory": [...],
  "objects": [...],
  "rooms": [...],
  "branches": [...],
  "decision_overlay": {...},
  "quality_checks": {...},
  "provenance": {...}
}
```

### 4.2 Agent layer

Required fields:

- `coord_rc`: full-map row/col。
- `heading_deg`: 如果当前 pose 可用则保存；没有就 `null`。
- `source`: `graph.agent_frontier_coord` / `world_state.pose` / fallback。
- `alignment_check`: agent 是否落在 known/free/near-free cell 上。

Acceptance:

- 若 `agent.coord_rc` 与 free/known map 距离超过阈值，例如 2-4 cells，`quality_checks.agent_on_known_area=false`，debug 图必须醒目标黄/红，不再安静画错。

### 4.3 Object layer

Required fields:

- `id`: deterministic，如 `O001`。
- `caption`: graph node caption。
- `center_rc`: full-map row/col。
- `target_relevance`: 复用 `smoothnav/world_state.py:61-79` 的 goal-caption score。
- `num_detections`: 当前已有。
- `source`: `graph.nodes[].center`。
- `confidence`: 第一版可由 `num_detections` 和 target relevance 派生；不能伪称 detector confidence。
- `display_priority`: 用于 map 上 label 取舍，避免 label 爆炸。

Optional / later:

- `bbox_rc` / `mask_rle`：若能从 node object bbox 或 segmentation 投影取得。
- `first_seen_step` / `last_seen_step`：若 controller/graph delta 已记录。
- `room_id`: 若 node 属于 room node。

Renderer behavior:

- Map 上只画 top-K object labels，例如 target relevance 高、检测次数高、离 branch 近的物体。
- 其余 objects 放到右侧 legend / side panel，不全部压在地图上。
- target-relevant objects 用暖色边框，普通 objects 用蓝/青色。

### 4.4 Room / region layer

Required fields for v1:

- `id`: deterministic，如 `R001`。
- `caption`: room caption。
- `object_ids`: room 内 object ids。
- `center_rc`: 从 member object centers 的 median/mean 估计。
- `bbox_rc`: 从 member object centers 外扩得到；标记为 `hypothesis_bbox`。
- `source`: `graph.room_nodes`。
- `confidence`: `hypothesis` / low / medium；不要和 GT room mask 混淆。

Explicit non-goal for v1:

- 不画精确 room polygon，除非数据源明确提供 room mask / room bbox。

Renderer behavior:

- Room hypothesis 画虚线框或半透明弱色，不画实心区域。
- 图例写清楚：`R# = graph room hypothesis, not GT room boundary`。

### 4.5 Branch layer

Base fields already available from `smoothnav/frontier_branching.py:124-276` and replay snapshot:

- `id`
- `centroid`
- `representative_coord`
- `candidate_points`
- `direction_from_agent`
- `candidate_point_count`
- `score_terms_aggregate`

New semantic fields:

- `nearby_object_ids`: radius-based nearest observed objects。
- `nearby_room_ids`: nearest room hypotheses。
- `semantic_target_affinity`: simple transparent score from target/object/room text match。
- `visit_state`: `unvisited` / `recently_selected` / `revisited` / `dead_end_candidate`。
- `expected_information_gain`: v1 可先填 `unknown`，offline replay 后补 `observed_new_objects_after_n_steps` / `observed_new_area_after_n_steps`。

Decision overlay fields:

- `dry_selected_branch_id`
- `mllm_selected_branch_id`
- `final_selected_branch_id`
- `planner_prior_applied`
- `grounding_selected_frontier`
- `result_after_window`: replay-only，例如 `new_object_count_next_20_steps`、`new_area_delta_next_20_steps`、`same_room_loop`。

### 4.6 Provenance and hash

Every frame should include:

- `source_capsule_dir` or `run_dir`。
- `world_state_hash`。
- `frontier_branches_hash`。
- `maps_npz_hash` or full_map shape/hash。
- `renderer_version`。
- `prompt_hash` if used online。
- `image_hash` after render。

This lets a failed MLLM decision be reproduced without re-running Habitat.

---

## 5. 具体代码修改计划

### Phase 0 — 坐标契约与质量门禁（必须先做）

Files:

- Add `smoothnav/semantic_bev.py`
- Add `tests/test_semantic_bev_coordinates.py`
- Possibly add small helpers in `smoothnav/frontier_branching.py`, but avoid bloating it further.

Steps:

1. Define `CoordinateFrame` / helper functions:
   - `as_full_map_rc(coord)`
   - `xy_from_rc(coord, crop, scale)`
   - `content_crop_bounds(mask, branches, agent, objects, rooms)`
2. Move or wrap current crop/coordinate logic from `smoothnav/frontier_branching.py:350-527` into reusable helpers, but keep old API compatible.
3. Add quality checks:
   - agent within map bounds。
   - agent near known/free cell。
   - object centers within map bounds。
   - branch representative/candidate coords within map bounds。
   - crop contains agent + selected branches + displayed objects。
4. Add synthetic unit tests:
   - A 20x20 map, known free square, obstacle line, agent/object/branch coords；assert rendered pixel/JSON coords are not vertically flipped。
   - Regression test for prior robot-outside-room bug: if agent provided in flipped frame, quality check fails; if provided in full-map frame, passes。

Acceptance criteria:

- `SemanticBEVFrame.quality_checks.agent_on_known_or_near_known == true` for known-good synthetic case。
- If agent is deliberately row-flipped, test detects mismatch。
- No online episode required。

### Phase 1 — Object semantic overlay from existing graph

Files:

- `smoothnav/semantic_bev.py`
- `smoothnav/world_state.py`
- `tests/test_semantic_bev_objects.py`

Steps:

1. Extend `summarize_objects()` beyond `smoothnav/world_state.py:51-81`:
   - Add deterministic object id.
   - Preserve raw node index.
   - Add safe `center_rc` with coordinate frame metadata。
   - Add optional `room_caption` / `room_id` if membership can be inferred from `room_nodes`。
   - Keep current `target_relevance` fields。
2. Implement `build_semantic_bev_frame(..., world_state, frontier_branches, scoring_snapshot, full_map, episode_id, step_idx)`。
3. Implement `select_display_objects()`:
   - Sort by target relevance, detection count, proximity to selected/mllm/dry branches, proximity to agent。
   - Cap labels, e.g. map labels <= 12; legend can list more。
4. Render object markers:
   - Target-relevant: orange/red ring。
   - Non-target: cyan/blue dot。
   - Low confidence: small hollow dot。
5. Add side legend:
   - `O1 cabinet det=3 rel=0.10`
   - `O2 tv? det=1 rel=0.95`

Acceptance criteria:

- For a synthetic world state with 3 objects, JSON contains 3 object entries and PNG contains non-background colored markers at expected coords。
- If object center is missing, object goes to legend as `unlocalized` and is not drawn on map。
- Existing `world_state.summary()` remains backward-compatible.

### Phase 2 — Room / region hypothesis overlay

Files:

- `smoothnav/semantic_bev.py`
- `smoothnav/world_state.py`
- `tests/test_semantic_bev_rooms.py`

Steps:

1. Extend `summarize_rooms()` beyond `smoothnav/world_state.py:33-48`:
   - Deterministic room id。
   - Member object ids/captions。
   - `center_rc` from member centers if available。
   - `bbox_rc` from member centers + margin if at least 2 localized objects。
   - `geometry_type`: `member_object_hypothesis`。
2. Render room hypotheses as dashed boxes / center labels, not solid room masks。
3. Add legend statement that room geometry is a graph hypothesis, not GT room boundary。

Acceptance criteria:

- Room labels appear only when at least one member object has a localized center。
- Room bbox/hypothesis includes source and confidence。
- Renderer never presents room hypothesis as ground-truth segmentation。

### Phase 3 — Branch semantic affinity and visit/dead-end state

Files:

- `smoothnav/semantic_bev.py`
- `smoothnav/frontier_branching.py` only if needed for branch helper fields。
- `smoothnav/controller_state.py` or existing trace/capsule readers if branch history is already available。
- `tests/test_semantic_bev_branch_affinity.py`

Steps:

1. For each branch, compute nearest observed objects/rooms in BEV coordinates:
   - `nearby_object_ids` within radius, e.g. 50 map cells, top 3。
   - `nearby_room_ids` nearest 1-2 room hypotheses。
2. Compute transparent `semantic_target_affinity`:
   - object caption target match via existing `smoothnav/target_matching.py`。
   - room caption target prior using existing or simple text matching。
   - Do not let this silently affect scoring yet; first it is visualization/debug metadata。
3. Attach branch decision state:
   - dry selected branch from `dry_debug.selected_frontier_branch_id` in `smoothnav/strategy_grounding.py:299-310`。
   - MLLM selected branch from `mllm_result.verdict.selected_branch_id`。
   - final selected branch from `frontier_scoring.summarize_frontier_selection()` score breakdown (`smoothnav/frontier_scoring.py:211-228`)。
4. Attach branch outcome when doing replay:
   - `new_object_count_next_K_steps`
   - `new_area_delta_next_K_steps`
   - `same_room_loop`
   - `dead_end_revisit`

Acceptance criteria:

- A branch near a target-like object receives `nearby_object_ids` and visible label in debug JSON。
- Dry / MLLM / final branch are separately represented, not overwritten into one marker。
- Replay-only outcome fields are absent or null online, never fabricated。

### Phase 4 — Semantic renderer: planner image vs debug image

Files:

- `smoothnav/semantic_bev.py`
- Keep `smoothnav/frontier_branching.py:457-640` as legacy renderer / fallback.
- `tests/test_semantic_bev_renderer.py`

Renderer modes:

#### `mode="planner"`

Goal: 给 MLLM 的图，低噪声。

Layers:

1. unknown/free/obstacle base。
2. robot + heading。
3. branch IDs and representative markers。
4. top semantic object labels。
5. room hypothesis center labels only if not cluttered。
6. compact legend with color contract。

Do not include:

- dry/final outcome labels。
- too多 score 数字。
- 过多 historical traces。

#### `mode="debug"`

Goal: 人类和 evaluator 审查。

Additional layers:

1. dry-selected branch marker。
2. MLLM-selected branch marker。
3. final-selected branch marker。
4. branch visit/dead-end states。
5. trajectory / visited path if available。
6. object/room side panel。
7. quality check warnings。
8. future replay outcome labels。

Acceptance criteria:

- Same `SemanticBEVFrame` can render both images。
- Image legend explicitly explains object, room, branch, selected markers。
- If no semantic objects exist, image says `No localized semantic objects in frame` rather than pretending semantic coverage。

### Phase 5 — Capsule and trace integration

Files:

- `smoothnav/task_frame_capsule.py`
- `smoothnav/tracing.py`
- `smoothnav/strategy_grounding.py`
- Tests: `tests/test_task_frame_capsule.py`, `tests/test_strategy_grounding_gen2.py`

Steps:

1. Update capsule writer near `smoothnav/task_frame_capsule.py:134-148`:
   - Build `semantic_bev_frame.json` when `world_state`, `frontier_branches`, `scoring_input`, and `full_map` exist。
   - Save `semantic_annotated_bev_planner.png`。
   - Save `semantic_annotated_bev_debug.png`。
   - Keep old `bev_annotated_branches.png` for compatibility initially。
2. Add frame/image hashes to `capsule_index.json`。
3. Update `RunTracer.record_mllm_frontier_call()` payload (`smoothnav/tracing.py:156-159`) to include semantic BEV artifact path/hash, not full image bytes。
4. Update `strategy_grounding.py:256-266` to pass semantic frame/path to MLLM planner once available。

Acceptance criteria:

- Old capsules still load。
- New capsules include three BEV files: legacy branch image, planner semantic image, debug semantic image。
- `capsule_index.json` lists files and hashes。

### Phase 6 — MLLM planner integration with semantic BEV

Files:

- `smoothnav/bev_frontier_planner.py`
- `smoothnav/mllm_frontier_contract.py`
- `smoothnav/strategy_grounding.py`
- Tests: `tests/test_bev_frontier_mllm_planner.py`, `tests/test_mllm_frontier_contract.py`

Steps:

1. Change `BEVFrontierMLLMPlanner.plan()` at `smoothnav/bev_frontier_planner.py:210-215`:
   - Prefer `render_semantic_annotated_bev(frame, mode="planner")`。
   - Fall back to `render_branch_bev_image()` only if semantic frame unavailable。
2. Extend prompt at `smoothnav/bev_frontier_planner.py:116-147`:
   - Add explicit legend: `O# object`, `R# room hypothesis`, `B# branch`, marker colors。
   - Add `SEMANTIC_BEV_SUMMARY` JSON with visible objects/rooms/quality checks。
   - Warn model: room boxes are hypotheses; choose only branch IDs from allowed set。
3. Keep output contract branch-ID-only；do not accept MLLM raw coordinate。
4. Validate that selected branch exists and is actionable before creating prior scores。
5. Record whether semantic image had object/room coverage; use this in later analysis。

Acceptance criteria:

- Unit test confirms prompt contains semantic legend and branch ID constraints。
- If `quality_checks.agent_on_known_or_near_known=false`, MLLM planner should either skip or mark low-confidence depending on configured policy。
- Existing branch-ID validation remains strict。

### Phase 7 — Offline capsule replay evaluator before any online run

Files:

- Add `scripts/replay_semantic_bev_capsules.py` or extend `analysis/s22b_capsule_review_20260427/evaluate_s22b_capsules.py`
- Add `analysis/s23_semantic_bev_review_20260427/README.md` generated by script。

Target frames:

- s22b step 31
- s22b step 43
- s22b step 98
- s22b step 145

Steps:

1. Load each capsule directory。
2. Read `world_state.json`, `frontier_branches.json`, `scoring_input.json`, `maps.npz`。
3. Build `SemanticBEVFrame`。
4. Render planner/debug images。
5. Produce per-frame evaluator JSON:
   - `agent_alignment_ok`
   - `object_count_localized`
   - `target_relevant_object_count`
   - `room_hypothesis_count`
   - `branch_count`
   - `branch_semantic_hint_coverage`
   - `dry_branch`
   - `mllm_branch`
   - `final_branch`
   - `same_room_loop_evidence`
   - `new_area_after_step`
   - `new_object_after_step`
6. Generate review markdown with side-by-side links to image + input JSON。

Acceptance criteria:

- No API call required。
- No Habitat runtime required。
- The four target frames produce valid semantic frames/images。
- If semantic information is absent, evaluator says exactly which upstream source is empty: `world_state.object_summary empty` / `object centers missing` / `room nodes missing`。

### Phase 8 — Only after offline acceptance: one online smoke

Scope:

- One known failed episode / one profile only。
- Not a suite。
- Purpose: verify instrumentation and whether branch choice visibly improves, not claim SR improvement。

Acceptance criteria before online:

- All Phase 0-7 tests/replay pass。
- Manual review of generated four BEV images confirms they contain useful semantic information and robot alignment is plausible。
- Prompt/image artifact is saved for every MLLM call。

Online outputs to inspect:

- `mllm_frontier_calls/*.jsonl`
- `task_frame_capsules/*/semantic_bev_frame.json`
- `semantic_annotated_bev_planner.png`
- `semantic_annotated_bev_debug.png`
- step traces for branch selected / no-progress。

---

## 6. Acceptance Criteria（整体）

### 6.1 Semantic coverage

- For frames where `world_state.object_summary` has localized objects, `semantic_bev_frame.json.objects[*].center_rc` is non-empty and corresponding object markers appear in PNG。
- For frames with graph room nodes and localized member objects, `rooms[*].geometry_type == "member_object_hypothesis"` and debug image shows room hypothesis label/box。
- If no localized objects/rooms exist, frame explicitly reports missing upstream semantic evidence, not silent sparse image。

### 6.2 Coordinate correctness

- `agent.coord_rc`、branch representative coords、object centers 都在 map bounds 内。
- `quality_checks.agent_on_known_or_near_known == true` for valid capsules after current coordinate fix。
- Synthetic row-flip regression test fails when fed flipped agent coords。

### 6.3 Planner/debug separation

- Planner image is not cluttered and contains only task-relevant semantic markers plus branch IDs。
- Debug image contains dry / MLLM / final branch overlays and branch outcome annotations when replay data exists。

### 6.4 Replay-first workflow

- s22b four target steps can be regenerated offline from capsule without online rerun。
- Offline evaluator generates machine-readable JSON and human-readable markdown。
- No API key or raw secret is written to docs/logs/artifacts。

### 6.5 Backward compatibility

- Existing tests for `frontier_branching`, `bev_frontier_mllm_planner`, `task_frame_capsule` continue to pass。
- Existing `bev_annotated_branches.png` remains available at least until semantic renderer is stable。

---

## 7. Verification Plan

### 7.1 Unit tests

Run locally/remote:

```bash
python -m unittest discover -s tests -p "test_semantic_bev_*.py"
python -m unittest discover -s tests -p "test_task_frame_capsule.py"
python -m unittest discover -s tests -p "test_bev_frontier_mllm_planner.py"
python -m unittest discover -s tests -p "test_frontier_branching.py"
```

Specific tests to add:

1. `test_coordinate_frame_agent_not_flipped`
2. `test_object_markers_rendered_from_world_state`
3. `test_room_hypothesis_not_gt_boundary`
4. `test_branch_decision_overlay_dry_mllm_final`
5. `test_capsule_writes_semantic_bev_artifacts`
6. `test_prompt_contains_semantic_bev_legend`
7. `test_no_semantic_source_reports_explicit_warning`

### 7.2 Offline replay verification

```bash
python scripts/replay_semantic_bev_capsules.py \
  --capsule-root analysis/s22b_capsule_review_20260427/raw/run/task_frame_capsules \
  --steps 31 43 98 145 \
  --out analysis/s23_semantic_bev_review_20260427
```

Expected outputs:

- `analysis/s23_semantic_bev_review_20260427/summary.json`
- `analysis/s23_semantic_bev_review_20260427/review.md`
- per-step `semantic_bev_frame.json`
- per-step `semantic_annotated_bev_planner.png`
- per-step `semantic_annotated_bev_debug.png`

### 7.3 Manual review checklist

For each of step 31 / 43 / 98 / 145:

- Is `A robot` inside explored/free topology?
- Are object labels present where `world_state.object_summary` says they should be?
- Are room labels clearly marked as hypotheses?
- Does each branch B# have a visible representative marker?
- Which branch is dry / MLLM / final?
- Did chosen branch later produce new area/new object?
- Is failure a planner semantic choice issue, branch decomposition issue, map semantic absence issue, or low-level execution issue?

### 7.4 Remote smoke after offline pass only

```bash
ssh 10.176.56.73 'cd /mnt/sdd/xxy/SmoothNav && source /mnt/sdd/xxy/miniconda3/etc/profile.d/conda.sh && conda activate unigoal && python -m unittest discover -s tests -p "test_semantic_bev_*.py"'
```

Then one online episode only if offline artifacts are correct.

---

## 8. Risks and Mitigations

### Risk 1: Object centers are in a different coordinate convention

Mitigation:

- Every object center goes through `SemanticBEVFrame` quality checks。
- Compare object centers against known/free map and branch/object relative layout。
- Add synthetic tests and replay warnings。

### Risk 2: Room hypothesis misleads MLLM

Mitigation:

- Room overlay must be visually dashed/weak and prompt must say `room hypothesis, not ground-truth boundary`。
- MLLM still chooses branch IDs; final scoring can override invalid branch。

### Risk 3: Image becomes too cluttered for MLLM

Mitigation:

- Separate planner/debug modes。
- Planner mode shows top-K objects/rooms only。
- Full details remain in JSON summary and debug image。

### Risk 4: Semantic labels are stale or hallucinated by graph

Mitigation:

- Store `source`, `num_detections`, and `confidence` per object。
- Make target-relevance visible but do not auto-score branch from it until replay confirms usefulness。

### Risk 5: This turns into a dense semantic-map project too early

Mitigation:

- Phase 1 explicitly uses existing graph only。
- Dense semantic channels / VLMaps-like heatmaps are schema extensions, not implementation blockers。

### Risk 6: Online test consumes time/API without isolating bug

Mitigation:

- No online run until s22b capsule replay produces usable semantic images and evaluator summary。
- One online episode only, with artifacts saved。

---

## 9. ADR

### Decision

Implement `SemanticBEVFrame v1` and semantic BEV renderer as a separate module (`smoothnav/semantic_bev.py`), fed by existing `WorldState`, frontier branch snapshots, scoring snapshots, and full_map. Use it first for offline capsule replay, then for the BEV MLLM planner.

### Drivers

- Current BEV images are too sparse and cannot support spatial semantic reasoning。
- Mature semantic navigation systems use semantic channels, object/room maps, trajectory/current-agent markers, and long-term goal overlays, not branch-only occupancy pictures。
- SmoothNav needs replayable instrumentation to avoid repeated full online episode runs。

### Alternatives considered

1. Keep current branch-only renderer and rely on text prompt.
   - Rejected: this is exactly the current failure mode; image lacks object/room evidence。
2. Immediately integrate VLMaps/LSeg dense open-vocabulary map.
   - Rejected for next step: high dependency and runtime cost; would widen scope before current branch-grounding bug is isolated。
3. Ask MLLM to output raw BEV coordinates.
   - Rejected: loses executable candidate contract and reintroduces coordinate/actionability bugs。
4. Draw room boxes as if they were ground truth.
   - Rejected: SmoothNav current room nodes are graph hypotheses, not room masks。

### Why chosen

The staged schema+renderer approach gives immediate diagnostic value on saved capsules, preserves the branch-ID execution contract, and leaves a clear path toward stronger dense semantics later.

### Consequences

- More artifacts per capsule, but less need for repeated online reruns。
- Initial semantic BEV is still graph-instance based, not dense semantic prediction。
- Requires careful coordinate tests before trusting MLLM image decisions。

### Follow-ups

1. Implement Phase 0-2 first and regenerate s22b four frames。
2. If object/room overlay is useful, integrate planner image into `BEVFrontierMLLMPlanner`。
3. If graph semantic coverage is too sparse, then design Phase B semantic channel projection。
4. Only after offline success, run one known-failed online episode.

---

## 10. Execution handoff recommendation

Recommended route:

- `$ralph` single-owner execution for Phase 0-4 because coordinate/schema/rendering are tightly coupled。
- Use separate reviewer/verifier after implementation to inspect generated images and test coverage。

Suggested task order:

1. Implement `smoothnav/semantic_bev.py` schema + coordinate tests。
2. Add object overlay from `world_state.object_summary`。
3. Add room hypothesis overlay。
4. Add branch decision overlay。
5. Integrate capsule writer。
6. Build s22b offline replay script。
7. Manual review generated images。
8. Only then wire semantic image into online MLLM planner。

