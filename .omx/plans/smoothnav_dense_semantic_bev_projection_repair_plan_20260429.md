# SmoothNav Dense Semantic BEV Projection Repair Plan

Date: 2026-04-29
Scope: 修正 SmoothNav MLLM frontier planner 的 BEV 输入构造，使其按 SemExp/PONI 成熟工作方式从上游语义投影/对象点云 footprint 生成栅格级语义占用，而不是用中心点、文字标签或稀疏通道 bbox-fill 冒充 dense BEV。

## 0. Current Observation

Observation: 当前 `dense_semantic_bev.png` 的问题不只是渲染风格，而是上游状态表达不够成熟：

1. `smoothnav/semantic_bev.py:380-640` 的 `build_dense_semantic_raster()` 已能读取 `full_map[4:]`，但默认 `semantic_bbox_fill=True` (`smoothnav/semantic_bev.py:386`) 会把稀疏 semantic seed 的连通分量外接矩形填成类别区域 (`smoothnav/semantic_bev.py:501-552`)。这不是 SemExp/PONI 的做法，会产生错误物体尺寸/位置。
2. 当前 replay 调用没有显式关闭 bbox-fill：`scripts/replay_semantic_bev_capsules.py:255` 直接调用 `build_dense_semantic_raster(full_map, objects=...)`。
3. 图中“物体框”目前主要来自两类不稳定来源：
   - sparse `full_map[4:]` 的 bbox-fill；
   - `world_state.object_summary[*].bbox_rc`，但旧 capsule 中该字段为空，新代码只是在 `smoothnav/world_state.py:122-168` 尝试从 graph pcd/bbox 估计。
4. 成熟工作不是从点/文字推框：SemExp 将 image semantic mask 经 depth point cloud splat 到 BEV (`analysis/external_repos/semexp/model.py:188-244`)，PONI 将 semantic mesh/point vertices rasterize 到 BEV cell (`analysis/external_repos/poni/scripts/create_semantic_maps.py:560-620`)。

## 1. Design Principles

1. Primary dense BEV 只能来自真实空间投影：`obs[:,4:]` semantic mask projection、graph object pcd footprint、或显式保存的 per-object BEV mask。
2. 禁止把 sparse seed 的外接矩形当作 true semantic map。
3. object bbox 可以作为低置信 object footprint，但来源必须是 3D mask/pcd/bbox 投影，不是中心点放大。
4. `dense_semantic_bev.png` 只显示栅格语义场；debug label、branch score、warning 留在 debug overlay。
5. replay 必须区分旧 capsule 不可恢复的信息与新 capsule 可验证的信息。

## 2. Target Architecture

### Upstream semantic map path

`obs RGB-D + semantic mask/logits` -> `base_UniGoal/src/map/bev_mapping.py::Mapping.forward()` -> `local_map/full_map[4:]` -> `build_dense_semantic_raster()` -> dense semantic BEV.

Required changes:
- audit/verify `obs[:,4:]` before mapping (`base_UniGoal/src/map/bev_mapping.py:84-86`);
- add semantic projection diagnostics around `agent_height_proj[:,1:]` and `agent_view[:,4:]` (`base_UniGoal/src/map/bev_mapping.py:137-141`);
- preserve per-category pixel counts before/after `torch.max` map fusion (`base_UniGoal/src/map/bev_mapping.py:177-179`).

### Object footprint path

`segmentation detections mask + depth` -> `create_object_pcd()` -> graph object `pcd`/`bbox` (`base_UniGoal/src/graph/utils/utils.py:267-285`) -> serialized BEV footprint/mask -> `build_dense_semantic_raster(... objects=...)`.

Required changes:
- promote `world_state.py` pcd/bbox projection from bbox-only to true occupancy footprint where possible;
- save object footprint masks or compact RLE/indices into capsule, not just center/bbox;
- render object footprint only when `footprint_source in {graph_pcd_cells, graph_3d_bbox}` and label confidence is available.

### Capsule/replay path

Online run -> `world_state.summary()` (`smoothnav/types.py:577-596`) + `maps.npz` (`smoothnav/task_frame_capsule.py:130-137`) + optional new `object_footprints.npz/json` -> offline replay (`scripts/replay_semantic_bev_capsules.py:203-270`).

Required changes:
- capsule writer must store the new semantic/object projection diagnostics;
- replay must not invent footprints when old capsules lack the raw data.

## 3. Implementation Phases

### P0 — Stop wrong dense-BEV inflation

Files:
- `smoothnav/semantic_bev.py`
- `scripts/replay_semantic_bev_capsules.py`
- `tests/test_semantic_bev_renderer.py`

Steps:
1. Change `build_dense_semantic_raster(... semantic_bbox_fill=False)` default.
2. Ensure replay uses `semantic_bbox_fill=False` for `dense_semantic_bev.png`.
3. If keeping bbox-fill, move it to explicit debug-only artifact, e.g. `semantic_bbox_fill_debug.png/json`, with source `debug_sparse_seed_bbox_fill`; do not include it in primary `display_semantic_source`.
4. Update tests: default dense map must not fill sparse seeds; explicit debug bbox-fill test can remain.

Acceptance:
- `semantic_bbox_footprint_source == none` in primary dense BEV unless an explicit debug path is invoked.
- No object/branch/room labels on dense map remain true.
- Old s22b step98/145 images may look sparse; JSON must say why.

### P1 — Add trustworthy upstream semantic projection diagnostics

Files:
- `base_UniGoal/src/map/bev_mapping.py`
- `smoothnav/world_state.py`
- `smoothnav/task_frame_capsule.py`
- tests under `tests/`

Steps:
1. In `Mapping.forward()`, compute and optionally expose:
   - `obs_semantic_active_count`, `obs_semantic_pixel_count_by_category` before AvgPool (`base_UniGoal/src/map/bev_mapping.py:84-86`);
   - `local_semantic_pixel_count_by_category` after `agent_height_proj[:,1:] / cat_pred_threshold` (`base_UniGoal/src/map/bev_mapping.py:139-141`);
   - `full_map_semantic_pixel_count_by_category` after fusion (`base_UniGoal/src/map/bev_mapping.py:177-179`).
2. Add lightweight `BEV_Map.semantic_projection_summary(env_idx=0)` to expose these counts.
3. Include this diagnostic in `world_state.summary()` or capsule metadata.

Acceptance:
- For any replay/online capsule, we can answer: semantic input was empty, projection erased it, or renderer threshold hid it.
- If `obs[:,4:]` is empty, logs say upstream segmentation/logits are missing.

### P2 — Serialize real object BEV footprints

Files:
- `smoothnav/world_state.py`
- `smoothnav/task_frame_capsule.py`
- `smoothnav/semantic_bev.py`
- possibly `base_UniGoal/src/graph/graph.py` / graph utility code if current node object loses pcd.

Steps:
1. Replace bbox-only `_object_bev_footprint()` with two-level output:
   - `footprint_rc_indices`: sampled or RLE-encoded cell coordinates projected from object `pcd.points`;
   - `bbox_rc`: derived from those cells, used only as summary/legend metadata.
2. Add `footprint_source` values:
   - `graph_pcd_cells` for projected pcd cells;
   - `graph_3d_bbox_cells` for fallback bbox corner/box fill;
   - `none` if unavailable.
3. Add bounds/shape validation and confidence:
   - require minimum point/cell count;
   - reject enormous footprints exceeding e.g. 20% of explored area unless class is wall/floor/background;
   - reject object footprint if bbox center is far from graph node center.
4. Save full compact footprint payload into capsule, preferably `object_footprints.json` + optional `object_footprints.npz` for masks.
5. Update `build_dense_semantic_raster()` to paint `footprint_rc_indices` first; only fill `bbox_rc` for explicit `graph_3d_bbox_cells` fallback.

Acceptance:
- Dense BEV object areas are derived from projected 3D support, not guessed boxes.
- JSON lists object count, class, `num_detections`, pcd points/cells, bbox, confidence, rejection reason if not drawn.

### P3 — Repair online capture and replay contracts

Files:
- `smoothnav/main.py`
- `smoothnav/task_frame_capsule.py`
- `smoothnav/tracing.py`
- `scripts/replay_semantic_bev_capsules.py`

Steps:
1. Pass `bev_map.semantic_projection_summary()` and object footprints into `tracer.record_task_frame_capsule()` alongside `full_map/local_map` (`smoothnav/main.py:765-768`).
2. `write_task_frame_capsule()` stores:
   - `dense_semantic_bev.json/png` at capture time;
   - `semantic_projection_summary.json`;
   - `object_footprints.json/npz` if available;
   - old debug overlay under debug names only.
3. Replay reads these artifacts. If absent in old capsules, reports `unrecoverable_old_capsule_missing_object_footprints` rather than trying to hallucinate.

Acceptance:
- New single-step capsule can be replayed and reproduce identical dense BEV metadata.
- Old capsule replay remains honest and does not fabricate missing object extent.

### P4 — Renderer cleanup after data contract is correct

Files:
- `smoothnav/semantic_bev.py`

Steps:
1. Keep primary `dense_semantic_bev.png` as pure raster + thin agent/trajectory/frontier.
2. Legend shows only present classes and source counts.
3. If semantic channels are empty but object footprints exist, image title/JSON must distinguish `true_semantic_source=empty_map_channels` and `object_footprint_source=graph_pcd_cells`.
4. Add optional separate debug layers only after primary artifact is correct.

Acceptance:
- Image resembles SemExp/PONI: category-colored cell clusters at actual top-down positions.
- No branch/object text boxes in primary raster.

### P5 — Verification ladder

1. Unit tests:
   - sparse semantic seed does not inflate by default;
   - real semantic channel mask paints exact cells;
   - object `footprint_rc_indices` paints exact cells;
   - invalid/huge/out-of-bounds footprints are rejected;
   - empty channels report empty.
2. Replay old s22b steps 31/43/98/145:
   - expected: honest sparse/empty result; no fake object boxes.
3. Run one minimal online capsule capture on the known failed scene/step policy:
   - expected: new `semantic_projection_summary` + object footprints appear if upstream graph stores pcd.
4. Human visual inspection:
   - object count and footprint list must match JSON;
   - colored occupied cells align with geometry and agent position.
5. Only after this passes, reconnect dense BEV to MLLM prompt/decision-state.

## 4. Risks and Mitigations

Risk: Graph nodes may not retain pcd/bbox at `world_state.summary()` time.
Mitigation: Serialize object footprints immediately after graph update or add a graph snapshot/footprint extractor before summary conversion.

Risk: Semantic channel categories in `full_map[4:]` do not match labels used by graph captions.
Mitigation: add a single category registry and save it in every dense BEV JSON.

Risk: 3D bbox fallback creates too-large furniture rectangles.
Mitigation: prefer pcd cell indices; use bbox fill only as marked fallback with area clamps.

Risk: Old capsules cannot recover mature dense BEV.
Mitigation: explicitly mark them unrecoverable for object extent and use them only for renderer/diagnostic regression, not final visual proof.

## 5. ADR

Decision: Treat SemExp/PONI-style dense BEV as a projection/data-contract problem, not a renderer beautification problem.

Drivers:
- Mature methods rasterize actual semantic observations into map cells.
- MLLM planner needs spatial state, not debug labels.
- False dense boxes are worse than sparse honest maps because they mislead branch choice.

Alternatives considered:
- Keep semantic bbox-fill from sparse map seeds: rejected because it caused wrong size/location boxes and is not used by mature work.
- Draw object center blobs: rejected because it remains pseudo support, not object occupancy.
- Train/import PONI potential network now: rejected because current blocker is input map correctness, not value learning.

Consequences:
- Old capsule images may initially become less visually dense, but more truthful.
- Need one fresh online capsule to verify real object footprints after upstream serialization is fixed.

Follow-ups:
- After dense semantic/object projection is correct, revisit target-conditioned value map and MLLM prompt coupling.
