# Semantic BEV 构造链路对比：SemExp / PONI vs SmoothNav

Date: 2026-04-29  
Scope: 本文只整理 **planner / policy 可用的 semantic BEV 状态表示**。它不讨论低层 executor，也不把 debug overlay 当作成熟 semantic BEV。  
Primary local references:

- SemExp local code: `analysis/external_repos/semexp/`
- PONI local code: `analysis/external_repos/poni/`
- SmoothNav current code: `base_UniGoal/src/map/bev_mapping.py`, `smoothnav/semantic_bev.py`, `smoothnav/task_frame_capsule.py`

## 0. 总结：成熟工作里的 BEV 是“状态场”，不是“调试图”

**Observation:** 从给定的两个成熟工作示例图以及本地代码看，SemExp / PONI 风格的 semantic BEV 共同点不是“标签更多”或“图更好看”，而是：

1. BEV 主体是 top-down raster：每个 map cell 都有明确含义。
2. 几何状态和语义状态被编码成 dense / pseudo-dense grid：unknown / free / obstacle / object category 等类别直接占据空间格子。
3. object category 不是用中心点文字表示，而是在 BEV 上形成一块有面积的 colored cell cluster。
4. policy / planner 消费的是 cell-level map state；可视化只是这个状态的 palette rendering。
5. 文字标签、debug warning、branch id、score table 通常不进入 semantic raster 主图。

**Claim:** 如果希望 SmoothNav 的 MLLM frontier planner 输入接近成熟工作的 BEV 表达，首要目标不是继续堆 prompt 或 side panel，而是保证上游能产生可信的 **cell-level semantic occupancy / footprint**，并在图像中以类别颜色稳定占据对应空间区域。

Evidence: SemExp 直接从 `obs[:,4:]` semantic channels 投影到 `local_map[:,4:]`，可视化时对每个 cell 取 semantic argmax；PONI 直接从 Habitat/3D scene semantic point cloud 生成 `map_semantic`，保存为每个 top-down cell 的 semantic id。详见下文代码路径。

---

## 1. SemExp 如何一步步构造 semantic BEV

### 1.1 输入：RGB-D + instance segmentation mask

SemExp 的语义来自 Detectron2 Mask R-CNN。每一帧 RGB 先被分割成 COCO 类别实例 mask，然后累加进固定类别通道。

Relevant code:

- `analysis/external_repos/semexp/agents/utils/semantic_prediction.py:39-48`
- `analysis/external_repos/semexp/constants.py:38-72`

流程：

1. 初始化 `semantic_input = zeros(H, W, 15 + 1)`。
2. 遍历 detector 输出的 `pred_classes`。
3. 只保留 COCO 到 ObjectNav 类别表里的类别。
4. 对每个实例，把 `pred_masks[j]` 加到对应 category channel。
5. 返回 `semantic_input`，作为 observation 的 `obs[:,4:]` 语义通道。

类别表是 15 个 ObjectNav 语义类：

- chair
- couch
- potted plant
- bed
- toilet
- tv
- dining-table
- oven
- sink
- refrigerator
- book
- clock
- vase
- cup
- bottle

**Observation:** SemExp 的 object category 从一开始就是 per-pixel mask channel，而不是 object center label。

### 1.2 RGB-D 投影：从 image mask 到 3D voxel，再到 top-down map

Relevant code:

- `analysis/external_repos/semexp/model.py:178-243`
- `analysis/external_repos/semexp/model.py:264-283`

核心步骤：

1. 取深度图：`depth = obs[:,3,:,:]`。
2. 由深度和相机内参生成点云：`get_point_cloud_from_z_t(...)`。
3. 把点云变换到 agent camera view，再变换到 agent-centered map frame。
4. 将点云坐标归一化到 voxel grid。
5. 对 `obs[:,4:]` semantic channels 做 pooling 后作为 voxel splatting 的 feature。
6. 用 `splat_feat_nd` 把几何和语义 feature splat 到 3D voxel。
7. 沿高度方向投影：
   - `agent_height_proj` 用于 obstacle / semantic map。
   - `all_height_proj` 用于 explored map。
8. 写入 agent-view map：
   - channel 0: obstacle map
   - channel 1: explored map
   - channel 4+: semantic category maps
9. 用 pose transform 把当前 agent-view map 旋转/平移到累计 map 中。
10. 与历史 map 做 max merge，形成持续累积的 local map。

**Observation:** SemExp 的 semantic BEV 不是把 2D bbox 画到地图上，而是用 depth 将 image-level semantic mask 投影到 top-down grid。

### 1.3 地图状态：full/local maps 和 policy 输入

Relevant code:

- `analysis/external_repos/semexp/main.py:95-106`
- `analysis/external_repos/semexp/main.py:260-286`
- `analysis/external_repos/semexp/main.py:382-438`（持续更新逻辑）

SemExp 初始化的 map channel 数是：

```text
num_channels = num_sem_categories + 4
```

其中：

| Channel | 含义 |
|---:|---|
| 0 | obstacle |
| 1 | explored |
| 2 | current agent location |
| 3 | past agent locations / visited |
| 4+ | semantic category channels |

Policy 的 global input 由三部分组成：

1. `local_map[:,0:4]`: 当前 local 几何状态。
2. pooled `full_map[:,0:4]`: 下采样的全局几何上下文。
3. `local_map[:,4:]`: local semantic category map。

**Observation:** 语义不是 side note，而是 policy input channel 的一部分。

### 1.4 目标类别如何使用 semantic map

Relevant code:

- `analysis/external_repos/semexp/main.py:500-508`

如果当前 local map 中目标类别 channel 非空，SemExp 会把该类别 channel 二值化为 goal map：

```text
cn = goal_cat_id + 4
if local_map[:, cn].sum() != 0:
    goal_maps = local_map[:, cn] > 0
```

**Observation:** 对于“目标已可见/已投影”的情况，目标 grounding 是直接从 cell-level semantic map 得到，而不是从文字 planner 猜测得到。

### 1.5 SemExp BEV 图中包含哪些元素

Relevant code:

- `analysis/external_repos/semexp/agents/sem_exp.py:338-392`
- `analysis/external_repos/semexp/constants.py:74-94`

可视化流程：

1. 取 `map_pred`、`exp_pred`、`goal`、`sem_map_pred`。
2. `sem_map_pred` 来自 `local_map[e,4:,:,:].argmax(0)`。
3. 对没有 semantic category 的 cell，根据几何状态填为：
   - out-of-bounds / unknown
   - obstacle
   - explored/free
4. visited trajectory 覆盖为固定类别。
5. goal mask 覆盖为固定类别。
6. 用 fixed palette 渲染成 RGB image。
7. 最近邻 resize，保持 raster cell 结构。
8. 与 RGB observation 放在同一可视化画布中。

SemExp semantic BEV 主体元素：

| 元素 | 是否在 semantic raster 主图中 | 表示方式 |
|---|---:|---|
| unknown / out-of-bounds | 是 | 固定背景颜色 |
| obstacle / wall | 是 | 固定颜色 cell |
| explored / free | 是 | 固定颜色 cell |
| object category | 是 | category color 的 cell cluster |
| visited trajectory | 是 | fixed overlay class |
| goal mask | 是 | fixed overlay class |
| object text label | 否 | 不在 dense raster 主体中 |
| branch id / branch score | 否 | SemExp 无此 debug layer |
| side panel warning | 否 | 不进入 semantic map |

---

## 2. PONI 如何一步步构造 semantic BEV

PONI 与 SemExp 的关键差异是：PONI 论文核心使用 offline semantic map / potential function 学习。它不是只依赖在线 Mask R-CNN 投影，而是从 3D scene semantic point cloud / mesh 预生成 ground-truth-like semantic top-down map，再训练 potential function network。

### 2.1 输入：scene-level semantic vertices / point cloud

Relevant code:

- `analysis/external_repos/poni/scripts/create_semantic_maps.py:403-665`

PONI 的 semantic map 生成以整场景几何和语义为输入：

1. 读取每个 scene 的 semantic point cloud / mesh vertices。
2. 按 floor 高度切分楼层。
3. 选择当前 floor 上的 floor、wall、object vertices。
4. 保留每个 vertex 的：
   - 3D position
   - object instance id
   - semantic category id

### 2.2 从 3D vertices 投影成 top-down cell map

Relevant code:

- `analysis/external_repos/poni/scripts/create_semantic_maps.py:501-625`

关键步骤：

1. 根据 floor height 选出当前楼层的 floor / wall / object vertices。
2. 对 floor / wall 的高度做下移，使 object vertices 在同一 top-down cell 内能优先成为最高点。
3. 以固定 resolution 将 x/z 平面离散成 map cell。
4. 对每个 vertex 计算它落入的 `(map_z, map_x)` cell。
5. 对每个 cell 用 `scatter_max` 选择最高 vertex。
6. 最高 vertex 的 semantic id 成为这个 cell 的 `map_semantic`。
7. 最高 vertex 的 object id 成为这个 cell 的 `map_instance`。
8. 同时保存 `map_heights` 和 valid `mask`。

**Observation:** PONI 的语义图是一个真正的 `H x W` integer semantic id map，每个 cell 只对应一个最终 semantic class。

### 2.3 PONI 保存哪些地图数据

Relevant code:

- `analysis/external_repos/poni/scripts/create_semantic_maps.py:638-661`

PONI 保存到 HDF5 的字段包括：

| 字段 | 含义 |
|---|---|
| `wall_sem_id` | wall category id |
| `floor_sem_id` | floor category id |
| `out-of-bounds_sem_id` | out-of-bounds category id |
| `{floor_id}/mask` | 有效 top-down cells |
| `{floor_id}/map_heights` | 每个 cell 的最高点高度 |
| `{floor_id}/map_instance` | 每个 cell 的 object instance id |
| `{floor_id}/map_semantic` | 每个 cell 的 semantic id |
| `{floor_id}/map_semantic_rgb` | palette-rendered semantic map |

**Observation:** PONI 不需要从文字 room hypothesis 画框；它在地图层面已经有 scene-level cell semantic id。

### 2.4 PONI 如何可视化 semantic map

Relevant code:

- `analysis/external_repos/poni/scripts/create_semantic_maps.py:385-400`

可视化流程非常直接：

1. 将 `map_semantic` 转为 int class map。
2. 用 fixed `COLOR_PALETTE` 生成 PIL palette image。
3. 将 class id flatten 后写入 palette image。
4. 转为 RGB。
5. 拼接 legend。

**Observation:** PONI 的可视化是 semantic id map 的直接渲染。它不是在灰色 occupancy 上后加 object label。

### 2.5 PONI 的 potential function 如何用 semantic BEV

Relevant code:

- `analysis/external_repos/poni/poni/dataset.py:83-99`
- `analysis/external_repos/poni/poni/dataset.py:300-350`
- `analysis/external_repos/poni/semexp/model_pf.py:222-259`

数据和模型输入流程：

1. 从 HDF5 读取 `map_semantic`。
2. 转换成 one-hot semantic map。
3. 根据 agent pose 做 crop / rotation，生成 agent-centric partial semantic map。
4. 计算 explored / unknown / frontier / visibility masks。
5. Potential Function model 输入由以下通道组成：
   - free map
   - obstacle map
   - semantic category maps
6. 对目标类别输出 target-conditioned potential field。
7. 运行时可基于 potential field、unexplored mask、distance map 选择下一步搜索位置。

**Observation:** PONI 图中的 “Predicted potential function” 不是普通 heat overlay，而是基于 semantic map 预测的 target-conditioned spatial value field。

### 2.6 PONI BEV 图中包含哪些元素

PONI-style BEV 元素：

| 元素 | 是否在 semantic raster / state 中 | 表示方式 |
|---|---:|---|
| navigable / floor / free | 是 | floor/free semantic or free map channel |
| wall / obstacle | 是 | wall/obstacle category or obstacle map channel |
| out-of-bounds / unknown | 是 | mask / special semantic id |
| object category | 是 | category color cell clusters |
| object instance id | 内部保存 | `map_instance`，不一定直接显示文字 |
| height | 内部保存 | `map_heights` |
| frontier | 用于 potential training/scoring | frontier mask |
| predicted potential field | 是，单独 panel | continuous [0,1] value heatmap |
| branch id / score / warning | 否 | 不属于 semantic BEV raster |

---

## 3. 成熟工作 BEV 的共同实现模式

把 SemExp 和 PONI 放在一起看，可以抽象成下面的标准链路：

```text
semantic evidence
  -> metric projection / map discretization
  -> per-cell semantic state
  -> cumulative map state
  -> policy/planner input channels
  -> palette visualization
```

### 3.1 关键共同点

**Observation:** 两者都避免把 semantic BEV 退化成“中心点 + 文本标签”。

| 维度 | SemExp | PONI |
|---|---|---|
| 语义来源 | 在线 Mask R-CNN instance masks | 离线 scene semantic point cloud / mesh |
| 投影方式 | RGB-D mask -> 3D voxel -> top-down | 3D vertices -> top-down cell |
| map state | `num_sem_categories + 4` channels | `map_semantic` integer map + one-hot |
| object 表达 | semantic category channel 中的 cell cluster | `map_semantic` / `map_instance` cell cluster |
| 可视化 | category argmax + palette | semantic id + palette |
| planner/policy 输入 | map channels | free/obstacle/semantic channels + potential field |
| debug text | 不进入 semantic raster | 不进入 semantic raster |

### 3.2 成熟 BEV 主图通常包含什么

最低元素：

1. Unknown / out-of-bounds
2. Explored / navigable free area
3. Obstacle / wall
4. Semantic object categories as colored cell clusters
5. Agent pose or trajectory overlay（通常很轻）
6. Goal / potential field panel（如果是 target-conditioned planning）
7. Legend / palette

通常不包含：

1. `O#` object labels
2. `R#` room hypothesis boxes
3. `B#` branch labels
4. branch score strings
5. warning text
6. 大 side panel
7. 未定位 room hypothesis 的空间框

---

## 4. SmoothNav 当前如何构造 semantic BEV

SmoothNav 当前有两条相关链路：

1. **原始 SemExp-like map channel 链路**：RGB-D semantic mask 投影到 `full_map[4:]`。
2. **新增 instance-depth footprint 链路**：把 detector instance bbox / mask support 通过 depth/pose 投影成 planner-facing object footprint，作为 source-tagged footprint evidence，不直接改导航 map channel。

### 4.1 语义检测输入：Mask R-CNN 输出 semantic channels 和 bbox

Relevant code:

- `base_UniGoal/src/utils/visualization/semantic_prediction.py:48-60`
- `base_UniGoal/src/agent/unigoal/agent.py:872-965`

流程：

1. Detectron2 对 RGB 做 instance segmentation。
2. 对每个目标类别：
   - 把 `pred_masks[j]` 加到 semantic channel。
   - 保存 detector bbox。
3. `Agent.preprocess_obs()` 将 RGB、depth、semantic channels 拼接成 observation：

```text
state = [rgb, depth, sem_seg_pred]
```

4. SmoothNav 额外保存 `last_semantic_instances`，包含：
   - detector id
   - category index / label
   - confidence
   - mapping-frame bbox
   - semantic seed pixel count

**Observation:** SmoothNav 和 SemExp 在最底层都依赖 Detectron2 per-pixel semantic mask；但 SmoothNav 当前还额外尝试保留 detector bbox 以修复 sparse semantic map 下 object footprint 过小的问题。

### 4.2 原始 BEV map 投影：`full_map[4:]` semantic channels

Relevant code:

- `base_UniGoal/src/map/bev_mapping.py:103-185`

流程与 SemExp 近似：

1. depth -> point cloud。
2. point cloud -> camera / pose transform。
3. semantic channels `obs[:,4:]` 作为 feature。
4. `splat_feat_nd` 到 3D voxel。
5. height projection：
   - channel 0: obstacle / collision map
   - channel 1: explored map
   - channel 4+: semantic category projection
6. 当前 view map 旋转/平移后与历史 local/full map 合并。

SmoothNav map channel contract：

| Channel | 含义 |
|---:|---|
| 0 | obstacle / collision |
| 1 | explored / free |
| 2 | current agent |
| 3 | visited trajectory |
| 4+ | semantic category channels |

**Observation:** 原始 SmoothNav semantic BEV 本质上已经有 SemExp-like 通道设计；当前问题不是完全没有 semantic channels，而是实际 episode/capsule 中这些 channels 往往非常稀疏，无法自然形成成熟工作中那种 object-sized footprint。

### 4.3 新增 instance-depth footprint 链路

Relevant code:

- `base_UniGoal/src/map/bev_mapping.py:444-681`
- `base_UniGoal/src/map/bev_mapping.py:787-812`
- `smoothnav/world_state.py:371-420`
- `smoothnav/semantic_bev.py:2104-2140`

新增链路的目的：当 `full_map[4:]` 只投影出墙边/物体表面少量 semantic pixels 时，仍然能从 detector instance bbox/mask + depth 生成更接近 object footprint 的 BEV 支持区域。

流程：

1. `smoothnav/main.py` 将 agent 保存的 `last_semantic_instances` 写入 `infos["semantic_instances"]`。
2. `BEV_Map.mapping()` 先运行常规 mapping，再调用 `_project_semantic_instance_footprints(...)`。
3. 对每个 instance：
   - 找到 category channel。
   - 在 detector bbox 内检查 semantic mask seed。
   - 如果 bbox 内有 semantic seed，用 `semantic_mask_in_detector_bbox`。
   - 如果没有 seed，则低置信使用 `detector_bbox_depth_fallback`。
4. 将 instance mask / bbox support 作为 feature，经 depth point cloud 和 pose transform 投影到 BEV。
5. 对投影后的 support 取 top-down footprint。
6. 根据 category minimum footprint size 做有限扩张。
7. 尝试把 bbox shift 到 explored non-obstacle interior，避免贴墙/压墙。
8. 输出 source-tagged footprint：
   - `footprint_source = semantic_instance_depth_bbox_cells`
   - `bbox_rc`
   - `raw_surface_bbox_rc`
   - `footprint_rc_indices`
   - `footprint_pixel_count`
   - `seed_source`

**Observation:** 这条链路是 SmoothNav 向成熟 BEV 靠近的关键修复：它开始让 object 在 BEV 中有面积，而不只是中心点标签。但它仍是 deterministic footprint repair，不等同于 PONI 那种 scene-level ground-truth semantic map。

### 4.4 SmoothNav dense semantic raster 构造

Relevant code:

- `smoothnav/semantic_bev.py:51-68`
- `smoothnav/semantic_bev.py:143-183`
- `smoothnav/semantic_bev.py:541-874`
- `smoothnav/semantic_bev.py:877-923`

`build_dense_semantic_raster()` 的当前逻辑：

1. 创建 `class_id_map: H x W`。
2. 用 base geometry 初始化：
   - unknown: dark gray
   - explored/free: light color
   - obstacle/wall: black
3. 读取 `full_map[4:]`：
   - 对每个 cell 取 semantic max。
   - 超过 threshold 才写入 semantic category class id。
   - 统计 `semantic_coverage_ratio`、`true_semantic_pixel_count`、`per_category_pixel_count`。
4. 可选 `semantic_bbox_fill` fallback：
   - 从 sparse semantic channel connected component 生成 bbox。
   - 扩张到 category minimum footprint。
   - source 标记为 `semantic_channel_bbox_fill`。
   - 不把它计入 true semantic coverage。
5. 读取 objects 中的 trusted footprints：
   - `footprint_rc_indices` 优先。
   - 只有可信来源的 bbox 才能转成 footprint cells。
   - trusted source 包括 `semantic_instance_depth_bbox_cells`、`graph_pcd_bbox_cells` 等。
   - center-only / label-only object 不允许画成 dense semantic footprint。
6. 返回 metadata：
   - true semantic pixel count
   - object footprint pixel count
   - semantic bbox fallback pixel count
   - display semantic source
   - rejected object footprints
   - warnings

**Observation:** 当前 renderer 已经区分三类证据：

1. `map_channels`: 真正从 `full_map[4:]` 来的 semantic projection。
2. `graph_object_footprints` / `semantic_instance_depth_bbox_cells`: 有 source-tag 的 object footprint。
3. `semantic_channel_bbox_fill`: old-capsule fallback / diagnostic，不是真 semantic coverage。

### 4.5 SmoothNav dense semantic BEV 如何渲染

Relevant code:

- `smoothnav/semantic_bev.py:955-1075`
- `smoothnav/semantic_bev.py:1078-1105`

`render_dense_semantic_bev()` 的当前规则：

1. 将 `class_id_map` 用 fixed palette 转成 RGB。
2. 可选 crop 到内容区域。
3. 只画很薄的 trajectory / frontier pixels。
4. agent 用绿色 marker。
5. legend 只显示当前出现的 class name 和颜色。
6. 明确禁止在 dense map 主图里画：
   - `O#` object text labels
   - `R#` room hypothesis boxes
   - `B#` frontier branch labels
   - branch scores
   - warning text
   - side panel debug text

`dense_semantic_raster_metadata()` 中硬性记录：

```text
unlocalized_rooms_drawn_count = 0
object_labels_drawn_on_dense_map_count = 0
branch_labels_drawn_on_dense_map_count = 0
```

**Observation:** 当前 dense renderer 的图像规范已经接近成熟工作：主图是 cell raster，不是 debug overlay。剩余风险主要在上游 semantic / footprint evidence 是否足够正确，而不是 renderer 是否能画标签。

### 4.6 SmoothNav capsule / replay 输出哪些 BEV artifact

Relevant code:

- `smoothnav/task_frame_capsule.py:157-245`
- `scripts/replay_semantic_bev_capsules.py:256-322`

当前 capsule / replay 会输出多种图，职责不同：

| Artifact | 作用 | 是否应作为 mature semantic BEV 主图 |
|---|---|---:|
| `dense_semantic_bev.png/json` | 真实 semantic channels + trusted object footprint raster | 是，主检查对象 |
| `semantic_footprint_bev.png/json` | dense raster + old-capsule semantic bbox fallback | 可辅助检查，但要看 source |
| `pseudo_semantic_support_bev.png/json` | object center prior 支持图 | 否，不能冒充 true semantic map |
| `debug_overlay_full.png` / `legacy_bev_annotated_branches.png` | O/R/B labels、branch score、warning 等调试图 | 否 |
| `decision_state_bev_planner.png` | planner decision-state 综合视图 | 否，不等同 dense semantic raster |
| `semantic_field_panel.png` | decision-state semantic panel | 辅助 |
| `target_value_field_panel.png` | target value / potential field panel | 辅助 |

**Observation:** SmoothNav 当前图像体系已经把 debug overlay 和 dense semantic raster 分开；人工审查时不应把 debug 图当成 mature semantic BEV 成果。

---

## 5. SmoothNav 当前 BEV 图中包含哪些内容

### 5.1 `dense_semantic_bev.png`

应该包含：

1. unknown / unexplored cells
2. explored/free cells
3. obstacle/wall cells
4. `full_map[4:]` 中真实语义 category cells
5. trusted object footprints（如果在线 capsule 已保存）
6. thin visited trajectory overlay
7. agent marker
8. compact legend

不应该包含：

1. branch labels
2. object text labels
3. room hypothesis boxes
4. branch score
5. warning text
6. side panel
7. unlocalized room region

### 5.2 `semantic_footprint_bev.png`

应该包含：

1. 与 dense semantic BEV 相同的 base geometry。
2. true semantic cells。
3. trusted object footprints。
4. 额外 old-capsule fallback：`semantic_channel_bbox_fill`。

**Observation:** 如果 JSON 中 `semantic_bbox_footprint_source = semantic_channel_bbox_fill`，这说明图中某些 object-sized patch 是由 sparse semantic channel 的 connected component bbox 扩张得到，不能解释为上游已经产生真实 object footprint。

### 5.3 `debug_overlay_full.png` / legacy annotated BEV

可以包含：

1. O# object labels
2. R# room hypothesis labels
3. B# branch candidates
4. branch scores
5. warning / coordinate diagnostics
6. side panel summaries

**Observation:** 这些信息有调试价值，但它们不符合 SemExp / PONI style semantic raster 的要求。

---

## 6. 当前证据：SmoothNav 与成熟工作之间还差在哪里

### 6.1 已有修复进展

**Observation:** SmoothNav 当前已经完成了从“debug overlay 图”到“dense raster artifact”的结构性拆分：

- `dense_semantic_bev.png` 不再画 O/R/B labels。
- true semantic coverage 与 fallback / footprint coverage 分开统计。
- 未定位 room hypothesis 不画成空间区域。
- 新增 online instance-depth footprint path，把 detector instance 通过 depth/pose 投影到 BEV footprint。

### 6.2 旧 capsule 上的限制

**Observation:** 在旧 s22b step145 replay 中，旧 capsule 没有保存新的 online detector instance footprint，因此只能验证 fallback renderer，不能验证新的 instance-depth online path。

Recorded old-capsule metrics from `analysis/s35_bev_footprint_repair_review_20260429/step_000145/semantic_footprint_bev.json`:

```json
{
  "true_semantic_pixel_count": 50,
  "object_footprint_pixel_count": 0,
  "semantic_bbox_footprint_pixel_count": 1213,
  "semantic_bbox_footprint_source": "semantic_channel_bbox_fill",
  "display_semantic_source": "map_channels+semantic_channel_bbox_fill"
}
```

Interpretation:

- `true_semantic_pixel_count=50`：真实 semantic channel 非常稀疏。
- `object_footprint_pixel_count=0`：旧 capsule 没有可用的 trusted object footprint。
- `semantic_bbox_footprint_pixel_count=1213`：图中的大 patch 来自 fallback bbox fill，不是 mature-work-style ground-truth semantic projection。

### 6.3 新 online path 的 smoke evidence

**Observation:** Synthetic BEV_Map projection smoke 产生了一个 chair footprint：

- Artifact: `analysis/s35_bev_footprint_repair_review_20260429/synthetic_instance_projection.json`
- Result: one `chair` footprint, `footprint_pixel_count=136`, `bbox_rc=[231,252,247,259]`

**Hypothesis:** 下一次在线 capsule 如果 `semantic_instance_footprints` 非空，`dense_semantic_bev.png` 应能显示 source-tagged object footprint patch；如果仍然很稀疏或贴墙，则问题应继续定位到 depth projection / pose frame / bbox-to-footprint expansion / interior shift，而不是 prompt。

---

## 7. Side-by-side 对比表

| 维度 | SemExp | PONI | SmoothNav 当前 |
|---|---|---|---|
| 语义来源 | 在线 Mask R-CNN masks | 离线 scene semantic vertices / mesh | 在线 Mask R-CNN masks + detector bbox records |
| 投影核心 | depth point cloud + voxel splat | x/z discretization + highest vertex per cell | depth point cloud + voxel splat；另有 instance-depth bbox/seed footprint path |
| 语义状态 | `local_map[:,4:]` semantic channels | `map_semantic` integer id + one-hot | `full_map[4:]` semantic channels；新增 `semantic_instance_footprints` |
| object 空间表达 | semantic channel cell clusters | semantic / instance id cell clusters | true semantic cells 通常稀疏；新增 trusted footprint cells 改善 object area |
| 可视化主图 | palette raster | palette raster + potential field panel | `dense_semantic_bev.png` raster；debug overlay 分离 |
| text labels | 不进入 semantic raster | 不进入 semantic raster | dense map 禁止；debug 图保留 |
| room hypothesis | 不用文字 room box 做语义图 | scene semantic map 内隐含空间功能 | unlocalized room 不画；localized room 目前主要在 decision/debug 层 |
| target value | goal semantic map / policy | learned potential function | 已有 target value panel，但本文件重点是 semantic BEV |
| 当前主要风险 | segmentation / projection noise | offline map dependency | semantic channels 稀疏；instance footprint online evidence 需要真实 episode 验证 |

---

## 8. 对下一轮审查最有用的检查项

为了判断 SmoothNav 是否真正接近成熟工作 BEV，而不是只是“看起来有 patch”，下一轮建议按下面顺序检查：

1. `dense_semantic_bev.json`
   - `true_semantic_pixel_count`
   - `semantic_coverage_ratio`
   - `per_category_pixel_count`
   - `semantic_source`
2. `semantic_instance_footprints.json`
   - `projected_instance_count`
   - each instance `footprint_source`
   - `footprint_pixel_count`
   - `bbox_rc`
   - `raw_surface_bbox_rc`
   - `seed_source`
3. `object_footprints.json`
   - 是否有 nonzero trusted `footprint_rc_indices`
   - 是否存在 rejected object footprints
4. 图像人工检查：
   - object patch 是否在 explored/free interior，而不是墙面边缘。
   - patch 尺寸是否接近该类别物理 footprint。
   - patch 类别颜色是否与 legend 一致。
   - 是否没有 O/R/B labels 污染 dense map。
5. 如果 patch 错位：
   - 优先检查 pose frame / full-map rc offset / local-map boundary。
6. 如果 patch 贴墙：
   - 优先检查 semantic mask 是否只来自物体可见表面；检查 `_shift_bbox_to_local_interior` 和 obstacle/explored masks。
7. 如果物体数量极少：
   - 检查 detector output 数量、category filter、confidence threshold、`last_semantic_instances` 是否写入 `infos`。

---

## 9. 结论

**Claim:** SemExp / PONI 的 BEV 图之所以“成熟”，不是因为绘制了更多文本，而是因为它们把语义变成了 top-down cell-level state：类别直接占据地图空间。

Evidence:

- SemExp：`obs[:,4:]` semantic mask channels 经 RGB-D voxel projection 写入 `local_map[:,4:]`，可视化时由 `argmax` 生成 dense semantic raster。
- PONI：scene semantic vertices 被离散到 `map_semantic`，每个 cell 保存 semantic id，并进一步作为 potential function network 的输入。

**Claim:** SmoothNav 当前的正确路线是继续强化上游 object footprint / semantic projection，而不是把更多 label 或 prompt 文字塞进 BEV 图。

Evidence:

- 当前 `dense_semantic_bev.png` 已经按 raster-only 规则渲染，问题主要转移到 `full_map[4:]` 是否足够稠密，以及 `semantic_instance_depth_bbox_cells` 是否在在线 episode 中稳定地产生正确 footprint。

**Hypothesis:** 若下一轮真实在线 capsule 中 `semantic_instance_footprints` 非空且 footprint 在 explored interior 内与物体尺寸一致，SmoothNav 的 BEV 输入会比旧的 debug overlay 明显更接近 SemExp / PONI 的 decision-state semantic map；若仍贴墙/错位/数量过少，下一步应优先修复 projection frame 和 instance evidence persistence，而不是修改 MLLM planner prompt。
