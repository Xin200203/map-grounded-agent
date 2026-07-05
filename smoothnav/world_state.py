"""World-state assembly for SmoothNav's Layer 0 world model."""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from smoothnav.target_matching import score_caption_against_goal
from smoothnav.types import WorldState


def _safe_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _graph_map_size(graph) -> Optional[int]:
    for key in ("map_size", "global_width", "global_height"):
        value = getattr(graph, key, None)
        try:
            if value is not None:
                return int(value)
        except Exception:
            pass
    args = getattr(graph, "args", None)
    for key in ("map_size", "global_width", "global_height"):
        value = getattr(args, key, None)
        try:
            if value is not None:
                return int(value)
        except Exception:
            pass
    return None


def _graph_map_resolution(graph) -> float:
    for owner in (graph, getattr(graph, "args", None)):
        value = getattr(owner, "map_resolution", None)
        try:
            if value is not None and float(value) > 0:
                return float(value)
        except Exception:
            pass
    return 5.0


def _points_to_map_rc(points, *, map_size: Optional[int], map_resolution: float) -> Optional[np.ndarray]:
    """Project graph-world x/y points into the canonical full-map BEV rc frame."""

    try:
        arr = np.asarray(points, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 2:
        return None
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[finite]
    if arr.size == 0:
        return None
    scale = 100.0 / float(map_resolution or 5.0)
    # BEV_Map writes the full_map in image row/col order with
    #   row = world_y / resolution, col = world_x / resolution.
    # Graph node.center uses a different graph/FMM convention
    #   [world_x / res, map_size - 1 - world_y / res].
    rows = arr[:, 1] * scale
    cols = arr[:, 0] * scale
    rc = np.stack([rows, cols], axis=1)
    if map_size is not None:
        in_bounds = (
            (rc[:, 0] >= 0)
            & (rc[:, 0] < map_size)
            & (rc[:, 1] >= 0)
            & (rc[:, 1] < map_size)
        )
        rc = rc[in_bounds]
    return rc if rc.size else None


def _graph_center_to_full_map_rc(center, *, map_size: Optional[int]) -> Optional[List[int]]:
    if center is None:
        return None
    try:
        if hasattr(center, "tolist"):
            center = center.tolist()
        if len(center) < 2:
            return None
        graph_r = int(round(float(center[0])))
        graph_c = int(round(float(center[1])))
    except Exception:
        return None
    if map_size is None:
        return [graph_r, graph_c]
    return [int(map_size - 1 - graph_c), int(graph_r)]


def _clip_bbox_rc(bbox: List[int], map_size: Optional[int]) -> Optional[List[int]]:
    if len(bbox) != 4:
        return None
    r0, c0, r1, c1 = [int(v) for v in bbox]
    if map_size is not None:
        r0 = max(0, min(map_size - 1, r0))
        r1 = max(0, min(map_size - 1, r1))
        c0 = max(0, min(map_size - 1, c0))
        c1 = max(0, min(map_size - 1, c1))
    if r1 < r0:
        r0, r1 = r1, r0
    if c1 < c0:
        c0, c1 = c1, c0
    if r1 < r0 or c1 < c0:
        return None
    return [r0, c0, r1, c1]


def _bbox_from_projected_rc(rc: np.ndarray, *, map_size: Optional[int], pad: int = 2, min_extent: int = 3) -> Optional[List[int]]:
    if rc is None or rc.size == 0:
        return None
    r0 = int(math.floor(float(np.min(rc[:, 0])))) - int(pad)
    c0 = int(math.floor(float(np.min(rc[:, 1])))) - int(pad)
    r1 = int(math.ceil(float(np.max(rc[:, 0])))) + int(pad)
    c1 = int(math.ceil(float(np.max(rc[:, 1])))) + int(pad)
    if r1 - r0 < min_extent:
        extra = int(math.ceil((min_extent - (r1 - r0)) / 2.0))
        r0 -= extra
        r1 += extra
    if c1 - c0 < min_extent:
        extra = int(math.ceil((min_extent - (c1 - c0)) / 2.0))
        c0 -= extra
        c1 += extra
    return _clip_bbox_rc([r0, c0, r1, c1], map_size)


def _bbox_cells(bbox_rc: Optional[List[int]], *, map_size: Optional[int], max_cells: int = 12000) -> List[List[int]]:
    if bbox_rc is None:
        return []
    clipped = _clip_bbox_rc(list(bbox_rc), map_size)
    if clipped is None:
        return []
    r0, c0, r1, c1 = [int(v) for v in clipped]
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    if area <= 0 or area > int(max_cells):
        return []
    return [[int(r), int(c)] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _unique_projected_cells(rc: Optional[np.ndarray], *, map_size: Optional[int]) -> List[List[int]]:
    if rc is None or rc.size == 0:
        return []
    cells = set()
    for row, col in np.asarray(rc, dtype=float)[:, :2]:
        if not (np.isfinite(row) and np.isfinite(col)):
            continue
        r = int(round(float(row)))
        c = int(round(float(col)))
        if map_size is not None and not (0 <= r < int(map_size) and 0 <= c < int(map_size)):
            continue
        cells.add((r, c))
    return [[int(r), int(c)] for r, c in sorted(cells)]


def _object_bev_footprint(node, graph) -> Dict[str, Any]:
    """Summarize an object's BEV footprint from graph pcd/bbox without labels.

    Mature BEV maps paint projected cells.  We therefore serialize compact BEV
    footprint cells derived from graph point-cloud / 3-D bbox support instead of
    leaving replay to infer object extent from a center label.
    """

    obj = getattr(node, "object", None) or {}
    map_size = _graph_map_size(graph)
    map_resolution = _graph_map_resolution(graph)
    points = None
    try:
        pcd = obj.get("pcd") if isinstance(obj, dict) else None
        points = np.asarray(pcd.points, dtype=float) if pcd is not None else None
    except Exception:
        points = None
    rc = _points_to_map_rc(points, map_size=map_size, map_resolution=map_resolution) if points is not None else None
    source = ""
    seed_cells = _unique_projected_cells(rc, map_size=map_size) if rc is not None else []
    if rc is None:
        try:
            bbox_obj = obj.get("bbox") if isinstance(obj, dict) else None
            if bbox_obj is not None and hasattr(bbox_obj, "get_box_points"):
                rc = _points_to_map_rc(
                    np.asarray(bbox_obj.get_box_points(), dtype=float),
                    map_size=map_size,
                    map_resolution=map_resolution,
                )
                source = "graph_3d_bbox_cells"
        except Exception:
            rc = None
    else:
        source = "graph_pcd_bbox_cells"
    bbox_rc = _bbox_from_projected_rc(rc, map_size=map_size) if rc is not None else None
    footprint_cells = _bbox_cells(bbox_rc, map_size=map_size)
    rejected_reason = ""
    if bbox_rc is not None and not footprint_cells:
        rejected_reason = "footprint_bbox_too_large_or_empty"
    pixel_count = int(len(footprint_cells))
    detection_xyxy_count = 0
    pixel_area_sum = 0
    try:
        detection_xyxy_count = len(obj.get("xyxy", []) or []) if isinstance(obj, dict) else 0
        pixel_area_sum = int(sum(float(v) for v in (obj.get("pixel_area", []) or []))) if isinstance(obj, dict) else 0
    except Exception:
        pass
    return {
        "bbox_rc": bbox_rc,
        "footprint_source": source if bbox_rc is not None else "",
        "footprint_pixel_count": pixel_count,
        "footprint_confidence": 0.75 if source == "graph_pcd_bbox_cells" else (0.60 if source == "graph_3d_bbox_cells" else 0.0),
        "footprint_rc_indices": footprint_cells,
        "footprint_encoding": "rc_indices_from_projected_bbox" if footprint_cells else "",
        "footprint_seed_cell_count": int(len(seed_cells)),
        "pcd_point_count": int(len(points)) if points is not None and getattr(points, "ndim", 0) == 2 else 0,
        "footprint_rejected_reason": rejected_reason,
        "detection_xyxy_count": int(detection_xyxy_count),
        "detection_pixel_area_sum": int(pixel_area_sum),
    }


def summarize_frontiers(graph, limit: int = 8) -> List[Dict[str, Any]]:
    frontiers = getattr(graph, "frontier_locations_16", None)
    if frontiers is None:
        frontiers = getattr(graph, "frontier_locations", None)
    result = []
    for idx, coord in enumerate(_safe_list(frontiers)[:limit]):
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            result.append({"rank": idx + 1, "coord": [int(coord[0]), int(coord[1])]})
    return result


def summarize_rooms(graph, limit: int = 20) -> List[Dict[str, Any]]:
    rooms = []
    map_size = _graph_map_size(graph)
    for idx, room_node in enumerate(getattr(graph, "room_nodes", [])[:limit], start=1):
        nodes = list(getattr(room_node, "nodes", []) or [])
        localized_centers = []
        for node in nodes:
            center = getattr(node, "center", None)
            if center is not None and len(center) >= 2:
                converted = _graph_center_to_full_map_rc(center, map_size=map_size)
                if converted is not None:
                    localized_centers.append(converted)
        center_rc = None
        bbox_rc = None
        if localized_centers:
            rows = [coord[0] for coord in localized_centers]
            cols = [coord[1] for coord in localized_centers]
            center_rc = [
                int(round(sum(rows) / float(len(rows)))),
                int(round(sum(cols) / float(len(cols)))),
            ]
            pad = 10
            bbox_rc = [
                int(min(rows) - pad),
                int(min(cols) - pad),
                int(max(rows) + pad),
                int(max(cols) + pad),
            ]
        rooms.append(
            {
                "id": f"R{idx:03d}",
                "caption": getattr(room_node, "caption", ""),
                "object_count": len(nodes),
                "objects": [
                    getattr(node, "caption", "")
                    for node in nodes[:10]
                    if getattr(node, "caption", "")
                ],
                "center_rc": center_rc,
                "bbox_rc": bbox_rc,
                "geometry_type": (
                    "member_object_hypothesis"
                    if center_rc is not None
                    else "unlocalized_hypothesis"
                ),
                "confidence": (
                    "medium"
                    if len(localized_centers) >= 2
                    else ("low" if len(localized_centers) == 1 else "unlocalized")
                ),
                "source": "graph.room_nodes",
            }
        )
    return rooms


def summarize_objects(graph, limit: int = 30) -> List[Dict[str, Any]]:
    objects = []
    goal_text = (
        getattr(graph, "text_goal", None)
        or getattr(graph, "obj_goal", None)
        or ""
    )
    for idx, node in enumerate(getattr(graph, "nodes", [])[:limit], start=1):
        center = getattr(node, "center", None)
        caption = getattr(node, "caption", "")
        match = score_caption_against_goal(caption, goal_text)
        detections = None
        try:
            detections = int(getattr(node, "object", {}).get("num_detections", 0) or 0)
        except Exception:
            detections = None
        footprint = _object_bev_footprint(node, graph)
        graph_center = (
            [int(center[0]), int(center[1])]
            if center is not None and len(center) >= 2
            else None
        )
        center_rc = _graph_center_to_full_map_rc(graph_center, map_size=_graph_map_size(graph))
        objects.append(
            {
                "id": f"O{idx:03d}",
                "node_index": idx - 1,
                "caption": caption,
                "center": center_rc,
                "center_rc": center_rc,
                "graph_center_rc": graph_center,
                "coordinate_frame": "full_map_rc_unflipped",
                "num_detections": detections,
                "target_relevance": match["score"],
                "target_match_reason": match["reason"],
                "target_primary_categories": match["primary_categories"],
                "source": "graph.nodes",
                "bbox_rc": footprint["bbox_rc"],
                "footprint_source": footprint["footprint_source"],
                "footprint_pixel_count": footprint["footprint_pixel_count"],
                "footprint_confidence": footprint["footprint_confidence"],
                "footprint_rc_indices": footprint["footprint_rc_indices"],
                "footprint_encoding": footprint["footprint_encoding"],
                "footprint_seed_cell_count": footprint["footprint_seed_cell_count"],
                "pcd_point_count": footprint["pcd_point_count"],
                "footprint_rejected_reason": footprint["footprint_rejected_reason"],
                "detection_xyxy_count": footprint["detection_xyxy_count"],
                "detection_pixel_area_sum": footprint["detection_pixel_area_sum"],
                "confidence": min(
                    1.0,
                    0.25
                    + 0.15 * float(detections or 0)
                    + 0.40 * float(match["score"] or 0.0),
                ),
            }
        )
    return objects


def summarize_visible_targets(agent=None) -> List[Dict[str, Any]]:
    if agent is None:
        return []
    override = getattr(agent, "last_override_info", {}) or {}
    if not override.get("visible_target_override"):
        return []
    goal = override.get("adopted_goal_summary")
    return [
        {
            "source": override.get("adopted_goal_source", "visible_target"),
            "goal": goal,
        }
    ]


def summarize_semantic_instance_footprints(bev_map, limit: int = 30) -> List[Dict[str, Any]]:
    """Expose detector-depth BEV footprints as localized semantic objects."""

    if bev_map is None or not hasattr(bev_map, "semantic_instance_footprints"):
        return []
    try:
        footprints = list(bev_map.semantic_instance_footprints(env_idx=0) or [])
    except Exception:
        footprints = []
    out = []
    for idx, item in enumerate(footprints[:limit], start=1):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_rc")
        center_rc = None
        try:
            if bbox is not None and len(bbox) >= 4:
                center_rc = [
                    int(round((float(bbox[0]) + float(bbox[2])) / 2.0)),
                    int(round((float(bbox[1]) + float(bbox[3])) / 2.0)),
                ]
        except Exception:
            center_rc = None
        out.append(
            {
                "id": str(item.get("id") or f"SI{idx:03d}"),
                "node_index": None,
                "caption": str(item.get("caption") or item.get("category_label") or "object"),
                "center": center_rc,
                "center_rc": center_rc,
                "coordinate_frame": "full_map_rc_unflipped",
                "num_detections": 1,
                "target_relevance": 0.0,
                "target_match_reason": "",
                "target_primary_categories": [],
                "source": "bev_map.semantic_instance_footprints",
                "bbox_rc": item.get("bbox_rc"),
                "raw_surface_bbox_rc": item.get("raw_surface_bbox_rc"),
                "footprint_source": item.get("footprint_source") or "semantic_instance_depth_bbox_cells",
                "footprint_pixel_count": int(item.get("footprint_pixel_count", 0) or 0),
                "footprint_confidence": float(item.get("footprint_confidence", 0.0) or 0.0),
                "footprint_rc_indices": list(item.get("footprint_rc_indices") or []),
                "footprint_encoding": str(item.get("footprint_encoding", "") or ""),
                "footprint_seed_cell_count": int(item.get("footprint_seed_cell_count", 0) or 0),
                "pcd_point_count": 0,
                "footprint_rejected_reason": "",
                "detection_xyxy_count": 1,
                "detection_pixel_area_sum": 0,
                "seed_source": item.get("seed_source", ""),
                "confidence": float(item.get("footprint_confidence", item.get("confidence", 0.0)) or 0.0),
            }
        )
    return out


def build_world_state(
    *,
    step_idx: int,
    world_epoch: int = 0,
    pose,
    local_pose,
    graph,
    bev_map,
    controller_state,
    graph_delta,
    agent=None,
) -> WorldState:
    frontier_summary = summarize_frontiers(graph)
    frontier_locations = getattr(graph, "frontier_locations_16", None)
    if frontier_locations is None:
        frontier_locations = getattr(graph, "frontier_locations", [])
    semantic_projection_summary = {
        "available": False,
        "reason": "bev_map_missing_semantic_projection_summary",
    }
    try:
        if bev_map is not None and hasattr(bev_map, "semantic_projection_summary"):
            semantic_projection_summary = dict(bev_map.semantic_projection_summary(env_idx=0) or {})
            semantic_projection_summary.setdefault("available", True)
        elif bev_map is not None:
            semantic_projection_summary["reason"] = "bev_map_has_no_semantic_projection_summary_method"
    except Exception as exc:
        semantic_projection_summary = {
            "available": False,
            "reason": "semantic_projection_summary_error",
            "error": str(exc),
        }
    semantic_instance_footprints = summarize_semantic_instance_footprints(bev_map)
    semantic_projection_summary["semantic_instance_footprint_count"] = int(
        len(semantic_instance_footprints)
    )
    return WorldState(
        step_idx=int(step_idx),
        world_epoch=int(world_epoch),
        pose=pose,
        local_pose=local_pose,
        graph=graph,
        bev_map=bev_map,
        explored_regions=list(getattr(controller_state, "explored_regions", []) or []),
        frontier_count=len(_safe_list(frontier_locations)),
        frontier_summary=frontier_summary,
        room_summary=summarize_rooms(graph),
        object_summary=summarize_objects(graph),
        visible_targets=summarize_visible_targets(agent),
        visible_target_summary=summarize_visible_targets(agent),
        semantic_projection_summary=semantic_projection_summary,
        semantic_instance_footprints=semantic_instance_footprints,
        stuck_signal=bool(getattr(graph_delta, "stuck", False)),
        no_progress_steps=int(getattr(controller_state, "no_progress_steps", 0) or 0),
        graph_delta=graph_delta,
    )


class WorldStateBuilder:
    """Single writer for monotonic WorldState epochs."""

    def __init__(self):
        self.world_epoch = 0
        self._last_signature = None

    def reset(self) -> None:
        self.world_epoch = 0
        self._last_signature = None

    def build(self, **kwargs) -> WorldState:
        graph = kwargs.get("graph")
        graph_delta = kwargs.get("graph_delta")
        frontier_locations = getattr(graph, "frontier_locations_16", None)
        if frontier_locations is None:
            frontier_locations = getattr(graph, "frontier_locations", [])
        signature = (
            len(getattr(graph, "nodes", []) or []),
            len(_safe_list(frontier_locations)),
            tuple(getattr(graph_delta, "event_types", []) or []),
            bool(getattr(graph_delta, "stuck", False)),
            bool(getattr(graph_delta, "frontier_reached", False)),
        )
        if self._last_signature is None or signature != self._last_signature:
            self.world_epoch += 1
            self._last_signature = signature
        kwargs["world_epoch"] = self.world_epoch
        return build_world_state(**kwargs)
