"""Planner decision-state BEV construction for SmoothNav.

This module is the structured state builder behind the MLLM frontier planner.
It deliberately does more than draw a nicer image: it turns the current BEV map,
graph/object summaries, room hypotheses, and frontier branches into a replayable
decision state with separated geometry, semantic, target-value, and branch
candidate fields.

The implementation is deterministic and heuristic ("PONI-lite / VLFM-lite").
It does **not** train or import a learned potential network.  It also does not
pretend sparse graph/object hints are dense semantics: every pseudo-dense field
has a source and confidence label.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DECISION_STATE_SCHEMA_VERSION = "smoothnav.bev_decision_state.v1"
TARGET_VALUE_SCHEMA_VERSION = "smoothnav.target_value_field.v1"
SEMANTIC_FIELD_SCHEMA_VERSION = "smoothnav.semantic_field.v1"
BRANCH_TABLE_SCHEMA_VERSION = "smoothnav.branch_decision_table.v2"
CANONICAL_FRAME_NAME = "full_map_rc_unflipped"

SEMANTIC_CHANNEL_OFFSET = 4
SEMANTIC_CONFIDENT_THRESHOLD = 0.5
SEMANTIC_CATEGORY_NAMES = (
    "chair",
    "couch",
    "potted plant",
    "bed",
    "toilet",
    "tv",
    "dining-table",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "cup",
    "bottle",
    "other",
)

EVIDENCE_LEVELS = ("direct", "anchor", "room_prior", "geometry_only")
_EVIDENCE_RANK = {level: idx for idx, level in enumerate(EVIDENCE_LEVELS)}

TRUSTED_FOOTPRINT_SOURCES = {
    "graph_pcd_bbox_cells",
    "graph_pcd_cells",
    "graph_3d_bbox_cells",
    "graph_3d_bbox",
    "graph_object_bbox",
    "semantic_instance_depth_bbox_cells",
}

TARGET_ANCHOR_PRIORS = {
    "tv": {
        "couch": 0.90,
        "chair": 0.50,
        "dining-table": 0.35,
        "bed": 0.30,
        "book": 0.20,
        "vase": 0.15,
        "clock": 0.15,
        "cabinet": 0.18,
    },
    "couch": {"tv": 0.60, "chair": 0.45, "dining-table": 0.30},
    "chair": {"dining-table": 0.55, "couch": 0.45, "tv": 0.30},
    "bed": {"tv": 0.25, "chair": 0.20},
    "dining-table": {"chair": 0.75, "couch": 0.20},
    "sink": {"refrigerator": 0.55, "oven": 0.50, "dining-table": 0.25},
    "refrigerator": {"sink": 0.50, "oven": 0.45, "dining-table": 0.25},
    "oven": {"sink": 0.55, "refrigerator": 0.45, "dining-table": 0.25},
    "toilet": {"sink": 0.45},
}


def evidence_rank(level: Any) -> int:
    return _EVIDENCE_RANK.get(str(level or "geometry_only"), _EVIDENCE_RANK["geometry_only"])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _coord_rc(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("canonical_coord_rc") or value.get("canonical_representative_coord_rc") or value.get("center_rc") or value.get("coord_rc") or value.get("representative_coord") or value.get("center") or value.get("coord")
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if len(value) < 2:
            return None
        return [_safe_int(value[0]), _safe_int(value[1])]
    except Exception:
        return None


def _map_to_numpy(map_like: Any) -> Optional[np.ndarray]:
    if map_like is None:
        return None
    try:
        if hasattr(map_like, "detach"):
            map_like = map_like.detach().cpu().numpy()
        arr = np.asarray(map_like)
    except Exception:
        return None
    if arr.dtype == object:
        return None
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def _map_shape(map_like: Any) -> Optional[Tuple[int, int]]:
    arr = _map_to_numpy(map_like)
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 3:
        return int(arr.shape[-2]), int(arr.shape[-1])
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    return None


def _map_layers(map_like: Any) -> Dict[str, Any]:
    arr = _map_to_numpy(map_like)
    if arr is None or arr.size == 0:
        return {"arr": None, "semantic_channels": None}
    if arr.ndim != 3:
        return {"arr": arr, "semantic_channels": None}
    known = np.any(arr > 0, axis=0)
    obstacle = arr[0] > 0 if arr.shape[0] > 0 else np.zeros(arr.shape[-2:], dtype=bool)
    explored = arr[1] > 0 if arr.shape[0] > 1 else np.logical_and(known, ~obstacle)
    visited = arr[3] > 0 if arr.shape[0] > 3 else np.zeros(arr.shape[-2:], dtype=bool)
    semantic = arr[SEMANTIC_CHANNEL_OFFSET:] if arr.shape[0] > SEMANTIC_CHANNEL_OFFSET else None
    return {
        "arr": arr,
        "known": known.astype(bool),
        "obstacle": obstacle.astype(bool),
        "free": explored.astype(bool),
        "explored": known.astype(bool),
        "unknown": (~known).astype(bool),
        "visited": visited.astype(bool),
        "semantic_channels": semantic,
    }


def _normalize_field(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / peak, 0.0, 1.0).astype(np.float32)


def _add_radial(field: np.ndarray, center: Any, *, weight: float, radius: int) -> None:
    rc = _coord_rc(center)
    if rc is None:
        return
    row, col = int(rc[0]), int(rc[1])
    h, w = field.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        return
    weight = _clip01(weight)
    if weight <= 0:
        return
    radius = max(1, int(radius))
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    dist = np.sqrt((yy - row) ** 2 + (xx - col) ** 2)
    blob = np.maximum(0.0, 1.0 - dist / float(radius)) * float(weight)
    field[r0:r1, c0:c1] = np.maximum(field[r0:r1, c0:c1], blob)


def _field_stats(field: Optional[np.ndarray]) -> Dict[str, Any]:
    if field is None:
        return {"available": False, "positive_pixel_count": 0, "peak": 0.0, "mean_positive": 0.0}
    arr = np.asarray(field, dtype=np.float32)
    positive = arr > 0
    return {
        "available": True,
        "shape": [int(arr.shape[0]), int(arr.shape[1])] if arr.ndim >= 2 else list(arr.shape),
        "positive_pixel_count": int(np.count_nonzero(positive)),
        "peak": float(arr.max()) if arr.size else 0.0,
        "mean_positive": float(arr[positive].mean()) if np.any(positive) else 0.0,
    }


def _category_name(index: int) -> str:
    idx = int(index)
    if 0 <= idx < len(SEMANTIC_CATEGORY_NAMES):
        return SEMANTIC_CATEGORY_NAMES[idx]
    return f"semantic_{idx}"


def _normalize_label(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _semantic_channel_index(label: Any) -> Optional[int]:
    wanted = _normalize_label(label)
    if not wanted:
        return None
    synonyms = {
        "tv": {"tv", "television", "monitor", "screen"},
        "couch": {"couch", "sofa", "loveseat"},
        "potted plant": {"potted plant", "plant"},
        "dining-table": {"dining table", "dining-table", "table", "desk"},
        "refrigerator": {"refrigerator", "fridge"},
        "oven": {"oven", "stove"},
        "chair": {"chair", "seat"},
        "bed": {"bed"},
        "toilet": {"toilet"},
        "sink": {"sink", "basin"},
        "book": {"book", "bookshelf"},
    }
    for idx, name in enumerate(SEMANTIC_CATEGORY_NAMES):
        candidates = {_normalize_label(name)}
        candidates.update(_normalize_label(item) for item in synonyms.get(name, set()))
        if wanted in candidates:
            return idx
    return None


def _target_indices(target_description: str, objects: Sequence[Mapping[str, Any]]) -> List[int]:
    labels: List[str] = []
    text = str(target_description or "").lower()
    for name in SEMANTIC_CATEGORY_NAMES:
        if _normalize_label(name) and _normalize_label(name) in _normalize_label(text):
            labels.append(name)
    if "television" in text or "tv" in text:
        labels.append("tv")
    for obj in objects or []:
        for label in obj.get("target_primary_categories", []) or []:
            labels.append(str(label))
        if _safe_float(obj.get("target_relevance"), 0.0) >= 0.75:
            labels.append(str(obj.get("caption") or ""))
    out: List[int] = []
    for label in labels:
        idx = _semantic_channel_index(label)
        if idx is not None and idx not in out:
            out.append(idx)
    return out


def _target_prior_weights(target_indices: Sequence[int]) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    for idx in target_indices:
        target = _category_name(int(idx))
        weights[int(idx)] = max(weights.get(int(idx), 0.0), 1.0)
        for anchor_label, weight in (TARGET_ANCHOR_PRIORS.get(target) or {}).items():
            anchor_idx = _semantic_channel_index(anchor_label)
            if anchor_idx is not None:
                weights[int(anchor_idx)] = max(weights.get(int(anchor_idx), 0.0), float(weight))
    return weights


def _direction_from_agent(coord: Any, agent_rc: Any) -> str:
    center = _coord_rc(coord)
    agent = _coord_rc(agent_rc)
    if center is None or agent is None:
        return ""
    dr = float(center[0]) - float(agent[0])
    dc = float(center[1]) - float(agent[1])
    if abs(dr) >= abs(dc):
        return "south" if dr > 0 else "north"
    return "east" if dc > 0 else "west"


def _distance(a: Any, b: Any) -> Optional[float]:
    aa = _coord_rc(a)
    bb = _coord_rc(b)
    if aa is None or bb is None:
        return None
    try:
        return float(math.dist([float(aa[0]), float(aa[1])], [float(bb[0]), float(bb[1])]))
    except Exception:
        return None


def _mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
    if mask is None or not np.any(mask):
        return None
    rr, cc = np.where(mask)
    return [int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())]


def _bbox_rc(value: Any, shape: Optional[Tuple[int, int]] = None) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("bbox_rc") or value.get("bev_bbox_rc") or value.get("map_bbox_rc") or value.get("bbox")
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if len(value) != 4:
            return None
        vals = [_safe_int(v) for v in value]
    except Exception:
        return None
    r0, c0, r1, c1 = [int(v) for v in vals]
    if r1 < r0:
        r0, r1 = r1, r0
    if c1 < c0:
        c0, c1 = c1, c0
    if shape is not None:
        h, w = int(shape[0]), int(shape[1])
        r0 = max(0, min(h - 1, r0))
        r1 = max(0, min(h - 1, r1))
        c0 = max(0, min(w - 1, c0))
        c1 = max(0, min(w - 1, c1))
    if r1 < r0 or c1 < c0:
        return None
    return [r0, c0, r1, c1]


def _mask_components(mask: np.ndarray) -> List[np.ndarray]:
    if mask is None or not np.any(mask):
        return []
    coords = np.column_stack(np.where(mask))
    if int(coords.shape[0]) > 20000:
        return [coords.astype(int)]
    remaining = {(int(r), int(c)) for r, c in coords}
    out: List[List[Tuple[int, int]]] = []
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    while remaining:
        start = remaining.pop()
        queue = [start]
        comp = [start]
        while queue:
            row, col = queue.pop()
            for dr, dc in neighbors:
                nxt = (row + dr, col + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
                    comp.append(nxt)
        out.append(comp)
    return [np.asarray(comp, dtype=int) for comp in out]


def _object_footprint_mask(
    obj: Mapping[str, Any],
    *,
    shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Return trusted object footprint cells without inventing center blobs.

    This accepts only serialized BEV cells or a bbox that explicitly came from a
    3-D/depth projection path.  A graph/object center by itself is deliberately
    not enough to create a dense footprint.
    """

    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    source = str(obj.get("footprint_source") or "").strip()
    cells = (
        obj.get("footprint_rc_indices")
        or obj.get("footprint_rcs")
        or obj.get("footprint_cells_rc")
        or obj.get("bev_footprint_rc_indices")
        or []
    )
    try:
        if hasattr(cells, "tolist"):
            cells = cells.tolist()
        for coord in cells or []:
            if hasattr(coord, "tolist"):
                coord = coord.tolist()
            if not isinstance(coord, Sequence) or isinstance(coord, (str, bytes)) or len(coord) < 2:
                continue
            row, col = _safe_int(coord[0]), _safe_int(coord[1])
            if 0 <= row < h and 0 <= col < w:
                mask[int(row), int(col)] = True
    except Exception:
        mask[:, :] = False
    bbox = _bbox_rc(obj.get("bbox_rc"), shape=shape)
    if not np.any(mask) and bbox is not None and source in TRUSTED_FOOTPRINT_SOURCES:
        r0, c0, r1, c1 = bbox
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    count = int(np.count_nonzero(mask))
    record = {
        "footprint_source": source or "none",
        "footprint_pixel_count": count,
        "bbox_rc": bbox if bbox is not None else (_mask_bbox(mask) if count else None),
        "trusted": bool(count > 0 and (source in TRUSTED_FOOTPRINT_SOURCES or obj.get("footprint_rc_indices"))),
    }
    if count <= 0 or not record["trusted"]:
        return None, record
    return mask, record


def _sample_field(field: np.ndarray, coords: Sequence[Any], *, radius: int = 10) -> Tuple[float, float]:
    values: List[float] = []
    h, w = field.shape[:2]
    for coord in coords:
        rc = _coord_rc(coord)
        if rc is None:
            continue
        row, col = int(rc[0]), int(rc[1])
        if not (0 <= row < h and 0 <= col < w):
            continue
        radius = max(0, int(radius))
        r0, r1 = max(0, row - radius), min(h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(w, col + radius + 1)
        patch = field[r0:r1, c0:c1]
        if patch.size:
            values.append(float(np.max(patch)))
            values.append(float(np.mean(patch)))
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.max(values))


def _branch_coords(branch: Mapping[str, Any]) -> List[List[int]]:
    coords: List[List[int]] = []
    rep = _coord_rc(branch.get("canonical_representative_coord_rc") or branch.get("representative_coord") or branch.get("centroid"))
    if rep is not None:
        coords.append(rep)
    for point in branch.get("candidate_points", []) or []:
        if not isinstance(point, Mapping):
            continue
        coord = _coord_rc(point.get("coord"))
        if coord is not None:
            coords.append(coord)
    return coords


def _room_prior_weight(target_description: str, room_caption: str) -> float:
    goal = str(target_description or "").lower()
    room = str(room_caption or "").lower()
    if not goal or not room:
        return 0.0
    rules = (
        (("tv", "television", "monitor"), ("living", "lounge"), 0.70),
        (("tv", "television", "monitor"), ("bedroom", "bed room"), 0.45),
        (("couch", "sofa"), ("living", "lounge"), 0.75),
        (("bed",), ("bedroom", "bed room"), 0.85),
        (("sink", "refrigerator", "fridge", "oven", "stove"), ("kitchen",), 0.80),
        (("toilet", "bathtub"), ("bathroom", "bath room", "washroom"), 0.85),
        (("dining", "table"), ("dining", "kitchen"), 0.55),
    )
    best = 0.0
    for goal_terms, room_terms, weight in rules:
        if any(term in goal for term in goal_terms) and any(term in room for term in room_terms):
            best = max(best, float(weight))
    return best


def _localized_room(room: Mapping[str, Any]) -> bool:
    geom = str(room.get("geometry_type") or "")
    status = str(room.get("spatialization_status") or "")
    return bool(room.get("localized") and room.get("center_rc") is not None and geom != "unlocalized_hypothesis" and status != "unlocalized_prior")


def build_geometry_field(full_map: Any, *, branches: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    layers = _map_layers(full_map)
    shape = _map_shape(full_map)
    if shape is None:
        empty = np.zeros((0, 0), dtype=np.float32)
        return {
            "schema_version": "smoothnav.geometry_field.v1",
            "available": False,
            "reason": "missing_map_shape",
        }, {
            "obstacle": empty,
            "free": empty,
            "explored": empty,
            "unknown": empty,
            "visited": empty,
            "frontier_mask": empty,
        }
    h, w = shape
    obstacle = np.asarray(layers.get("obstacle") if layers.get("obstacle") is not None else np.zeros(shape), dtype=bool)
    free = np.asarray(layers.get("free") if layers.get("free") is not None else np.zeros(shape), dtype=bool)
    explored = np.asarray(layers.get("explored") if layers.get("explored") is not None else np.logical_or(obstacle, free), dtype=bool)
    unknown = np.asarray(layers.get("unknown") if layers.get("unknown") is not None else ~explored, dtype=bool)
    visited = np.asarray(layers.get("visited") if layers.get("visited") is not None else np.zeros(shape), dtype=bool)
    frontier_mask = np.zeros((h, w), dtype=np.float32)
    for branch in branches or []:
        for coord in _branch_coords(branch):
            _add_radial(frontier_mask, coord, weight=1.0, radius=2)
    fields = {
        "obstacle": obstacle.astype(np.float32),
        "free": free.astype(np.float32),
        "explored": explored.astype(np.float32),
        "unknown": unknown.astype(np.float32),
        "visited": visited.astype(np.float32),
        "frontier_mask": _normalize_field(frontier_mask),
    }
    summary = {
        "schema_version": "smoothnav.geometry_field.v1",
        "available": True,
        "map_shape": [int(h), int(w)],
        "obstacle_pixel_count": int(np.count_nonzero(obstacle)),
        "free_pixel_count": int(np.count_nonzero(free)),
        "explored_pixel_count": int(np.count_nonzero(explored)),
        "unknown_pixel_count": int(np.count_nonzero(unknown)),
        "visited_pixel_count": int(np.count_nonzero(visited)),
        "frontier_pixel_count": int(np.count_nonzero(frontier_mask > 0)),
    }
    return summary, fields


def build_semantic_field(
    full_map: Any,
    *,
    objects: Sequence[Mapping[str, Any]],
    rooms: Sequence[Mapping[str, Any]],
    target_description: str,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    shape = _map_shape(full_map)
    if shape is None:
        empty = np.zeros((0, 0), dtype=np.float32)
        return {
            "schema_version": SEMANTIC_FIELD_SCHEMA_VERSION,
            "semantic_reliability": "empty",
            "active_category_count": 0,
            "category_masks": [],
            "localized_objects": [],
            "object_supports": [],
            "localized_rooms": [],
            "room_supports": [],
            "unlocalized_room_hypotheses": [],
        }, {"true_semantic": empty, "object_support": empty, "room_support": empty, "semantic_support": empty}

    h, w = shape
    layers = _map_layers(full_map)
    sem = layers.get("semantic_channels")
    target_indices = _target_indices(target_description, objects)
    target_prior_weights = _target_prior_weights(target_indices)
    true_semantic = np.zeros((h, w), dtype=np.float32)
    category_masks: List[Dict[str, Any]] = []
    semantic_mask_fields: Dict[str, np.ndarray] = {}
    if sem is not None and np.any(sem > SEMANTIC_CONFIDENT_THRESHOLD):
        for idx in range(int(sem.shape[0])):
            channel = np.asarray(sem[idx], dtype=np.float32)
            if not np.any(channel > SEMANTIC_CONFIDENT_THRESHOLD):
                continue
            mask = channel > SEMANTIC_CONFIDENT_THRESHOLD
            masked_channel = np.where(mask, channel, 0.0).astype(np.float32)
            true_semantic = np.maximum(true_semantic, _normalize_field(masked_channel))
            semantic_mask_fields[f"category_mask_{idx}"] = mask.astype(np.float32)
            components = []
            for comp in _mask_components(mask):
                if comp.size == 0:
                    continue
                rows = comp[:, 0]
                cols = comp[:, 1]
                bbox = [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())]
                bbox_area = int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
                pixel_count = int(len(rows))
                components.append(
                    {
                        "bbox_rc": bbox,
                        "center_rc": [int(round(float(rows.mean()))), int(round(float(cols.mean())))],
                        "pixel_count": pixel_count,
                        "bbox_area": bbox_area,
                        "bbox_inflation_ratio": round(float(bbox_area) / float(max(1, pixel_count)), 4),
                    }
                )
            bbox_all = _mask_bbox(mask)
            bbox_area_all = 0
            if bbox_all is not None:
                bbox_area_all = int((bbox_all[2] - bbox_all[0] + 1) * (bbox_all[3] - bbox_all[1] + 1))
            category_masks.append(
                {
                    "id": f"C{len(category_masks) + 1:03d}",
                    "category_index": int(idx),
                    "channel_index": int(SEMANTIC_CHANNEL_OFFSET + idx),
                    "label": _category_name(idx),
                    "source": "true_semantic_channel",
                    "confidence": 1.0,
                    "pixel_count": int(np.count_nonzero(mask)),
                    "bbox_rc": bbox_all,
                    "bbox_area": int(bbox_area_all),
                    "bbox_inflation_ratio": round(float(bbox_area_all) / float(max(1, np.count_nonzero(mask))), 4),
                    "component_count": int(len(components)),
                    "components": components[:12],
                    "threshold": float(SEMANTIC_CONFIDENT_THRESHOLD),
                    "is_direct_target_category": bool(int(idx) in {int(i) for i in target_indices}),
                    "target_prior_weight": float(target_prior_weights.get(int(idx), 0.0)),
                }
            )
    total_sem_pixels = sum(int(item.get("pixel_count", 0) or 0) for item in category_masks)
    if total_sem_pixels <= 0:
        reliability = "empty"
    elif total_sem_pixels < max(256, int(h * w * 0.002)):
        reliability = "sparse"
    else:
        reliability = "dense"

    object_support = np.zeros((h, w), dtype=np.float32)
    object_supports: List[Dict[str, Any]] = []
    localized_objects: List[Dict[str, Any]] = []
    for idx, obj in enumerate(objects or [], start=1):
        center = _coord_rc(obj.get("center_rc"))
        footprint_mask, footprint_record = _object_footprint_mask(obj, shape=(h, w))
        if center is None and footprint_record.get("bbox_rc") is not None:
            bbox = footprint_record.get("bbox_rc") or []
            if len(bbox) >= 4:
                center = [int(round((float(bbox[0]) + float(bbox[2])) / 2.0)), int(round((float(bbox[1]) + float(bbox[3])) / 2.0))]
        if center is None:
            continue
        rel = _safe_float(obj.get("target_relevance"), 0.0)
        confidence = max(0.15, min(0.65, _safe_float(obj.get("confidence"), 0.25) * 0.65))
        radius = 14 if rel < 0.75 else 22
        source = "object_center_prior"
        if footprint_mask is not None and np.any(footprint_mask):
            object_support[footprint_mask] = np.maximum(
                object_support[footprint_mask],
                max(confidence, rel * 0.85),
            )
            source = "object_projected_footprint"
        else:
            _add_radial(object_support, center, weight=max(confidence, rel * 0.85), radius=radius)
        record = {
            "id": str(obj.get("id") or f"O{idx:03d}"),
            "caption": str(obj.get("caption") or ""),
            "center_rc": center,
            "source": source,
            "confidence": round(float(confidence), 4),
            "target_relevance": round(float(rel), 4),
            "support_radius_px": int(0 if source == "object_projected_footprint" else radius),
            "bbox_rc": footprint_record.get("bbox_rc"),
            "footprint_pixel_count": int(footprint_record.get("footprint_pixel_count", 0) or 0),
            "footprint_source": footprint_record.get("footprint_source"),
            "is_direct_target_object": bool(rel >= 0.75),
        }
        localized_objects.append(record)
        object_supports.append(record)

    room_support = np.zeros((h, w), dtype=np.float32)
    localized_rooms: List[Dict[str, Any]] = []
    room_supports: List[Dict[str, Any]] = []
    unlocalized_rooms: List[Dict[str, Any]] = []
    for idx, room in enumerate(rooms or [], start=1):
        room_id = str(room.get("id") or f"R{idx:03d}")
        if not _localized_room(room):
            unlocalized_rooms.append(
                {
                    "id": room_id,
                    "caption": str(room.get("caption") or ""),
                    "source": str(room.get("source") or ""),
                    "confidence": room.get("confidence"),
                    "geometry_type": str(room.get("geometry_type") or "unlocalized_hypothesis"),
                    "spatialization_status": "unlocalized_prior",
                    "object_captions": list(room.get("object_captions") or []),
                }
            )
            continue
        center = _coord_rc(room.get("center_rc"))
        weight = _room_prior_weight(target_description, str(room.get("caption") or ""))
        confidence = 0.38 if str(room.get("confidence")) == "low" else 0.55
        bbox = room.get("bbox_rc")
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            r0, c0, r1, c1 = [_safe_int(v) for v in bbox[:4]]
            r0, r1 = max(0, min(r0, r1)), min(h, max(r0, r1) + 1)
            c0, c1 = max(0, min(c0, c1)), min(w, max(c0, c1) + 1)
            if r0 < r1 and c0 < c1:
                room_support[r0:r1, c0:c1] = np.maximum(room_support[r0:r1, c0:c1], max(confidence, weight))
        else:
            _add_radial(room_support, center, weight=max(confidence, weight), radius=42)
        record = {
            "id": room_id,
            "caption": str(room.get("caption") or ""),
            "center_rc": center,
            "bbox_rc": list(bbox) if isinstance(bbox, Sequence) and len(bbox) >= 4 else None,
            "source": "localized_room_or_function_area",
            "confidence": round(float(confidence), 4),
            "room_prior_weight": round(float(weight), 4),
            "object_ids": list(room.get("object_ids") or []),
        }
        localized_rooms.append(record)
        room_supports.append(record)

    semantic_support = np.maximum.reduce([
        _normalize_field(true_semantic),
        _normalize_field(object_support) * 0.55,
        _normalize_field(room_support) * 0.45,
    ])
    fields = {
        "true_semantic": _normalize_field(true_semantic),
        "object_support": _normalize_field(object_support),
        "room_support": _normalize_field(room_support),
        "semantic_support": _normalize_field(semantic_support),
    }
    fields.update(semantic_mask_fields)
    summary = {
        "schema_version": SEMANTIC_FIELD_SCHEMA_VERSION,
        "active_category_count": int(len(category_masks)),
        "semantic_reliability": reliability,
        "category_masks": category_masks,
        "localized_objects": localized_objects,
        "object_supports": object_supports,
        "localized_rooms": localized_rooms,
        "room_supports": room_supports,
        "unlocalized_room_hypotheses": unlocalized_rooms,
        "notes": [
            "true semantic channels are dense only when full_map[4:] has non-zero masks",
            "object_support and room_support are pseudo-dense priors with explicit source/confidence",
            "unlocalized room hypotheses are never rendered as spatial masks",
        ],
    }
    return summary, fields


def build_target_value_field(
    *,
    map_shape: Tuple[int, int],
    semantic_field: Mapping[str, Any],
    semantic_fields: Mapping[str, np.ndarray],
    branches: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    target_description: str,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    h, w = int(map_shape[0]), int(map_shape[1])
    direct = np.zeros((h, w), dtype=np.float32)
    anchor = np.zeros((h, w), dtype=np.float32)
    room_prior = np.zeros((h, w), dtype=np.float32)
    geometry = np.zeros((h, w), dtype=np.float32)
    revisit_penalty_field = np.zeros((h, w), dtype=np.float32)
    deadend_penalty_field = np.zeros((h, w), dtype=np.float32)

    target_indices = _target_indices(target_description, objects)
    priors = _target_prior_weights(target_indices)

    for item in semantic_field.get("category_masks", []) or []:
        idx = _safe_int(item.get("category_index"), -1)
        mask = np.asarray(semantic_fields.get(f"category_mask_{idx}", np.zeros((h, w), dtype=np.float32))) > 0
        if not np.any(mask):
            continue
        weight = _safe_float(item.get("target_prior_weight"), 0.0)
        target_field = direct if item.get("is_direct_target_category") else anchor
        field_weight = 1.0 if item.get("is_direct_target_category") else weight
        if field_weight <= 0:
            continue
        target_field[mask] = np.maximum(target_field[mask], field_weight)
        # Build value/potential around actual semantic cells, not around the
        # category-wide bbox.  This is the key distinction from the previous
        # failure mode: sparse wall/edge pixels no longer inflate into a large
        # rectangular "object".
        for comp in item.get("components", []) or []:
            center = comp.get("center_rc") if isinstance(comp, Mapping) else None
            pixels = _safe_int((comp or {}).get("pixel_count"), 0) if isinstance(comp, Mapping) else 0
            radius = max(5, min(18, int(round(math.sqrt(max(1, pixels)) * 1.8))))
            _add_radial(target_field, center, weight=field_weight * 0.55, radius=radius)

    for obj in objects or []:
        center = _coord_rc(obj.get("center_rc"))
        footprint_mask, footprint_record = _object_footprint_mask(obj, shape=(h, w))
        if center is None and footprint_record.get("bbox_rc") is not None:
            bbox = footprint_record.get("bbox_rc") or []
            if len(bbox) >= 4:
                center = [int(round((float(bbox[0]) + float(bbox[2])) / 2.0)), int(round((float(bbox[1]) + float(bbox[3])) / 2.0))]
        rel = _safe_float(obj.get("target_relevance"), 0.0)
        idx = _semantic_channel_index(obj.get("caption"))
        field = None
        weight = 0.0
        if rel >= 0.75:
            field = direct
            weight = max(0.75, rel)
        elif idx is not None and _safe_float(priors.get(idx), 0.0) > 0:
            field = anchor
            weight = min(0.65, _safe_float(priors.get(idx), 0.0) * 0.65)
        if field is None or weight <= 0:
            continue
        if footprint_mask is not None and np.any(footprint_mask):
            field[footprint_mask] = np.maximum(field[footprint_mask], weight)
            if center is not None:
                _add_radial(field, center, weight=weight * 0.45, radius=18)
        elif center is not None:
            _add_radial(field, center, weight=weight, radius=30 if field is anchor else 36)

    for room in semantic_field.get("room_supports", []) or []:
        weight = _safe_float(room.get("room_prior_weight"), 0.0)
        if weight <= 0:
            continue
        bbox = room.get("bbox_rc")
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            r0, c0, r1, c1 = [_safe_int(v) for v in bbox[:4]]
            r0, r1 = max(0, min(r0, r1)), min(h, max(r0, r1) + 1)
            c0, c1 = max(0, min(c0, c1)), min(w, max(c0, c1) + 1)
            if r0 < r1 and c0 < c1:
                room_prior[r0:r1, c0:c1] = np.maximum(room_prior[r0:r1, c0:c1], weight)
        else:
            _add_radial(room_prior, room.get("center_rc"), weight=weight, radius=42)

    branch_rows: List[Dict[str, Any]] = []
    for branch in branches or []:
        coords = _branch_coords(branch)
        rep = coords[0] if coords else None
        terms = dict(branch.get("score_terms_aggregate") or {})
        candidate_count = int(branch.get("candidate_point_count", 0) or 0)
        info_gain = max(
            _clip01(terms.get("novelty_score_max")),
            min(1.0, math.log1p(max(0, candidate_count)) / math.log(9.0)),
        )
        actionability = _clip01(terms.get("actionability_score_max"))
        repeat = max(_clip01(terms.get("repeat_penalty_max")), _clip01(terms.get("recent_penalty_max")))
        deadend = _clip01(repeat + (0.35 if candidate_count <= 0 else 0.0))
        geom_value = _clip01(0.58 * info_gain + 0.34 * actionability - 0.40 * repeat - 0.25 * deadend)
        frontier_final_score = _optional_float(terms.get("final_score_max"))
        frontier_score_terms = {
            "final": frontier_final_score,
            "base": _optional_float(terms.get("base_score_max")),
            "semantic_bias": _optional_float(terms.get("bias_score_max")),
            "target_progress": _optional_float(terms.get("target_progress_score_max")),
            "planner_prior": _optional_float(terms.get("planner_prior_score_max")),
            "novelty": _optional_float(terms.get("novelty_score_max")),
            "actionability": _optional_float(terms.get("actionability_score_max")),
            "repeat_penalty": _optional_float(terms.get("repeat_penalty_max")),
            "recent_penalty": _optional_float(terms.get("recent_penalty_max")),
        }
        _add_radial(geometry, rep, weight=geom_value, radius=24)
        _add_radial(revisit_penalty_field, rep, weight=repeat, radius=20)
        _add_radial(deadend_penalty_field, rep, weight=deadend, radius=20)
        # Fill component scores below after all fields are normalized.
        branch_rows.append(
            {
                "id": str(branch.get("id") or ""),
                "canonical_direction": str(branch.get("canonical_direction_from_agent") or branch.get("map_direction_from_agent") or _direction_from_agent(rep, branch.get("canonical_agent_coord_rc"))),
                "candidate_count": candidate_count,
                "representative_coord": rep,
                "canonical_coord_rc": rep,
                "info_gain": round(float(info_gain), 4),
                "revisit_penalty": round(float(repeat), 4),
                "dead_end_risk": round(float(deadend), 4),
                "frontier_score_available": frontier_final_score is not None,
                "frontier_final_score": round(float(frontier_final_score), 4) if frontier_final_score is not None else None,
                "frontier_score_terms": {
                    key: (round(float(val), 4) if val is not None else None)
                    for key, val in frontier_score_terms.items()
                },
                # First-class aliases for table consumers that should compare
                # the PONI-lite value against Graph.get_goal's executable score
                # instead of silently treating them as one ranking.
                "frontier_base_score": (
                    round(float(frontier_score_terms["base"]), 4)
                    if frontier_score_terms["base"] is not None
                    else None
                ),
                "frontier_bias_score": (
                    round(float(frontier_score_terms["semantic_bias"]), 4)
                    if frontier_score_terms["semantic_bias"] is not None
                    else None
                ),
                "frontier_target_progress_score": (
                    round(float(frontier_score_terms["target_progress"]), 4)
                    if frontier_score_terms["target_progress"] is not None
                    else None
                ),
                "frontier_planner_prior_score": (
                    round(float(frontier_score_terms["planner_prior"]), 4)
                    if frontier_score_terms["planner_prior"] is not None
                    else None
                ),
                "frontier_actionability_score": (
                    round(float(frontier_score_terms["actionability"]), 4)
                    if frontier_score_terms["actionability"] is not None
                    else None
                ),
            }
        )

    direct = _normalize_field(direct)
    anchor = _normalize_field(anchor)
    room_prior = _normalize_field(room_prior)
    geometry = _normalize_field(geometry)
    revisit_penalty_field = _normalize_field(revisit_penalty_field)
    deadend_penalty_field = _normalize_field(deadend_penalty_field)
    combined_raw = (
        1.00 * direct
        + 0.74 * anchor
        + 0.55 * room_prior
        + 0.36 * geometry
        - 0.28 * revisit_penalty_field
        - 0.18 * deadend_penalty_field
    )
    combined = _normalize_field(np.clip(combined_raw, 0.0, None))

    for row, branch in zip(branch_rows, branches or []):
        coords = _branch_coords(branch)
        direct_mean, direct_max = _sample_field(direct, coords, radius=12)
        anchor_mean, anchor_max = _sample_field(anchor, coords, radius=12)
        room_mean, room_max = _sample_field(room_prior, coords, radius=12)
        geom_mean, geom_max = _sample_field(geometry, coords, radius=12)
        combined_mean, combined_max = _sample_field(combined, coords, radius=12)
        if direct_max >= 0.25:
            evidence = "direct"
        elif anchor_max >= 0.20:
            evidence = "anchor"
        elif room_max >= 0.20:
            evidence = "room_prior"
        else:
            evidence = "geometry_only"
        risk_flags: List[str] = []
        if row["revisit_penalty"] >= 0.35:
            risk_flags.append("revisit_risk")
        if row["dead_end_risk"] >= 0.35:
            risk_flags.append("dead_end_risk")
        if row["candidate_count"] <= 0:
            risk_flags.append("non_actionable")
        semantic_support = []
        for item in semantic_field.get("localized_objects", []) or []:
            dist = _distance(row.get("representative_coord"), item.get("center_rc"))
            if dist is not None and dist <= 100:
                semantic_support.append({"id": item.get("id"), "source": item.get("source"), "distance": round(dist, 2)})
        for item in semantic_field.get("localized_rooms", []) or []:
            dist = _distance(row.get("representative_coord"), item.get("center_rc"))
            if dist is not None and dist <= 150:
                semantic_support.append({"id": item.get("id"), "source": item.get("source"), "distance": round(dist, 2)})
        row.update(
            {
                "schema_version": BRANCH_TABLE_SCHEMA_VERSION,
                "target_value_mean": round(float(combined_mean), 4),
                "target_value_max": round(float(combined_max), 4),
                # Backward-compatible aliases consumed by existing prompt/evaluator.
                "target_value_score": round(float(combined_max), 4),
                "direct_value": round(float(direct_max), 4),
                "anchor_value": round(float(anchor_max), 4),
                "room_prior_value": round(float(room_max), 4),
                "geometry_value": round(float(geom_max), 4),
                "direct_score": round(float(direct_max), 4),
                "anchor_score": round(float(anchor_max), 4),
                "room_prior_score": round(float(room_max), 4),
                "info_gain_score": row["info_gain"],
                "semantic_support": semantic_support[:8],
                "evidence_level": evidence,
                "target_visible": bool(evidence == "direct"),
                "risk_flags": risk_flags,
            }
        )

    branch_rows.sort(
        key=lambda item: (
            -float(item.get("target_value_max", 0.0) or 0.0),
            evidence_rank(item.get("evidence_level")),
            str(item.get("id", "")),
        )
    )
    for rank, row in enumerate(branch_rows, start=1):
        row["target_value_rank"] = int(rank)

    frontier_ranked = sorted(
        [row for row in branch_rows if row.get("frontier_final_score") is not None],
        key=lambda item: (-float(item.get("frontier_final_score") or 0.0), str(item.get("id", ""))),
    )
    for rank, row in enumerate(frontier_ranked, start=1):
        row["frontier_final_score_rank"] = int(rank)
        row["executable_decision_rank"] = int(rank)
    for row in branch_rows:
        row.setdefault("frontier_final_score_rank", None)
        row.setdefault("executable_decision_rank", None)
        row["executable_decision_score"] = row.get("frontier_final_score")
        row["executable_decision_source"] = (
            "graph_frontier_final_score"
            if row.get("frontier_final_score") is not None
            else "target_value_fallback"
        )
    if not frontier_ranked:
        for rank, row in enumerate(branch_rows, start=1):
            row["executable_decision_rank"] = int(rank)
            row["executable_decision_score"] = row.get("target_value_score")

    component_peaks = {
        "direct": float(direct.max()) if direct.size else 0.0,
        "anchor": float(anchor.max()) if anchor.size else 0.0,
        "room_prior": float(room_prior.max()) if room_prior.size else 0.0,
        "geometry_only": float(geometry.max()) if geometry.size else 0.0,
    }
    active_sources = [name for name, peak in component_peaks.items() if peak > 0.05]
    if not active_sources:
        source = "geometry_only"
    elif len(active_sources) > 1:
        source = "mixed"
    else:
        source = active_sources[0]
    strongest = "geometry_only"
    if component_peaks["direct"] > 0.05:
        strongest = "direct"
    elif component_peaks["anchor"] > 0.05:
        strongest = "anchor"
    elif component_peaks["room_prior"] > 0.05:
        strongest = "room_prior"
    confidence = max(component_peaks.values()) if component_peaks else 0.0
    if strongest == "geometry_only":
        confidence = min(confidence, 0.45)
    summary = {
        "schema_version": TARGET_VALUE_SCHEMA_VERSION,
        "available": True,
        "source": source,
        "strongest_source": strongest,
        "confidence": round(float(confidence), 4),
        "combined_value": _field_stats(combined),
        "direct_target_value": _field_stats(direct),
        "anchor_value": _field_stats(anchor),
        "room_prior_value": _field_stats(room_prior),
        "geometry_fallback_value": _field_stats(geometry),
        "component_peaks": component_peaks,
        "evidence_counts": {
            level: sum(1 for row in branch_rows if row.get("evidence_level") == level)
            for level in EVIDENCE_LEVELS
        },
        "branch_ranking": [
            {
                "id": row.get("id"),
                "target_value_score": row.get("target_value_max"),
                "target_value_max": row.get("target_value_max"),
                "target_value_mean": row.get("target_value_mean"),
                "evidence_level": row.get("evidence_level"),
                "target_visible": row.get("target_visible"),
                "frontier_final_score": row.get("frontier_final_score"),
                "frontier_final_score_rank": row.get("frontier_final_score_rank"),
                "frontier_actionability_score": row.get("frontier_actionability_score"),
            }
            for row in branch_rows[:12]
        ],
        "executable_decision_ranking": [
            {
                "id": row.get("id"),
                "executable_decision_score": row.get("executable_decision_score"),
                "executable_decision_rank": row.get("executable_decision_rank"),
                "executable_decision_source": row.get("executable_decision_source"),
                "frontier_final_score": row.get("frontier_final_score"),
                "target_value_score": row.get("target_value_score"),
                "evidence_level": row.get("evidence_level"),
            }
            for row in sorted(
                branch_rows,
                key=lambda item: (
                    int(item.get("executable_decision_rank") or 10**6),
                    str(item.get("id") or ""),
                ),
            )[:12]
        ],
        "notes": [
            "combined_value is deterministic PONI-lite/VLFM-lite value, not a learned model",
            "executable_decision_ranking is Graph.get_goal/frontier-score aligned when final scores are available",
            "geometry fallback remains valid when semantic evidence is empty, but branch evidence stays geometry_only",
        ],
    }
    fields = {
        "combined_value": combined,
        "direct_target_value": direct,
        "anchor_value": anchor,
        "room_prior_value": room_prior,
        "geometry_fallback_value": geometry,
        "revisit_penalty": revisit_penalty_field,
        "deadend_penalty": deadend_penalty_field,
    }
    return summary, fields, branch_rows


def build_bev_decision_state(
    *,
    full_map: Any,
    step_idx: Optional[int],
    target_description: str,
    agent: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    rooms: Sequence[Mapping[str, Any]],
    strategy: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Build a structured planner BEV decision state and its dense fields."""

    shape = _map_shape(full_map)
    geometry_summary, geometry_fields = build_geometry_field(full_map, branches=branches)
    semantic_summary, semantic_fields = build_semantic_field(
        full_map,
        objects=objects,
        rooms=rooms,
        target_description=target_description,
    )
    if shape is None:
        target_summary = {
            "schema_version": TARGET_VALUE_SCHEMA_VERSION,
            "available": False,
            "reason": "missing_map_shape",
            "branch_ranking": [],
        }
        target_fields = {}
        branch_table: List[Dict[str, Any]] = []
    else:
        target_summary, target_fields, branch_table = build_target_value_field(
            map_shape=shape,
            semantic_field=semantic_summary,
            semantic_fields=semantic_fields,
            branches=branches,
            objects=objects,
            target_description=target_description,
        )

    canonical_frame = {
        "coord_type": CANONICAL_FRAME_NAME,
        "agent_rc": _coord_rc(agent.get("coord_rc")),
        "north_arrow": "image_up_is_north",
        "row_axis": "south_down_image_y",
        "col_axis": "east_right_image_x",
        "map_shape": list(shape) if shape is not None else None,
        "notes": [
            "Planner-critical prompt exposes only this frame",
            "raw/local/repaired coordinates are debug-only and excluded from branch_table",
        ],
    }
    state = {
        "schema_version": DECISION_STATE_SCHEMA_VERSION,
        "step_idx": step_idx,
        "target_description": str(target_description or ""),
        "canonical_frame": canonical_frame,
        # Backward-compatible name used by prior summary code.
        "coordinate_contract": {
            "frame": CANONICAL_FRAME_NAME,
            "agent_coord_rc": canonical_frame["agent_rc"],
            "branch_coord_key": "representative_coord",
            "branch_direction_key": "canonical_direction",
            "hidden_from_planner_prompt": ["raw_coord_rc", "map_current_channel_coord_rc", "direction_from_agent"],
        },
        "strategy_context": {
            "target_region": str(getattr(strategy, "target_region", "") or (strategy or {}).get("target_region", "") if isinstance(strategy, Mapping) else getattr(strategy, "target_region", "") or ""),
            "anchor_object": str(getattr(strategy, "anchor_object", "") or (strategy or {}).get("anchor_object", "") if isinstance(strategy, Mapping) else getattr(strategy, "anchor_object", "") or ""),
        },
        "geometry_field": geometry_summary,
        "semantic_field": semantic_summary,
        "target_value_field": target_summary,
        "target_value_state": target_summary,
        "branch_table": branch_table,
        "branch_decision_table": branch_table,
        "array_artifacts": {
            "recommended_npz_name": "planner_decision_state_fields.npz",
            "field_keys": sorted(
                [f"geometry_{key}" for key in geometry_fields]
                + [f"semantic_{key}" for key in semantic_fields]
                + [f"target_{key}" for key in target_fields]
            ),
        },
    }
    fields: Dict[str, np.ndarray] = {}
    fields.update({f"geometry_{k}": v.astype(np.float32) for k, v in geometry_fields.items()})
    fields.update({f"semantic_{k}": v.astype(np.float32) for k, v in semantic_fields.items()})
    fields.update({f"target_{k}": v.astype(np.float32) for k, v in target_fields.items()})
    return _jsonable(state), fields


def build_decision_state_from_frame(full_map: Any, frame: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    return build_bev_decision_state(
        full_map=full_map,
        step_idx=frame.get("step_idx"),
        target_description=str((frame.get("planner_context") or {}).get("goal_description") or ""),
        agent=dict(frame.get("agent") or {}),
        branches=list(frame.get("branches") or []),
        objects=list(frame.get("objects") or []),
        rooms=list(frame.get("rooms") or []),
        strategy=dict(frame.get("planner_context") or {}),
    )


def save_decision_state_fields_npz(path: str | Path, fields: Mapping[str, np.ndarray]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): np.asarray(v, dtype=np.float32) for k, v in fields.items()}
    np.savez_compressed(path, **serializable)
    return str(path)


def decision_state_summary_for_prompt(decision_state: Mapping[str, Any], *, max_branches: int = 12) -> Dict[str, Any]:
    """Compact decision-state facts for the MLLM prompt."""

    semantic = dict(decision_state.get("semantic_field") or {})
    target = dict(decision_state.get("target_value_field") or decision_state.get("target_value_state") or {})
    return {
        "schema_version": "smoothnav.bev_decision_state_prompt_summary.v1",
        "canonical_frame": dict(decision_state.get("canonical_frame") or {}),
        "geometry_field": dict(decision_state.get("geometry_field") or {}),
        "semantic_field": {
            "active_category_count": semantic.get("active_category_count"),
            "semantic_reliability": semantic.get("semantic_reliability"),
            "category_masks": list(semantic.get("category_masks") or [])[:8],
            "object_supports": list(semantic.get("object_supports") or [])[:10],
            "room_supports": list(semantic.get("room_supports") or [])[:6],
            "unlocalized_room_hypotheses": list(semantic.get("unlocalized_room_hypotheses") or [])[:8],
        },
        "target_value_field": {
            "source": target.get("source"),
            "strongest_source": target.get("strongest_source"),
            "confidence": target.get("confidence"),
            "component_peaks": target.get("component_peaks", {}),
            "evidence_counts": target.get("evidence_counts", {}),
            "branch_ranking": list(target.get("branch_ranking") or [])[:max_branches],
            "notes": target.get("notes", []),
        },
        "branch_table": list(decision_state.get("branch_table") or [])[:max_branches],
    }


def write_decision_state_json(path: str | Path, decision_state: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(decision_state), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return str(path)
