"""Semantic annotated BEV frame assembly and rendering.

This module deliberately separates the semantic BEV *data contract* from the
Pillow renderer.  Runtime code can save/replay the JSON frame without an image,
and both human debug images and MLLM planner images are rendered from the same
contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # target matching is lightweight, but keep this module import-safe.
    from smoothnav.target_matching import score_caption_against_goal
except Exception:  # pragma: no cover
    score_caption_against_goal = None

try:  # keep replay import-safe when only renderer dependencies are available.
    from smoothnav.bev_decision_state import (
        BRANCH_TABLE_SCHEMA_VERSION,
        DECISION_STATE_SCHEMA_VERSION,
        TARGET_VALUE_SCHEMA_VERSION,
        build_bev_decision_state,
        build_decision_state_from_frame,
        evidence_rank,
        save_decision_state_fields_npz,
    )
except Exception:  # pragma: no cover
    BRANCH_TABLE_SCHEMA_VERSION = "smoothnav.branch_decision_table.v1"
    DECISION_STATE_SCHEMA_VERSION = "smoothnav.planner_decision_state.v2"
    TARGET_VALUE_SCHEMA_VERSION = "smoothnav.target_conditioned_value.v0"
    build_bev_decision_state = None
    build_decision_state_from_frame = None
    evidence_rank = None
    save_decision_state_fields_npz = None

SCHEMA_VERSION = "smoothnav.semantic_bev_frame.v1"
RENDERER_VERSION = "smoothnav.semantic_bev_renderer.v1"
COORDINATE_FRAME_NAME = "full_map_rc_unflipped"

# BEV_Map follows the SemExp/ObjectNav convention used in
# base_UniGoal/src/map/bev_mapping.py:
#   0 obstacle, 1 explored/free, 2 current agent, 3 past agent, 4+ categories.
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

_CATEGORY_SYNONYMS = {
    "chair": ("chair", "seat"),
    "couch": ("couch", "sofa", "loveseat"),
    "potted plant": ("potted plant", "plant"),
    "bed": ("bed", "bedroom"),
    "toilet": ("toilet", "bathroom"),
    "tv": ("tv", "television", "monitor", "tv monitor", "tv_monitor", "screen"),
    "dining-table": ("dining-table", "dining table", "table", "desk"),
    "oven": ("oven", "stove"),
    "sink": ("sink", "basin"),
    "refrigerator": ("refrigerator", "fridge"),
    "book": ("book", "bookshelf"),
    "clock": ("clock",),
    "vase": ("vase",),
    "cup": ("cup", "mug"),
    "bottle": ("bottle",),
}

_GOAL_ANCHOR_PRIORS = {
    # TV is usually discovered through living-room / bedroom anchors before the
    # TV itself is visible.  These are weak search priors, not direct detections.
    "tv": {
        "couch": 0.90,
        "chair": 0.50,
        "dining-table": 0.35,
        "bed": 0.30,
        "book": 0.20,
        "vase": 0.15,
        "clock": 0.15,
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

_SEMANTIC_PALETTE = (
    (244, 67, 54),    # chair
    (255, 152, 0),    # couch
    (76, 175, 80),    # potted plant
    (103, 58, 183),   # bed
    (0, 150, 136),    # toilet
    (233, 30, 99),    # tv
    (121, 85, 72),    # dining-table
    (255, 87, 34),    # oven
    (3, 169, 244),    # sink
    (0, 188, 212),    # refrigerator
    (139, 195, 74),   # book
    (205, 220, 57),   # clock
    (156, 39, 176),   # vase
    (255, 235, 59),   # cup
    (96, 125, 139),   # bottle
    (158, 158, 158),  # other
)

DENSE_UNKNOWN_CLASS_ID = 0
DENSE_FREE_CLASS_ID = 1
DENSE_OBSTACLE_CLASS_ID = 2
DENSE_SEMANTIC_CLASS_OFFSET = 10

DENSE_BASE_CLASS_COLORS = {
    DENSE_UNKNOWN_CLASS_ID: (52, 52, 52),     # unknown / unexplored
    DENSE_FREE_CLASS_ID: (238, 238, 238),     # explored free
    DENSE_OBSTACLE_CLASS_ID: (12, 12, 12),    # wall / obstacle
}

# Stable SemExp/PONI-style semantic raster palette.  These colors are used for
# cell-level semantic classes only; object/room/branch labels belong to debug
# views and must not be drawn on the dense semantic map.
DENSE_SEMANTIC_COLOR_BY_LABEL = {
    "chair": (255, 152, 0),           # orange
    "couch": (139, 230, 118),         # light green
    "potted plant": (76, 175, 80),
    "bed": (66, 133, 244),            # blue
    "toilet": (129, 212, 250),        # light blue
    "tv": (0, 188, 212),              # cyan
    "dining-table": (121, 85, 72),    # brown
    "oven": (255, 87, 34),
    "sink": (21, 101, 192),           # dark blue
    "refrigerator": (0, 150, 136),    # teal
    "cabinet": (156, 39, 176),        # purple if a future channel exposes it
    "book": (139, 195, 74),
    "clock": (205, 220, 57),
    "vase": (171, 71, 188),
    "cup": (255, 235, 59),
    "bottle": (96, 125, 139),
    "other": (158, 158, 158),
}

DENSE_MIN_FOOTPRINT_CELLS_BY_LABEL = {
    "chair": (8, 8),
    "couch": (10, 18),
    "potted plant": (6, 6),
    "bed": (16, 24),
    "toilet": (8, 8),
    "tv": (4, 10),
    "dining-table": (14, 18),
    "oven": (8, 8),
    "sink": (8, 8),
    "refrigerator": (12, 8),
    "book": (4, 4),
    "clock": (4, 4),
    "vase": (5, 5),
    "cup": (3, 3),
    "bottle": (3, 3),
    "cabinet": (10, 14),
    "windows": (4, 18),
    "mirror": (6, 10),
    "other": (6, 6),
}

_OBJECT_COLOR = (0, 188, 212)
_TARGET_OBJECT_COLOR = (255, 112, 67)
_ROOM_COLOR = (126, 87, 194)
_AGENT_COLOR = (0, 190, 0)
_DRY_COLOR = (255, 255, 255)
_MLLM_COLOR = (255, 64, 220)
_FINAL_COLOR = (0, 255, 140)
_WARNING_COLOR = (255, 80, 80)
_CURRENT_CHANNEL_COLOR = (64, 210, 255)
_VISITED_CHANNEL_COLOR = (255, 214, 64)
_TARGET_HEAT_COLOR = (255, 64, 0)

_COORD_NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().numpy().tolist()
        except Exception:
            pass
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


def canonical_json_hash(payload: Any) -> str:
    """Stable SHA1 hash for JSON-compatible payloads."""

    text = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _base_masks(map_like: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    arr = _map_to_numpy(map_like)
    if arr is None or arr.size == 0:
        return None, None, None
    if arr.ndim == 3:
        known = np.any(arr > 0, axis=0)
        obstacle = arr[0] > 0 if arr.shape[0] > 0 else np.zeros(arr.shape[-2:], dtype=bool)
        free = arr[1] > 0 if arr.shape[0] > 1 else np.logical_and(known, ~obstacle)
        return known.astype(bool), free.astype(bool), obstacle.astype(bool)
    if arr.ndim == 2:
        known = arr > 0
        return known.astype(bool), known.astype(bool), np.zeros_like(known, dtype=bool)
    return None, None, None


def _map_layers(map_like: Any) -> Dict[str, Any]:
    """Decode the SemExp-like BEV channel convention without assuming GT semantics."""

    arr = _map_to_numpy(map_like)
    if arr is None or arr.size == 0:
        return {"arr": None, "current_mask": None, "visited_mask": None, "semantic_channels": None}
    if arr.ndim != 3:
        return {"arr": arr, "current_mask": None, "visited_mask": None, "semantic_channels": None}
    return {
        "arr": arr,
        "obstacle": arr[0] if arr.shape[0] > 0 else None,
        "explored": arr[1] if arr.shape[0] > 1 else None,
        "current_mask": arr[2] > 0 if arr.shape[0] > 2 else None,
        "visited_mask": arr[3] > 0 if arr.shape[0] > 3 else None,
        "semantic_channels": arr[SEMANTIC_CHANNEL_OFFSET:] if arr.shape[0] > SEMANTIC_CHANNEL_OFFSET else None,
    }


def _dense_semantic_color_for_label(label: str, fallback_index: int = 0) -> Tuple[int, int, int]:
    normalized = str(label or "").strip().lower()
    if normalized in DENSE_SEMANTIC_COLOR_BY_LABEL:
        return DENSE_SEMANTIC_COLOR_BY_LABEL[normalized]
    return _SEMANTIC_PALETTE[int(fallback_index) % len(_SEMANTIC_PALETTE)]


def _dense_class_definitions(category_names: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    definitions: Dict[int, Dict[str, Any]] = {
        DENSE_UNKNOWN_CLASS_ID: {
            "id": DENSE_UNKNOWN_CLASS_ID,
            "label": "unknown",
            "kind": "geometry",
            "color_rgb": list(DENSE_BASE_CLASS_COLORS[DENSE_UNKNOWN_CLASS_ID]),
        },
        DENSE_FREE_CLASS_ID: {
            "id": DENSE_FREE_CLASS_ID,
            "label": "free_explored",
            "kind": "geometry",
            "color_rgb": list(DENSE_BASE_CLASS_COLORS[DENSE_FREE_CLASS_ID]),
        },
        DENSE_OBSTACLE_CLASS_ID: {
            "id": DENSE_OBSTACLE_CLASS_ID,
            "label": "obstacle_wall",
            "kind": "geometry",
            "color_rgb": list(DENSE_BASE_CLASS_COLORS[DENSE_OBSTACLE_CLASS_ID]),
        },
    }
    for idx, label in enumerate(category_names):
        class_id = DENSE_SEMANTIC_CLASS_OFFSET + int(idx)
        definitions[class_id] = {
            "id": class_id,
            "label": str(label),
            "kind": "semantic_channel",
            "channel_index": int(SEMANTIC_CHANNEL_OFFSET + idx),
            "category_index": int(idx),
            "color_rgb": list(_dense_semantic_color_for_label(str(label), idx)),
        }
    return definitions


def _dense_label_key(label: Any) -> str:
    normalized = " ".join(str(label or "").strip().lower().replace("_", " ").split())
    aliases = {
        "sofa": "couch",
        "loveseat": "couch",
        "tv monitor": "tv",
        "television": "tv",
        "monitor": "tv",
        "table": "dining-table",
        "dining table": "dining-table",
        "fridge": "refrigerator",
        "cupboard": "cabinet",
        "closet": "cabinet",
        "window": "windows",
        "windows": "windows",
    }
    return aliases.get(normalized, normalized)


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
    if any(v is None for v in vals):
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


def _expand_bbox_to_label_min_size(
    bbox: Sequence[int],
    *,
    label: str,
    shape: Tuple[int, int],
) -> List[int]:
    """Expand a sparse surface bbox to a conservative class-sized footprint box."""

    base = _bbox_rc(list(bbox), shape=shape)
    if base is None:
        return list(bbox)
    h, w = int(shape[0]), int(shape[1])
    r0, c0, r1, c1 = [int(v) for v in base]
    min_h, min_w = DENSE_MIN_FOOTPRINT_CELLS_BY_LABEL.get(
        str(label or "other").strip().lower(),
        DENSE_MIN_FOOTPRINT_CELLS_BY_LABEL["other"],
    )
    target_h = max(int(r1 - r0 + 1), int(min_h))
    target_w = max(int(c1 - c0 + 1), int(min_w))
    center_r = int(round((r0 + r1) / 2.0))
    center_c = int(round((c0 + c1) / 2.0))
    nr0 = center_r - target_h // 2
    nr1 = nr0 + target_h - 1
    nc0 = center_c - target_w // 2
    nc1 = nc0 + target_w - 1
    if nr0 < 0:
        nr1 -= nr0
        nr0 = 0
    if nc0 < 0:
        nc1 -= nc0
        nc0 = 0
    if nr1 >= h:
        delta = nr1 - h + 1
        nr0 -= delta
        nr1 -= delta
    if nc1 >= w:
        delta = nc1 - w + 1
        nc0 -= delta
        nc1 -= delta
    return _bbox_rc([nr0, nc0, nr1, nc1], shape=shape) or base


def _footprint_cells_from_object(item: Mapping[str, Any], *, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return validated BEV cells from an upstream object footprint payload.

    Mature BEV renderers paint cells that come from semantic/depth projection.
    This helper therefore accepts explicit rc cells only; callers may separately
    decide whether a trusted 3-D bbox fallback should be densified.
    """

    raw = (
        item.get("footprint_rc_indices")
        or item.get("footprint_rcs")
        or item.get("footprint_cells_rc")
        or item.get("bev_footprint_rc_indices")
    )
    if raw is None:
        return []
    height, width = int(shape[0]), int(shape[1])
    cells: set[Tuple[int, int]] = set()
    try:
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        for coord in raw:
            if hasattr(coord, "tolist"):
                coord = coord.tolist()
            if not isinstance(coord, Sequence) or len(coord) < 2:
                continue
            row = _safe_int(coord[0])
            col = _safe_int(coord[1])
            if row is None or col is None:
                continue
            row_i, col_i = int(row), int(col)
            if 0 <= row_i < height and 0 <= col_i < width:
                cells.add((row_i, col_i))
    except Exception:
        return []
    return sorted(cells)


def _shift_bbox_to_explored_interior(
    bbox: Sequence[int],
    *,
    obstacle_mask: Optional[np.ndarray],
    explored_mask: Optional[np.ndarray],
    shape: Tuple[int, int],
    max_shift: Optional[int] = None,
) -> List[int]:
    """Move a surface-component bbox toward explored non-wall interior cells.

    Semantic channels built from RGB-D masks often project visible object
    surfaces onto the same cells as obstacle/wall contours.  This is useful as
    evidence but visually wrong as an object footprint.  For the planner-facing
    footprint artifact, choose a nearby same-size rectangle that maximizes
    explored non-obstacle support and penalizes wall/unknown overlap.
    """

    base = _bbox_rc(list(bbox), shape=shape)
    if base is None or explored_mask is None or obstacle_mask is None:
        return list(base or bbox)
    h, w = int(shape[0]), int(shape[1])
    r0, c0, r1, c1 = [int(v) for v in base]
    bh = int(r1 - r0 + 1)
    bw = int(c1 - c0 + 1)
    if bh <= 0 or bw <= 0:
        return base
    max_shift = int(max_shift if max_shift is not None else max(4, min(24, round(max(bh, bw) * 1.5))))
    best_bbox = base
    best_score = -1e18
    # Prefer small moves when scores tie, so already-good components stay put.
    for dr in range(-max_shift, max_shift + 1):
        nr0 = r0 + dr
        nr1 = nr0 + bh - 1
        if nr0 < 0 or nr1 >= h:
            continue
        for dc in range(-max_shift, max_shift + 1):
            nc0 = c0 + dc
            nc1 = nc0 + bw - 1
            if nc0 < 0 or nc1 >= w:
                continue
            explored_patch = np.asarray(explored_mask[nr0 : nr1 + 1, nc0 : nc1 + 1], dtype=bool)
            obstacle_patch = np.asarray(obstacle_mask[nr0 : nr1 + 1, nc0 : nc1 + 1], dtype=bool)
            interior = np.logical_and(explored_patch, ~obstacle_patch)
            unknown = ~explored_patch
            # Penalize wall/obstacle overlap strongly.  The footprint image is a
            # decision-state approximation; it should sit in room interior, not
            # trace wall edges like a raw depth surface projection.
            score = (
                3.0 * float(np.count_nonzero(interior))
                - 4.0 * float(np.count_nonzero(obstacle_patch))
                - 1.5 * float(np.count_nonzero(unknown))
                - 0.03 * float(abs(dr) + abs(dc))
            )
            if score > best_score:
                best_score = score
                best_bbox = [int(nr0), int(nc0), int(nr1), int(nc1)]
    return best_bbox


def build_dense_semantic_raster(
    full_map: Any,
    semantic_category_names: Optional[Sequence[str]] = None,
    semantic_threshold: float = 0.5,
    objects: Optional[Sequence[Mapping[str, Any]]] = None,
    include_object_footprints: bool = True,
    semantic_bbox_fill: bool = False,
    semantic_bbox_fill_threshold: float = 0.2,
    semantic_bbox_fill_mode: str = "component_bbox",
) -> Dict[str, Any]:
    """Build a true dense semantic class raster from BEV map channels.

    The returned ``class_id_map`` is an HxW integer grid.  Its primary semantic
    source is ``full_map[4:]``.  Graph object footprints may contribute only
    when upstream has serialized a BEV footprint/bbox from 3-D object support.
    Graph object centers, room text, target priors, branch labels, and warnings
    are not allowed to create semantic cells here.
    """

    category_names = list(semantic_category_names or SEMANTIC_CATEGORY_NAMES)
    object_items = list(objects or [])
    display_category_names = list(category_names)
    if include_object_footprints:
        for obj in object_items:
            if _bbox_rc(obj.get("bbox_rc") if isinstance(obj, Mapping) else None) is None:
                continue
            label = _dense_label_key((obj or {}).get("caption") if isinstance(obj, Mapping) else "")
            if label and label not in display_category_names:
                display_category_names.append(label)
    arr = _map_to_numpy(full_map)
    warnings: List[str] = []
    if arr is None or arr.size == 0:
        warnings.extend(["map_unavailable", "semantic_channels_empty", "dense_semantic_bev_cannot_show_object_regions_without_semantic_projection"])
        print("[DenseSemanticBEV] semantic channels empty: check obs[:,4:] projection in BEV_Map and upstream segmentation/logits.")
        return {
            "schema_version": "smoothnav.dense_semantic_raster.v1",
            "class_id_map": np.zeros((0, 0), dtype=np.int16),
            "map_shape": [0, 0],
            "semantic_threshold": float(semantic_threshold),
            "semantic_coverage_ratio": 0.0,
            "semantic_channel_active_count": 0,
            "true_semantic_pixel_count": 0,
            "per_category_pixel_count": {str(name): 0 for name in display_category_names},
            "object_footprint_pixel_count": 0,
            "object_footprint_per_category_pixel_count": {},
            "object_footprint_source": "none",
            "object_footprints": [],
            "semantic_bbox_footprint_pixel_count": 0,
            "semantic_bbox_footprint_per_category_pixel_count": {},
            "semantic_bbox_footprint_source": "none",
            "semantic_bbox_footprints": [],
            "display_semantic_pixel_count": 0,
            "display_semantic_coverage_ratio": 0.0,
            "display_semantic_source": "empty",
            "semantic_source": "empty_map_channels",
            "warnings": warnings,
            "class_definitions": _dense_class_definitions(display_category_names),
        }
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        warnings.extend(["invalid_map_rank", "semantic_channels_empty", "dense_semantic_bev_cannot_show_object_regions_without_semantic_projection"])
        print("[DenseSemanticBEV] semantic channels empty: check obs[:,4:] projection in BEV_Map and upstream segmentation/logits.")
        return {
            "schema_version": "smoothnav.dense_semantic_raster.v1",
            "class_id_map": np.zeros((0, 0), dtype=np.int16),
            "map_shape": [0, 0],
            "semantic_threshold": float(semantic_threshold),
            "semantic_coverage_ratio": 0.0,
            "semantic_channel_active_count": 0,
            "true_semantic_pixel_count": 0,
            "per_category_pixel_count": {str(name): 0 for name in display_category_names},
            "object_footprint_pixel_count": 0,
            "object_footprint_per_category_pixel_count": {},
            "object_footprint_source": "none",
            "object_footprints": [],
            "semantic_bbox_footprint_pixel_count": 0,
            "semantic_bbox_footprint_per_category_pixel_count": {},
            "semantic_bbox_footprint_source": "none",
            "semantic_bbox_footprints": [],
            "display_semantic_pixel_count": 0,
            "display_semantic_coverage_ratio": 0.0,
            "display_semantic_source": "empty",
            "semantic_source": "empty_map_channels",
            "warnings": warnings,
            "class_definitions": _dense_class_definitions(display_category_names),
        }

    height, width = int(arr.shape[-2]), int(arr.shape[-1])
    class_id_map = np.full((height, width), DENSE_UNKNOWN_CLASS_ID, dtype=np.int16)
    if arr.shape[0] > 1:
        class_id_map[np.asarray(arr[1]) > 0] = DENSE_FREE_CLASS_ID
        explored_mask_for_footprints = np.asarray(arr[1]) > 0
    else:
        known = np.any(arr > 0, axis=0)
        class_id_map[known] = DENSE_FREE_CLASS_ID
        explored_mask_for_footprints = known.astype(bool)
    if arr.shape[0] > 0:
        obstacle_mask_for_footprints = np.asarray(arr[0]) > 0
        class_id_map[obstacle_mask_for_footprints] = DENSE_OBSTACLE_CLASS_ID
    else:
        obstacle_mask_for_footprints = np.zeros((height, width), dtype=bool)

    per_category_pixel_count = {str(name): 0 for name in display_category_names}
    active_count = 0
    semantic_pixel_mask = np.zeros((height, width), dtype=bool)
    semantic_bbox_footprint_mask = np.zeros((height, width), dtype=bool)
    semantic_bbox_footprint_per_category_pixel_count: Dict[str, int] = {}
    semantic_bbox_footprints: List[Dict[str, Any]] = []
    sem = arr[SEMANTIC_CHANNEL_OFFSET:] if arr.shape[0] > SEMANTIC_CHANNEL_OFFSET else None
    threshold = float(semantic_threshold)
    bbox_threshold = float(semantic_bbox_fill_threshold)
    if sem is not None and sem.size > 0:
        sem = np.asarray(sem, dtype=np.float32)
        usable = min(int(sem.shape[0]), len(category_names))
        if usable > 0:
            sem_crop = sem[:usable]
            sem_max = np.max(sem_crop, axis=0)
            sem_idx = np.argmax(sem_crop, axis=0)
            semantic_pixel_mask = sem_max > threshold
            for idx in range(usable):
                channel_mask = sem_crop[idx] > threshold
                count = int(np.count_nonzero(channel_mask))
                per_category_pixel_count[str(category_names[idx])] = count
                if count > 0:
                    active_count += 1
                display_mask = np.logical_and(semantic_pixel_mask, sem_idx == idx)
                if np.any(display_mask):
                    class_id_map[display_mask] = DENSE_SEMANTIC_CLASS_OFFSET + int(idx)
                if semantic_bbox_fill:
                    bbox_seed_mask = sem_crop[idx] > bbox_threshold
                    if not np.any(bbox_seed_mask):
                        continue
                    label = str(category_names[idx])
                    class_id = DENSE_SEMANTIC_CLASS_OFFSET + int(idx)
                    for comp in _mask_components(bbox_seed_mask):
                        if comp.size == 0:
                            continue
                        rows = comp[:, 0]
                        cols = comp[:, 1]
                        seed_count = int(len(rows))
                        if seed_count < 3:
                            continue
                        bbox = _bbox_rc(
                            [
                                int(rows.min()),
                                int(cols.min()),
                                int(rows.max()),
                                int(cols.max()),
                            ],
                            shape=(height, width),
                        )
                        if bbox is None:
                            continue
                        raw_bbox = list(bbox)
                        bbox_fill_mode = str(semantic_bbox_fill_mode or "component_bbox")
                        if bbox_fill_mode in {"interior_bbox", "category_footprint_bbox"}:
                            bbox = _expand_bbox_to_label_min_size(
                                bbox,
                                label=label,
                                shape=(height, width),
                            )
                        if bbox_fill_mode == "interior_bbox":
                            bbox = _shift_bbox_to_explored_interior(
                                bbox,
                                obstacle_mask=obstacle_mask_for_footprints,
                                explored_mask=explored_mask_for_footprints,
                                shape=(height, width),
                            )
                        r0, c0, r1, c1 = bbox
                        fill_count = int((r1 - r0 + 1) * (c1 - c0 + 1))
                        # Guard against the exact failure mode where sparse,
                        # disconnected evidence for a category gets inflated
                        # into one large, wrong rectangle.  Connected-component
                        # bboxes should stay close to their seed support.
                        max_multiplier = 30 if bbox_fill_mode in {"interior_bbox", "category_footprint_bbox"} else 12
                        if fill_count > max(64, seed_count * max_multiplier):
                            continue
                        fill_mask = np.zeros((height, width), dtype=bool)
                        fill_mask[r0 : r1 + 1, c0 : c1 + 1] = True
                        class_id_map[fill_mask] = class_id
                        semantic_bbox_footprint_mask |= fill_mask
                        semantic_bbox_footprint_per_category_pixel_count[label] = (
                            int(semantic_bbox_footprint_per_category_pixel_count.get(label, 0)) + fill_count
                        )
                        semantic_bbox_footprints.append(
                            {
                                "id": f"SBF{len(semantic_bbox_footprints) + 1:03d}",
                                "category_label": label,
                                "category_index": int(idx),
                                "bbox_rc": bbox,
                                "raw_surface_bbox_rc": raw_bbox,
                                "seed_pixel_count": seed_count,
                                "pixel_count": fill_count,
                                "source": "semantic_channel_component_bbox_fill",
                                "fill_mode": bbox_fill_mode,
                                "threshold": float(bbox_threshold),
                            }
                        )

    object_footprint_mask = np.zeros((height, width), dtype=bool)
    object_footprint_per_category_pixel_count: Dict[str, int] = {}
    object_footprints: List[Dict[str, Any]] = []
    rejected_object_footprints: List[Dict[str, Any]] = []
    object_summary_count = int(len(object_items))
    if include_object_footprints:
        for obj_idx, obj in enumerate(object_items, start=1):
            item = obj if isinstance(obj, Mapping) else {}
            raw_source = str(item.get("footprint_source") or "").strip()
            footprint_cells = _footprint_cells_from_object(item, shape=(height, width))
            bbox = _bbox_rc(item.get("bbox_rc"), shape=(height, width))
            source = raw_source or ("graph_object_cells" if footprint_cells else "")
            # A bbox-only footprint is allowed only when it was explicitly
            # produced from upstream 3-D object support.  Center-derived or
            # missing-source boxes are rejected so replay cannot hallucinate a
            # SemExp/PONI-style dense semantic map from labels alone.
            if not footprint_cells and bbox is not None:
                trusted_bbox_sources = {
                    "graph_pcd_bbox_cells",
                    "graph_pcd_cells",
                    "graph_3d_bbox_cells",
                    "graph_3d_bbox",
                    "graph_object_bbox",
                    "semantic_instance_depth_bbox_cells",
                }
                if source in trusted_bbox_sources:
                    r0, c0, r1, c1 = bbox
                    footprint_cells = [
                        (int(r), int(c))
                        for r in range(int(r0), int(r1) + 1)
                        for c in range(int(c0), int(c1) + 1)
                    ]
                else:
                    rejected_object_footprints.append(
                        {
                            "id": item.get("id") or f"O{obj_idx:03d}",
                            "caption": item.get("caption") or item.get("label") or "",
                            "reason": "bbox_without_trusted_3d_footprint_source",
                            "bbox_rc": bbox,
                            "source": source or "none",
                        }
                    )
            if not footprint_cells:
                continue
            label = _dense_label_key(item.get("caption") or item.get("label") or "object")
            if not label:
                label = "object"
            if label not in display_category_names:
                display_category_names.append(label)
                per_category_pixel_count.setdefault(label, 0)
            cat_idx = display_category_names.index(label)
            class_id = DENSE_SEMANTIC_CLASS_OFFSET + int(cat_idx)
            mask = np.zeros((height, width), dtype=bool)
            rr = [int(cell[0]) for cell in footprint_cells]
            cc = [int(cell[1]) for cell in footprint_cells]
            mask[rr, cc] = True
            count = int(np.count_nonzero(mask))
            if count <= 0:
                continue
            if bbox is None:
                bbox = _bbox_rc([min(rr), min(cc), max(rr), max(cc)], shape=(height, width))
            explored_overlap = int(np.count_nonzero(np.logical_and(mask, explored_mask_for_footprints)))
            obstacle_overlap = int(np.count_nonzero(np.logical_and(mask, obstacle_mask_for_footprints)))
            free_overlap = int(np.count_nonzero(np.logical_and(mask, np.logical_and(explored_mask_for_footprints, ~obstacle_mask_for_footprints))))
            unknown_overlap = int(count - explored_overlap)
            unknown_ratio = float(unknown_overlap) / float(max(1, count))
            seed_count = _safe_int(item.get("footprint_seed_cell_count"), None)
            inflation_ratio = float(count) / float(max(1, int(seed_count))) if seed_count else None
            if unknown_ratio > 0.80:
                rejected_object_footprints.append(
                    {
                        "id": item.get("id") or f"O{obj_idx:03d}",
                        "caption": item.get("caption") or item.get("label") or "",
                        "reason": "footprint_mostly_unknown_unexplored",
                        "bbox_rc": bbox,
                        "source": source or "none",
                        "pixel_count": int(count),
                        "unknown_overlap_ratio": round(float(unknown_ratio), 4),
                    }
                )
                continue
            class_id_map[mask] = class_id
            object_footprint_mask |= mask
            object_footprint_per_category_pixel_count[label] = (
                int(object_footprint_per_category_pixel_count.get(label, 0)) + count
            )
            object_footprints.append(
                {
                    "id": item.get("id") or f"O{obj_idx:03d}",
                    "caption": item.get("caption") or item.get("label") or "",
                    "category_label": label,
                    "bbox_rc": bbox,
                    "pixel_count": count,
                    "source": source or "graph_object_cells",
                    "confidence": _safe_float(item.get("footprint_confidence"), item.get("confidence", 0.0)) or 0.0,
                    "footprint_encoding": str(item.get("footprint_encoding") or ("rc_indices" if item.get("footprint_rc_indices") else "bbox_fill")),
                    "seed_cell_count": seed_count,
                    "bbox_to_seed_inflation_ratio": round(float(inflation_ratio), 4) if inflation_ratio is not None else None,
                    "explored_overlap_ratio": round(float(explored_overlap) / float(max(1, count)), 4),
                    "free_overlap_ratio": round(float(free_overlap) / float(max(1, count)), 4),
                    "obstacle_overlap_ratio": round(float(obstacle_overlap) / float(max(1, count)), 4),
                    "unknown_overlap_ratio": round(float(unknown_overlap) / float(max(1, count)), 4),
                }
            )

    true_semantic_pixel_count = int(np.count_nonzero(semantic_pixel_mask))
    semantic_coverage_ratio = (
        float(true_semantic_pixel_count) / float(max(1, height * width))
    )
    semantic_source = "map_channels" if true_semantic_pixel_count > 0 else "empty_map_channels"
    if true_semantic_pixel_count <= 0:
        warnings.extend([
            "semantic_channels_empty",
            "dense_semantic_bev_cannot_show_object_regions_without_semantic_projection",
        ])
        print("[DenseSemanticBEV] semantic channels empty: check obs[:,4:] projection in BEV_Map and upstream segmentation/logits.")
    object_footprint_pixel_count = int(np.count_nonzero(object_footprint_mask))
    if object_summary_count > 0 and object_footprint_pixel_count <= 0:
        warnings.append("object_summaries_missing_projected_footprints")
    semantic_bbox_footprint_pixel_count = int(np.count_nonzero(semantic_bbox_footprint_mask))
    display_semantic_mask = np.logical_or(
        np.logical_or(semantic_pixel_mask, semantic_bbox_footprint_mask),
        object_footprint_mask,
    )
    display_semantic_pixel_count = int(np.count_nonzero(display_semantic_mask))
    display_sources = []
    if true_semantic_pixel_count > 0:
        display_sources.append("map_channels")
    if semantic_bbox_footprint_pixel_count > 0:
        display_sources.append("semantic_channel_bbox_fill")
    if object_footprint_pixel_count > 0:
        display_sources.append("graph_object_footprints")
    display_semantic_source = "+".join(display_sources) if display_sources else semantic_source

    return {
        "schema_version": "smoothnav.dense_semantic_raster.v1",
        "class_id_map": class_id_map,
        "map_shape": [height, width],
        "semantic_threshold": float(threshold),
        "semantic_coverage_ratio": float(semantic_coverage_ratio),
        "semantic_channel_active_count": int(active_count),
        "true_semantic_pixel_count": int(true_semantic_pixel_count),
        "per_category_pixel_count": per_category_pixel_count,
        "object_footprint_pixel_count": int(object_footprint_pixel_count),
        "object_footprint_per_category_pixel_count": object_footprint_per_category_pixel_count,
        "object_footprint_source": "graph_object_footprints" if object_footprint_pixel_count > 0 else "none",
        "object_footprints": object_footprints,
        "rejected_object_footprints": rejected_object_footprints,
        "semantic_bbox_footprint_pixel_count": int(semantic_bbox_footprint_pixel_count),
        "semantic_bbox_footprint_per_category_pixel_count": semantic_bbox_footprint_per_category_pixel_count,
        "semantic_bbox_footprint_source": "semantic_channel_bbox_fill" if semantic_bbox_footprint_pixel_count > 0 else "none",
        "semantic_bbox_footprints": semantic_bbox_footprints,
        "display_semantic_pixel_count": int(display_semantic_pixel_count),
        "display_semantic_coverage_ratio": float(display_semantic_pixel_count) / float(max(1, height * width)),
        "display_semantic_source": display_semantic_source,
        "semantic_source": semantic_source,
        "warnings": warnings,
        "class_definitions": _dense_class_definitions(display_category_names),
    }


def dense_semantic_raster_metadata(
    dense_raster: Mapping[str, Any],
    *,
    unlocalized_room_hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
    pseudo_support: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return JSON-safe metadata for a dense semantic raster."""

    pseudo_support = dict(pseudo_support or {})
    metadata = {
        "schema_version": "smoothnav.dense_semantic_bev_artifact.v1",
        "map_shape": list(dense_raster.get("map_shape") or []),
        "semantic_threshold": dense_raster.get("semantic_threshold"),
        "semantic_source": dense_raster.get("semantic_source"),
        "semantic_coverage_ratio": dense_raster.get("semantic_coverage_ratio", 0.0),
        "semantic_channel_active_count": dense_raster.get("semantic_channel_active_count", 0),
        "true_semantic_pixel_count": dense_raster.get("true_semantic_pixel_count", 0),
        "per_category_pixel_count": dict(dense_raster.get("per_category_pixel_count") or {}),
        "object_footprint_pixel_count": dense_raster.get("object_footprint_pixel_count", 0),
        "object_footprint_per_category_pixel_count": dict(dense_raster.get("object_footprint_per_category_pixel_count") or {}),
        "object_footprint_source": dense_raster.get("object_footprint_source", "none"),
        "object_footprints": [_jsonable(item) for item in (dense_raster.get("object_footprints") or [])],
        "rejected_object_footprints": [_jsonable(item) for item in (dense_raster.get("rejected_object_footprints") or [])],
        "semantic_bbox_footprint_pixel_count": dense_raster.get("semantic_bbox_footprint_pixel_count", 0),
        "semantic_bbox_footprint_per_category_pixel_count": dict(dense_raster.get("semantic_bbox_footprint_per_category_pixel_count") or {}),
        "semantic_bbox_footprint_source": dense_raster.get("semantic_bbox_footprint_source", "none"),
        "semantic_bbox_footprints": [_jsonable(item) for item in (dense_raster.get("semantic_bbox_footprints") or [])],
        "display_semantic_pixel_count": dense_raster.get("display_semantic_pixel_count", dense_raster.get("true_semantic_pixel_count", 0)),
        "display_semantic_coverage_ratio": dense_raster.get("display_semantic_coverage_ratio", dense_raster.get("semantic_coverage_ratio", 0.0)),
        "display_semantic_source": dense_raster.get("display_semantic_source", dense_raster.get("semantic_source")),
        "warnings": list(dense_raster.get("warnings") or []),
        "class_definitions": {
            str(k): _jsonable(v)
            for k, v in dict(dense_raster.get("class_definitions") or {}).items()
        },
        "unlocalized_room_hypotheses": [_jsonable(item) for item in (unlocalized_room_hypotheses or [])],
        "pseudo_support_pixel_count": int(pseudo_support.get("pseudo_support_pixel_count", 0) or 0),
        "pseudo_support_enabled": bool(pseudo_support.get("pseudo_support_enabled", False)),
        "pseudo_support_source": str(pseudo_support.get("pseudo_support_source", "")),
        "pseudo_support_objects": list(pseudo_support.get("pseudo_support_objects", []) or []),
        # Hard guarantees for the dense map renderer.  These labels may exist
        # in debug_overlay_full.png, but never in dense_semantic_bev.png.
        "unlocalized_rooms_drawn_count": 0,
        "object_labels_drawn_on_dense_map_count": 0,
        "branch_labels_drawn_on_dense_map_count": 0,
    }
    return metadata


def _class_id_color_table(dense_raster: Mapping[str, Any]) -> Dict[int, Tuple[int, int, int]]:
    table: Dict[int, Tuple[int, int, int]] = {}
    for key, value in dict(dense_raster.get("class_definitions") or {}).items():
        try:
            class_id = int(key)
        except Exception:
            class_id = int(value.get("id", 0) or 0) if isinstance(value, Mapping) else 0
        color = tuple((value.get("color_rgb") if isinstance(value, Mapping) else None) or (255, 255, 255))
        table[class_id] = (int(color[0]), int(color[1]), int(color[2]))
    return table


def _dense_rgb_from_class_map(dense_raster: Mapping[str, Any]) -> Optional[np.ndarray]:
    class_id_map = dense_raster.get("class_id_map")
    if class_id_map is None:
        return None
    class_id_map = np.asarray(class_id_map, dtype=np.int16)
    if class_id_map.ndim != 2:
        return None
    rgb = np.zeros((class_id_map.shape[0], class_id_map.shape[1], 3), dtype=np.uint8)
    color_table = _class_id_color_table(dense_raster)
    for class_id in np.unique(class_id_map):
        rgb[class_id_map == int(class_id)] = np.asarray(
            color_table.get(int(class_id), (255, 255, 255)),
            dtype=np.uint8,
        )
    return rgb


def render_dense_semantic_bev(
    dense_raster: Mapping[str, Any],
    agent_pose: Any = None,
    trajectory: Optional[Sequence[Any]] = None,
    frontier_mask: Any = None,
    save_path: Optional[str | Path] = None,
    *,
    scale: int = 1,
    crop_to_content: bool = False,
    crop_margin: int = 24,
) -> Any:
    """Render a SemExp/PONI-style dense semantic BEV raster.

    The main map contains only cell-level occupancy/semantic colors plus thin
    agent/trajectory/frontier overlays.  No O#/B#/R# labels, scores, warnings,
    or side-panel debug text are drawn.
    """

    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover
        return None
    rgb = _dense_rgb_from_class_map(dense_raster)
    if rgb is None:
        return None
    class_id_map_full = np.asarray(dense_raster.get("class_id_map"), dtype=np.int16)
    r0 = c0 = 0
    if crop_to_content and class_id_map_full.ndim == 2 and class_id_map_full.size:
        content = class_id_map_full != DENSE_UNKNOWN_CLASS_ID
        agent_rc = _coord_rc(agent_pose)
        if agent_rc is not None and 0 <= agent_rc[0] < content.shape[0] and 0 <= agent_rc[1] < content.shape[1]:
            content[int(agent_rc[0]), int(agent_rc[1])] = True
        if np.any(content):
            rr, cc = np.where(content)
            margin = max(0, int(crop_margin))
            r0 = max(0, int(rr.min()) - margin)
            r1 = min(content.shape[0], int(rr.max()) + margin + 1)
            c0 = max(0, int(cc.min()) - margin)
            c1 = min(content.shape[1], int(cc.max()) + margin + 1)
            rgb = rgb[r0:r1, c0:c1]
            class_id_map_full = class_id_map_full[r0:r1, c0:c1]
    image = Image.fromarray(rgb, mode="RGB")
    scale = max(1, int(scale or 1))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)

    def xy(coord: Any) -> Optional[Tuple[int, int]]:
        rc = _coord_rc(coord)
        if rc is None:
            return None
        row = float(rc[0]) - float(r0)
        col = float(rc[1]) - float(c0)
        if row < 0 or col < 0 or row >= rgb.shape[0] or col >= rgb.shape[1]:
            return None
        return int(round(col * scale)), int(round(row * scale))

    if trajectory:
        points = [xy(item) for item in trajectory]
        points = [item for item in points if item is not None]
        # Capsule replay stores visited cells as a sampled mask, not an
        # ordered trajectory polyline.  Draw individual thin cells to avoid
        # fabricating scanline paths across the dense semantic field.
        for x, y in points:
            if scale <= 1:
                draw.point((x, y), fill=(255, 214, 64))
            else:
                draw.rectangle((x, y, x + scale - 1, y + scale - 1), fill=(255, 214, 64))

    if frontier_mask is not None:
        try:
            fm = np.asarray(frontier_mask)
            if crop_to_content and fm.ndim == 2:
                fm = fm[r0 : r0 + rgb.shape[0], c0 : c0 + rgb.shape[1]]
            if fm.ndim == 2 and fm.shape[:2] == rgb.shape[:2]:
                rr, cc = np.where(fm > 0)
                # Draw sparse frontier pixels only; no branch labels.
                for row, col in zip(rr[:: max(1, len(rr) // 1200)], cc[:: max(1, len(cc) // 1200)]):
                    x, y = int(col * scale), int(row * scale)
                    draw.rectangle((x, y, x + max(1, scale - 1), y + max(1, scale - 1)), fill=(64, 200, 255))
        except Exception:
            pass

    agent_xy = xy(agent_pose)
    if agent_xy is not None:
        radius = max(3, 4 * scale)
        x, y = agent_xy
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 210, 0), outline=(0, 0, 0), width=max(1, scale))

    # Compact legend: class names and colors only, no debug data.
    present_ids = [int(item) for item in np.unique(class_id_map_full).tolist()]
    definitions = dict(dense_raster.get("class_definitions") or {})
    color_table = _class_id_color_table(dense_raster)
    legend_items = []
    for class_id in present_ids:
        definition = definitions.get(class_id) or definitions.get(str(class_id)) or {}
        label = str(definition.get("label") or f"class_{class_id}")
        if class_id >= DENSE_SEMANTIC_CLASS_OFFSET or class_id in (DENSE_UNKNOWN_CLASS_ID, DENSE_FREE_CLASS_ID, DENSE_OBSTACLE_CLASS_ID):
            legend_items.append((label, color_table.get(class_id, (255, 255, 255))))
    legend_items = legend_items[:20]
    if legend_items:
        font = _font()
        row_h = 16
        legend_w = max(160, max(len(label) * 7 for label, _ in legend_items) + 30)
        legend_h = max(28, row_h * len(legend_items) + 8)
        composed = Image.new("RGB", (image.width + legend_w, max(image.height, legend_h)), (45, 45, 45))
        composed.paste(image, (0, 0))
        draw = ImageDraw.Draw(composed)
        lx = image.width + 8
        ly = 8
        for idx, (label, color) in enumerate(legend_items):
            y = ly + idx * row_h
            draw.rectangle((lx, y + 2, lx + 10, y + 12), fill=color, outline=(0, 0, 0))
            draw.text((lx + 16, y), label[:24], fill=(235, 235, 235), font=font)
        image = composed

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
    return image


def save_dense_semantic_bev_artifacts(
    png_path: str | Path,
    json_path: str | Path,
    dense_raster: Mapping[str, Any],
    *,
    agent_pose: Any = None,
    trajectory: Optional[Sequence[Any]] = None,
    frontier_mask: Any = None,
    unlocalized_room_hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
    pseudo_support: Optional[Mapping[str, Any]] = None,
    scale: int = 1,
    crop_to_content: bool = False,
    crop_margin: int = 24,
) -> Dict[str, Any]:
    image = render_dense_semantic_bev(
        dense_raster,
        agent_pose=agent_pose,
        trajectory=trajectory,
        frontier_mask=frontier_mask,
        save_path=png_path,
        scale=scale,
        crop_to_content=crop_to_content,
        crop_margin=crop_margin,
    )
    metadata = dense_semantic_raster_metadata(
        dense_raster,
        unlocalized_room_hypotheses=unlocalized_room_hypotheses,
        pseudo_support=pseudo_support,
    )
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {
        "image": str(png_path) if image is not None else "",
        "json": str(json_path),
        "metadata": metadata,
    }


def build_pseudo_semantic_support_raster(
    full_map: Any,
    objects: Sequence[Mapping[str, Any]],
    *,
    radius: int = 10,
) -> Dict[str, Any]:
    """Build a separate low-confidence object-center support raster.

    This is intentionally not a true dense semantic map and must not affect
    semantic_coverage_ratio.
    """

    shape = _map_shape(full_map)
    if shape is None:
        support = np.zeros((0, 0), dtype=np.float32)
    else:
        support = np.zeros(shape, dtype=np.float32)
    support_objects = []
    for obj in objects or []:
        center = _coord_rc(obj.get("center_rc"))
        if center is None or support.size == 0:
            continue
        confidence = max(0.10, min(0.45, _safe_float(obj.get("confidence"), 0.25) * 0.45))
        _add_radial_heat(support, center, weight=confidence, radius=radius)
        support_objects.append(
            {
                "id": obj.get("id"),
                "caption": obj.get("caption"),
                "center_rc": center,
                "source": "object_center_prior",
                "confidence": float(confidence),
                "support_radius_px": int(radius),
            }
        )
    return {
        "schema_version": "smoothnav.pseudo_semantic_support_raster.v1",
        "pseudo_support_enabled": bool(support_objects),
        "pseudo_support_source": "object_center_prior" if support_objects else "",
        "pseudo_support_map": support,
        "pseudo_support_pixel_count": int(np.count_nonzero(support > 0)),
        "pseudo_support_objects": support_objects,
    }


def render_pseudo_semantic_support_bev(
    pseudo_support: Mapping[str, Any],
    full_map: Any,
    *,
    agent_pose: Any = None,
    save_path: Optional[str | Path] = None,
    scale: int = 1,
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover
        return None
    base = _geometry_canvas_from_map(full_map)
    if base is None:
        return None
    support = np.asarray(pseudo_support.get("pseudo_support_map"), dtype=np.float32)
    _render_color_field_overlay(base, support, color=(0, 188, 212), alpha=0.55)
    image = Image.fromarray(base, mode="RGB")
    scale = max(1, int(scale or 1))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    agent_xy = None
    rc = _coord_rc(agent_pose)
    if rc is not None:
        agent_xy = (int(rc[1] * scale), int(rc[0] * scale))
    if agent_xy is not None:
        r = max(3, 4 * scale)
        draw.ellipse((agent_xy[0] - r, agent_xy[1] - r, agent_xy[0] + r, agent_xy[1] + r), fill=(0, 210, 0), outline=(0, 0, 0), width=max(1, scale))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
    return image


def _mask_center(mask: Optional[np.ndarray]) -> Optional[List[int]]:
    if mask is None or not np.any(mask):
        return None
    rr, cc = np.where(mask)
    return [int(round(float(rr.mean()))), int(round(float(cc.mean())))]


def _sample_mask_coords(mask: Optional[np.ndarray], *, limit: int = 80) -> List[List[int]]:
    if mask is None or not np.any(mask):
        return []
    rr, cc = np.where(mask)
    count = int(len(rr))
    if count <= 0:
        return []
    stride = max(1, int(math.ceil(count / float(max(1, limit)))))
    return [[int(r), int(c)] for r, c in zip(rr[::stride], cc[::stride])][:limit]


def _normalize_label(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _category_name(index: int) -> str:
    idx = int(index)
    if 0 <= idx < len(SEMANTIC_CATEGORY_NAMES):
        return SEMANTIC_CATEGORY_NAMES[idx]
    return f"semantic_{idx}"


def _semantic_channel_index_for_label(label: str) -> Optional[int]:
    wanted = _normalize_label(label)
    if not wanted:
        return None
    for idx, name in enumerate(SEMANTIC_CATEGORY_NAMES):
        candidates = {_normalize_label(name)}
        candidates.update(_normalize_label(s) for s in _CATEGORY_SYNONYMS.get(name, ()))
        if wanted in candidates:
            return idx
    return None


def _label_matches_goal(label: str, goal_text: str) -> bool:
    goal = _normalize_label(goal_text)
    if not goal:
        return False
    label_norm = _normalize_label(label)
    candidates = {label_norm}
    for name, synonyms in _CATEGORY_SYNONYMS.items():
        if _normalize_label(name) == label_norm:
            candidates.update(_normalize_label(s) for s in synonyms)
    return any(candidate and candidate in goal for candidate in candidates)


def _target_category_indices(goal_description: str, objects: Sequence[Mapping[str, Any]]) -> List[int]:
    """Infer intended target category indices from goal text and stored matching hints."""

    labels: List[str] = []
    for name in SEMANTIC_CATEGORY_NAMES:
        if _label_matches_goal(name, goal_description):
            labels.append(name)
    for obj in objects or []:
        for raw in obj.get("target_primary_categories", []) or []:
            label = str(raw or "").strip()
            if label:
                labels.append(label)
        reason = str(obj.get("target_match_reason", "") or "")
        if "television" in reason.lower() or "tv" in reason.lower():
            labels.append("tv")
    # The target-matcher in older capsules writes "television" as a primary
    # category even when goal_description is absent.  Normalize through synonyms.
    out: List[int] = []
    for label in labels:
        idx = _semantic_channel_index_for_label(label)
        if idx is not None and idx not in out:
            out.append(idx)
    return out


def _semantic_channel_stats(map_like: Any, *, target_category_indices: Sequence[int]) -> Dict[str, Any]:
    layers = _map_layers(map_like)
    sem = layers.get("semantic_channels")
    active = []
    total_semantic_pixels = 0
    if sem is not None:
        for idx in range(int(sem.shape[0])):
            channel = np.asarray(sem[idx], dtype=float)
            nz = channel > SEMANTIC_CONFIDENT_THRESHOLD
            nonzero = int(np.count_nonzero(nz))
            total_semantic_pixels += nonzero
            if nonzero <= 0:
                continue
            active.append(
                {
                    "category_index": int(idx),
                    "channel_index": int(SEMANTIC_CHANNEL_OFFSET + idx),
                    "label": _category_name(idx),
                    "pixel_count": int(nonzero),
                    "activation_sum": float(channel[nz].sum()),
                    "activation_max": float(channel[nz].max()),
                    "is_direct_target_category": bool(int(idx) in {int(i) for i in target_category_indices}),
                }
            )
    active.sort(key=lambda item: (-float(item["activation_sum"]), str(item["label"])))
    target_pixels = 0
    target_sum = 0.0
    if sem is not None:
        for idx in target_category_indices:
            if 0 <= int(idx) < int(sem.shape[0]):
                channel = np.asarray(sem[int(idx)], dtype=float)
                nz = channel > SEMANTIC_CONFIDENT_THRESHOLD
                target_pixels += int(np.count_nonzero(nz))
                target_sum += float(channel[nz].sum())
    return {
        "channel_offset": SEMANTIC_CHANNEL_OFFSET,
        "semantic_channel_count": int(sem.shape[0]) if sem is not None else 0,
        "category_names": list(SEMANTIC_CATEGORY_NAMES),
        "active_category_count": int(len(active)),
        "total_semantic_pixels": int(total_semantic_pixels),
        "active_categories": active[:12],
        "has_semantic_channels": bool(total_semantic_pixels > 0),
        "target_category_indices": [int(i) for i in target_category_indices],
        "target_category_labels": [_category_name(int(i)) for i in target_category_indices],
        "direct_target_pixel_count": int(target_pixels),
        "direct_target_activation_sum": float(target_sum),
        "semantic_confident_threshold": float(SEMANTIC_CONFIDENT_THRESHOLD),
    }


def _mask_components(mask: np.ndarray) -> List[np.ndarray]:
    if mask is None or not np.any(mask):
        return []
    coords = np.column_stack(np.where(mask))
    # Runtime BEV calls should not spend planner time flood-filling very large
    # semantic masks.  Large masks are already useful as a category region even
    # without exact connected-component splitting.
    if int(coords.shape[0]) > 20000:
        return [coords.astype(int)]
    coord_set = {(int(r), int(c)) for r, c in coords}
    components: List[List[Tuple[int, int]]] = []
    while coord_set:
        start = coord_set.pop()
        queue = [start]
        comp = [start]
        while queue:
            row, col = queue.pop()
            for dr, dc in _COORD_NEIGHBORS_8:
                key = (row + dr, col + dc)
                if key in coord_set:
                    coord_set.remove(key)
                    queue.append(key)
                    comp.append(key)
        components.append(comp)
    return [np.asarray(comp, dtype=int) for comp in components]


def _semantic_regions_from_map(
    map_like: Any,
    *,
    target_category_indices: Sequence[int],
    max_regions_per_category: int = 4,
) -> List[Dict[str, Any]]:
    layers = _map_layers(map_like)
    sem = layers.get("semantic_channels")
    if sem is None or not np.any(sem > SEMANTIC_CONFIDENT_THRESHOLD):
        return []
    prior_weights = _target_prior_weights(target_category_indices)
    regions: List[Dict[str, Any]] = []
    serial = 1
    for idx in range(int(sem.shape[0])):
        channel = np.asarray(sem[idx], dtype=float)
        if not np.any(channel > SEMANTIC_CONFIDENT_THRESHOLD):
            continue
        components = []
        for comp in _mask_components(channel > SEMANTIC_CONFIDENT_THRESHOLD):
            if comp.size == 0:
                continue
            rows = comp[:, 0]
            cols = comp[:, 1]
            values = channel[rows, cols]
            components.append((float(values.sum()), comp, values))
        components.sort(key=lambda item: -item[0])
        for activation_sum, comp, values in components[: max(1, int(max_regions_per_category))]:
            rows = comp[:, 0]
            cols = comp[:, 1]
            label = _category_name(idx)
            prior_weight = float(prior_weights.get(int(idx), 0.0))
            regions.append(
                {
                    "id": f"S{serial:03d}",
                    "category_index": int(idx),
                    "channel_index": int(SEMANTIC_CHANNEL_OFFSET + idx),
                    "label": label,
                    "center_rc": [
                        int(round(float(rows.mean()))),
                        int(round(float(cols.mean()))),
                    ],
                    "bbox_rc": [
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ],
                    "pixel_count": int(len(rows)),
                    "activation_sum": float(activation_sum),
                    "activation_max": float(values.max()) if values.size else 0.0,
                    "target_prior_weight": float(prior_weight),
                    "is_direct_target_category": bool(int(idx) in {int(i) for i in target_category_indices}),
                    "is_target_anchor_category": bool(prior_weight > 0.0 and int(idx) not in {int(i) for i in target_category_indices}),
                    "source": "full_map.semantic_channel",
                }
            )
            serial += 1
    regions.sort(
        key=lambda item: (
            -float(item.get("is_direct_target_category", False)),
            -float(item.get("target_prior_weight", 0.0) or 0.0),
            -float(item.get("activation_sum", 0.0) or 0.0),
            str(item.get("id", "")),
        )
    )
    for idx, region in enumerate(regions, start=1):
        region["id"] = f"S{idx:03d}"
    return regions


def _add_radial_heat(heatmap: np.ndarray, center: Sequence[int], *, weight: float, radius: int = 32) -> None:
    rc = _coord_rc(center)
    if rc is None:
        return
    row, col = int(rc[0]), int(rc[1])
    h, w = heatmap.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        return
    radius = max(1, int(radius))
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    dist = np.sqrt((yy - row) ** 2 + (xx - col) ** 2)
    patch = np.maximum(0.0, 1.0 - dist / float(radius)) * float(weight)
    heatmap[r0:r1, c0:c1] = np.maximum(heatmap[r0:r1, c0:c1], patch)


def _target_prior_weights(target_indices: Sequence[int]) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    for target_idx in target_indices:
        target_label = _category_name(int(target_idx))
        weights[int(target_idx)] = max(weights.get(int(target_idx), 0.0), 1.0)
        for anchor_label, weight in (_GOAL_ANCHOR_PRIORS.get(target_label) or {}).items():
            anchor_idx = _semantic_channel_index_for_label(anchor_label)
            if anchor_idx is None:
                continue
            weights[int(anchor_idx)] = max(weights.get(int(anchor_idx), 0.0), float(weight))
    return weights


def _target_heatmap_from_map_and_objects(
    map_like: Any,
    *,
    objects: Sequence[Mapping[str, Any]],
    target_category_indices: Sequence[int],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Build a target/query heatmap from direct semantic channels and weak anchors.

    This is intentionally explicit about source strength.  A TV heatmap from a
    TV channel is direct evidence; a heatmap from couch/chair/table channels is
    only a co-occurrence prior that should guide exploration, not target claim.
    """

    shape = _map_shape(map_like)
    if shape is None:
        return None, {"source": "unavailable", "reason": "missing_map_shape"}
    heatmap = np.zeros(shape, dtype=np.float32)
    contributions: List[Dict[str, Any]] = []
    layers = _map_layers(map_like)
    sem = layers.get("semantic_channels")
    prior_weights = _target_prior_weights(target_category_indices)
    direct_pixels = 0
    prior_pixels = 0
    if sem is not None and prior_weights:
        for idx, weight in sorted(prior_weights.items()):
            if not (0 <= int(idx) < int(sem.shape[0])):
                continue
            channel = np.asarray(sem[int(idx)], dtype=np.float32)
            confident_mask = channel > SEMANTIC_CONFIDENT_THRESHOLD
            if not np.any(confident_mask):
                continue
            max_value = float(channel[confident_mask].max())
            normalized = channel / max(1e-6, max_value)
            pixels = int(np.count_nonzero(confident_mask))
            if float(weight) >= 0.99 and int(idx) in {int(i) for i in target_category_indices}:
                direct_pixels += pixels
                source = "direct_target_semantic_channel"
            else:
                prior_pixels += pixels
                source = "semantic_cooccurrence_prior_channel"
            normalized = np.where(confident_mask, normalized, 0.0)
            weighted = normalized * float(weight)
            heatmap = np.maximum(heatmap, weighted)
            contributions.append(
                {
                    "source": source,
                    "category_index": int(idx),
                    "channel_index": int(SEMANTIC_CHANNEL_OFFSET + idx),
                    "label": _category_name(idx),
                    "weight": float(weight),
                    "pixel_count": int(pixels),
                    "activation_sum": float(channel[confident_mask].sum()),
                    "threshold": float(SEMANTIC_CONFIDENT_THRESHOLD),
                }
            )

    object_direct = 0
    object_prior = 0
    for obj in objects or []:
        center = obj.get("center_rc")
        if center is None:
            continue
        caption = str(obj.get("caption", "") or "")
        rel = float(obj.get("target_relevance", 0.0) or 0.0)
        idx = _semantic_channel_index_for_label(caption)
        weight = 0.0
        source = ""
        if rel > 0.25:
            weight = max(0.7, min(1.0, rel))
            source = "direct_target_graph_object"
            object_direct += 1
        elif idx is not None and idx in prior_weights and prior_weights[idx] < 0.99:
            weight = float(prior_weights[idx]) * 0.75
            source = "graph_object_cooccurrence_prior"
            object_prior += 1
        if weight > 0:
            _add_radial_heat(heatmap, center, weight=weight, radius=36)
            contributions.append(
                {
                    "source": source,
                    "object_id": obj.get("id"),
                    "label": caption,
                    "center_rc": center,
                    "weight": float(weight),
                }
            )

    if not np.any(heatmap > 0):
        return None, {
            "source": "none",
            "target_category_indices": [int(i) for i in target_category_indices],
            "target_category_labels": [_category_name(int(i)) for i in target_category_indices],
            "positive_pixel_count": 0,
            "contributions": [],
        }
    heatmap = np.clip(heatmap, 0.0, 1.0)
    source = "direct_target"
    if direct_pixels <= 0 and object_direct <= 0:
        source = "semantic_prior_only"
    elif prior_pixels > 0 or object_prior > 0:
        source = "direct_plus_prior"
    return heatmap, {
        "source": source,
        "target_category_indices": [int(i) for i in target_category_indices],
        "target_category_labels": [_category_name(int(i)) for i in target_category_indices],
        "direct_target_pixel_count": int(direct_pixels),
        "prior_pixel_count": int(prior_pixels),
        "graph_direct_object_count": int(object_direct),
        "graph_prior_object_count": int(object_prior),
        "positive_pixel_count": int(np.count_nonzero(heatmap > 0)),
        "peak": float(heatmap.max()),
        "contributions": contributions[:12],
    }


def _sample_heatmap_score(heatmap: Optional[np.ndarray], coord: Any, *, radius: int = 24) -> float:
    rc = _coord_rc(coord)
    if heatmap is None or rc is None:
        return 0.0
    row, col = int(rc[0]), int(rc[1])
    h, w = heatmap.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        return 0.0
    radius = max(0, int(radius))
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    if r0 >= r1 or c0 >= c1:
        return 0.0
    patch = heatmap[r0:r1, c0:c1]
    if patch.size == 0:
        return 0.0
    return float(np.max(patch))


def _direction_from_agent_coord(coord: Any, agent_coord: Any) -> str:
    center = _coord_rc(coord)
    agent = _coord_rc(agent_coord)
    if center is None or agent is None:
        return ""
    dr = float(center[0]) - float(agent[0])
    dc = float(center[1]) - float(agent[1])
    if abs(dr) >= abs(dc):
        return "south" if dr > 0 else "north"
    return "east" if dc > 0 else "west"


def _value_from(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    return obj if isinstance(obj, Mapping) else {}


def _dict_or_empty(obj: Any) -> Dict[str, Any]:
    return dict(obj) if isinstance(obj, Mapping) else {}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    f = _safe_float(value)
    if f is None:
        return default
    return int(round(f))


def _coord_rc(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("center_rc") or value.get("coord_rc") or value.get("center") or value.get("coord")
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if len(value) < 2:
            return None
        row = _safe_int(value[0])
        col = _safe_int(value[1])
        if row is None or col is None:
            return None
        return [row, col]
    except Exception:
        return None


def _in_bounds(coord: Optional[Sequence[int]], shape: Optional[Tuple[int, int]]) -> bool:
    if coord is None or shape is None:
        return False
    try:
        row, col = int(coord[0]), int(coord[1])
        return 0 <= row < int(shape[0]) and 0 <= col < int(shape[1])
    except Exception:
        return False


def _near_mask(coord: Optional[Sequence[int]], mask: Optional[np.ndarray], radius: int = 4) -> bool:
    if coord is None or mask is None:
        return False
    row, col = int(coord[0]), int(coord[1])
    h, w = mask.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        return False
    if bool(mask[row, col]):
        return True
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    return bool(np.any(mask[r0:r1, c0:c1]))


def _branch_color(index: int) -> Tuple[int, int, int]:
    palette = (
        (255, 193, 7),
        (33, 150, 243),
        (244, 67, 54),
        (156, 39, 176),
        (76, 175, 80),
        (255, 112, 67),
        (0, 188, 212),
        (205, 220, 57),
        (121, 85, 72),
        (233, 30, 99),
    )
    return palette[int(index) % len(palette)]


def _distance(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return math.dist([float(a[0]), float(a[1])], [float(b[0]), float(b[1])])
    except Exception:
        return None


def _extract_objects_from_summary(
    object_summary: Iterable[Any],
    *,
    map_shape: Optional[Tuple[int, int]],
    goal_description: str = "",
) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    for idx, raw in enumerate(object_summary or [], start=1):
        item = raw if isinstance(raw, Mapping) else {}
        caption = str(item.get("caption") or item.get("label") or "").strip()
        center_source_key = (
            "center_rc"
            if item.get("center_rc") is not None
            else ("coord_rc" if item.get("coord_rc") is not None else ("center" if item.get("center") is not None else "coord"))
        )
        center = _coord_rc(item.get("center_rc") or item.get("coord_rc") or item.get("center") or item.get("coord"))
        coordinate_frame = str(item.get("coordinate_frame") or item.get("frame") or "").strip()
        if (
            center is not None
            and center_source_key in {"center", "coord"}
            and coordinate_frame not in {COORDINATE_FRAME_NAME, "full_map_rc_unflipped"}
            and map_shape is not None
        ):
            # Older world_state.object_summary entries stored graph node centers
            # in Graph/FMM frame: [world_x/res, map_h-1-world_y/res].  Convert
            # them before using them as BEV image row/col coordinates.
            center = [int(map_shape[0] - 1 - int(center[1])), int(center[0])]
        bbox = _bbox_rc(item.get("bbox_rc") or item.get("bev_bbox_rc") or item.get("map_bbox_rc"), shape=map_shape)
        target_relevance = _safe_float(item.get("target_relevance"), 0.0) or 0.0
        if goal_description and score_caption_against_goal is not None and not item.get("target_match_reason"):
            try:
                match = score_caption_against_goal(caption, goal_description)
                target_relevance = float(match.get("score", target_relevance) or 0.0)
            except Exception:
                pass
        detections = _safe_int(item.get("num_detections"), 0) or 0
        obj_id = str(item.get("id") or item.get("object_id") or f"O{idx:03d}")
        confidence = min(1.0, 0.25 + 0.15 * detections + 0.40 * target_relevance)
        objects.append(
            {
                "id": obj_id,
                "node_index": int(item.get("node_index", idx - 1) or idx - 1),
                "caption": caption or f"object_{idx}",
                "center_rc": center,
                "bbox_rc": bbox,
                "footprint_source": str(item.get("footprint_source", "") or ""),
                "footprint_pixel_count": _safe_int(item.get("footprint_pixel_count"), 0) or 0,
                "footprint_confidence": _safe_float(item.get("footprint_confidence"), None),
                "footprint_rc_indices": _jsonable(item.get("footprint_rc_indices") or item.get("footprint_rcs") or []),
                "footprint_encoding": str(item.get("footprint_encoding", "") or ""),
                "footprint_seed_cell_count": _safe_int(item.get("footprint_seed_cell_count"), 0) or 0,
                "pcd_point_count": _safe_int(item.get("pcd_point_count"), 0) or 0,
                "footprint_rejected_reason": str(item.get("footprint_rejected_reason", "") or ""),
                "detection_xyxy_count": _safe_int(item.get("detection_xyxy_count"), 0) or 0,
                "detection_pixel_area_sum": _safe_int(item.get("detection_pixel_area_sum"), 0) or 0,
                "num_detections": detections,
                "target_relevance": float(target_relevance),
                "target_match_reason": str(item.get("target_match_reason", "") or ""),
                "target_primary_categories": list(item.get("target_primary_categories", []) or []),
                "source": str(item.get("source", "world_state.object_summary")),
                "coordinate_frame": COORDINATE_FRAME_NAME,
                "raw_coordinate_frame": coordinate_frame or ("graph_object_rc_flipped_yx" if center_source_key in {"center", "coord"} else COORDINATE_FRAME_NAME),
                "confidence": float(confidence),
                "localized": bool(center is not None and _in_bounds(center, map_shape) if map_shape is not None else center is not None),
                "within_map_bounds": bool(_in_bounds(center, map_shape)) if center is not None and map_shape is not None else None,
                "display_priority": float(10.0 * target_relevance + min(detections, 10)),
            }
        )
    return objects


def _extract_objects_from_graph(graph: Any, *, map_shape: Optional[Tuple[int, int]], goal_description: str = "") -> List[Dict[str, Any]]:
    try:
        from smoothnav.world_state import summarize_objects as _summarize_objects

        graph_summary = _summarize_objects(graph)
        if graph_summary:
            return _extract_objects_from_summary(
                graph_summary,
                map_shape=map_shape,
                goal_description=goal_description,
            )
    except Exception:
        pass
    nodes = list(getattr(graph, "nodes", []) or [])
    summary = []
    goal_text = goal_description or getattr(graph, "text_goal", None) or getattr(graph, "obj_goal", "") or ""
    for idx, node in enumerate(nodes):
        caption = str(getattr(node, "caption", "") or "")
        center = _coord_rc(getattr(node, "center", None))
        detections = 0
        try:
            detections = int(getattr(node, "object", {}).get("num_detections", 0) or 0)
        except Exception:
            detections = 0
        match = {"score": 0.0, "reason": "", "primary_categories": []}
        if score_caption_against_goal is not None:
            try:
                match = score_caption_against_goal(caption, goal_text)
            except Exception:
                pass
        summary.append(
            {
                "id": f"O{idx + 1:03d}",
                "node_index": idx,
                "caption": caption,
                "center": center,
                "num_detections": detections,
                "target_relevance": match.get("score", 0.0),
                "target_match_reason": match.get("reason", ""),
                "target_primary_categories": match.get("primary_categories", []),
                "source": "graph.nodes",
            }
        )
    return _extract_objects_from_summary(summary, map_shape=map_shape, goal_description=goal_description)


def _room_objects_from_summary(room: Mapping[str, Any], objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = [str(name).strip().lower() for name in (room.get("objects") or []) if str(name).strip()]
    if not wanted:
        return []
    result = []
    used = set()
    for wanted_caption in wanted:
        for obj in objects:
            if obj["id"] in used:
                continue
            if str(obj.get("caption", "")).strip().lower() == wanted_caption:
                result.append(obj)
                used.add(obj["id"])
                break
    return result


def _extract_rooms_from_summary(room_summary: Iterable[Any], objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rooms: List[Dict[str, Any]] = []
    for idx, raw in enumerate(room_summary or [], start=1):
        item = raw if isinstance(raw, Mapping) else {}
        caption = str(item.get("caption") or item.get("label") or f"room_{idx}").strip()
        member_objects = _room_objects_from_summary(item, objects)
        coords = [obj["center_rc"] for obj in member_objects if obj.get("center_rc") is not None]
        center = None
        bbox = None
        if coords:
            arr = np.asarray(coords, dtype=float)
            center = [int(round(float(arr[:, 0].mean()))), int(round(float(arr[:, 1].mean())))]
            pad = 10
            bbox = [
                int(round(float(arr[:, 0].min()))) - pad,
                int(round(float(arr[:, 1].min()))) - pad,
                int(round(float(arr[:, 0].max()))) + pad,
                int(round(float(arr[:, 1].max()))) + pad,
            ]
        room_id = str(item.get("id") or item.get("room_id") or f"R{idx:03d}")
        localized = bool(center is not None)
        spatialization_status = "localized_region" if localized else "unlocalized_prior"
        rooms.append(
            {
                "id": room_id,
                "caption": caption,
                "object_ids": [obj["id"] for obj in member_objects],
                "object_captions": list(item.get("objects") or []),
                "object_count": int(item.get("object_count", len(member_objects)) or 0),
                "center_rc": center,
                "bbox_rc": bbox,
                "geometry_type": "member_object_hypothesis" if center is not None else "unlocalized_hypothesis",
                "spatialization_status": spatialization_status,
                "planner_evidence_role": "spatial_room_prior" if localized else "text_room_prior_only",
                "source": str(item.get("source", "world_state.room_summary")),
                "confidence": "medium" if len(coords) >= 2 else ("low" if len(coords) == 1 else "unlocalized"),
                "localized": localized,
            }
        )
    return rooms


def _extract_rooms_from_graph(graph: Any, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    node_to_object = {}
    for obj in objects:
        node_to_object[int(obj.get("node_index", -1))] = obj.get("caption", "")
    for idx, room_node in enumerate(getattr(graph, "room_nodes", []) or [], start=1):
        captions = []
        for node in list(getattr(room_node, "nodes", []) or []):
            caption = str(getattr(node, "caption", "") or "")
            if caption:
                captions.append(caption)
        summary.append(
            {
                "id": f"R{idx:03d}",
                "caption": str(getattr(room_node, "caption", "") or f"room_{idx}"),
                "object_count": len(captions),
                "objects": captions,
                "source": "graph.room_nodes",
            }
        )
    return _extract_rooms_from_summary(summary, objects)


def _known_coords_from_layers(known_mask: Optional[np.ndarray], map_shape: Optional[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    if known_mask is None or map_shape is None or not np.any(known_mask):
        return [], []
    rr, cc = np.where(known_mask)
    return [int(rr.min()), int(rr.max())], [int(cc.min()), int(cc.max())]


def _content_crop(
    *,
    map_shape: Optional[Tuple[int, int]],
    known_mask: Optional[np.ndarray],
    agent: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    rooms: Sequence[Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    semantic_regions: Sequence[Mapping[str, Any]] = (),
    margin: int = 24,
) -> Dict[str, int]:
    if map_shape is None:
        return {"r0": 0, "r1": 0, "c0": 0, "c1": 0}
    h, w = map_shape
    rows, cols = _known_coords_from_layers(known_mask, map_shape)

    def add(coord: Any) -> None:
        rc = _coord_rc(coord)
        if rc is None:
            return
        rows.append(int(rc[0]))
        cols.append(int(rc[1]))

    add(agent.get("coord_rc"))
    for obj in objects:
        add(obj.get("center_rc"))
    for room in rooms:
        add(room.get("center_rc"))
        bbox = room.get("bbox_rc")
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            rows.extend([int(bbox[0]), int(bbox[2])])
            cols.extend([int(bbox[1]), int(bbox[3])])
    for region in semantic_regions:
        add(region.get("center_rc"))
        bbox = region.get("bbox_rc")
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            rows.extend([int(bbox[0]), int(bbox[2])])
            cols.extend([int(bbox[1]), int(bbox[3])])
    for branch in branches:
        add(branch.get("representative_coord") or branch.get("centroid"))
        add(branch.get("centroid"))
        for point in branch.get("candidate_points", []) or []:
            if isinstance(point, Mapping):
                add(point.get("coord"))
    if not rows or not cols:
        return {"r0": 0, "r1": int(h), "c0": 0, "c1": int(w)}
    pad = max(0, int(margin))
    return {
        "r0": max(0, min(rows) - pad),
        "r1": min(int(h), max(rows) + pad + 1),
        "c0": max(0, min(cols) - pad),
        "c1": min(int(w), max(cols) + pad + 1),
    }


def _branch_decision_overlay(
    *,
    scoring_input: Optional[Mapping[str, Any]],
    grounding_result: Optional[Mapping[str, Any]],
    planner_result: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    scoring_input = _dict_or_empty(scoring_input)
    grounding_result = _dict_or_empty(grounding_result)
    planner_result = _dict_or_empty(planner_result)
    graph_debug = dict(grounding_result.get("graph_debug") or {})
    mllm = dict(graph_debug.get("mllm_frontier") or {})
    score_breakdown = dict(
        grounding_result.get("selected_frontier_score_breakdown")
        or graph_debug.get("selected_frontier_score_breakdown")
        or {}
    )
    verdict = dict(planner_result.get("verdict") or {})
    selected_from_plan = (
        str(verdict.get("selected_branch_id") or "")
        or str(planner_result.get("selected_branch_id") or "")
        or str(planner_result.get("selected_frontier_id") or "")
    )
    mllm_selected = str(mllm.get("selected_branch_id") or selected_from_plan or scoring_input.get("planner_selected_branch_id") or "")
    final_selected = str(score_breakdown.get("branch_id") or grounding_result.get("selected_frontier_branch_id") or graph_debug.get("selected_frontier_branch_id") or "")
    dry_selected = str(mllm.get("dry_selected_branch_id") or "")
    if not dry_selected and not mllm:
        dry_selected = final_selected
    return {
        "dry_selected_branch_id": dry_selected,
        "mllm_selected_branch_id": mllm_selected,
        "final_selected_branch_id": final_selected,
        "planner_prior_applied": bool(mllm.get("applied", bool(scoring_input.get("planner_selected_branch_id")))),
        "mllm_status": str(mllm.get("status") or planner_result.get("status") or ""),
        "mllm_reason": str(mllm.get("reason") or planner_result.get("reason") or ""),
        "ranked_branch_ids": list(mllm.get("ranked_branch_ids") or verdict.get("ranked_branch_ids") or []),
        "allowed_branch_ids": list(mllm.get("allowed_branch_ids") or planner_result.get("allowed_branch_ids") or []),
        "dry_selected_frontier": _jsonable(mllm.get("dry_selected_frontier")),
        "final_selected_frontier": _jsonable(grounding_result.get("selected_frontier") or graph_debug.get("selected_frontier")),
        "prompt_hash": str(mllm.get("prompt_hash") or planner_result.get("prompt_hash") or ""),
    }


def _annotate_branches(
    branches_payload: Mapping[str, Any],
    *,
    objects: Sequence[Mapping[str, Any]],
    rooms: Sequence[Mapping[str, Any]],
    decision_overlay: Mapping[str, Any],
    semantic_regions: Sequence[Mapping[str, Any]] = (),
    agent_coord: Any = None,
    target_heatmap: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dry = str(decision_overlay.get("dry_selected_branch_id") or "")
    mllm = str(decision_overlay.get("mllm_selected_branch_id") or "")
    final = str(decision_overlay.get("final_selected_branch_id") or "")
    for idx, raw in enumerate(branches_payload.get("branches", []) or []):
        if not isinstance(raw, Mapping):
            continue
        branch = dict(raw)
        bid = str(branch.get("id") or f"B{idx + 1}")
        rep = _coord_rc(branch.get("representative_coord") or branch.get("centroid"))
        canonical_direction = _direction_from_agent_coord(rep, agent_coord)
        branch["canonical_representative_coord_rc"] = rep
        branch["canonical_agent_coord_rc"] = _coord_rc(agent_coord)
        branch["canonical_direction_from_agent"] = canonical_direction
        branch["canonical_coordinate_frame"] = COORDINATE_FRAME_NAME
        near_objs = []
        for obj in objects:
            dist = _distance(rep, obj.get("center_rc"))
            if dist is None:
                continue
            near_objs.append((dist, obj))
        near_objs.sort(key=lambda item: (item[0], -float(item[1].get("target_relevance", 0.0) or 0.0), str(item[1].get("id", ""))))
        near_rooms = []
        for room in rooms:
            dist = _distance(rep, room.get("center_rc"))
            if dist is None:
                continue
            near_rooms.append((dist, room))
        near_rooms.sort(key=lambda item: (item[0], str(item[1].get("id", ""))))
        near_regions = []
        for region in semantic_regions or []:
            dist = _distance(rep, region.get("center_rc"))
            if dist is None:
                continue
            near_regions.append((dist, region))
        near_regions.sort(
            key=lambda item: (
                item[0],
                -float(item[1].get("target_prior_weight", 0.0) or 0.0),
                str(item[1].get("id", "")),
            )
        )
        branch["nearby_object_ids"] = [obj["id"] for dist, obj in near_objs[:3] if dist <= 90]
        branch["nearby_room_ids"] = [room["id"] for dist, room in near_rooms[:2] if dist <= 140]
        branch["nearby_semantic_region_ids"] = [region["id"] for dist, region in near_regions[:4] if dist <= 120]
        branch["nearby_semantic_region_labels"] = [
            region["label"] for dist, region in near_regions[:4] if dist <= 120
        ]
        branch["nearest_object_distance"] = near_objs[0][0] if near_objs else None
        branch["nearest_room_distance"] = near_rooms[0][0] if near_rooms else None
        branch["nearest_semantic_region_distance"] = near_regions[0][0] if near_regions else None
        branch["map_direction_from_agent"] = canonical_direction
        heat_score = _sample_heatmap_score(target_heatmap, rep, radius=12)
        candidate_heat_scores = [
            _sample_heatmap_score(target_heatmap, point.get("coord"), radius=8)
            for point in branch.get("candidate_points", []) or []
            if isinstance(point, Mapping)
        ]
        if candidate_heat_scores:
            heat_score = max(heat_score, max(candidate_heat_scores))
        branch["target_heatmap_score"] = float(round(float(heat_score), 4))
        target_affinity = 0.0
        for _, obj in near_objs[:5]:
            rel = float(obj.get("target_relevance", 0.0) or 0.0)
            if rel > target_affinity:
                target_affinity = rel
        for dist, region in near_regions[:5]:
            weight = float(region.get("target_prior_weight", 0.0) or 0.0)
            if weight <= 0:
                continue
            # Nearby semantic anchors are weaker than direct target heat unless
            # the branch representative actually overlaps the target heatmap.
            attenuation = max(0.25, 1.0 - min(float(dist), 120.0) / 140.0)
            pixel_count = max(0, int(region.get("pixel_count", 0) or 0))
            # A one-pixel semantic blip should not dominate branch semantics.
            # Mature BEV planners reason over cell clusters/fields; scale
            # branch affinity by the support size while keeping compact but
            # confident object fragments usable.
            support_scale = min(1.0, math.sqrt(float(pixel_count)) / 4.0) if pixel_count > 0 else 0.0
            target_affinity = max(target_affinity, weight * attenuation * support_scale)
        terms = dict(branch.get("score_terms_aggregate") or {})
        target_affinity = max(
            target_affinity,
            float(terms.get("target_progress_score_max") or 0.0),
            float(terms.get("planner_prior_score_max") or 0.0) * 0.25,
            float(heat_score),
        )
        branch["semantic_target_affinity"] = float(round(target_affinity, 4))
        roles = []
        if bid and bid == dry:
            roles.append("dry")
        if bid and bid == mllm:
            roles.append("mllm")
        if bid and bid == final:
            roles.append("final")
        branch["decision_roles"] = roles
        candidate_count = int(branch.get("candidate_point_count", 0) or 0)
        actionable = bool(candidate_count > 0 and float(terms.get("actionability_score_max") or 0.0) > 0.0)
        branch["visit_state"] = "non_actionable" if not actionable else ("selected" if roles else "candidate")
        out.append(branch)
    return out


def _localized_world_summary(world_state: Any, graph: Any, goal_description: str, map_shape: Optional[Tuple[int, int]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if world_state is not None:
        object_summary = _value_from(world_state, "object_summary", []) or []
        room_summary = _value_from(world_state, "room_summary", []) or []
        semantic_instance_footprints = _value_from(world_state, "semantic_instance_footprints", []) or []
    else:
        object_summary = []
        room_summary = []
        semantic_instance_footprints = []
    if object_summary:
        objects = _extract_objects_from_summary(object_summary, map_shape=map_shape, goal_description=goal_description)
    else:
        objects = _extract_objects_from_graph(graph, map_shape=map_shape, goal_description=goal_description) if graph is not None else []
    if semantic_instance_footprints:
        existing_keys = {
            (
                str(obj.get("caption", "")).strip().lower(),
                tuple(obj.get("bbox_rc") or []),
            )
            for obj in objects
        }
        for obj in _extract_objects_from_summary(
            semantic_instance_footprints,
            map_shape=map_shape,
            goal_description=goal_description,
        ):
            key = (str(obj.get("caption", "")).strip().lower(), tuple(obj.get("bbox_rc") or []))
            if key in existing_keys:
                continue
            objects.append(obj)
            existing_keys.add(key)
    if room_summary:
        rooms = _extract_rooms_from_summary(room_summary, objects)
    else:
        rooms = _extract_rooms_from_graph(graph, objects) if graph is not None else []
    return objects, rooms


def _target_value_selection_contract(
    target_value_summary: Mapping[str, Any],
    branch_decision_table: Sequence[Mapping[str, Any]],
    decision_overlay: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Explain how the target-value ranking relates to the executed branch.

    The target-conditioned value map is a semantic/exploration state estimate.
    Graph.get_goal still owns the executable frontier scorer.  This contract
    makes ranking drift explicit instead of letting offline replay report every
    value mismatch as an unexplained planner bug.
    """

    summary = dict(target_value_summary or {})
    table = [dict(row) for row in (branch_decision_table or [])]
    rows_by_id = {str(row.get("id") or ""): row for row in table if row.get("id") is not None}

    value_ranking = list(summary.get("branch_ranking") or [])
    if not value_ranking and table:
        value_ranking = sorted(
            table,
            key=lambda row: (
                -float(row.get("target_value_score") or row.get("target_value_max") or 0.0),
                evidence_rank(row.get("evidence_level")) if evidence_rank is not None else 99,
                str(row.get("id") or ""),
            ),
        )
    final_id = str(decision_overlay.get("final_selected_branch_id") or "")
    mllm_id = str(decision_overlay.get("mllm_selected_branch_id") or "")
    dry_id = str(decision_overlay.get("dry_selected_branch_id") or "")
    top_value_id = str((value_ranking[0] or {}).get("id") or "") if value_ranking else ""

    frontier_ranked = sorted(
        [row for row in table if _safe_float(row.get("frontier_final_score")) is not None],
        key=lambda row: (-float(row.get("frontier_final_score") or 0.0), str(row.get("id") or "")),
    )
    top_frontier_id = str((frontier_ranked[0] or {}).get("id") or "") if frontier_ranked else ""
    final_row = rows_by_id.get(final_id, {})
    top_value_row = rows_by_id.get(top_value_id, {})

    def _score(row: Mapping[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            value = _safe_float(row.get(key))
            if value is not None:
                return float(value)
        return None

    top_value_score = _score(top_value_row, "target_value_score", "target_value_max")
    final_value_score = _score(final_row, "target_value_score", "target_value_max")
    value_gap = (
        max(0.0, float(top_value_score) - float(final_value_score))
        if top_value_score is not None and final_value_score is not None
        else None
    )
    final_frontier_score = _score(final_row, "frontier_final_score")
    top_frontier_score = _score(frontier_ranked[0], "frontier_final_score") if frontier_ranked else None
    final_actionability = _score(final_row, "frontier_actionability_score", "geometry_value")
    top_value_actionability = _score(top_value_row, "frontier_actionability_score", "geometry_value")
    final_executable_rank = _safe_int(
        final_row.get("executable_decision_rank") or final_row.get("frontier_final_score_rank")
    )

    aligned = bool(final_id and top_value_id and final_id == top_value_id)
    drift_is_explained = bool(aligned or not final_id or not top_value_id)
    override_reason = "target_value_aligned" if aligned else ""
    if not aligned and final_id and top_value_id:
        if value_gap is not None and value_gap <= 0.08:
            drift_is_explained = True
            override_reason = "within_target_value_tolerance"
        elif top_frontier_id and final_id == top_frontier_id:
            drift_is_explained = True
            override_reason = "graph_frontier_score_contract"
        elif (
            final_actionability is not None
            and top_value_actionability is not None
            and final_actionability - top_value_actionability >= 0.15
        ):
            drift_is_explained = True
            override_reason = "actionability_contract"
        else:
            drift_is_explained = False
            override_reason = "unexplained_rank_drift"
    elif not final_id:
        override_reason = "missing_final_branch"
    elif not top_value_id:
        override_reason = "missing_target_value_ranking"

    contract = {
        "schema_version": "smoothnav.target_value_selection_contract.v1",
        "final_branch_id": final_id,
        "mllm_branch_id": mllm_id,
        "dry_branch_id": dry_id,
        "top_target_value_branch_id": top_value_id,
        "top_frontier_score_branch_id": top_frontier_id,
        "top_executable_decision_branch_id": top_frontier_id,
        "final_executable_decision_rank": final_executable_rank,
        "target_value_aligned": aligned,
        "drift_is_explained": bool(drift_is_explained),
        "override_reason": override_reason,
        "target_value_gap": round(float(value_gap), 4) if value_gap is not None else None,
        "final_target_value_score": (
            round(float(final_value_score), 4) if final_value_score is not None else None
        ),
        "top_target_value_score": (
            round(float(top_value_score), 4) if top_value_score is not None else None
        ),
        "final_frontier_score": (
            round(float(final_frontier_score), 4) if final_frontier_score is not None else None
        ),
        "top_frontier_score": (
            round(float(top_frontier_score), 4) if top_frontier_score is not None else None
        ),
        "final_actionability_score": (
            round(float(final_actionability), 4) if final_actionability is not None else None
        ),
        "top_target_value_actionability_score": (
            round(float(top_value_actionability), 4) if top_value_actionability is not None else None
        ),
        "contract_note": (
            "target_value ranks semantic search value; Graph.get_goal final branch may override it "
            "when executable frontier score or actionability is stronger"
        ),
    }
    summary["selection_contract"] = contract
    return summary, contract


def build_semantic_bev_frame(
    *,
    full_map: Any = None,
    frontier_branches: Optional[Mapping[str, Any]] = None,
    world_state: Any = None,
    graph: Any = None,
    scoring_input: Optional[Mapping[str, Any]] = None,
    grounding_result: Optional[Mapping[str, Any]] = None,
    planner_result: Optional[Mapping[str, Any]] = None,
    strategy: Any = None,
    episode_id: Optional[int] = None,
    step_idx: Optional[int] = None,
    goal_description: str = "",
    source: str = "runtime",
    crop_margin: int = 24,
) -> Dict[str, Any]:
    """Build a replayable semantic BEV frame JSON payload."""

    frontier_branches = dict(frontier_branches or {})
    scoring_input = _dict_or_empty(scoring_input)
    grounding_result = _dict_or_empty(grounding_result)
    planner_result = _dict_or_empty(planner_result)
    shape = _map_shape(full_map)
    known_mask, free_mask, obstacle_mask = _base_masks(full_map)
    layers = _map_layers(full_map)
    current_mask = layers.get("current_mask")
    visited_mask = layers.get("visited_mask")
    map_current_coord = _mask_center(current_mask)
    if step_idx is None:
        step_idx = _value_from(world_state, "step_idx", None)
    raw_agent_coord = _coord_rc(scoring_input.get("agent_coord"))
    if raw_agent_coord is None:
        raw_agent_coord = _coord_rc((frontier_branches or {}).get("agent_coord"))
    if raw_agent_coord is None:
        raw_agent_coord = _coord_rc(_value_from(world_state, "agent_coord", None))
    objects, rooms = _localized_world_summary(world_state, graph, goal_description, shape)
    target_category_indices = _target_category_indices(goal_description, objects)
    semantic_layers = _semantic_channel_stats(
        full_map,
        target_category_indices=target_category_indices,
    )
    semantic_regions = _semantic_regions_from_map(
        full_map,
        target_category_indices=target_category_indices,
    )
    target_heatmap, target_heatmap_summary = _target_heatmap_from_map_and_objects(
        full_map,
        objects=objects,
        target_category_indices=target_category_indices,
    )
    raw_agent_on_known = bool(_near_mask(raw_agent_coord, known_mask, radius=4)) if known_mask is not None else None
    current_on_known = bool(_near_mask(map_current_coord, known_mask, radius=4)) if known_mask is not None else None
    agent_channel_distance = _distance(raw_agent_coord, map_current_coord)
    agent_coord = raw_agent_coord
    agent_source = "scoring_input.agent_coord" if scoring_input.get("agent_coord") is not None else "unknown"
    agent_display_repaired = False
    if map_current_coord is not None and current_on_known is not False:
        # Older capsules stored scoring_input.agent_coord in the vertically
        # flipped FMM frame while frontier branches stayed in full-map frame.
        # The current-location map channel is the authoritative BEV-frame
        # marker for visualization, so use it when the two disagree materially.
        if agent_coord is None or raw_agent_on_known is False or (agent_channel_distance is not None and agent_channel_distance > 12.0):
            agent_coord = map_current_coord
            agent_source = "full_map.channel_2_current_agent"
            agent_display_repaired = bool(raw_agent_coord is not None and raw_agent_coord != map_current_coord)
    decision_overlay = _branch_decision_overlay(
        scoring_input=scoring_input,
        grounding_result=grounding_result,
        planner_result=planner_result,
    )
    agent = {
        "coord_rc": agent_coord,
        "raw_coord_rc": raw_agent_coord,
        "map_current_channel_coord_rc": map_current_coord,
        "agent_channel_distance": agent_channel_distance,
        "display_coord_source": agent_source,
        "display_repaired_from_map_current_channel": bool(agent_display_repaired),
        "heading_deg": None,
        "source": agent_source,
        "within_map_bounds": bool(_in_bounds(agent_coord, shape)) if shape is not None else None,
        "on_known_or_near_known": bool(_near_mask(agent_coord, known_mask, radius=4)) if known_mask is not None else None,
        "on_free_or_near_free": bool(_near_mask(agent_coord, free_mask, radius=4)) if free_mask is not None else None,
        "raw_on_known_or_near_known": raw_agent_on_known,
    }
    branches = _annotate_branches(
        frontier_branches,
        objects=objects,
        rooms=rooms,
        decision_overlay=decision_overlay,
        semantic_regions=semantic_regions,
        agent_coord=agent_coord,
        target_heatmap=target_heatmap,
    )
    target_value_summary: Dict[str, Any] = {
        "schema_version": TARGET_VALUE_SCHEMA_VERSION,
        "available": False,
        "reason": "bev_decision_state_builder_unavailable",
        "branch_ranking": [],
    }
    branch_decision_table: List[Dict[str, Any]] = []
    decision_state_payload: Dict[str, Any] = {}
    selection_contract: Dict[str, Any] = {}
    if build_bev_decision_state is not None:
        decision_state_payload, _decision_fields = build_bev_decision_state(
            full_map=full_map,
            step_idx=step_idx,
            target_description=goal_description,
            agent=agent,
            branches=branches,
            objects=objects,
            rooms=rooms,
            strategy=strategy,
        )
        target_value_summary = dict(
            decision_state_payload.get("target_value_field")
            or decision_state_payload.get("target_value_state")
            or {}
        )
        branch_decision_table = list(
            decision_state_payload.get("branch_table")
            or decision_state_payload.get("branch_decision_table")
            or []
        )
        branch_table_by_id = {str(item.get("id") or ""): item for item in branch_decision_table}
        for branch in branches:
            row = branch_table_by_id.get(str(branch.get("id") or ""))
            if not row:
                continue
            branch["target_value_score"] = row.get("target_value_score")
            branch["target_value_mean"] = row.get("target_value_mean")
            branch["target_value_max"] = row.get("target_value_max")
            branch["target_value_rank"] = row.get("target_value_rank")
            branch["evidence_level"] = row.get("evidence_level")
            branch["target_visible"] = row.get("target_visible")
            branch["decision_table_row"] = row
        target_value_summary, selection_contract = _target_value_selection_contract(
            target_value_summary,
            branch_decision_table,
            decision_overlay,
        )
        if decision_state_payload:
            decision_state_payload["target_value_field"] = target_value_summary
            decision_state_payload["target_value_state"] = target_value_summary
            decision_state_payload["selection_contract"] = selection_contract
    heat_ranked = sorted(
        [
            {
                "id": str(branch.get("id", "")),
                "target_heatmap_score": float(branch.get("target_heatmap_score", 0.0) or 0.0),
                "semantic_target_affinity": float(branch.get("semantic_target_affinity", 0.0) or 0.0),
            }
            for branch in branches
        ],
        key=lambda item: (-item["target_heatmap_score"], -item["semantic_target_affinity"], item["id"]),
    )
    for rank, item in enumerate(heat_ranked, start=1):
        for branch in branches:
            if str(branch.get("id", "")) == item["id"]:
                branch["target_heatmap_rank"] = int(rank)
                break
    target_heatmap_summary["branch_ranking"] = heat_ranked[:8]
    crop = _content_crop(
        map_shape=shape,
        known_mask=known_mask,
        agent=agent,
        objects=objects,
        rooms=rooms,
        branches=branches,
        semantic_regions=semantic_regions,
        margin=crop_margin,
    )
    localized_objects = [obj for obj in objects if obj.get("center_rc") is not None]
    localized_rooms = [room for room in rooms if room.get("center_rc") is not None]
    quality_warnings = []
    if agent["within_map_bounds"] is False:
        quality_warnings.append("agent_out_of_map_bounds")
    if agent["on_known_or_near_known"] is False:
        quality_warnings.append("agent_not_on_known_or_near_known")
    if agent_display_repaired:
        quality_warnings.append("agent_display_coord_repaired_from_map_current_channel")
    if raw_agent_coord is not None and agent_channel_distance is not None and agent_channel_distance > 12.0:
        quality_warnings.append(f"raw_agent_coord_disagrees_with_map_current_channel:{agent_channel_distance:.1f}")
    if objects and not localized_objects:
        quality_warnings.append("objects_present_but_unlocalized")
    if rooms and not localized_rooms:
        quality_warnings.append("rooms_present_but_unlocalized")
    if semantic_layers.get("semantic_channel_count", 0) > 0 and not semantic_layers.get("has_semantic_channels"):
        quality_warnings.append("semantic_channels_present_but_empty")
    if target_category_indices and int(semantic_layers.get("direct_target_pixel_count", 0) or 0) <= 0:
        quality_warnings.append("direct_target_semantic_channel_empty")
    if target_heatmap_summary.get("source") == "semantic_prior_only":
        quality_warnings.append("target_heatmap_uses_prior_not_direct_target")
    if target_value_summary.get("available") and not any(
        str(item.get("evidence_level")) in {"direct", "anchor", "room_prior"}
        for item in branch_decision_table
    ):
        quality_warnings.append("target_value_geometry_only")
    if selection_contract and selection_contract.get("drift_is_explained") is False:
        quality_warnings.append("target_value_final_branch_unexplained_drift")
    out_of_bounds_objects = [obj["id"] for obj in objects if obj.get("within_map_bounds") is False]
    if out_of_bounds_objects:
        quality_warnings.append("object_centers_out_of_bounds:" + ",".join(out_of_bounds_objects[:8]))
    trajectory = _sample_mask_coords(visited_mask, limit=80)
    current_channel_pixels = int(np.count_nonzero(current_mask)) if current_mask is not None else 0
    visited_channel_pixels = int(np.count_nonzero(visited_mask)) if visited_mask is not None else 0
    localized_rooms = [room for room in rooms if room.get("center_rc") is not None]
    unlocalized_rooms = [room for room in rooms if room.get("center_rc") is None]
    localized_objects = [obj for obj in objects if obj.get("center_rc") is not None]
    if decision_state_payload:
        decision_state = dict(decision_state_payload)
        semantic_field_state = dict(decision_state.get("semantic_field") or {})
        room_state = {
            "schema_version": "smoothnav.room_spatialization_state.v1",
            "localized_room_regions": list(semantic_field_state.get("localized_rooms") or []),
            "unlocalized_room_priors": list(semantic_field_state.get("unlocalized_room_hypotheses") or []),
        }
        decision_state["room_state"] = room_state
        decision_state["quality_flags"] = quality_warnings
    else:
        room_state = {
            "schema_version": "smoothnav.room_spatialization_state.v1",
            "localized_room_regions": [
                {
                    "id": room.get("id"),
                    "caption": room.get("caption"),
                    "center_rc": room.get("center_rc"),
                    "bbox_rc": room.get("bbox_rc"),
                    "confidence": room.get("confidence"),
                    "source": room.get("source"),
                    "object_ids": room.get("object_ids", []),
                }
                for room in localized_rooms
            ],
            "unlocalized_room_priors": [
                {
                    "id": room.get("id"),
                    "caption": room.get("caption"),
                    "confidence": room.get("confidence"),
                    "source": room.get("source"),
                    "object_captions": room.get("object_captions", []),
                }
                for room in unlocalized_rooms
            ],
        }
        decision_state = {
            "schema_version": DECISION_STATE_SCHEMA_VERSION,
            "coordinate_contract": {
                "frame": COORDINATE_FRAME_NAME,
                "agent_coord_rc": agent_coord,
                "branch_coord_key": "canonical_representative_coord_rc",
                "branch_direction_key": "canonical_direction_from_agent",
                "hidden_from_planner_prompt": [
                    "raw_coord_rc",
                    "map_current_channel_coord_rc",
                    "direction_from_agent",
                ],
            },
            "geometry_field": {"map_shape": list(shape) if shape is not None else None},
            "semantic_field": {
                "active_category_count": int(semantic_layers.get("active_category_count", 0) or 0),
                "semantic_reliability": "dense" if semantic_layers.get("has_semantic_channels") else "empty",
                "localized_objects": localized_objects,
                "localized_rooms": room_state["localized_room_regions"],
                "unlocalized_room_hypotheses": room_state["unlocalized_room_priors"],
            },
            "room_state": room_state,
            "target_value_field": target_value_summary,
            "target_value_state": target_value_summary,
            "branch_table": branch_decision_table,
            "branch_decision_table": branch_decision_table,
            "quality_flags": quality_warnings,
        }
    frame: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step_idx": step_idx,
        "coordinate_frame": {
            "name": COORDINATE_FRAME_NAME,
            "row_axis": "south_down_image_y",
            "col_axis": "east_right_image_x",
            "origin": "full_map[0,0]",
            "map_shape": list(shape) if shape is not None else None,
            "crop": crop,
        },
        "base_layers": {
            "has_obstacle": bool(obstacle_mask is not None),
            "has_free": bool(free_mask is not None),
            "has_known": bool(known_mask is not None),
            "has_semantic_channels": bool(semantic_layers.get("has_semantic_channels")),
            "has_language_query_heatmap": bool(target_heatmap is not None),
            "semantic_channel_count": int(semantic_layers.get("semantic_channel_count", 0) or 0),
            "active_semantic_category_count": int(semantic_layers.get("active_category_count", 0) or 0),
            "current_agent_channel_pixels": int(current_channel_pixels),
            "past_agent_channel_pixels": int(visited_channel_pixels),
        },
        "semantic_layers": semantic_layers,
        "semantic_regions": semantic_regions,
        "target_query_heatmap": target_heatmap_summary,
        "agent": agent,
        "trajectory": trajectory,
        "trajectory_summary": {
            "source": "full_map.channel_3_past_agent",
            "past_agent_pixel_count": int(visited_channel_pixels),
            "current_agent_pixel_count": int(current_channel_pixels),
            "sample_count": int(len(trajectory)),
        },
        "objects": objects,
        "rooms": rooms,
        "branches": branches,
        "room_state": room_state,
        "target_value_state": target_value_summary,
        "branch_decision_table": branch_decision_table,
        "selection_contract": selection_contract,
        "decision_state": decision_state,
        "decision_overlay": decision_overlay,
        "quality_checks": {
            "agent_within_map_bounds": agent["within_map_bounds"],
            "agent_on_known_or_near_known": agent["on_known_or_near_known"],
            "agent_on_free_or_near_free": agent["on_free_or_near_free"],
            "object_count": len(objects),
            "localized_object_count": len(localized_objects),
            "room_count": len(rooms),
            "localized_room_hypothesis_count": len(localized_rooms),
            "unlocalized_room_prior_count": len(unlocalized_rooms),
            "semantic_region_count": len(semantic_regions),
            "target_anchor_semantic_region_count": sum(1 for item in semantic_regions if item.get("target_prior_weight", 0.0)),
            "branch_count": len(branches),
            "branches_with_semantic_hints": sum(
                1
                for b in branches
                if b.get("nearby_object_ids") or b.get("nearby_room_ids") or b.get("nearby_semantic_region_ids")
            ),
            "branch_evidence_counts": dict(target_value_summary.get("evidence_counts") or {}),
            "warnings": quality_warnings,
        },
        "planner_context": {
            "target_region": str(getattr(strategy, "target_region", "") or _value_from(strategy, "target_region", "") or ""),
            "anchor_object": str(getattr(strategy, "anchor_object", "") or _value_from(strategy, "anchor_object", "") or ""),
            "goal_description": str(goal_description or ""),
        },
        "provenance": {
            "source": str(source or "runtime"),
            "renderer_version": RENDERER_VERSION,
            "world_state_hash": canonical_json_hash(_jsonable(world_state)) if world_state is not None else "",
            "frontier_branches_hash": canonical_json_hash(frontier_branches),
            "scoring_input_hash": canonical_json_hash(scoring_input) if scoring_input else "",
        },
    }
    out = _jsonable(frame)
    out["provenance"]["frame_hash"] = canonical_json_hash(out)
    return out


def semantic_bev_summary_for_prompt(frame: Mapping[str, Any], *, max_objects: int = 10, max_rooms: int = 6) -> Dict[str, Any]:
    """Compact semantic BEV facts for MLLM prompt text."""

    objects = list(frame.get("objects", []) or [])
    objects.sort(key=lambda item: (-float(item.get("target_relevance", 0.0) or 0.0), -float(item.get("display_priority", 0.0) or 0.0), str(item.get("id", ""))))
    rooms = list(frame.get("rooms", []) or [])
    branches = list(frame.get("branches", []) or [])
    decision_state = dict(frame.get("decision_state") or {})
    room_state = dict(frame.get("room_state") or {})
    quality_checks = dict(frame.get("quality_checks") or {})
    raw_warnings = list(quality_checks.get("warnings") or [])
    planner_warnings = [
        warning
        for warning in raw_warnings
        if not str(warning).startswith("agent_display_coord_repaired")
        and not str(warning).startswith("raw_agent_coord_disagrees_with_map_current_channel")
    ]
    quality_checks["warnings"] = planner_warnings
    if len(planner_warnings) != len(raw_warnings):
        quality_checks["debug_coordinate_warnings_hidden_from_prompt"] = int(
            len(raw_warnings) - len(planner_warnings)
        )
    return {
        "schema_version": "smoothnav.semantic_bev_prompt_summary.v1",
        "coordinate_frame": dict(frame.get("coordinate_frame") or {}),
        "coordinate_contract": dict(decision_state.get("coordinate_contract") or {}),
        "base_layers": dict(frame.get("base_layers") or {}),
        "quality_checks": quality_checks,
        "semantic_layers": {
            "active_category_count": (frame.get("semantic_layers") or {}).get("active_category_count"),
            "active_categories": (frame.get("semantic_layers") or {}).get("active_categories", [])[:8],
            "target_category_labels": (frame.get("semantic_layers") or {}).get("target_category_labels", []),
            "direct_target_pixel_count": (frame.get("semantic_layers") or {}).get("direct_target_pixel_count", 0),
        },
        "semantic_regions": [
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "center_rc": item.get("center_rc"),
                "bbox_rc": item.get("bbox_rc"),
                "pixel_count": item.get("pixel_count"),
                "target_prior_weight": item.get("target_prior_weight"),
                "is_direct_target_category": item.get("is_direct_target_category"),
                "is_target_anchor_category": item.get("is_target_anchor_category"),
            }
            for item in list(frame.get("semantic_regions", []) or [])[:12]
        ],
        "target_query_heatmap": {
            "source": (frame.get("target_query_heatmap") or {}).get("source"),
            "target_category_labels": (frame.get("target_query_heatmap") or {}).get("target_category_labels", []),
            "positive_pixel_count": (frame.get("target_query_heatmap") or {}).get("positive_pixel_count", 0),
            "direct_target_pixel_count": (frame.get("target_query_heatmap") or {}).get("direct_target_pixel_count", 0),
            "prior_pixel_count": (frame.get("target_query_heatmap") or {}).get("prior_pixel_count", 0),
            "branch_ranking": (frame.get("target_query_heatmap") or {}).get("branch_ranking", [])[:8],
        },
        "target_value_state": {
            "schema_version": (frame.get("target_value_state") or {}).get("schema_version"),
            "available": (frame.get("target_value_state") or {}).get("available"),
            "source": (frame.get("target_value_state") or {}).get("source"),
            "strongest_source": (frame.get("target_value_state") or {}).get("strongest_source"),
            "confidence": (frame.get("target_value_state") or {}).get("confidence"),
            "heat_source": (frame.get("target_value_state") or {}).get("heat_source"),
            "positive_pixel_count": (
                (frame.get("target_value_state") or {}).get("positive_pixel_count")
                or ((frame.get("target_value_state") or {}).get("combined_value") or {}).get("positive_pixel_count", 0)
            ),
            "peak": (
                (frame.get("target_value_state") or {}).get("peak")
                or ((frame.get("target_value_state") or {}).get("combined_value") or {}).get("peak", 0.0)
            ),
            "component_peaks": (frame.get("target_value_state") or {}).get("component_peaks", {}),
            "evidence_counts": (frame.get("target_value_state") or {}).get("evidence_counts", {}),
            "branch_ranking": (frame.get("target_value_state") or {}).get("branch_ranking", [])[:8],
            "executable_decision_ranking": (frame.get("target_value_state") or {}).get(
                "executable_decision_ranking",
                [],
            )[:8],
            "selection_contract": (frame.get("target_value_state") or {}).get(
                "selection_contract",
                frame.get("selection_contract", {}),
            ),
        },
        "trajectory_summary": dict(frame.get("trajectory_summary") or {}),
        "agent": {
            "coord_rc": (frame.get("agent") or {}).get("coord_rc"),
            "source": (frame.get("agent") or {}).get("source"),
            "within_map_bounds": (frame.get("agent") or {}).get("within_map_bounds"),
            "on_known_or_near_known": (frame.get("agent") or {}).get("on_known_or_near_known"),
            "on_free_or_near_free": (frame.get("agent") or {}).get("on_free_or_near_free"),
        },
        "object_labels": [
            {
                "id": item.get("id"),
                "caption": item.get("caption"),
                "center_rc": item.get("center_rc"),
                "target_relevance": item.get("target_relevance"),
                "num_detections": item.get("num_detections"),
            }
            for item in objects[:max_objects]
        ],
        "room_hypotheses": [
            {
                "id": item.get("id"),
                "caption": item.get("caption"),
                "center_rc": item.get("center_rc"),
                "geometry_type": item.get("geometry_type"),
                "spatialization_status": item.get("spatialization_status"),
                "confidence": item.get("confidence"),
                "object_ids": item.get("object_ids"),
            }
            for item in rooms[:max_rooms]
        ],
        "room_state": {
            "localized_room_regions": list(room_state.get("localized_room_regions") or [])[:max_rooms],
            "unlocalized_room_priors": list(room_state.get("unlocalized_room_priors") or [])[:max_rooms],
        },
        "branch_decision_table": [
            {
                "id": item.get("id"),
                "canonical_direction": item.get("canonical_direction"),
                "canonical_coord_rc": item.get("canonical_coord_rc"),
                "candidate_count": item.get("candidate_count"),
                "target_value_score": item.get("target_value_score"),
                "target_value_rank": item.get("target_value_rank"),
                "frontier_final_score": item.get("frontier_final_score"),
                "frontier_final_score_rank": item.get("frontier_final_score_rank"),
                "frontier_actionability_score": item.get("frontier_actionability_score"),
                "executable_decision_score": item.get("executable_decision_score"),
                "executable_decision_rank": item.get("executable_decision_rank"),
                "executable_decision_source": item.get("executable_decision_source"),
                "evidence_level": item.get("evidence_level"),
                "target_visible": item.get("target_visible"),
                "semantic_support": item.get("semantic_support"),
                "info_gain_score": item.get("info_gain_score"),
                "revisit_penalty": item.get("revisit_penalty"),
                "dead_end_risk": item.get("dead_end_risk"),
            }
            for item in list(frame.get("branch_decision_table", []) or [])[:12]
        ],
        "branch_semantic_hints": [
            {
                "id": item.get("id"),
                "canonical_direction": item.get("canonical_direction_from_agent") or item.get("map_direction_from_agent"),
                "canonical_coord_rc": item.get("canonical_representative_coord_rc") or item.get("representative_coord"),
                "nearby_object_ids": item.get("nearby_object_ids", []),
                "nearby_room_ids": item.get("nearby_room_ids", []),
                "nearby_semantic_region_ids": item.get("nearby_semantic_region_ids", []),
                "nearby_semantic_region_labels": item.get("nearby_semantic_region_labels", []),
                "semantic_target_affinity": item.get("semantic_target_affinity"),
                "target_heatmap_score": item.get("target_heatmap_score"),
                "target_heatmap_rank": item.get("target_heatmap_rank"),
                "target_value_score": item.get("target_value_score"),
                "target_value_rank": item.get("target_value_rank"),
                "evidence_level": item.get("evidence_level"),
                "decision_roles": item.get("decision_roles", []),
                "visit_state": item.get("visit_state", ""),
            }
            for item in branches
        ],
        "decision_overlay": dict(frame.get("decision_overlay") or {}),
    }


def _font():
    try:
        from PIL import ImageFont

        return ImageFont.load_default()
    except Exception:  # pragma: no cover
        return None


def _draw_label(draw: Any, xy: Tuple[int, int], text: str, *, fill: Tuple[int, int, int], font: Any, image_size: Optional[Tuple[int, int]] = None, background: Tuple[int, int, int] = (35, 35, 35)) -> None:
    x, y = int(xy[0]), int(xy[1])
    try:
        bbox = draw.textbbox((x, y), text, font=font)
    except Exception:  # pragma: no cover
        try:
            w, h = draw.textsize(text, font=font)
        except Exception:
            w, h = len(text) * 6, 12
        bbox = (x, y, x + w, y + h)
    if image_size is not None:
        width, height = image_size
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = min(max(4, x), max(4, int(width) - text_w - 6))
        y = min(max(4, y), max(4, int(height) - text_h - 6))
        try:
            bbox = draw.textbbox((x, y), text, font=font)
        except Exception:  # pragma: no cover
            bbox = (x, y, x + text_w, y + text_h)
    pad = 2
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=background, outline=(220, 220, 220))
    draw.text((x, y), text, fill=fill, font=font)


def _draw_dashed_rect(draw: Any, rect: Tuple[int, int, int, int], *, fill: Tuple[int, int, int], width: int = 2, dash: int = 8) -> None:
    x0, y0, x1, y1 = [int(v) for v in rect]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    for x in range(x0, x1, dash * 2):
        draw.line((x, y0, min(x + dash, x1), y0), fill=fill, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=fill, width=width)
    for y in range(y0, y1, dash * 2):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=fill, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=fill, width=width)


def _blend_mask(canvas: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    if mask is None or not np.any(mask):
        return
    alpha = max(0.0, min(1.0, float(alpha)))
    color_arr = np.asarray(color, dtype=np.float32)
    base = canvas[mask].astype(np.float32)
    canvas[mask] = np.clip(base * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)


def _render_target_heatmap_overlay(canvas: np.ndarray, heatmap: Optional[np.ndarray], *, alpha: float = 0.45) -> None:
    if heatmap is None or not np.any(heatmap > 0):
        return
    heat = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    mask = heat > 0
    color = np.zeros_like(canvas, dtype=np.float32)
    color[:, :, 0] = 255.0
    color[:, :, 1] = np.clip(210.0 * (1.0 - heat), 40.0, 210.0)
    color[:, :, 2] = 0.0
    local_alpha = (heat * float(alpha))[:, :, None]
    base = canvas.astype(np.float32)
    blended = base * (1.0 - local_alpha) + color * local_alpha
    canvas[mask] = np.clip(blended[mask], 0, 255).astype(np.uint8)


def _render_color_field_overlay(
    canvas: np.ndarray,
    field: Optional[np.ndarray],
    *,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> None:
    if field is None:
        return
    arr = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    if arr.shape[:2] != canvas.shape[:2] or not np.any(arr > 0):
        return
    mask = arr > 0
    color_arr = np.zeros_like(canvas, dtype=np.float32)
    color_arr[:, :, 0] = float(color[0])
    color_arr[:, :, 1] = float(color[1])
    color_arr[:, :, 2] = float(color[2])
    local_alpha = (arr * float(alpha))[:, :, None]
    base = canvas.astype(np.float32)
    blended = base * (1.0 - local_alpha) + color_arr * local_alpha
    canvas[mask] = np.clip(blended[mask], 0, 255).astype(np.uint8)


def _canvas_from_map(full_map: Any, frame: Optional[Mapping[str, Any]] = None) -> Optional[np.ndarray]:
    arr = _map_to_numpy(full_map)
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 3:
        height, width = arr.shape[-2], arr.shape[-1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, :] = np.array([70, 70, 70], dtype=np.uint8)
        known = np.any(arr > 0, axis=0)
        canvas[known] = np.array([150, 150, 150], dtype=np.uint8)
        if arr.shape[0] > 1:
            canvas[arr[1] > 0] = np.array([232, 232, 232], dtype=np.uint8)
        if arr.shape[0] > SEMANTIC_CHANNEL_OFFSET:
            sem = arr[SEMANTIC_CHANNEL_OFFSET:]
            if np.any(sem > 0):
                sem_max = np.max(sem, axis=0)
                sem_idx = np.argmax(sem, axis=0)
                sem_mask = sem_max > 0
                for idx in np.unique(sem_idx[sem_mask]):
                    idx_int = int(idx)
                    color = _SEMANTIC_PALETTE[idx_int % len(_SEMANTIC_PALETTE)]
                    _blend_mask(canvas, np.logical_and(sem_mask, sem_idx == idx_int), color, alpha=0.48)
        if arr.shape[0] > 3:
            _blend_mask(canvas, arr[3] > 0, _VISITED_CHANNEL_COLOR, alpha=0.42)
        if frame is not None and (frame.get("base_layers") or {}).get("has_language_query_heatmap"):
            target_indices = ((frame.get("target_query_heatmap") or {}).get("target_category_indices") or [])
            heatmap, _ = _target_heatmap_from_map_and_objects(
                full_map,
                objects=list(frame.get("objects", []) or []),
                target_category_indices=[int(i) for i in target_indices],
            )
            _render_target_heatmap_overlay(canvas, heatmap, alpha=0.55)
        if arr.shape[0] > 2:
            _blend_mask(canvas, arr[2] > 0, _CURRENT_CHANNEL_COLOR, alpha=0.72)
        if arr.shape[0] > 0:
            canvas[arr[0] > 0] = np.array([20, 20, 20], dtype=np.uint8)
        return canvas
    if arr.ndim == 2:
        canvas = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        canvas[:, :, :] = np.array([70, 70, 70], dtype=np.uint8)
        canvas[arr > 0] = np.array([232, 232, 232], dtype=np.uint8)
        return canvas
    return None


def _geometry_canvas_from_map(full_map: Any) -> Optional[np.ndarray]:
    arr = _map_to_numpy(full_map)
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 3:
        height, width = arr.shape[-2], arr.shape[-1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, :] = np.array([70, 70, 70], dtype=np.uint8)
        known = np.any(arr > 0, axis=0)
        canvas[known] = np.array([150, 150, 150], dtype=np.uint8)
        if arr.shape[0] > 1:
            canvas[arr[1] > 0] = np.array([232, 232, 232], dtype=np.uint8)
        if arr.shape[0] > 3:
            _blend_mask(canvas, arr[3] > 0, _VISITED_CHANNEL_COLOR, alpha=0.35)
        if arr.shape[0] > 2:
            _blend_mask(canvas, arr[2] > 0, _CURRENT_CHANNEL_COLOR, alpha=0.72)
        if arr.shape[0] > 0:
            canvas[arr[0] > 0] = np.array([20, 20, 20], dtype=np.uint8)
        return canvas
    if arr.ndim == 2:
        canvas = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        canvas[:, :, :] = np.array([70, 70, 70], dtype=np.uint8)
        canvas[arr > 0] = np.array([232, 232, 232], dtype=np.uint8)
        return canvas
    return None


def _crop_canvas(canvas: np.ndarray, frame: Mapping[str, Any], *, crop_to_content: bool = True) -> Tuple[np.ndarray, Dict[str, int]]:
    height, width = canvas.shape[:2]
    crop = dict((frame.get("coordinate_frame") or {}).get("crop") or {})
    if not crop_to_content or not crop:
        crop = {"r0": 0, "r1": height, "c0": 0, "c1": width}
    r0 = int(crop.get("r0", 0) or 0)
    r1 = int(crop.get("r1", height) or height)
    c0 = int(crop.get("c0", 0) or 0)
    c1 = int(crop.get("c1", width) or width)
    r0, r1 = max(0, r0), min(height, max(r0 + 1, r1))
    c0, c1 = max(0, c0), min(width, max(c0 + 1, c1))
    return canvas[r0:r1, c0:c1], {"r0": r0, "r1": r1, "c0": c0, "c1": c1}


def _render_branch_value_overlay(canvas: np.ndarray, frame: Mapping[str, Any]) -> None:
    table = list(frame.get("branch_decision_table") or [])
    if not table:
        return
    heat = np.zeros(canvas.shape[:2], dtype=np.float32)
    for row in table:
        coord = row.get("canonical_coord_rc")
        value = float(row.get("target_value_score", 0.0) or 0.0)
        if value <= 0:
            continue
        _add_radial_heat(heat, coord, weight=value, radius=24)
    _render_target_heatmap_overlay(canvas, heat, alpha=0.62)


def _evidence_color(level: Any) -> Tuple[int, int, int]:
    level = str(level or "geometry_only")
    if level == "direct":
        return (255, 80, 60)
    if level == "anchor":
        return (255, 180, 40)
    if level == "room_prior":
        return (180, 120, 255)
    return (210, 210, 210)


def _draw_panel_overlays(
    image: Any,
    frame: Mapping[str, Any],
    *,
    crop: Mapping[str, int],
    scale: int,
    panel: str,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    font = _font()
    r0, c0 = int(crop.get("r0", 0)), int(crop.get("c0", 0))
    map_w, map_h = image.size

    def xy(coord: Any) -> Optional[Tuple[int, int]]:
        rc = _coord_rc(coord)
        if rc is None:
            return None
        x = int(round((float(rc[1]) - c0) * scale))
        y = int(round((float(rc[0]) - r0) * scale))
        if x < -20 or y < -20 or x > map_w + 20 or y > map_h + 20:
            return None
        return x, y

    if panel in {"geometry", "value"}:
        for idx, branch in enumerate(frame.get("branches", []) or []):
            pt = xy(branch.get("canonical_representative_coord_rc") or branch.get("representative_coord") or branch.get("centroid"))
            if pt is None:
                continue
            row = dict(branch.get("decision_table_row") or {})
            level = row.get("evidence_level") or branch.get("evidence_level")
            color = _evidence_color(level) if panel == "value" else _branch_color(idx)
            r = max(4, 4 * scale)
            draw.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), fill=color, outline=(20, 20, 20), width=max(1, scale))
            if panel == "value":
                label = f"{branch.get('id')} V={float(row.get('target_value_score', branch.get('target_value_score', 0)) or 0):.2f} {str(level)[:4]}"
            else:
                label = f"{branch.get('id')} {branch.get('canonical_direction_from_agent') or branch.get('map_direction_from_agent','')}"
            _draw_label(draw, (pt[0] + r + 2, pt[1] - r), label, fill=color, font=font, image_size=(map_w, map_h))

    # The semantic panel is now a dense raster state, not a debug overlay.
    # Object/room/region labels remain available in debug_overlay_full.png and
    # JSON, but are intentionally not drawn here because they make the MLLM
    # planner read labels instead of spatial cells.

    agent = dict(frame.get("agent") or {})
    agent_pt = xy(agent.get("coord_rc"))
    if agent_pt is not None:
        r = max(6, 6 * scale)
        draw.ellipse((agent_pt[0] - r, agent_pt[1] - r, agent_pt[0] + r, agent_pt[1] + r), fill=_AGENT_COLOR, outline=(0, 0, 0), width=max(1, scale))
        _draw_label(draw, (agent_pt[0] + r + 2, agent_pt[1]), "A", fill=(0, 255, 0), font=font, image_size=(map_w, map_h), background=(20, 55, 20))


def render_decision_state_bev(
    frame: Mapping[str, Any],
    full_map: Any,
    *,
    scale: int = 1,
    crop_to_content: bool = True,
) -> Any:
    """Render a multi-panel planner BEV: geometry, semantic, value, table."""

    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover
        return None

    geometry = _geometry_canvas_from_map(full_map)
    if geometry is None:
        return None
    decision_fields: Dict[str, np.ndarray] = {}
    if build_decision_state_from_frame is not None:
        try:
            _rebuilt_state, decision_fields = build_decision_state_from_frame(full_map, frame)
        except Exception:
            decision_fields = {}
    dense_raster = build_dense_semantic_raster(
        full_map,
        objects=list(frame.get("objects", []) or []),
        semantic_bbox_fill=False,
    )
    semantic = _dense_rgb_from_class_map(dense_raster)
    if semantic is None:
        semantic = geometry.copy()
    value = _geometry_canvas_from_map(full_map)
    if value is None:
        value = geometry.copy()
    if "target_combined_value" in decision_fields:
        _render_target_heatmap_overlay(value, decision_fields.get("target_combined_value"), alpha=0.68)
    else:
        _render_branch_value_overlay(value, frame)

    geometry, crop = _crop_canvas(geometry, frame, crop_to_content=crop_to_content)
    semantic, _ = _crop_canvas(semantic, frame, crop_to_content=crop_to_content)
    value, _ = _crop_canvas(value, frame, crop_to_content=crop_to_content)
    scale = max(1, int(scale or 1))

    panels = []
    for name, canvas in (("geometry", geometry), ("semantic", semantic), ("value", value)):
        img = Image.fromarray(canvas, mode="RGB")
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        _draw_panel_overlays(img, frame, crop=crop, scale=scale, panel=name)
        panels.append((name, img))

    panel_w = max(img.width for _, img in panels)
    panel_h = max(img.height for _, img in panels)
    table_w = 430
    header_h = 22
    composed = Image.new("RGB", (panel_w * 3 + table_w, panel_h + header_h), (42, 42, 42))
    draw = ImageDraw.Draw(composed)
    font = _font()
    headers = {
        "geometry": "A Geometry: occupancy/free/trajectory + branches",
        "semantic": "B Dense semantic raster: true channels + trusted footprints",
        "value": "C Target-conditioned value field",
    }
    x = 0
    for name, img in panels:
        draw.rectangle((x, 0, x + panel_w - 1, header_h - 1), fill=(32, 32, 32), outline=(110, 110, 110))
        draw.text((x + 6, 5), headers[name][:55], fill=(245, 245, 245), font=font)
        composed.paste(img, (x, header_h))
        x += panel_w

    table_x = panel_w * 3
    draw.rectangle((table_x, 0, composed.width - 1, composed.height - 1), fill=(38, 38, 38), outline=(130, 130, 130))
    y = 8

    def line(text: str, color: Tuple[int, int, int] = (235, 235, 235)) -> None:
        nonlocal y
        draw.text((table_x + 8, y), str(text)[:68], fill=color, font=font)
        y += 14

    line("D Branch decision table", (255, 255, 255))
    line(f"step={frame.get('step_idx')} schema={(frame.get('decision_state') or {}).get('schema_version','')}", (210, 230, 255))
    qc = dict(frame.get("quality_checks") or {})
    warnings = list(qc.get("warnings") or [])
    line("warnings: " + (",".join(warnings[:3]) if warnings else "none"), _WARNING_COLOR if warnings else (160, 255, 160))
    tv = dict(frame.get("target_value_state") or {})
    tv_peak = tv.get("peak")
    if tv_peak is None:
        tv_peak = (tv.get("combined_value") or {}).get("peak", 0.0)
    line(f"value: {tv.get('source','')} strongest={tv.get('strongest_source','')} peak={float(tv_peak or 0):.2f}", (255, 190, 120))
    line(f"evidence counts: {tv.get('evidence_counts', {})}", (210, 210, 210))
    room_state = dict(frame.get("room_state") or {})
    line(f"rooms localized={len(room_state.get('localized_room_regions') or [])} unlocalized={len(room_state.get('unlocalized_room_priors') or [])}", _ROOM_COLOR)
    y += 4
    line("ID dir  V   evidence     info revisit risk support", (240, 240, 180))
    for row in list(frame.get("branch_decision_table", []) or [])[:14]:
        level = str(row.get("evidence_level") or "")
        color = _evidence_color(level)
        raw_support = row.get("semantic_support") or {}
        if isinstance(raw_support, Mapping):
            ids = (raw_support.get("direct_ids") or [])[:1] + (raw_support.get("anchor_ids") or [])[:2] + (raw_support.get("room_ids") or [])[:1]
        else:
            ids = [str(item.get("id") if isinstance(item, Mapping) else item) for item in list(raw_support)[:4]]
        line(
            f"{row.get('id'):>2} {str(row.get('canonical_direction',''))[:5]:<5} "
            f"{float(row.get('target_value_score',0) or 0):.2f} {level:<12} "
            f"{float(row.get('info_gain_score',0) or 0):.2f} "
            f"{float(row.get('revisit_penalty',0) or 0):.2f} "
            f"{float(row.get('dead_end_risk',0) or 0):.2f} {','.join(ids)}",
            color,
        )
    y += 4
    line("Unlocalized room priors (not drawn as spatial masks):", (180, 120, 255))
    for room in list(room_state.get("unlocalized_room_priors") or [])[:8]:
        line(f" {room.get('id')} {room.get('caption')} {room.get('confidence')}", (180, 120, 255))
    return composed


def render_decision_state_field_panel(
    frame: Mapping[str, Any],
    full_map: Any,
    *,
    panel: str = "semantic",
    scale: int = 1,
    crop_to_content: bool = True,
) -> Any:
    """Render one decision-state field panel for focused inspection."""

    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover
        return None
    panel = str(panel or "semantic").lower()
    geometry = _geometry_canvas_from_map(full_map)
    if geometry is None:
        return None
    decision_fields: Dict[str, np.ndarray] = {}
    if build_decision_state_from_frame is not None:
        try:
            _state, decision_fields = build_decision_state_from_frame(full_map, frame)
        except Exception:
            decision_fields = {}
    if panel == "geometry":
        canvas = geometry
    elif panel == "value":
        canvas = geometry.copy()
        if "target_combined_value" in decision_fields:
            _render_target_heatmap_overlay(canvas, decision_fields.get("target_combined_value"), alpha=0.68)
        else:
            _render_branch_value_overlay(canvas, frame)
    else:
        dense_raster = build_dense_semantic_raster(
            full_map,
            objects=list(frame.get("objects", []) or []),
            semantic_bbox_fill=False,
        )
        canvas = _dense_rgb_from_class_map(dense_raster)
        if canvas is None:
            canvas = geometry.copy()
        panel = "semantic"
    canvas, crop = _crop_canvas(canvas, frame, crop_to_content=crop_to_content)
    scale = max(1, int(scale or 1))
    image = Image.fromarray(canvas, mode="RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    _draw_panel_overlays(image, frame, crop=crop, scale=scale, panel="value" if panel == "value" else panel)
    header_h = 24
    title = {
        "geometry": "Geometry field",
        "semantic": "Dense semantic raster: true map channels + trusted projected footprints",
        "value": "Target-conditioned value / potential field",
    }.get(panel, panel)
    composed = Image.new("RGB", (image.width, image.height + header_h), (42, 42, 42))
    composed.paste(image, (0, header_h))
    draw = ImageDraw.Draw(composed)
    draw.rectangle((0, 0, composed.width - 1, header_h - 1), fill=(32, 32, 32), outline=(110, 110, 110))
    draw.text((6, 6), title[:95], fill=(245, 245, 245), font=_font())
    return composed


def render_semantic_annotated_bev(
    frame: Mapping[str, Any],
    full_map: Any,
    *,
    mode: str = "planner",
    scale: int = 2,
    crop_to_content: bool = True,
) -> Any:
    """Render a semantic annotated BEV image from a frame.

    Returns a PIL Image or None if Pillow/map conversion is unavailable.
    """

    if str(mode).lower() in {"decision", "decision_state"}:
        return render_decision_state_bev(
            frame,
            full_map,
            scale=scale,
            crop_to_content=crop_to_content,
        )
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover
        return None
    canvas = _canvas_from_map(full_map, frame=frame)
    if canvas is None:
        return None
    height, width = canvas.shape[:2]
    crop = dict((frame.get("coordinate_frame") or {}).get("crop") or {})
    if not crop_to_content or not crop:
        crop = {"r0": 0, "r1": height, "c0": 0, "c1": width}
    r0 = int(crop.get("r0", 0) or 0)
    r1 = int(crop.get("r1", height) or height)
    c0 = int(crop.get("c0", 0) or 0)
    c1 = int(crop.get("c1", width) or width)
    r0, r1 = max(0, r0), min(height, max(r0 + 1, r1))
    c0, c1 = max(0, c0), min(width, max(c0 + 1, c1))
    canvas = canvas[r0:r1, c0:c1]
    image = Image.fromarray(canvas, mode="RGB")
    scale = max(1, int(scale or 1))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    panel_w = 330 if str(mode).lower() == "debug" else 280
    composed = Image.new("RGB", (image.width + panel_w, image.height), (42, 42, 42))
    composed.paste(image, (0, 0))
    draw = ImageDraw.Draw(composed)
    font = _font()
    map_w, map_h = image.size

    def xy(coord: Any) -> Optional[Tuple[int, int]]:
        rc = _coord_rc(coord)
        if rc is None:
            return None
        x = int(round((float(rc[1]) - c0) * scale))
        y = int(round((float(rc[0]) - r0) * scale))
        if x < -20 or y < -20 or x > map_w + 20 or y > map_h + 20:
            return None
        return x, y

    # Dense semantic-channel regions: these are the missing visual link between
    # BEV pixels and planner text.  They are drawn before graph rooms/objects so
    # graph evidence remains visible on top.
    for region in frame.get("semantic_regions", []) or []:
        center_xy = xy(region.get("center_rc"))
        if center_xy is None:
            continue
        idx = int(region.get("category_index", 0) or 0)
        color = _SEMANTIC_PALETTE[idx % len(_SEMANTIC_PALETTE)]
        bbox = region.get("bbox_rc")
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            x0 = int(round((float(bbox[1]) - c0) * scale))
            y0 = int(round((float(bbox[0]) - r0) * scale))
            x1 = int(round((float(bbox[3]) - c0) * scale))
            y1 = int(round((float(bbox[2]) - r0) * scale))
            pad = max(3, 3 * scale)
            draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=color, width=max(1, scale + 1))
        prior = float(region.get("target_prior_weight", 0.0) or 0.0)
        label = f"{region.get('id')} {region.get('label')}"
        if prior > 0:
            label += f" w={prior:.2f}"
        _draw_label(
            draw,
            (center_xy[0] + 5, center_xy[1] + 5),
            label[:32],
            fill=color,
            font=font,
            image_size=(map_w, map_h),
            background=(35, 30, 25) if prior > 0 else (35, 35, 35),
        )

    # Room hypotheses: intentionally weak/dashed to avoid presenting them as GT masks.
    for room in frame.get("rooms", []) or []:
        bbox = room.get("bbox_rc")
        center_xy = xy(room.get("center_rc"))
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            x0 = int(round((float(bbox[1]) - c0) * scale))
            y0 = int(round((float(bbox[0]) - r0) * scale))
            x1 = int(round((float(bbox[3]) - c0) * scale))
            y1 = int(round((float(bbox[2]) - r0) * scale))
            _draw_dashed_rect(draw, (x0, y0, x1, y1), fill=_ROOM_COLOR, width=max(1, scale), dash=8 * scale)
        if center_xy is not None:
            _draw_label(
                draw,
                (center_xy[0] + 4, center_xy[1] + 4),
                f"{room.get('id')} {room.get('caption', '')}"[:34],
                fill=_ROOM_COLOR,
                font=font,
                image_size=(map_w, map_h),
                background=(50, 40, 70),
            )

    # Branch candidate points and representatives.
    branch_by_id = {}
    for idx, branch in enumerate(frame.get("branches", []) or []):
        bid = str(branch.get("id") or f"B{idx + 1}")
        branch_by_id[bid] = branch
        color = _branch_color(idx)
        terms = dict(branch.get("score_terms_aggregate") or {})
        candidate_count = int(branch.get("candidate_point_count", 0) or 0)
        actionable = bool(candidate_count > 0 and float(terms.get("actionability_score_max") or 0.0) > 0.0)
        point_radius = max(2, 2 * scale)
        if str(mode).lower() == "debug":
            for point in branch.get("candidate_points", []) or []:
                if not isinstance(point, Mapping):
                    continue
                pt = xy(point.get("coord"))
                if pt is None:
                    continue
                draw.rectangle(
                    (pt[0] - point_radius, pt[1] - point_radius, pt[0] + point_radius, pt[1] + point_radius),
                    fill=color if bool(point.get("actionable", True)) else (120, 120, 120),
                    outline=(20, 20, 20),
                )
        coord_xy = xy(branch.get("representative_coord") or branch.get("centroid"))
        if coord_xy is None:
            continue
        r = max(5, 5 * scale)
        if actionable:
            draw.ellipse((coord_xy[0] - r, coord_xy[1] - r, coord_xy[0] + r, coord_xy[1] + r), fill=color, outline=(20, 20, 20), width=max(1, scale))
        else:
            draw.ellipse((coord_xy[0] - r, coord_xy[1] - r, coord_xy[0] + r, coord_xy[1] + r), fill=(150, 150, 150), outline=(255, 0, 0), width=max(1, scale))
            draw.line((coord_xy[0] - r, coord_xy[1] - r, coord_xy[0] + r, coord_xy[1] + r), fill=(255, 0, 0), width=max(1, scale))
            draw.line((coord_xy[0] - r, coord_xy[1] + r, coord_xy[0] + r, coord_xy[1] - r), fill=(255, 0, 0), width=max(1, scale))
        roles = "/".join(branch.get("decision_roles", []) or [])
        role_suffix = f" [{roles}]" if roles and str(mode).lower() == "debug" else ""
        final_score = terms.get("final_score_max")
        score_text = "" if final_score is None else f" F={float(final_score):.1f}"
        direction = branch.get("map_direction_from_agent") or branch.get("direction_from_agent", "")
        heat = float(branch.get("target_heatmap_score", 0.0) or 0.0)
        heat_text = f" H={heat:.2f}" if heat > 0 else ""
        label = f"{bid} {direction} cand={candidate_count}{score_text}{heat_text}{role_suffix}".strip()
        _draw_label(draw, (coord_xy[0] + r + 2, coord_xy[1] - r), label, fill=color, font=font, image_size=(map_w, map_h))

    # Decision overlay rings for debug mode.
    if str(mode).lower() == "debug":
        decision = dict(frame.get("decision_overlay") or {})
        for role, bid, color, radius_extra in (
            ("dry", decision.get("dry_selected_branch_id"), _DRY_COLOR, 8),
            ("mllm", decision.get("mllm_selected_branch_id"), _MLLM_COLOR, 14),
            ("final", decision.get("final_selected_branch_id"), _FINAL_COLOR, 20),
        ):
            branch = branch_by_id.get(str(bid or ""))
            if not branch:
                continue
            pt = xy(branch.get("representative_coord") or branch.get("centroid"))
            if pt is None:
                continue
            r = max(radius_extra, radius_extra * scale // 2)
            draw.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), outline=color, width=max(2, scale + 1))
            _draw_label(draw, (pt[0] - r, pt[1] - r - 16), role.upper(), fill=color, font=font, image_size=(map_w, map_h), background=(30, 30, 30))

    # Objects after branches so labels are visible.
    objects = list(frame.get("objects", []) or [])
    objects.sort(key=lambda item: (-float(item.get("target_relevance", 0.0) or 0.0), -float(item.get("display_priority", 0.0) or 0.0), str(item.get("id", ""))))
    max_labels = 16 if str(mode).lower() == "debug" else 8
    for obj in objects[:max_labels]:
        pt = xy(obj.get("center_rc"))
        if pt is None:
            continue
        rel = float(obj.get("target_relevance", 0.0) or 0.0)
        color = _TARGET_OBJECT_COLOR if rel > 0.25 else _OBJECT_COLOR
        r = max(4, 4 * scale)
        draw.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), fill=color, outline=(0, 0, 0), width=max(1, scale))
        label = f"{obj.get('id')} {obj.get('caption', '')}"[:28]
        _draw_label(draw, (pt[0] + r + 2, pt[1] - r), label, fill=color, font=font, image_size=(map_w, map_h), background=(20, 45, 55))

    # Agent marker.
    agent_info = dict(frame.get("agent") or {})
    raw_agent_pt = xy(agent_info.get("raw_coord_rc"))
    agent_pt = xy(agent_info.get("coord_rc"))
    if raw_agent_pt is not None and agent_info.get("display_repaired_from_map_current_channel"):
        rr = max(5, 5 * scale)
        draw.line((raw_agent_pt[0] - rr, raw_agent_pt[1] - rr, raw_agent_pt[0] + rr, raw_agent_pt[1] + rr), fill=_WARNING_COLOR, width=max(1, scale))
        draw.line((raw_agent_pt[0] - rr, raw_agent_pt[1] + rr, raw_agent_pt[0] + rr, raw_agent_pt[1] - rr), fill=_WARNING_COLOR, width=max(1, scale))
        if str(mode).lower() == "debug":
            _draw_label(draw, (raw_agent_pt[0] + rr + 2, raw_agent_pt[1]), "raw A mismatch", fill=_WARNING_COLOR, font=font, image_size=(map_w, map_h), background=(70, 20, 20))
    if agent_pt is not None:
        r = max(7, 7 * scale)
        draw.ellipse((agent_pt[0] - r, agent_pt[1] - r, agent_pt[0] + r, agent_pt[1] + r), fill=_AGENT_COLOR, outline=(0, 0, 0), width=max(1, scale))
        agent_label = "A robot"
        if agent_info.get("display_repaired_from_map_current_channel"):
            agent_label += " (map-current)"
        _draw_label(draw, (agent_pt[0] + r + 2, agent_pt[1]), agent_label, fill=(0, 255, 0), font=font, image_size=(map_w, map_h), background=(20, 55, 20))

    # Top-left legend inside map.
    legend_lines = [
        "A=robot; B#=frontier branch",
        "O#=object; R#=room hypothesis",
        "semantic colors=map ch4+; orange=target/query heat",
        "cyan=current ch2; yellow=visited ch3",
        "room boxes are graph hypotheses, not GT masks",
        f"crop rows {r0}:{r1}, cols {c0}:{c1}",
    ]
    if str(mode).lower() == "debug":
        legend_lines.insert(2, "debug rings: dry=white, mllm=magenta, final=green")
    lx, ly = 6, 6
    line_h = 14
    legend_w = max(250, max(len(line) * 6 for line in legend_lines) + 12)
    legend_h = line_h * len(legend_lines) + 8
    draw.rectangle((lx, ly, lx + legend_w, ly + legend_h), fill=(35, 35, 35), outline=(230, 230, 230))
    for i, line in enumerate(legend_lines):
        draw.text((lx + 6, ly + 5 + i * line_h), line, fill=(245, 245, 245), font=font)

    # Compass.
    arrow_x = map_w - 34
    arrow_y = 18
    draw.line((arrow_x, arrow_y + 34, arrow_x, arrow_y), fill=(255, 255, 255), width=max(1, scale))
    draw.polygon([(arrow_x, arrow_y - 2), (arrow_x - 6, arrow_y + 10), (arrow_x + 6, arrow_y + 10)], fill=(255, 255, 255))
    draw.text((arrow_x - 4, arrow_y + 38), "N", fill=(255, 255, 255), font=font)

    # Side panel.
    panel_x = map_w + 8
    y = 8
    def panel_line(text: str, color: Tuple[int, int, int] = (235, 235, 235)) -> None:
        nonlocal y
        draw.text((panel_x, y), str(text)[:46], fill=color, font=font)
        y += 14

    draw.rectangle((map_w, 0, composed.width - 1, composed.height - 1), fill=(42, 42, 42))
    panel_line(f"Semantic BEV ({mode})", (255, 255, 255))
    panel_line(f"step={frame.get('step_idx')} objects={len(objects)} rooms={len(frame.get('rooms', []) or [])}")
    qc = dict(frame.get("quality_checks") or {})
    warnings = list(qc.get("warnings") or [])
    if warnings:
        panel_line("WARN: " + ";".join(warnings[:2]), _WARNING_COLOR)
    else:
        panel_line("quality: no warnings", (160, 255, 160))
    base_layers = dict(frame.get("base_layers") or {})
    panel_line(
        f"sem={base_layers.get('active_semantic_category_count', 0)}/{base_layers.get('semantic_channel_count', 0)} "
        f"traj={base_layers.get('past_agent_channel_pixels', 0)}",
        (210, 245, 255),
    )
    heat_summary = dict(frame.get("target_query_heatmap") or {})
    panel_line(
        f"target heat={heat_summary.get('source','none')} pix={heat_summary.get('positive_pixel_count',0)}",
        (255, 170, 120) if heat_summary.get("source") not in ("none", "", None) else _WARNING_COLOR,
    )
    decision = dict(frame.get("decision_overlay") or {})
    panel_line(f"dry={decision.get('dry_selected_branch_id','')} mllm={decision.get('mllm_selected_branch_id','')}", (255, 220, 255))
    panel_line(f"final={decision.get('final_selected_branch_id','')} applied={decision.get('planner_prior_applied')}", (200, 255, 220))
    active_categories = list((frame.get("semantic_layers") or {}).get("active_categories", []) or [])
    if active_categories:
        panel_line("Sem channels:", (210, 245, 255))
        for item in active_categories[:5]:
            panel_line(f" {item.get('label')} ch={item.get('channel_index')} pix={item.get('pixel_count')}", (210, 245, 255))
    semantic_regions = list(frame.get("semantic_regions", []) or [])
    anchor_regions = [item for item in semantic_regions if float(item.get("target_prior_weight", 0.0) or 0.0) > 0]
    if anchor_regions:
        panel_line("Target anchors:", (255, 190, 130))
        for item in anchor_regions[:5]:
            panel_line(
                f" {item.get('id')} {item.get('label')} w={float(item.get('target_prior_weight',0) or 0):.2f} {item.get('center_rc')}",
                (255, 190, 130),
            )
    y += 4
    panel_line("Objects:", _OBJECT_COLOR)
    if not objects:
        panel_line("  no localized semantic objects", _WARNING_COLOR)
    for obj in objects[:10 if str(mode).lower() == "debug" else 6]:
        loc = obj.get("center_rc") if obj.get("center_rc") is not None else "unloc"
        panel_line(f" {obj.get('id')} {obj.get('caption')} det={obj.get('num_detections')} rel={float(obj.get('target_relevance',0) or 0):.2f} {loc}", _TARGET_OBJECT_COLOR if float(obj.get("target_relevance", 0) or 0) > 0.25 else _OBJECT_COLOR)
    y += 4
    panel_line("Rooms:", _ROOM_COLOR)
    for room in list(frame.get("rooms", []) or [])[:8 if str(mode).lower() == "debug" else 4]:
        panel_line(f" {room.get('id')} {room.get('caption')} {room.get('confidence')}", _ROOM_COLOR)
    y += 4
    panel_line("Branches:", (230, 230, 120))
    for branch in list(frame.get("branches", []) or [])[:12 if str(mode).lower() == "debug" else 8]:
        roles = "/".join(branch.get("decision_roles", []) or [])
        hints = ",".join(
            (branch.get("nearby_object_ids") or [])[:1]
            + (branch.get("nearby_semantic_region_ids") or [])[:2]
            + (branch.get("nearby_room_ids") or [])[:1]
        )
        direction = branch.get("map_direction_from_agent") or branch.get("direction_from_agent", "")
        panel_line(
            f" {branch.get('id')} {direction} H={float(branch.get('target_heatmap_score',0) or 0):.2f} {roles} hints={hints}",
            (230, 230, 120),
        )

    return composed


def save_semantic_annotated_bev(
    path: str | Path,
    frame: Mapping[str, Any],
    full_map: Any,
    *,
    mode: str = "planner",
    scale: int = 2,
) -> Optional[str]:
    if str(mode).lower() in {"decision", "decision_state"}:
        image = render_decision_state_bev(frame, full_map, scale=scale)
        if image is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return str(path)
    image = render_semantic_annotated_bev(frame, full_map, mode=mode, scale=scale)
    if image is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def save_decision_state_bev(
    path: str | Path,
    frame: Mapping[str, Any],
    full_map: Any,
    *,
    scale: int = 1,
) -> Optional[str]:
    image = render_decision_state_bev(frame, full_map, scale=scale)
    if image is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def save_decision_state_field_panel(
    path: str | Path,
    frame: Mapping[str, Any],
    full_map: Any,
    *,
    panel: str = "semantic",
    scale: int = 1,
) -> Optional[str]:
    image = render_decision_state_field_panel(frame, full_map, panel=panel, scale=scale)
    if image is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)
