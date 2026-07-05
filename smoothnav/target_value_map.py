"""Target-conditioned BEV value utilities for planner decision state.

The functions here intentionally stay heuristic and dependency-light.  They do
not try to replace a learned PONI/SemExp policy; they build an inspectable
"PONI-lite" value field from the state SmoothNav already stores in replay
capsules.  The important contract is source attribution: a non-zero value field
must say whether it came from direct target evidence, semantic anchors,
localized room priors, or geometry-only exploration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DECISION_STATE_SCHEMA_VERSION = "smoothnav.planner_decision_state.v2"
TARGET_VALUE_SCHEMA_VERSION = "smoothnav.target_conditioned_value.v0"
BRANCH_TABLE_SCHEMA_VERSION = "smoothnav.branch_decision_table.v1"

EVIDENCE_LEVELS = ("direct", "anchor", "room_prior", "geometry_only")
_EVIDENCE_ORDER = {level: idx for idx, level in enumerate(EVIDENCE_LEVELS)}


def evidence_rank(level: Any) -> int:
    """Return a monotonic rank where stronger evidence has a lower number."""

    return _EVIDENCE_ORDER.get(str(level or "geometry_only"), _EVIDENCE_ORDER["geometry_only"])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        out = int(round(float(value)))
    except Exception:
        return int(default)
    return int(out)


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
        return [_safe_int(value[0]), _safe_int(value[1])]
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


def _distance(a: Any, b: Any) -> Optional[float]:
    aa = _coord_rc(a)
    bb = _coord_rc(b)
    if aa is None or bb is None:
        return None
    try:
        return float(math.dist([float(aa[0]), float(aa[1])], [float(bb[0]), float(bb[1])]))
    except Exception:
        return None


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _sample_map(value_map: Optional[np.ndarray], coord: Any, *, radius: int = 16) -> float:
    rc = _coord_rc(coord)
    if value_map is None or rc is None:
        return 0.0
    h, w = value_map.shape[:2]
    row, col = int(rc[0]), int(rc[1])
    if not (0 <= row < h and 0 <= col < w):
        return 0.0
    radius = max(0, int(radius))
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    patch = value_map[r0:r1, c0:c1]
    if patch.size == 0:
        return 0.0
    return float(np.max(patch))


def add_radial_value(value_map: np.ndarray, center: Any, *, weight: float, radius: int = 32) -> None:
    """Max-blend a radial potential into ``value_map``."""

    rc = _coord_rc(center)
    if rc is None or value_map is None:
        return
    row, col = int(rc[0]), int(rc[1])
    h, w = value_map.shape[:2]
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
    patch = np.maximum(0.0, 1.0 - dist / float(radius)) * weight
    value_map[r0:r1, c0:c1] = np.maximum(value_map[r0:r1, c0:c1], patch)


def _branch_coord(branch: Mapping[str, Any]) -> Optional[List[int]]:
    return _coord_rc(branch.get("canonical_representative_coord_rc") or branch.get("representative_coord") or branch.get("centroid"))


def _room_prior_weight(goal_description: str, room_caption: str) -> float:
    """Weak object-room commonsense prior for localized room regions only."""

    goal = str(goal_description or "").lower()
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


def _room_mask_value(value_map: np.ndarray, room: Mapping[str, Any], weight: float) -> None:
    bbox = room.get("bbox_rc")
    if isinstance(bbox, Sequence) and len(bbox) >= 4:
        h, w = value_map.shape[:2]
        r0, c0, r1, c1 = [_safe_int(v) for v in bbox[:4]]
        r0, r1 = max(0, min(r0, r1)), min(h, max(r0, r1) + 1)
        c0, c1 = max(0, min(c0, c1)), min(w, max(c0, c1) + 1)
        if r0 < r1 and c0 < c1:
            value_map[r0:r1, c0:c1] = np.maximum(value_map[r0:r1, c0:c1], _clip01(weight))
            return
    add_radial_value(value_map, room.get("center_rc"), weight=weight, radius=46)


def _nearby_score(rep: Any, items: Iterable[Mapping[str, Any]], *, id_key: str = "id", max_dist: float = 120.0, weight_key: str = "weight") -> Tuple[float, List[str]]:
    best = 0.0
    support: List[str] = []
    for item in items or []:
        dist = _distance(rep, item.get("center_rc"))
        if dist is None or dist > float(max_dist):
            continue
        raw_weight = _safe_float(item.get(weight_key), 0.0)
        attenuation = max(0.15, 1.0 - float(dist) / max(float(max_dist), 1.0))
        score = _clip01(raw_weight * attenuation)
        if score > 0:
            support.append(str(item.get(id_key) or ""))
            best = max(best, score)
    return best, [item for item in support if item]


def _object_support(rep: Any, objects: Sequence[Mapping[str, Any]]) -> Tuple[float, List[str], float, List[str]]:
    direct = 0.0
    direct_ids: List[str] = []
    anchor = 0.0
    anchor_ids: List[str] = []
    for obj in objects or []:
        dist = _distance(rep, obj.get("center_rc"))
        if dist is None or dist > 110.0:
            continue
        rel = _safe_float(obj.get("target_relevance"), 0.0)
        attenuation = max(0.20, 1.0 - float(dist) / 110.0)
        score = _clip01(rel * attenuation)
        if rel >= 0.75:
            direct = max(direct, score)
            direct_ids.append(str(obj.get("id") or ""))
        elif rel >= 0.35:
            anchor = max(anchor, max(score, rel * 0.5 * attenuation))
            anchor_ids.append(str(obj.get("id") or ""))
    return direct, [i for i in direct_ids if i], anchor, [i for i in anchor_ids if i]


def build_target_value_state(
    *,
    map_shape: Optional[Tuple[int, int]],
    target_heatmap: Optional[np.ndarray],
    target_heatmap_summary: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
    semantic_regions: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    rooms: Sequence[Mapping[str, Any]],
    goal_description: str = "",
) -> Tuple[Optional[np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    """Build a dense value map summary plus branch decision table.

    The returned dense map is for rendering only; the JSON-safe summary and
    branch table carry the replayable decision evidence.
    """

    if map_shape is None:
        return None, {
            "schema_version": TARGET_VALUE_SCHEMA_VERSION,
            "available": False,
            "reason": "missing_map_shape",
            "branch_ranking": [],
        }, []

    h, w = int(map_shape[0]), int(map_shape[1])
    value_map = np.zeros((h, w), dtype=np.float32)
    source_counts = {"direct": 0, "anchor": 0, "room_prior": 0, "geometry_only": 0}

    heat_source = str((target_heatmap_summary or {}).get("source") or "none")
    if target_heatmap is not None and np.any(target_heatmap > 0):
        heat = np.clip(np.asarray(target_heatmap, dtype=np.float32), 0.0, 1.0)
        # Direct heat is allowed to dominate; prior-only heat remains a useful
        # but weaker search potential.
        heat_weight = 0.85 if heat_source in {"direct_target", "direct_plus_prior"} else 0.58
        value_map = np.maximum(value_map, heat * float(heat_weight))

    direct_regions: List[Dict[str, Any]] = []
    anchor_regions: List[Dict[str, Any]] = []
    for region in semantic_regions or []:
        if region.get("is_direct_target_category"):
            direct_regions.append({**dict(region), "weight": 1.0})
        elif _safe_float(region.get("target_prior_weight"), 0.0) > 0.0:
            anchor_regions.append({**dict(region), "weight": _safe_float(region.get("target_prior_weight"), 0.0)})

    # Make graph objects and localized rooms visible in the value panel even
    # when the underlying semantic channels are sparse.
    for obj in objects or []:
        center = obj.get("center_rc")
        if center is None:
            continue
        rel = _safe_float(obj.get("target_relevance"), 0.0)
        if rel >= 0.75:
            add_radial_value(value_map, center, weight=0.92, radius=42)
        elif rel > 0.0:
            add_radial_value(value_map, center, weight=0.38 + 0.30 * rel, radius=34)

    localized_rooms: List[Dict[str, Any]] = []
    for room in rooms or []:
        if not room.get("localized"):
            continue
        weight = _room_prior_weight(goal_description, str(room.get("caption") or ""))
        if weight <= 0:
            continue
        localized_rooms.append({**dict(room), "weight": float(weight)})
        _room_mask_value(value_map, room, weight=0.30 + 0.35 * weight)

    table: List[Dict[str, Any]] = []
    for branch in branches or []:
        bid = str(branch.get("id") or "")
        rep = _branch_coord(branch)
        terms = dict(branch.get("score_terms_aggregate") or {})
        heat_score = _clip01(branch.get("target_heatmap_score"))
        if heat_score <= 0:
            heat_score = _sample_map(target_heatmap, rep, radius=24)

        region_direct_score, region_direct_ids = _nearby_score(rep, direct_regions, max_dist=105.0, weight_key="weight")
        region_anchor_score, region_anchor_ids = _nearby_score(rep, anchor_regions, max_dist=125.0, weight_key="weight")
        obj_direct_score, obj_direct_ids, obj_anchor_score, obj_anchor_ids = _object_support(rep, objects)
        room_score, room_ids = _nearby_score(rep, localized_rooms, max_dist=155.0, weight_key="weight")

        direct_score = max(
            region_direct_score,
            obj_direct_score,
            heat_score if heat_source in {"direct_target", "direct_plus_prior"} else 0.0,
            _clip01(terms.get("target_progress_score_max")) if heat_source in {"direct_target", "direct_plus_prior"} else 0.0,
        )
        anchor_score = max(
            region_anchor_score,
            obj_anchor_score,
            heat_score if heat_source == "semantic_prior_only" else 0.0,
        )
        room_prior_score = max(room_score, 0.0)
        info_gain = max(
            _clip01(terms.get("novelty_score_max")),
            min(1.0, math.log1p(max(0, int(branch.get("candidate_point_count", 0) or 0))) / math.log(9.0)),
        )
        actionability = _clip01(terms.get("actionability_score_max"))
        revisit_penalty = max(_clip01(terms.get("repeat_penalty_max")), _clip01(terms.get("recent_penalty_max")))
        dead_end_risk = _clip01(revisit_penalty + (0.35 if int(branch.get("candidate_point_count", 0) or 0) <= 0 else 0.0))

        semantic_utility = max(direct_score, 0.82 * anchor_score, 0.60 * room_prior_score)
        geometry_utility = 0.55 * info_gain + 0.35 * actionability
        value = _clip01(max(semantic_utility, 0.45 * geometry_utility) + 0.25 * heat_score - 0.35 * revisit_penalty)

        if direct_score >= 0.30:
            evidence_level = "direct"
            source_counts["direct"] += 1
        elif anchor_score >= 0.22:
            evidence_level = "anchor"
            source_counts["anchor"] += 1
        elif room_prior_score >= 0.22:
            evidence_level = "room_prior"
            source_counts["room_prior"] += 1
        else:
            evidence_level = "geometry_only"
            source_counts["geometry_only"] += 1

        support = {
            "direct_ids": sorted(set(region_direct_ids + obj_direct_ids)),
            "anchor_ids": sorted(set(region_anchor_ids + obj_anchor_ids)),
            "room_ids": sorted(set(room_ids)),
        }
        row = {
            "schema_version": BRANCH_TABLE_SCHEMA_VERSION,
            "id": bid,
            "canonical_coord_rc": rep,
            "canonical_direction": str(branch.get("canonical_direction_from_agent") or branch.get("map_direction_from_agent") or branch.get("direction_from_agent") or ""),
            "candidate_count": int(branch.get("candidate_point_count", 0) or 0),
            "target_value_score": round(float(value), 4),
            "evidence_level": evidence_level,
            "target_visible": bool(evidence_level == "direct"),
            "semantic_support": support,
            "direct_score": round(float(direct_score), 4),
            "anchor_score": round(float(anchor_score), 4),
            "room_prior_score": round(float(room_prior_score), 4),
            "info_gain_score": round(float(info_gain), 4),
            "actionability_score": round(float(actionability), 4),
            "revisit_penalty": round(float(revisit_penalty), 4),
            "dead_end_risk": round(float(dead_end_risk), 4),
            "source_breakdown": {
                "heat_source": heat_source,
                "target_heatmap_score": round(float(heat_score), 4),
                "score_terms": {
                    "target_progress": _safe_float(terms.get("target_progress_score_max"), 0.0),
                    "planner_prior": _safe_float(terms.get("planner_prior_score_max"), 0.0),
                    "novelty": _safe_float(terms.get("novelty_score_max"), 0.0),
                    "actionability": _safe_float(terms.get("actionability_score_max"), 0.0),
                    "repeat": _safe_float(terms.get("repeat_penalty_max"), 0.0),
                    "recent": _safe_float(terms.get("recent_penalty_max"), 0.0),
                },
            },
        }
        table.append(row)
        if _in_bounds(rep, (h, w)):
            # Branch values are rendered as a candidate interface, not as the
            # only state source.  Use smaller radius than target/room potentials.
            add_radial_value(value_map, rep, weight=max(value, 0.18 * info_gain), radius=18)

    table.sort(
        key=lambda item: (
            -float(item.get("target_value_score", 0.0) or 0.0),
            evidence_rank(item.get("evidence_level")),
            str(item.get("id", "")),
        )
    )
    for rank, row in enumerate(table, start=1):
        row["target_value_rank"] = int(rank)

    positive = value_map > 0
    summary = {
        "schema_version": TARGET_VALUE_SCHEMA_VERSION,
        "available": True,
        "source": "target_conditioned_value_v0",
        "heat_source": heat_source,
        "positive_pixel_count": int(np.count_nonzero(positive)),
        "peak": float(value_map.max()) if value_map.size else 0.0,
        "mean_positive": float(value_map[positive].mean()) if np.any(positive) else 0.0,
        "evidence_counts": source_counts,
        "branch_ranking": [
            {
                "id": item["id"],
                "target_value_score": item["target_value_score"],
                "evidence_level": item["evidence_level"],
                "target_visible": item["target_visible"],
            }
            for item in table[:12]
        ],
    }
    return np.clip(value_map, 0.0, 1.0), summary, table
