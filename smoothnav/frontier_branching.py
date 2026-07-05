"""Frontier branch decomposition and BEV annotation helpers.

The high-level planner should reason about map-level exploration branches
(doorways/openings/corridor continuations), while the low-level grounding code
still executes concrete frontier points.  This module keeps that boundary pure
and replayable: branch IDs are derived deterministically from SmoothNav-owned
frontier coordinates, and any MLLM output can be mapped back to candidate point
indices before scoring.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


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


def _as_2d_int_coords(coords: Any) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=int)
    arr = arr.reshape((-1, arr.shape[-1])) if arr.ndim != 2 else arr
    if arr.shape[1] < 2:
        return np.empty((0, 2), dtype=int)
    return np.rint(arr[:, :2]).astype(int)


def _score_array(values: Any, length: int, default: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full(length, float(default), dtype=float)
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return np.full(length, float(default), dtype=float)
    if arr.size == 0:
        return np.full(length, float(default), dtype=float)
    if arr.shape[0] != length:
        out = np.full(length, float(default), dtype=float)
        count = min(length, arr.shape[0])
        out[:count] = arr[:count]
        return out
    return arr.astype(float)


def _json_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _connected_components(coords: np.ndarray, *, connectivity: int = 8) -> List[List[int]]:
    """Return connected components as lists of local coordinate indices."""

    coords = _as_2d_int_coords(coords)
    if coords.size == 0:
        return []
    neighbor_offsets = _COORD_NEIGHBORS_8 if int(connectivity) == 8 else (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
    )
    coord_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for idx, (row, col) in enumerate(coords):
        coord_to_indices.setdefault((int(row), int(col)), []).append(int(idx))

    visited = set()
    components: List[List[int]] = []
    for start_idx, coord in enumerate(coords):
        start_key = (int(coord[0]), int(coord[1]))
        if start_key in visited:
            continue
        queue = [start_key]
        visited.add(start_key)
        component_indices: List[int] = []
        while queue:
            key = queue.pop(0)
            component_indices.extend(coord_to_indices.get(key, []))
            for dr, dc in neighbor_offsets:
                nkey = (key[0] + dr, key[1] + dc)
                if nkey in coord_to_indices and nkey not in visited:
                    visited.add(nkey)
                    queue.append(nkey)
        components.append(sorted(component_indices))
    return components


def _direction_from_agent(centroid: Sequence[float], agent_coord: Optional[Sequence[float]]) -> str:
    if agent_coord is None:
        return ""
    try:
        agent = np.asarray(agent_coord, dtype=float)[:2]
        center = np.asarray(centroid, dtype=float)[:2]
    except Exception:
        return ""
    if agent.shape[0] < 2 or center.shape[0] < 2:
        return ""
    delta = center - agent
    if not np.all(np.isfinite(delta)):
        return ""
    if abs(delta[0]) >= abs(delta[1]):
        return "south" if delta[0] > 0 else "north"
    return "east" if delta[1] > 0 else "west"


def build_frontier_branches(
    frontier_locations: Any,
    *,
    candidate_indices: Optional[Sequence[int]] = None,
    distances: Any = None,
    base_scores: Any = None,
    bias_scores: Any = None,
    novelty_scores: Any = None,
    actionability_scores: Any = None,
    repeat_penalties: Any = None,
    recent_penalties: Any = None,
    target_progress_scores: Any = None,
    planner_prior_scores: Any = None,
    final_scores: Any = None,
    agent_coord: Optional[Sequence[float]] = None,
    id_prefix: str = "B",
    connectivity: int = 8,
    max_candidate_points_per_branch: int = 8,
) -> Dict[str, Any]:
    """Group frontier candidate points into deterministic branch IDs.

    Parameters are aligned by local frontier index.  Coordinates should use the
    executable SmoothNav map coordinate convention (the same convention used by
    selected_frontier in traces, i.e. ``frontier_locations_16 - 1`` inside
    Graph.get_goal()).
    """

    coords = _as_2d_int_coords(frontier_locations)
    length = int(len(coords))
    if length == 0:
        return {
            "schema_version": "smoothnav.frontier_branches.v1",
            "branch_count": 0,
            "branches": [],
            "local_index_to_branch_id": {},
            "candidate_branch_ids": [],
        }

    candidate_set = (
        {int(i) for i in np.asarray(candidate_indices, dtype=int).tolist()}
        if candidate_indices is not None and len(np.asarray(candidate_indices).reshape(-1)) > 0
        else set(range(length))
    )
    distances_arr = _score_array(distances, length, default=0.0)
    base_arr = _score_array(base_scores, length, default=0.0)
    bias_arr = _score_array(bias_scores, length, default=0.0)
    novelty_arr = _score_array(novelty_scores, length, default=0.0)
    actionability_arr = _score_array(actionability_scores, length, default=0.0)
    repeat_arr = _score_array(repeat_penalties, length, default=0.0)
    recent_arr = _score_array(recent_penalties, length, default=0.0)
    target_progress_arr = _score_array(target_progress_scores, length, default=0.0)
    planner_prior_arr = _score_array(planner_prior_scores, length, default=0.0)
    final_arr = _score_array(final_scores, length, default=0.0)

    components = _connected_components(coords, connectivity=connectivity)

    sortable_components = []
    for component in components:
        component_coords = coords[component]
        centroid = component_coords.mean(axis=0)
        sortable_components.append((float(centroid[0]), float(centroid[1]), component))
    sortable_components.sort(key=lambda item: (item[0], item[1], len(item[2])))

    branches = []
    local_index_to_branch_id: Dict[str, str] = {}
    for branch_number, (_, _, component) in enumerate(sortable_components, start=1):
        branch_id = f"{id_prefix}{branch_number}"
        component_arr = np.asarray(component, dtype=int)
        component_coords = coords[component_arr]
        active_indices = [idx for idx in component if idx in candidate_set]
        rep_pool = np.asarray(active_indices if active_indices else component, dtype=int)
        if rep_pool.size:
            # Prefer the highest final score among executable candidates, with a
            # deterministic coordinate tie-breaker.
            rep_order = sorted(
                rep_pool.tolist(),
                key=lambda idx: (
                    -float(final_arr[int(idx)]),
                    float(distances_arr[int(idx)]),
                    int(coords[int(idx), 0]),
                    int(coords[int(idx), 1]),
                ),
            )
            representative_idx = int(rep_order[0])
        else:
            representative_idx = int(component[0])
        candidate_points = sorted(
            component,
            key=lambda idx: (
                idx not in candidate_set,
                -float(final_arr[int(idx)]),
                float(distances_arr[int(idx)]),
                int(coords[int(idx), 0]),
                int(coords[int(idx), 1]),
            ),
        )[: max(1, int(max_candidate_points_per_branch))]

        for idx in component:
            local_index_to_branch_id[str(int(idx))] = branch_id

        aggregate = {
            "distance_min": _json_float(np.min(distances_arr[component_arr])),
            "distance_mean": _json_float(np.mean(distances_arr[component_arr])),
            "base_score_max": _json_float(np.max(base_arr[component_arr])),
            "bias_score_max": _json_float(np.max(bias_arr[component_arr])),
            "novelty_score_max": _json_float(np.max(novelty_arr[component_arr])),
            "actionability_score_max": _json_float(np.max(actionability_arr[component_arr])),
            "target_progress_score_max": _json_float(np.max(target_progress_arr[component_arr])),
            "planner_prior_score_max": _json_float(np.max(planner_prior_arr[component_arr])),
            "final_score_max": _json_float(np.max(final_arr[component_arr])),
            "repeat_penalty_max": _json_float(np.max(repeat_arr[component_arr])),
            "recent_penalty_max": _json_float(np.max(recent_arr[component_arr])),
        }
        branches.append(
            {
                "id": branch_id,
                "component_label": int(branch_number),
                "pixel_count": int(len(component)),
                "candidate_point_count": int(len(active_indices)),
                "local_indices": [int(i) for i in component],
                "candidate_local_indices": [int(i) for i in active_indices],
                "centroid": [
                    _json_float(component_coords[:, 0].mean(), 0.0),
                    _json_float(component_coords[:, 1].mean(), 0.0),
                ],
                "representative_local_idx": representative_idx,
                "representative_coord": coords[representative_idx].astype(int).tolist(),
                "candidate_points": [
                    {
                        "local_idx": int(idx),
                        "coord": coords[int(idx)].astype(int).tolist(),
                        "final_score": _json_float(final_arr[int(idx)], 0.0),
                        "agent_distance": _json_float(distances_arr[int(idx)], 0.0),
                        "actionable": bool(int(idx) in candidate_set),
                    }
                    for idx in candidate_points
                ],
                "direction_from_agent": _direction_from_agent(
                    [component_coords[:, 0].mean(), component_coords[:, 1].mean()],
                    agent_coord,
                ),
                "score_terms_aggregate": aggregate,
            }
        )

    candidate_branch_ids = []
    for idx in sorted(candidate_set):
        branch_id = local_index_to_branch_id.get(str(int(idx)))
        if branch_id and branch_id not in candidate_branch_ids:
            candidate_branch_ids.append(branch_id)

    return {
        "schema_version": "smoothnav.frontier_branches.v1",
        "branch_count": int(len(branches)),
        "branches": branches,
        "local_index_to_branch_id": local_index_to_branch_id,
        "candidate_branch_ids": candidate_branch_ids,
    }


def branch_ids(frontier_branches: Mapping[str, Any]) -> List[str]:
    return [str(item.get("id")) for item in frontier_branches.get("branches", [])]


def build_branch_prior_scores(
    frontier_branches: Mapping[str, Any],
    *,
    selected_branch_id: str = "",
    ranked_branches: Optional[Iterable[Mapping[str, Any]]] = None,
    length: Optional[int] = None,
    default_selected_score: float = 1.0,
) -> np.ndarray:
    """Convert a branch-ranked planner output into per-frontier prior scores."""

    local_map = dict(frontier_branches.get("local_index_to_branch_id") or {})
    if length is None:
        max_idx = -1
        for key in local_map:
            try:
                max_idx = max(max_idx, int(key))
            except Exception:
                pass
        length = max_idx + 1
    scores = np.zeros(max(0, int(length)), dtype=float)
    branch_score: Dict[str, float] = {}
    if selected_branch_id:
        branch_score[str(selected_branch_id)] = float(default_selected_score)
    for rank, item in enumerate(ranked_branches or [], start=1):
        bid = str(item.get("id") or item.get("branch_id") or "")
        if not bid:
            continue
        raw_score = item.get("score")
        score = _json_float(raw_score)
        if score is None:
            score = max(0.0, float(default_selected_score) / float(rank + 1))
        branch_score[bid] = max(branch_score.get(bid, 0.0), float(score))
    for key, bid in local_map.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if 0 <= idx < len(scores):
            scores[idx] = float(branch_score.get(str(bid), 0.0))
    return scores


def branch_for_local_idx(frontier_branches: Mapping[str, Any], local_idx: int) -> str:
    return str((frontier_branches.get("local_index_to_branch_id") or {}).get(str(int(local_idx)), ""))


def _map_to_numpy(map_like: Any) -> Optional[np.ndarray]:
    if map_like is None:
        return None
    try:
        if hasattr(map_like, "detach"):
            map_like = map_like.detach().cpu().numpy()
        arr = np.asarray(map_like)
    except Exception:
        return None
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def _content_crop_bounds(
    known_mask: np.ndarray,
    frontier_branches: Mapping[str, Any],
    *,
    agent_coord: Optional[Sequence[float]] = None,
    margin: int = 24,
) -> Tuple[int, int, int, int]:
    """Return row/col crop bounds that keep map content and annotations visible.

    The original capsule BEV images rendered the complete global map canvas.
    In the hard `ep228` frames the useful explored area occupied only a small
    patch inside a mostly-unknown image, making the visual branch topology hard
    for both humans and MLLMs to read.  Cropping to the known map + branch
    annotations preserves the same coordinates in the text prompt while making
    the image itself informative.
    """

    height, width = known_mask.shape[:2]
    rows: list[int] = []
    cols: list[int] = []
    known = np.asarray(known_mask, dtype=bool)
    if known.any():
        rr, cc = np.where(known)
        rows.extend([int(rr.min()), int(rr.max())])
        cols.extend([int(cc.min()), int(cc.max())])

    def add_coord(coord: Any) -> None:
        try:
            if coord is None or len(coord) < 2:
                return
            row, col = float(coord[0]), float(coord[1])
            if not math.isfinite(row) or not math.isfinite(col):
                return
            rows.append(int(round(row)))
            cols.append(int(round(col)))
        except Exception:
            return

    add_coord(agent_coord)
    for branch in frontier_branches.get("branches", []) or []:
        add_coord(branch.get("representative_coord") or branch.get("centroid"))
        add_coord(branch.get("centroid"))
        for point in branch.get("candidate_points", []) or []:
            if isinstance(point, Mapping):
                add_coord(point.get("coord"))

    if not rows or not cols:
        return 0, height, 0, width

    pad = max(0, int(margin))
    r0 = max(0, min(rows) - pad)
    r1 = min(height, max(rows) + pad + 1)
    c0 = max(0, min(cols) - pad)
    c1 = min(width, max(cols) + pad + 1)
    if r1 <= r0 or c1 <= c0:
        return 0, height, 0, width
    return r0, r1, c0, c1


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
    )
    return palette[int(index) % len(palette)]


def _draw_label(
    draw: Any,
    xy: Tuple[int, int],
    text: str,
    *,
    fill: Tuple[int, int, int],
    font: Any,
    image_size: Optional[Tuple[int, int]] = None,
) -> None:
    """Draw a small readable label with a dark translucent-like backing."""

    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
    except Exception:  # pragma: no cover - older Pillow fallback
        w, h = draw.textsize(text, font=font)
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
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(35, 35, 35),
        outline=(230, 230, 230),
    )
    draw.text((x, y), text, fill=fill, font=font)


def render_branch_bev_image(
    full_map: Any,
    frontier_branches: Mapping[str, Any],
    *,
    agent_coord: Optional[Sequence[float]] = None,
    scale: int = 2,
    crop_to_content: bool = True,
    crop_margin: int = 24,
) -> Any:
    """Render an annotated BEV image for capsule/debug views.

    The renderer visualizes obstacle/free-like channels and overlays branch IDs,
    candidate points, actionability, coarse scores, and a legend.  It crops the
    mostly-empty global canvas by default so the BEV topology is readable in
    saved capsules and MLLM calls.
    If Pillow is unavailable or the map cannot be converted, it returns
    ``None``.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # pragma: no cover - Pillow is expected but optional
        return None

    arr = _map_to_numpy(full_map)
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 3:
        channels = arr
        height, width = channels.shape[-2], channels.shape[-1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, :] = np.array([70, 70, 70], dtype=np.uint8)  # unknown
        known_mask = np.any(channels > 0, axis=0)
        if channels.shape[0] > 1:
            free = channels[1] > 0
            canvas[free] = np.array([230, 230, 230], dtype=np.uint8)
        if channels.shape[0] > 0:
            obstacle = channels[0] > 0
            canvas[obstacle] = np.array([20, 20, 20], dtype=np.uint8)
    elif arr.ndim == 2:
        height, width = arr.shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, :] = np.array([70, 70, 70], dtype=np.uint8)
        canvas[arr > 0] = np.array([230, 230, 230], dtype=np.uint8)
        known_mask = arr > 0
    else:
        return None

    r0, r1, c0, c1 = (0, height, 0, width)
    if crop_to_content:
        r0, r1, c0, c1 = _content_crop_bounds(
            known_mask,
            frontier_branches,
            agent_coord=agent_coord,
            margin=crop_margin,
        )
        canvas = canvas[r0:r1, c0:c1]

    image = Image.fromarray(canvas, mode="RGB")
    scale = max(1, int(scale))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    def xy_from_coord(coord: Sequence[float]) -> Tuple[int, int]:
        row, col = float(coord[0]), float(coord[1])
        return int(round((col - c0) * scale)), int(round((row - r0) * scale))

    # Draw candidate points first, then representative branch markers.
    for branch_idx, branch in enumerate(frontier_branches.get("branches", []) or []):
        color = _branch_color(branch_idx)
        candidate_count = int(branch.get("candidate_point_count", 0) or 0)
        terms = branch.get("score_terms_aggregate", {}) or {}
        actionable = bool(candidate_count > 0 and float(terms.get("actionability_score_max") or 0.0) > 0.0)
        point_radius = max(2, 2 * scale)
        for point in branch.get("candidate_points", []) or []:
            if not isinstance(point, Mapping):
                continue
            coord = point.get("coord")
            if not coord or len(coord) < 2:
                continue
            try:
                x, y = xy_from_coord(coord)
            except Exception:
                continue
            draw.rectangle(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=color if bool(point.get("actionable", True)) else (120, 120, 120),
                outline=(20, 20, 20),
            )

        coord = branch.get("representative_coord") or branch.get("centroid")
        if not coord or len(coord) < 2:
            continue
        try:
            x, y = xy_from_coord(coord)
        except Exception:
            continue
        r = max(5, 5 * scale)
        if actionable:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(20, 20, 20), width=max(1, scale))
        else:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(150, 150, 150), outline=(255, 0, 0), width=max(1, scale))
            draw.line((x - r, y - r, x + r, y + r), fill=(255, 0, 0), width=max(1, scale))
            draw.line((x - r, y + r, x + r, y - r), fill=(255, 0, 0), width=max(1, scale))
        final_score = terms.get("final_score_max")
        score_text = "" if final_score is None else f" F={float(final_score):.1f}"
        label = (
            f"{branch.get('id', 'B?')} {branch.get('direction_from_agent', '')} "
            f"cand={candidate_count}{score_text}"
        ).strip()
        _draw_label(draw, (x + r + 2, y - r), label, fill=color, font=font, image_size=image.size)

    if agent_coord is not None:
        try:
            x, y = xy_from_coord(np.asarray(agent_coord, dtype=float)[:2])
            r = max(6, 6 * scale)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 180, 0), outline=(0, 0, 0))
            _draw_label(draw, (x + r + 2, y), "A robot", fill=(0, 255, 0), font=font, image_size=image.size)
        except Exception:
            pass

    # Lightweight legend and compass.  Keep it in-image so manual screenshots
    # and MLLM inputs carry the same interpretation.
    legend_lines = [
        "A=robot; colored B=frontier branch",
        "square=sample candidate; red X=non-actionable",
        f"crop rows {r0}:{r1}, cols {c0}:{c1}",
    ]
    legend_w = 0
    legend_h = 0
    try:
        boxes = [draw.textbbox((0, 0), line, font=font) for line in legend_lines]
        legend_w = max(box[2] - box[0] for box in boxes) + 10
        legend_h = sum(box[3] - box[1] + 4 for box in boxes) + 6
    except Exception:  # pragma: no cover
        legend_w, legend_h = 280, 58
    draw.rectangle((4, 4, 4 + legend_w, 4 + legend_h), fill=(35, 35, 35), outline=(230, 230, 230))
    y_text = 8
    for line in legend_lines:
        draw.text((9, y_text), line, fill=(245, 245, 245), font=font)
        y_text += 14
    arrow_x = image.width - 34
    arrow_y = 18
    draw.line((arrow_x, arrow_y + 34, arrow_x, arrow_y), fill=(255, 255, 255), width=max(1, scale))
    draw.polygon(
        [(arrow_x, arrow_y - 2), (arrow_x - 6, arrow_y + 10), (arrow_x + 6, arrow_y + 10)],
        fill=(255, 255, 255),
    )
    draw.text((arrow_x - 4, arrow_y + 38), "N", fill=(255, 255, 255), font=font)

    return image


def save_branch_bev_image(
    path: str | Path,
    full_map: Any,
    frontier_branches: Mapping[str, Any],
    *,
    agent_coord: Optional[Sequence[float]] = None,
    scale: int = 2,
    crop_to_content: bool = True,
    crop_margin: int = 24,
) -> Optional[str]:
    """Render and save an annotated BEV PNG."""

    image = render_branch_bev_image(
        full_map,
        frontier_branches,
        agent_coord=agent_coord,
        scale=scale,
        crop_to_content=crop_to_content,
        crop_margin=crop_margin,
    )
    if image is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)
