"""Pure helpers for Graph.get_goal frontier scoring and diagnostics."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def choose_frontier_locations(
    raw_frontier_locations,
    filtered_frontier_locations,
    *,
    allow_raw_frontier_fallback: bool = True,
):
    """Choose the frontier source, with an optional fallback to raw frontiers."""

    raw_frontier_locations = np.asarray(raw_frontier_locations)
    filtered_frontier_locations = np.asarray(filtered_frontier_locations)
    raw_frontier_count = int(len(raw_frontier_locations))
    filtered_frontier_count = int(len(filtered_frontier_locations))

    if filtered_frontier_count > 0:
        return {
            "frontier_locations": filtered_frontier_locations,
            "frontier_fallback_mode": "",
            "raw_frontier_count": raw_frontier_count,
            "filtered_frontier_count": filtered_frontier_count,
            "used_raw_frontier_fallback": False,
        }

    if allow_raw_frontier_fallback and raw_frontier_count > 0:
        return {
            "frontier_locations": raw_frontier_locations,
            "frontier_fallback_mode": "raw_frontier_fallback",
            "raw_frontier_count": raw_frontier_count,
            "filtered_frontier_count": filtered_frontier_count,
            "used_raw_frontier_fallback": True,
        }

    return {
        "frontier_locations": filtered_frontier_locations,
        "frontier_fallback_mode": "no_frontiers",
        "raw_frontier_count": raw_frontier_count,
        "filtered_frontier_count": filtered_frontier_count,
        "used_raw_frontier_fallback": False,
    }


def select_distance_candidate_indices(
    distances: Sequence[float],
    *,
    distance_threshold: float = 1.2,
    allow_relaxed_distance_fallback: bool = True,
):
    """Select frontier candidate indices after distance gating."""

    distances = np.asarray(distances, dtype=float)
    candidate_indices = np.where(distances >= float(distance_threshold))[0]
    if candidate_indices.size > 0:
        return {
            "candidate_indices": candidate_indices,
            "distance_threshold_used": float(distance_threshold),
            "candidate_fallback_mode": "",
            "used_relaxed_distance_fallback": False,
        }

    if allow_relaxed_distance_fallback and distances.size > 0:
        return {
            "candidate_indices": np.arange(len(distances)),
            "distance_threshold_used": 0.0,
            "candidate_fallback_mode": "relaxed_distance_threshold",
            "used_relaxed_distance_fallback": True,
        }

    return {
        "candidate_indices": candidate_indices,
        "distance_threshold_used": float(distance_threshold),
        "candidate_fallback_mode": "no_candidate_frontiers",
        "used_relaxed_distance_fallback": False,
    }


def select_bias_candidate_indices(
    num_candidates: int,
    bias_distances: Optional[Sequence[float]],
    *,
    bias_candidate_topk: int = 0,
    bias_candidate_radius: float = 0.0,
) -> Tuple[np.ndarray, bool]:
    """Return candidate frontier indices after optional bias-based filtering."""

    candidate_indices = np.arange(int(num_candidates))
    if num_candidates <= 0 or bias_distances is None:
        return candidate_indices, False

    bias_distances = np.asarray(bias_distances, dtype=float)
    radius_indices = np.array([], dtype=int)
    if bias_candidate_radius > 0:
        radius_indices = np.where(bias_distances <= bias_candidate_radius)[0]

    topk_indices = np.array([], dtype=int)
    if bias_candidate_topk > 0:
        bias_order = np.argsort(bias_distances)
        topk_indices = bias_order[: min(int(bias_candidate_topk), len(bias_order))]

    if radius_indices.size > 0 and topk_indices.size > 0:
        candidate_indices = np.intersect1d(radius_indices, topk_indices)
        if candidate_indices.size == 0:
            candidate_indices = radius_indices
    elif radius_indices.size > 0:
        candidate_indices = radius_indices
    elif topk_indices.size > 0:
        candidate_indices = topk_indices

    if candidate_indices.size == 0:
        candidate_indices = np.arange(int(num_candidates))

    return candidate_indices, bool(candidate_indices.size < int(num_candidates))


def summarize_frontier_selection(
    frontier_locations_16,
    distances_16,
    base_scores,
    bias_distances_16,
    bias_scores,
    final_scores,
    candidate_indices,
    *,
    topk_limit: int = 5,
    semantic_bias_weight: float = 1.0,
    last_selected_frontier=None,
    novelty_scores=None,
    actionability_scores=None,
    repeat_penalties=None,
    recent_penalties=None,
    frontier_novelty_weight: float = 0.0,
    local_actionability_weight: float = 0.0,
    frontier_repeat_penalty: float = 0.0,
    frontier_recent_penalty: float = 0.0,
    target_progress_scores=None,
    target_progress_weight: float = 0.0,
    target_progress_mode: str = "",
    planner_prior_scores=None,
    planner_prior_weight: float = 0.0,
    local_index_to_branch_id=None,
) -> Dict[str, Any]:
    """Summarize frontier ranking diagnostics for tracing and metrics."""

    frontier_locations_16 = np.asarray(frontier_locations_16)
    distances_16 = np.asarray(distances_16, dtype=float)
    base_scores = np.asarray(base_scores, dtype=float)
    bias_scores = np.asarray(bias_scores, dtype=float)
    final_scores = np.asarray(final_scores, dtype=float)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    bias_distances = None
    if bias_distances_16 is not None:
        bias_distances = np.asarray(bias_distances_16, dtype=float)
    novelty_scores = _coerce_score_array(novelty_scores, len(final_scores))
    actionability_scores = _coerce_score_array(
        actionability_scores, len(final_scores)
    )
    repeat_penalties = _coerce_score_array(repeat_penalties, len(final_scores))
    recent_penalties = _coerce_score_array(recent_penalties, len(final_scores))
    target_progress_scores = _coerce_score_array(
        target_progress_scores, len(final_scores)
    )
    planner_prior_scores = _coerce_score_array(
        planner_prior_scores, len(final_scores)
    )
    local_index_to_branch_id = dict(local_index_to_branch_id or {})

    order = candidate_indices[np.argsort(final_scores[candidate_indices])[::-1]]
    best_local_idx = int(order[0])
    selected_frontier = (frontier_locations_16[best_local_idx] - 1).tolist()
    selected_same = selected_frontier == last_selected_frontier

    topk_frontiers = []
    for rank, local_idx in enumerate(order[: max(1, int(topk_limit))], start=1):
        frontier_coord = frontier_locations_16[int(local_idx)] - 1
        topk_frontiers.append(
            {
                "rank": int(rank),
                "frontier": frontier_coord.tolist(),
                "agent_distance": float(distances_16[int(local_idx)]),
                "base_score": float(base_scores[int(local_idx)]),
                "bias_distance": (
                    None if bias_distances is None else float(bias_distances[int(local_idx)])
                ),
                "bias_score": float(bias_scores[int(local_idx)]),
                "novelty_score": float(novelty_scores[int(local_idx)]),
                "actionability_score": float(actionability_scores[int(local_idx)]),
                "repeat_penalty": float(repeat_penalties[int(local_idx)]),
                "recent_penalty": float(recent_penalties[int(local_idx)]),
                "target_progress_score": float(
                    target_progress_scores[int(local_idx)]
                ),
                "planner_prior_score": float(planner_prior_scores[int(local_idx)]),
                "branch_id": str(local_index_to_branch_id.get(str(int(local_idx)), "")),
                "final_score": float(final_scores[int(local_idx)]),
            }
        )

    ranked_scores = final_scores[order]
    if len(ranked_scores) >= 2:
        top1_top2_gap = float(ranked_scores[0] - ranked_scores[1])
    elif len(ranked_scores) == 1:
        top1_top2_gap = float(ranked_scores[0])
    else:
        top1_top2_gap = None

    score_breakdown = {
        "agent_distance": float(distances_16[best_local_idx]),
        "base_score": float(base_scores[best_local_idx]),
        "bias_distance": (
            None if bias_distances is None else float(bias_distances[best_local_idx])
        ),
        "bias_score": float(bias_scores[best_local_idx]),
        "novelty_score": float(novelty_scores[best_local_idx]),
        "actionability_score": float(actionability_scores[best_local_idx]),
        "repeat_penalty": float(repeat_penalties[best_local_idx]),
        "recent_penalty": float(recent_penalties[best_local_idx]),
        "target_progress_score": float(target_progress_scores[best_local_idx]),
        "target_progress_weight": float(target_progress_weight),
        "target_progress_mode": str(target_progress_mode or ""),
        "planner_prior_score": float(planner_prior_scores[best_local_idx]),
        "planner_prior_weight": float(planner_prior_weight),
        "branch_id": str(local_index_to_branch_id.get(str(int(best_local_idx)), "")),
        "final_score": float(final_scores[best_local_idx]),
        "semantic_bias_weight": float(semantic_bias_weight),
        "frontier_novelty_weight": float(frontier_novelty_weight),
        "local_actionability_weight": float(local_actionability_weight),
        "frontier_repeat_penalty": float(frontier_repeat_penalty),
        "frontier_recent_penalty": float(frontier_recent_penalty),
    }

    return {
        "best_local_idx": best_local_idx,
        "selected_frontier": selected_frontier,
        "selected_same": bool(selected_same),
        "topk_frontiers": topk_frontiers,
        "top1_top2_gap": top1_top2_gap,
        "base_score_std": float(np.std(base_scores[candidate_indices])),
        "bias_score_std": float(np.std(bias_scores[candidate_indices])),
        "selected_frontier_score_breakdown": score_breakdown,
    }


def _coerce_score_array(values, length: int) -> np.ndarray:
    if values is None:
        return np.zeros(int(length), dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape[0] != int(length):
        raise ValueError(
            f"Expected score array length {int(length)}, got {values.shape[0]}"
        )
    return values


def compute_frontier_unknown_novelty_scores(
    frontier_locations,
    unknown_map,
    *,
    radius: int = 6,
) -> np.ndarray:
    """Approximate information gain as unknown-space density near each frontier.

    This is intentionally lightweight: it does not try to predict the target
    distribution, but it prevents frontier selection from being driven only by
    distance-to-agent and distance-to-symbolic-bias.
    """

    frontier_locations = np.asarray(frontier_locations, dtype=int)
    if frontier_locations.size == 0:
        return np.zeros(0, dtype=float)
    if unknown_map is None:
        return np.zeros(len(frontier_locations), dtype=float)

    unknown = np.asarray(unknown_map).astype(bool)
    if unknown.ndim != 2:
        return np.zeros(len(frontier_locations), dtype=float)

    radius = max(1, int(radius))
    scores = []
    height, width = unknown.shape
    for row, col in frontier_locations:
        row = int(row)
        col = int(col)
        r0 = max(0, row - radius)
        r1 = min(height, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(width, col + radius + 1)
        patch = unknown[r0:r1, c0:c1]
        scores.append(float(patch.mean()) if patch.size else 0.0)
    return np.asarray(scores, dtype=float)


def compute_local_actionability_scores(
    frontier_locations,
    local_map_boundary,
    *,
    local_width: int,
    local_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return local-window validity mask and [0,1] actionability scores."""

    frontier_locations = np.asarray(frontier_locations, dtype=int)
    if frontier_locations.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    if local_map_boundary is None:
        valid = np.ones(len(frontier_locations), dtype=bool)
        return valid, valid.astype(float)

    try:
        row_offset = int(local_map_boundary[0, 0])
        col_offset = int(local_map_boundary[0, 2])
    except Exception:
        valid = np.ones(len(frontier_locations), dtype=bool)
        return valid, valid.astype(float)

    local_width = int(local_width)
    local_height = int(local_height)
    local_rows = frontier_locations[:, 0] - row_offset
    local_cols = frontier_locations[:, 1] - col_offset
    valid = (
        (local_rows >= 0)
        & (local_rows < local_width)
        & (local_cols >= 0)
        & (local_cols < local_height)
    )
    return valid, valid.astype(float)


def compute_frontier_repeat_penalties(
    frontier_locations,
    *,
    last_selected_frontier=None,
    recent_selected_frontiers=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mark exact last/recent frontier reuse for value-level downweighting."""

    frontier_locations = np.asarray(frontier_locations, dtype=int)
    if frontier_locations.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    last = (
        tuple(int(v) for v in last_selected_frontier)
        if last_selected_frontier is not None
        else None
    )
    recent = {
        tuple(int(v) for v in item)
        for item in (recent_selected_frontiers or [])
        if item is not None and len(item) >= 2
    }
    repeat_penalties = []
    recent_penalties = []
    for coord in frontier_locations:
        key = (int(coord[0]), int(coord[1]))
        repeat_penalties.append(1.0 if last is not None and key == last else 0.0)
        recent_penalties.append(1.0 if key in recent else 0.0)
    return np.asarray(repeat_penalties), np.asarray(recent_penalties)


def compose_frontier_value_scores(
    base_scores,
    bias_scores,
    *,
    semantic_bias_weight: float = 1.0,
    novelty_scores=None,
    actionability_scores=None,
    repeat_penalties=None,
    recent_penalties=None,
    frontier_novelty_weight: float = 0.0,
    local_actionability_weight: float = 0.0,
    frontier_repeat_penalty: float = 0.0,
    frontier_recent_penalty: float = 0.0,
    target_progress_scores=None,
    target_progress_weight: float = 0.0,
    planner_prior_scores=None,
    planner_prior_weight: float = 0.0,
) -> np.ndarray:
    """Compose the lightweight frontier value used for semantic anchoring."""

    base_scores = np.asarray(base_scores, dtype=float)
    bias_scores = np.asarray(bias_scores, dtype=float)
    length = len(base_scores)
    novelty_scores = _coerce_score_array(novelty_scores, length)
    actionability_scores = _coerce_score_array(actionability_scores, length)
    repeat_penalties = _coerce_score_array(repeat_penalties, length)
    recent_penalties = _coerce_score_array(recent_penalties, length)
    target_progress_scores = _coerce_score_array(target_progress_scores, length)
    planner_prior_scores = _coerce_score_array(planner_prior_scores, length)

    return (
        base_scores
        + float(semantic_bias_weight) * bias_scores
        + float(target_progress_weight) * target_progress_scores
        + float(planner_prior_weight) * planner_prior_scores
        + float(frontier_novelty_weight) * novelty_scores
        + float(local_actionability_weight) * actionability_scores
        - float(frontier_repeat_penalty) * repeat_penalties
        - float(frontier_recent_penalty) * recent_penalties
    )


def filter_candidate_indices_for_actionability(
    candidate_indices,
    local_valid_mask,
    *,
    relax_to_any_valid: bool = True,
) -> Tuple[np.ndarray, str]:
    """Prefer locally projectable frontiers while preserving fallback behavior."""

    candidate_indices = np.asarray(candidate_indices, dtype=int)
    local_valid_mask = np.asarray(local_valid_mask, dtype=bool)
    if candidate_indices.size == 0 or local_valid_mask.size == 0:
        return candidate_indices, ""

    valid_indices = np.where(local_valid_mask)[0]
    if valid_indices.size == 0:
        return candidate_indices, "no_local_actionable_frontiers"

    actionable_subset = np.intersect1d(candidate_indices, valid_indices)
    if actionable_subset.size > 0:
        if actionable_subset.size < candidate_indices.size:
            return actionable_subset, "filtered_to_local_actionable_subset"
        return candidate_indices, ""

    if relax_to_any_valid:
        return valid_indices, "relaxed_bias_to_local_actionable"

    return candidate_indices, "bias_subset_not_local_actionable"


def compute_target_progress_scores(
    frontier_locations,
    target_coord,
    *,
    agent_coord=None,
    target_distances=None,
    agent_target_distance=None,
    distance_scale: float = 20.0,
) -> Tuple[np.ndarray, str]:
    """Score how useful each frontier is for a persistent target anchor.

    Existing semantic bias scores are intentionally short-range and can saturate
    to zero when a target is outside the active local map or disconnected in the
    currently explored graph.  Target anchors need a softer full-map signal:

    - prefer frontiers that reduce distance to the target hypothesis;
    - otherwise keep a non-zero proximity potential that decays with distance;
    - fall back to Euclidean distances when FMM distances are missing or masked.

    This is a deterministic "PONI-lite" value term, not a learned potential.
    """

    frontier_locations = np.asarray(frontier_locations, dtype=float)
    if frontier_locations.size == 0:
        return np.zeros(0, dtype=float), "no_frontiers"
    if target_coord is None:
        return np.zeros(len(frontier_locations), dtype=float), "no_target"

    try:
        target = np.asarray(target_coord, dtype=float)[:2]
        if target.shape[0] < 2 or not np.all(np.isfinite(target)):
            return np.zeros(len(frontier_locations), dtype=float), "invalid_target"
    except Exception:
        return np.zeros(len(frontier_locations), dtype=float), "invalid_target"

    distance_scale = max(float(distance_scale or 20.0), 1e-6)
    distance_mode = "euclidean"

    frontier_target_dist = None
    if target_distances is not None:
        try:
            candidate = np.asarray(target_distances, dtype=float)
            if candidate.shape[0] == len(frontier_locations):
                # Some FMM implementations fill unreachable cells with a very
                # large finite sentinel.  Keep finite values, then selectively
                # replace pathological outliers with Euclidean distance below.
                frontier_target_dist = candidate.copy()
                distance_mode = "fmm"
        except Exception:
            frontier_target_dist = None

    euclidean_target_dist = np.linalg.norm(frontier_locations[:, :2] - target, axis=1)
    # Frontier distances in Graph.get_goal are in meters (FMM / 20).  Full-map
    # row/col deltas are map cells, so use the same /20 convention for fallback.
    euclidean_target_dist = euclidean_target_dist / 20.0

    if frontier_target_dist is None:
        frontier_target_dist = euclidean_target_dist
    else:
        finite = np.isfinite(frontier_target_dist)
        if finite.any():
            finite_values = frontier_target_dist[finite]
            # Replace extreme FMM sentinels with geometric fallback.  This
            # avoids a target anchor collapsing to all-zero value merely because
            # the exact target point is not yet connected in the traversible map.
            sentinel_threshold = max(
                float(np.nanmedian(finite_values)) * 4.0,
                float(np.nanmin(finite_values)) + distance_scale * 4.0,
                distance_scale * 8.0,
            )
            replace = (~finite) | (frontier_target_dist > sentinel_threshold)
            if replace.any():
                frontier_target_dist = frontier_target_dist.copy()
                frontier_target_dist[replace] = euclidean_target_dist[replace]
                distance_mode = "fmm_euclidean_fallback"
        else:
            frontier_target_dist = euclidean_target_dist
            distance_mode = "euclidean"

    if agent_target_distance is None:
        if agent_coord is not None:
            try:
                agent = np.asarray(agent_coord, dtype=float)[:2]
                if agent.shape[0] >= 2 and np.all(np.isfinite(agent)):
                    agent_target_distance = float(np.linalg.norm(agent - target) / 20.0)
            except Exception:
                agent_target_distance = None

    if agent_target_distance is None or not np.isfinite(agent_target_distance):
        agent_target_distance = float(np.nanmax(frontier_target_dist)) if len(frontier_target_dist) else 0.0

    # Progress is high when the candidate frontier is closer to the target than
    # the current agent location.  Proximity keeps a decaying, non-zero target
    # potential even when every frontier is still far away.
    progress = (
        float(agent_target_distance) - frontier_target_dist
    ) / max(distance_scale, float(agent_target_distance), 1.0)
    progress = np.clip(progress, 0.0, 1.0)
    proximity = 1.0 / (1.0 + np.maximum(frontier_target_dist, 0.0) / distance_scale)
    scores = np.maximum(progress, proximity)
    scores = np.clip(scores, 0.0, 1.0)
    return scores.astype(float), distance_mode


def _array_payload(values) -> list:
    if values is None:
        return []
    try:
        return np.asarray(values).tolist()
    except Exception:
        return []


def build_frontier_value_replay_snapshot(
    *,
    frontier_locations_16,
    distances_16,
    base_scores,
    bias_distances_16=None,
    bias_scores=None,
    novelty_scores=None,
    actionability_scores=None,
    repeat_penalties=None,
    recent_penalties=None,
    target_progress_scores=None,
    planner_prior_scores=None,
    candidate_indices=None,
    final_scores=None,
    bias_input=None,
    agent_coord=None,
    agent_target_distance=None,
    active_target_region: str = "",
    active_anchor_object: str = "",
    target_progress_active: bool = False,
    target_progress_mode: str = "",
    semantic_bias_weight: float = 1.0,
    frontier_novelty_weight: float = 0.0,
    local_actionability_weight: float = 0.0,
    frontier_repeat_penalty: float = 0.0,
    frontier_recent_penalty: float = 0.0,
    target_progress_weight: float = 0.0,
    target_progress_distance_scale: float = 20.0,
    planner_prior_weight: float = 0.0,
    planner_prior_source: str = "",
    planner_selected_branch_id: str = "",
    frontier_branches=None,
    actionability_filter_mode: str = "",
    selected_frontier=None,
    selected_frontier_score_breakdown=None,
) -> Dict[str, Any]:
    """Build a compact, JSON-ready replay snapshot for frontier-value checks.

    The snapshot is intentionally scoped to the semantic-grounding boundary:
    it stores the frontier candidates and scalar terms needed to replay
    Graph.get_goal's value ranking without Habitat, Detectron, or an LLM call.
    """

    return {
        "schema_version": "smoothnav.frontier_value_replay.v1",
        "stage": "grounding_frontier_value",
        "boundary": "semantic_strategy_to_geometric_frontier",
        "active_target_region": str(active_target_region or ""),
        "active_anchor_object": str(active_anchor_object or ""),
        "target_progress_active": bool(target_progress_active),
        "target_progress_mode": str(target_progress_mode or ""),
        "bias_input": _array_payload(bias_input) if bias_input is not None else None,
        "agent_coord": _array_payload(agent_coord) if agent_coord is not None else None,
        "agent_target_distance": agent_target_distance,
        "frontier_locations_16": _array_payload(frontier_locations_16),
        "distances_16": _array_payload(distances_16),
        "base_scores": _array_payload(base_scores),
        "bias_distances_16": (
            _array_payload(bias_distances_16) if bias_distances_16 is not None else None
        ),
        "bias_scores": _array_payload(bias_scores),
        "novelty_scores": _array_payload(novelty_scores),
        "actionability_scores": _array_payload(actionability_scores),
        "repeat_penalties": _array_payload(repeat_penalties),
        "recent_penalties": _array_payload(recent_penalties),
        "target_progress_scores": _array_payload(target_progress_scores),
        "planner_prior_scores": _array_payload(planner_prior_scores),
        "candidate_indices": _array_payload(candidate_indices),
        "final_scores": _array_payload(final_scores),
        "frontier_branches": dict(frontier_branches or {}),
        "weights": {
            "semantic_bias_weight": float(semantic_bias_weight),
            "frontier_novelty_weight": float(frontier_novelty_weight),
            "local_actionability_weight": float(local_actionability_weight),
            "frontier_repeat_penalty": float(frontier_repeat_penalty),
            "frontier_recent_penalty": float(frontier_recent_penalty),
            "target_progress_weight": float(target_progress_weight),
            "target_progress_distance_scale": float(target_progress_distance_scale),
            "planner_prior_weight": float(planner_prior_weight),
        },
        "planner_prior_source": str(planner_prior_source or ""),
        "planner_selected_branch_id": str(planner_selected_branch_id or ""),
        "actionability_filter_mode": str(actionability_filter_mode or ""),
        "selected_frontier": (
            _array_payload(selected_frontier) if selected_frontier is not None else None
        ),
        "selected_frontier_score_breakdown": dict(
            selected_frontier_score_breakdown or {}
        ),
    }


def replay_frontier_value_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Replay frontier-value ranking from a stored grounding snapshot."""

    weights = dict(snapshot.get("weights") or {})
    frontier_locations = np.asarray(
        snapshot.get("frontier_locations_16") or [], dtype=float
    )
    base_scores = np.asarray(snapshot.get("base_scores") or [], dtype=float)
    length = len(base_scores)
    if length == 0 or len(frontier_locations) == 0:
        return {
            "schema_version": "smoothnav.frontier_value_replay.result.v1",
            "status": "empty",
            "reason": "no_frontier_scores",
            "checks": {"has_candidates": False},
        }

    distances_16 = np.asarray(snapshot.get("distances_16") or np.zeros(length), dtype=float)
    def _snapshot_scores(key: str) -> np.ndarray:
        raw = snapshot.get(key)
        if raw is None or raw == []:
            return np.zeros(length, dtype=float)
        return _coerce_score_array(raw, length)

    bias_scores = _snapshot_scores("bias_scores")
    novelty_scores = _snapshot_scores("novelty_scores")
    actionability_scores = _snapshot_scores("actionability_scores")
    repeat_penalties = _snapshot_scores("repeat_penalties")
    recent_penalties = _snapshot_scores("recent_penalties")
    target_progress_scores = _snapshot_scores("target_progress_scores")
    planner_prior_scores = _snapshot_scores("planner_prior_scores")
    target_progress_mode = str(snapshot.get("target_progress_mode") or "")

    if (
        bool(snapshot.get("target_progress_active"))
        and not np.any(target_progress_scores > 0.0)
    ):
        target_progress_scores, target_progress_mode = compute_target_progress_scores(
            frontier_locations - 1,
            snapshot.get("bias_input"),
            agent_coord=snapshot.get("agent_coord"),
            target_distances=snapshot.get("bias_distances_16"),
            agent_target_distance=snapshot.get("agent_target_distance"),
            distance_scale=float(weights.get("target_progress_distance_scale", 20.0) or 20.0),
        )

    candidate_indices = np.asarray(snapshot.get("candidate_indices") or [], dtype=int)
    if candidate_indices.size == 0:
        candidate_indices = np.arange(length, dtype=int)

    final_scores = compose_frontier_value_scores(
        base_scores,
        bias_scores,
        semantic_bias_weight=float(weights.get("semantic_bias_weight", 1.0) or 1.0),
        novelty_scores=novelty_scores,
        actionability_scores=actionability_scores,
        repeat_penalties=repeat_penalties,
        recent_penalties=recent_penalties,
        frontier_novelty_weight=float(weights.get("frontier_novelty_weight", 0.0) or 0.0),
        local_actionability_weight=float(
            weights.get("local_actionability_weight", 0.0) or 0.0
        ),
        frontier_repeat_penalty=float(
            weights.get("frontier_repeat_penalty", 0.0) or 0.0
        ),
        frontier_recent_penalty=float(
            weights.get("frontier_recent_penalty", 0.0) or 0.0
        ),
        target_progress_scores=target_progress_scores,
        target_progress_weight=float(weights.get("target_progress_weight", 0.0) or 0.0),
        planner_prior_scores=planner_prior_scores,
        planner_prior_weight=float(weights.get("planner_prior_weight", 0.0) or 0.0),
    )
    frontier_branches = snapshot.get("frontier_branches") or {}
    local_index_to_branch_id = (
        frontier_branches.get("local_index_to_branch_id")
        if isinstance(frontier_branches, dict)
        else {}
    )
    summary = summarize_frontier_selection(
        frontier_locations,
        distances_16,
        base_scores,
        np.asarray(snapshot.get("bias_distances_16"), dtype=float)
        if snapshot.get("bias_distances_16") is not None
        else None,
        bias_scores,
        final_scores,
        candidate_indices,
        semantic_bias_weight=float(weights.get("semantic_bias_weight", 1.0) or 1.0),
        novelty_scores=novelty_scores,
        actionability_scores=actionability_scores,
        repeat_penalties=repeat_penalties,
        recent_penalties=recent_penalties,
        frontier_novelty_weight=float(weights.get("frontier_novelty_weight", 0.0) or 0.0),
        local_actionability_weight=float(
            weights.get("local_actionability_weight", 0.0) or 0.0
        ),
        frontier_repeat_penalty=float(
            weights.get("frontier_repeat_penalty", 0.0) or 0.0
        ),
        frontier_recent_penalty=float(
            weights.get("frontier_recent_penalty", 0.0) or 0.0
        ),
        target_progress_scores=target_progress_scores,
        target_progress_weight=float(weights.get("target_progress_weight", 0.0) or 0.0),
        target_progress_mode=target_progress_mode,
        planner_prior_scores=planner_prior_scores,
        planner_prior_weight=float(weights.get("planner_prior_weight", 0.0) or 0.0),
        local_index_to_branch_id=local_index_to_branch_id,
    )
    selected = dict(summary.get("selected_frontier_score_breakdown") or {})
    semantic_term = (
        float(weights.get("semantic_bias_weight", 1.0) or 1.0)
        * float(selected.get("bias_score") or 0.0)
        + float(weights.get("target_progress_weight", 0.0) or 0.0)
        * float(selected.get("target_progress_score") or 0.0)
        + float(weights.get("planner_prior_weight", 0.0) or 0.0)
        * float(selected.get("planner_prior_score") or 0.0)
    )
    final_score = float(selected.get("final_score") or 0.0)
    semantic_share = semantic_term / final_score if final_score > 0 else 0.0
    target_active = bool(snapshot.get("target_progress_active"))
    checks = {
        "has_candidates": bool(len(candidate_indices) > 0),
        "target_branch_active": target_active,
        "target_progress_nonzero": (
            not target_active or float(selected.get("target_progress_score") or 0.0) > 0.0
        ),
        "semantic_term_nonzero": (not target_active or semantic_term > 0.0),
        "selected_frontier_matches_snapshot": (
            snapshot.get("selected_frontier") is None
            or list(summary.get("selected_frontier") or [])
            == list(snapshot.get("selected_frontier") or [])
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": "smoothnav.frontier_value_replay.result.v1",
        "status": status,
        "checks": checks,
        "selected_frontier": summary.get("selected_frontier"),
        "selected_branch_id": selected.get("branch_id", ""),
        "selected_score_breakdown": selected,
        "frontier_branches": frontier_branches,
        "target_progress_mode": target_progress_mode,
        "semantic_term": semantic_term,
        "semantic_term_share": semantic_share,
        "top1_top2_gap": summary.get("top1_top2_gap"),
    }
