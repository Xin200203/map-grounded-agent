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
        "final_score": float(final_scores[best_local_idx]),
        "semantic_bias_weight": float(semantic_bias_weight),
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
