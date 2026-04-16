"""Strategy grounding helpers decoupled from the heavy runtime entrypoint."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroundingResult:
    """Structured result for one semantic-strategy grounding attempt."""

    success: bool
    changed: bool
    reason: str
    noop_reason: Optional[str] = None
    noop_type: str = ""
    bias_input: Optional[Tuple[int, int]] = None
    selected_frontier: Optional[Tuple[int, int]] = None
    selected_frontier_same_as_prev: bool = False
    selected_frontier_same_as_previous: bool = False
    selected_frontier_score: Optional[float] = None
    selected_frontier_score_breakdown: Dict[str, Any] = field(default_factory=dict)
    projected_goal: Optional[Tuple[int, int]] = None
    local_projection_valid: bool = False
    topk_frontier_scores: List[Dict[str, Any]] = field(default_factory=list)
    top1_top2_gap: Optional[float] = None
    base_score_std: Optional[float] = None
    bias_score_std: Optional[float] = None
    candidate_frontier_count_after_bias_filter: Optional[int] = None
    selected_from_bias_filtered_subset: bool = False
    graph_no_goal_reason: str = ""
    frontier_filter_fallback_mode: str = ""
    candidate_distance_fallback_mode: str = ""
    raw_frontier_count: Optional[int] = None
    filtered_frontier_count: Optional[int] = None
    used_raw_frontier_fallback: bool = False
    used_relaxed_distance_fallback: bool = False
    goal_before: List[int] = field(default_factory=list)
    goal_after: List[int] = field(default_factory=list)
    graph_debug: Dict[str, Any] = field(default_factory=dict)
    goal_epoch: int = 0
    task_epoch: int = 0
    belief_epoch: int = 0
    world_epoch: int = 0
    source_mode: str = ""
    candidate_family: str = "search"
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    primary_goal: Optional[Dict[str, Any]] = None
    geometric_actionability_confidence: float = 0.0
    candidate_entropy: Optional[float] = None
    ambiguity_type: Optional[str] = None
    failure_code: Optional[str] = None
    fallback_policy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "changed": bool(self.changed),
            "reason": self.reason,
            "noop_reason": self.noop_reason,
            "noop_type": self.noop_type,
            "bias_input": list(self.bias_input) if self.bias_input is not None else None,
            "selected_frontier": (
                list(self.selected_frontier)
                if self.selected_frontier is not None
                else None
            ),
            "selected_frontier_same_as_prev": bool(
                self.selected_frontier_same_as_prev
            ),
            "selected_frontier_same_as_previous": bool(
                self.selected_frontier_same_as_previous
            ),
            "selected_frontier_score": self.selected_frontier_score,
            "selected_frontier_score_breakdown": dict(
                self.selected_frontier_score_breakdown
            ),
            "projected_goal": (
                list(self.projected_goal) if self.projected_goal is not None else None
            ),
            "local_projection_valid": bool(self.local_projection_valid),
            "topk_frontier_scores": [dict(item) for item in self.topk_frontier_scores],
            "top1_top2_gap": self.top1_top2_gap,
            "base_score_std": self.base_score_std,
            "bias_score_std": self.bias_score_std,
            "candidate_frontier_count_after_bias_filter": (
                self.candidate_frontier_count_after_bias_filter
            ),
            "selected_from_bias_filtered_subset": bool(
                self.selected_from_bias_filtered_subset
            ),
            "graph_no_goal_reason": self.graph_no_goal_reason,
            "frontier_filter_fallback_mode": self.frontier_filter_fallback_mode,
            "candidate_distance_fallback_mode": self.candidate_distance_fallback_mode,
            "raw_frontier_count": self.raw_frontier_count,
            "filtered_frontier_count": self.filtered_frontier_count,
            "used_raw_frontier_fallback": bool(self.used_raw_frontier_fallback),
            "used_relaxed_distance_fallback": bool(
                self.used_relaxed_distance_fallback
            ),
            "goal_before": list(self.goal_before),
            "goal_after": list(self.goal_after),
            "graph_debug": dict(self.graph_debug),
            "goal_epoch": int(self.goal_epoch),
            "task_epoch": int(self.task_epoch),
            "belief_epoch": int(self.belief_epoch),
            "world_epoch": int(self.world_epoch),
            "source_mode": self.source_mode,
            "candidate_family": self.candidate_family,
            "candidates": [dict(item) for item in self.candidates],
            "primary_goal": (
                dict(self.primary_goal) if self.primary_goal is not None else None
            ),
            "geometric_actionability_confidence": float(
                self.geometric_actionability_confidence
            ),
            "candidate_entropy": self.candidate_entropy,
            "ambiguity_type": self.ambiguity_type,
            "failure_code": self.failure_code,
            "fallback_policy": self.fallback_policy,
        }


def apply_strategy(strategy, graph, bev_map, args, global_goals):
    """Ground a semantic strategy into a local-map goal update."""
    graph.set_full_map(bev_map.full_map)
    graph.set_full_pose(bev_map.full_pose)

    goal_before = list(global_goals)
    bias = strategy.bias_position if strategy else None
    goal = graph.get_goal(goal=bias)
    graph_debug = dict(getattr(graph, "last_goal_debug", {}) or {})
    selected_frontier = graph_debug.get("selected_frontier")
    if selected_frontier is not None:
        selected_frontier = tuple(selected_frontier)

    same_as_prev = bool(
        graph_debug.get(
            "selected_frontier_same_as_prev",
            graph_debug.get("selected_frontier_same_as_previous", False),
        )
    )
    selected_frontier_score_breakdown = dict(
        graph_debug.get("selected_frontier_score_breakdown", {}) or {}
    )
    topk_frontier_scores = list(graph_debug.get("topk_frontiers", []) or [])
    candidates = [
        {
            "rank": item.get("rank", idx + 1),
            "coord": item.get("coord") or item.get("frontier"),
            "score": item.get("score"),
            "family": "search",
        }
        for idx, item in enumerate(topk_frontier_scores)
    ]

    if goal is None:
        graph_no_goal_reason = graph_debug.get("no_goal_reason", "")
        return GroundingResult(
            success=False,
            changed=False,
            reason="get_goal_none",
            noop_reason="get_goal_none",
            noop_type="get_goal_none",
            bias_input=tuple(bias) if bias is not None else None,
            selected_frontier=selected_frontier,
            selected_frontier_same_as_prev=same_as_prev,
            selected_frontier_same_as_previous=same_as_prev,
            selected_frontier_score=graph_debug.get("selected_frontier_score"),
            selected_frontier_score_breakdown=selected_frontier_score_breakdown,
            local_projection_valid=False,
            topk_frontier_scores=topk_frontier_scores,
            top1_top2_gap=graph_debug.get("top1_top2_gap"),
            base_score_std=graph_debug.get("base_score_std"),
            bias_score_std=graph_debug.get("bias_score_std"),
            candidate_frontier_count_after_bias_filter=graph_debug.get(
                "candidate_frontier_count_after_bias_filter"
            ),
            selected_from_bias_filtered_subset=bool(
                graph_debug.get("selected_from_bias_filtered_subset", False)
            ),
            graph_no_goal_reason=graph_no_goal_reason,
            frontier_filter_fallback_mode=graph_debug.get(
                "frontier_filter_fallback_mode", ""
            ),
            candidate_distance_fallback_mode=graph_debug.get(
                "candidate_distance_fallback_mode", ""
            ),
            raw_frontier_count=graph_debug.get("raw_frontier_count"),
            filtered_frontier_count=graph_debug.get("filtered_frontier_count"),
            used_raw_frontier_fallback=bool(
                graph_debug.get("used_raw_frontier_fallback", False)
            ),
            used_relaxed_distance_fallback=bool(
                graph_debug.get("used_relaxed_distance_fallback", False)
            ),
            goal_before=goal_before,
            goal_after=goal_before,
            graph_debug=graph_debug,
            candidates=candidates,
            primary_goal=None,
            geometric_actionability_confidence=0.0,
            failure_code=graph_no_goal_reason or "get_goal_none",
            fallback_policy=(
                "hold_and_wait"
                if graph_no_goal_reason in {"no_frontiers", "stage_not_groundable_yet"}
                else "request_replan"
            ),
        )

    try:
        goal = list(goal)
        projected_goal = [
            int(goal[0] - bev_map.local_map_boundary[0, 0]),
            int(goal[1] - bev_map.local_map_boundary[0, 2]),
        ]
    except Exception:
        return GroundingResult(
            success=False,
            changed=False,
            reason="projection_invalid",
            noop_reason="projection_invalid",
            noop_type="projection_invalid",
            bias_input=tuple(bias) if bias is not None else None,
            selected_frontier=selected_frontier,
            selected_frontier_same_as_prev=same_as_prev,
            selected_frontier_same_as_previous=same_as_prev,
            selected_frontier_score=graph_debug.get("selected_frontier_score"),
            selected_frontier_score_breakdown=selected_frontier_score_breakdown,
            local_projection_valid=False,
            topk_frontier_scores=topk_frontier_scores,
            top1_top2_gap=graph_debug.get("top1_top2_gap"),
            base_score_std=graph_debug.get("base_score_std"),
            bias_score_std=graph_debug.get("bias_score_std"),
            candidate_frontier_count_after_bias_filter=graph_debug.get(
                "candidate_frontier_count_after_bias_filter"
            ),
            selected_from_bias_filtered_subset=bool(
                graph_debug.get("selected_from_bias_filtered_subset", False)
            ),
            graph_no_goal_reason=graph_debug.get("no_goal_reason", ""),
            frontier_filter_fallback_mode=graph_debug.get(
                "frontier_filter_fallback_mode", ""
            ),
            candidate_distance_fallback_mode=graph_debug.get(
                "candidate_distance_fallback_mode", ""
            ),
            raw_frontier_count=graph_debug.get("raw_frontier_count"),
            filtered_frontier_count=graph_debug.get("filtered_frontier_count"),
            used_raw_frontier_fallback=bool(
                graph_debug.get("used_raw_frontier_fallback", False)
            ),
            used_relaxed_distance_fallback=bool(
                graph_debug.get("used_relaxed_distance_fallback", False)
            ),
            goal_before=goal_before,
            goal_after=goal_before,
            graph_debug=graph_debug,
            candidates=candidates,
            failure_code="projection_invalid",
            fallback_policy="retry_or_replan",
        )

    if not all(math.isfinite(float(coord)) for coord in projected_goal):
        return GroundingResult(
            success=False,
            changed=False,
            reason="projection_invalid",
            noop_reason="projection_invalid",
            noop_type="projection_invalid",
            bias_input=tuple(bias) if bias is not None else None,
            selected_frontier=selected_frontier,
            selected_frontier_same_as_prev=same_as_prev,
            selected_frontier_same_as_previous=same_as_prev,
            selected_frontier_score=graph_debug.get("selected_frontier_score"),
            selected_frontier_score_breakdown=selected_frontier_score_breakdown,
            local_projection_valid=False,
            topk_frontier_scores=topk_frontier_scores,
            top1_top2_gap=graph_debug.get("top1_top2_gap"),
            base_score_std=graph_debug.get("base_score_std"),
            bias_score_std=graph_debug.get("bias_score_std"),
            candidate_frontier_count_after_bias_filter=graph_debug.get(
                "candidate_frontier_count_after_bias_filter"
            ),
            selected_from_bias_filtered_subset=bool(
                graph_debug.get("selected_from_bias_filtered_subset", False)
            ),
            graph_no_goal_reason=graph_debug.get("no_goal_reason", ""),
            frontier_filter_fallback_mode=graph_debug.get(
                "frontier_filter_fallback_mode", ""
            ),
            candidate_distance_fallback_mode=graph_debug.get(
                "candidate_distance_fallback_mode", ""
            ),
            raw_frontier_count=graph_debug.get("raw_frontier_count"),
            filtered_frontier_count=graph_debug.get("filtered_frontier_count"),
            used_raw_frontier_fallback=bool(
                graph_debug.get("used_raw_frontier_fallback", False)
            ),
            used_relaxed_distance_fallback=bool(
                graph_debug.get("used_relaxed_distance_fallback", False)
            ),
            goal_before=goal_before,
            goal_after=goal_before,
            graph_debug=graph_debug,
            candidates=candidates,
            failure_code="projection_invalid",
            fallback_policy="retry_or_replan",
        )

    in_bounds = (
        0 <= projected_goal[0] < args.local_width
        and 0 <= projected_goal[1] < args.local_height
    )
    if not in_bounds:
        return GroundingResult(
            success=False,
            changed=False,
            reason="out_of_local_window",
            noop_reason="out_of_local_window",
            noop_type="out_of_local_window",
            bias_input=tuple(bias) if bias is not None else None,
            selected_frontier=selected_frontier,
            selected_frontier_same_as_prev=same_as_prev,
            selected_frontier_same_as_previous=same_as_prev,
            selected_frontier_score=graph_debug.get("selected_frontier_score"),
            selected_frontier_score_breakdown=selected_frontier_score_breakdown,
            projected_goal=tuple(projected_goal),
            local_projection_valid=False,
            topk_frontier_scores=topk_frontier_scores,
            top1_top2_gap=graph_debug.get("top1_top2_gap"),
            base_score_std=graph_debug.get("base_score_std"),
            bias_score_std=graph_debug.get("bias_score_std"),
            candidate_frontier_count_after_bias_filter=graph_debug.get(
                "candidate_frontier_count_after_bias_filter"
            ),
            selected_from_bias_filtered_subset=bool(
                graph_debug.get("selected_from_bias_filtered_subset", False)
            ),
            graph_no_goal_reason=graph_debug.get("no_goal_reason", ""),
            frontier_filter_fallback_mode=graph_debug.get(
                "frontier_filter_fallback_mode", ""
            ),
            candidate_distance_fallback_mode=graph_debug.get(
                "candidate_distance_fallback_mode", ""
            ),
            raw_frontier_count=graph_debug.get("raw_frontier_count"),
            filtered_frontier_count=graph_debug.get("filtered_frontier_count"),
            used_raw_frontier_fallback=bool(
                graph_debug.get("used_raw_frontier_fallback", False)
            ),
            used_relaxed_distance_fallback=bool(
                graph_debug.get("used_relaxed_distance_fallback", False)
            ),
            goal_before=goal_before,
            goal_after=goal_before,
            graph_debug=graph_debug,
            candidates=candidates,
            primary_goal={"full_map_coord": goal, "local_map_coord": projected_goal},
            failure_code="out_of_local_window",
            fallback_policy="hold_or_move_local_map",
        )

    global_goals[0] = projected_goal[0]
    global_goals[1] = projected_goal[1]
    goal_after = list(global_goals)
    changed = goal_after != goal_before

    noop_reason = None
    noop_type = ""
    reason = "goal_updated"
    if not changed:
        if same_as_prev:
            noop_reason = "same_frontier_as_prev"
            noop_type = "same_frontier_as_prev"
            reason = "same_frontier_as_prev"
        else:
            noop_reason = "same_goal_as_prev"
            noop_type = "same_goal_as_prev"
            reason = "same_goal_as_prev"

    return GroundingResult(
        success=True,
        changed=changed,
        reason=reason,
        noop_reason=noop_reason,
        noop_type=noop_type,
        bias_input=tuple(bias) if bias is not None else None,
        selected_frontier=selected_frontier,
        selected_frontier_same_as_prev=same_as_prev,
        selected_frontier_same_as_previous=same_as_prev,
        selected_frontier_score=graph_debug.get("selected_frontier_score"),
        selected_frontier_score_breakdown=selected_frontier_score_breakdown,
        projected_goal=tuple(projected_goal),
        local_projection_valid=True,
        topk_frontier_scores=topk_frontier_scores,
        top1_top2_gap=graph_debug.get("top1_top2_gap"),
        base_score_std=graph_debug.get("base_score_std"),
        bias_score_std=graph_debug.get("bias_score_std"),
        candidate_frontier_count_after_bias_filter=graph_debug.get(
            "candidate_frontier_count_after_bias_filter"
        ),
        selected_from_bias_filtered_subset=bool(
            graph_debug.get("selected_from_bias_filtered_subset", False)
        ),
        graph_no_goal_reason=graph_debug.get("no_goal_reason", ""),
        frontier_filter_fallback_mode=graph_debug.get(
            "frontier_filter_fallback_mode", ""
        ),
        candidate_distance_fallback_mode=graph_debug.get(
            "candidate_distance_fallback_mode", ""
        ),
        raw_frontier_count=graph_debug.get("raw_frontier_count"),
        filtered_frontier_count=graph_debug.get("filtered_frontier_count"),
        used_raw_frontier_fallback=bool(
            graph_debug.get("used_raw_frontier_fallback", False)
        ),
        used_relaxed_distance_fallback=bool(
            graph_debug.get("used_relaxed_distance_fallback", False)
        ),
        goal_before=goal_before,
        goal_after=goal_after,
        graph_debug=graph_debug,
        candidates=candidates,
        primary_goal={"full_map_coord": goal, "local_map_coord": projected_goal},
        geometric_actionability_confidence=1.0 if changed else 0.35,
        failure_code=None if changed else noop_type or reason,
        fallback_policy=None if changed else "retry_or_replan",
    )
