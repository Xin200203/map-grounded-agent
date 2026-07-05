"""Strategy grounding helpers decoupled from the heavy runtime entrypoint."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from smoothnav.target_matching import caption_matches_label, score_caption_against_goal


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


def _resolve_direct_object_goal(strategy, graph):
    target_region = getattr(strategy, "target_region", "") or ""
    if not str(target_region).startswith("object:"):
        return None, []

    target_object = str(target_region).split("object:", 1)[1].strip().lower()
    goal_text = getattr(graph, "text_goal", None) or getattr(graph, "obj_goal", "")
    has_goal_text = bool(str(goal_text or "").strip())
    bias = getattr(strategy, "bias_position", None)
    candidates = []
    for node in getattr(graph, "nodes", []) or []:
        caption = str(getattr(node, "caption", "") or "").lower()
        center = getattr(node, "center", None)
        if not caption or center is None:
            continue
        if not caption_matches_label(target_object, caption):
            continue
        target_match = score_caption_against_goal(caption, goal_text)
        detections = 0
        try:
            detections = int(getattr(node, "object", {}).get("num_detections", 0) or 0)
        except Exception:
            detections = 0
        if bias is not None:
            try:
                distance_to_bias = math.dist(
                    [float(center[0]), float(center[1])],
                    [float(bias[0]), float(bias[1])],
                )
            except Exception:
                distance_to_bias = float("inf")
        else:
            distance_to_bias = 0.0
        candidates.append(
            {
                "caption": caption,
                "center": [int(center[0]), int(center[1])],
                "num_detections": detections,
                "distance_to_bias": distance_to_bias,
                "target_relevance": target_match["score"],
                "target_match_reason": target_match["reason"],
                "target_primary_categories": target_match["primary_categories"],
                "target_goal_text_present": has_goal_text,
            }
        )
    if not candidates:
        return None, []

    candidates.sort(
        key=lambda item: (
            item["distance_to_bias"],
            -item["num_detections"],
            item["center"][0],
            item["center"][1],
        )
    )
    return candidates[0]["center"], candidates


def _project_full_goal_to_local(goal, bev_map):
    try:
        return [
            int(goal[0] - bev_map.local_map_boundary[0, 0]),
            int(goal[1] - bev_map.local_map_boundary[0, 2]),
        ]
    except Exception:
        return None


def _graph_get_goal(graph, *, goal=None, record_selection_history=True):
    """Call Graph.get_goal with history control when the runtime supports it."""

    try:
        return graph.get_goal(
            goal=goal,
            record_selection_history=record_selection_history,
        )
    except TypeError:
        # Unit mocks and older Graph implementations may not accept the keyword.
        return graph.get_goal(goal=goal)


def _mllm_frontier_mode(args) -> str:
    return str(getattr(args, "mllm_frontier_planner_mode", "off") or "off").strip().lower()


def _is_mllm_frontier_enabled(args, planner) -> bool:
    return planner is not None and _mllm_frontier_mode(args) == "online"


def _ground_with_optional_mllm_frontier_prior(
    *,
    strategy,
    graph,
    bev_map,
    args,
    bias,
    mllm_frontier_planner=None,
    goal_description: str = "",
    trigger: str = "",
    episode_id: Optional[int] = None,
    step_idx: Optional[int] = None,
    trace_writer=None,
):
    """Ground through Graph.get_goal, optionally injecting an MLLM branch prior.

    The first pass is a dry run used only to expose deterministic frontier
    branches and an annotated BEV to the MLLM.  If the MLLM returns a valid
    branch-id plan, the final pass recomputes the frontier goal with a temporary
    per-frontier prior.  Selection history is updated only on the final pass.
    """

    if not _is_mllm_frontier_enabled(args, mllm_frontier_planner):
        return _graph_get_goal(graph, goal=bias), {}

    previous_prior = getattr(graph, "planner_frontier_prior", None)
    graph.planner_frontier_prior = None
    dry_goal = _graph_get_goal(
        graph,
        goal=bias,
        record_selection_history=False,
    )
    dry_snapshot = getattr(graph, "last_goal_replay_snapshot", None)
    dry_debug = dict(getattr(graph, "last_goal_debug", {}) or {})
    mllm_result = {
        "schema_version": "smoothnav.bev_frontier_mllm_planner.v1",
        "status": "skipped",
        "reason": "no_replay_snapshot",
        "dry_goal": dry_goal,
    }
    try:
        if dry_snapshot:
            mllm_result = mllm_frontier_planner.plan(
                goal_description=goal_description,
                strategy=strategy,
                graph=graph,
                snapshot=dry_snapshot,
                full_map=getattr(bev_map, "full_map", None),
                trigger=trigger,
                step_idx=step_idx,
            )
    except Exception as exc:  # pragma: no cover - online safety path
        mllm_result = {
            "schema_version": "smoothnav.bev_frontier_mllm_planner.v1",
            "status": "invalid",
            "reason": str(exc),
            "dry_goal": dry_goal,
        }

    verdict = dict(mllm_result.get("verdict") or {})
    raw_prior_scores = verdict.get("prior_scores") or []
    selected_branch_id = str(verdict.get("selected_branch_id") or "")
    evidence_level = str(verdict.get("evidence_level") or "geometry_only")
    evidence_gate_strength = {
        "direct": 1.0,
        "anchor": 0.85,
        "room_prior": 0.55,
        "geometry_only": 0.0,
    }.get(evidence_level, 0.0)
    prior_scores = [float(score) * float(evidence_gate_strength) for score in raw_prior_scores]
    applied = bool(
        verdict.get("valid")
        and selected_branch_id
        and prior_scores
        and evidence_gate_strength > 0.0
        and any(float(score) > 0.0 for score in prior_scores)
    )
    if applied:
        graph.planner_frontier_prior = {
            "source": "mllm_frontier_planner",
            "selected_branch_id": selected_branch_id,
            "planner_prior_scores": prior_scores,
            "evidence_level": evidence_level,
            "evidence_gate_strength": evidence_gate_strength,
        }
    else:
        graph.planner_frontier_prior = None

    goal = _graph_get_goal(
        graph,
        goal=bias,
        record_selection_history=True,
    )
    final_debug = dict(getattr(graph, "last_goal_debug", {}) or {})
    final_debug["mllm_frontier"] = {
        "enabled": True,
        "applied": applied,
        "status": mllm_result.get("status", ""),
        "reason": mllm_result.get("reason", ""),
        "selected_branch_id": selected_branch_id,
        "evidence_level": evidence_level,
        "evidence_gate_strength": evidence_gate_strength,
        "ranked_branch_ids": verdict.get("ranked_branch_ids", []),
        "allowed_branch_ids": mllm_result.get("allowed_branch_ids", []),
        "prompt_hash": mllm_result.get("prompt_hash", ""),
        "image_source": mllm_result.get("image_source", ""),
        "semantic_bev_quality": mllm_result.get("semantic_bev_quality", {}),
        "dry_selected_frontier": dry_debug.get("selected_frontier"),
        "dry_selected_branch_id": dry_debug.get("selected_frontier_branch_id", ""),
    }
    try:
        graph.last_goal_debug = final_debug
    except Exception:
        pass
    if trace_writer is not None and hasattr(trace_writer, "record_mllm_frontier_call"):
        trace_writer.record_mllm_frontier_call(
            episode_id if episode_id is not None else 0,
            {
                **mllm_result,
                "step_idx": step_idx,
                "trigger": trigger,
                "applied": applied,
                "strategy": {
                    "target_region": str(getattr(strategy, "target_region", "") or ""),
                    "anchor_object": str(getattr(strategy, "anchor_object", "") or ""),
                    "bias_position": bias,
                },
                "dry_debug": {
                    "selected_frontier": dry_debug.get("selected_frontier"),
                    "selected_frontier_branch_id": dry_debug.get(
                        "selected_frontier_branch_id", ""
                    ),
                    "topk_frontiers": dry_debug.get("topk_frontiers", []),
                },
                "final_debug": {
                    "selected_frontier": final_debug.get("selected_frontier"),
                    "selected_frontier_branch_id": final_debug.get(
                        "selected_frontier_branch_id", ""
                    ),
                    "planner_prior_weight": final_debug.get("planner_prior_weight"),
                    "planner_prior_union_count": final_debug.get(
                        "planner_prior_union_count"
                    ),
                    "planner_prior_union_mode": final_debug.get(
                        "planner_prior_union_mode", ""
                    ),
                    "topk_frontiers": final_debug.get("topk_frontiers", []),
                },
            },
        )
    graph.planner_frontier_prior = previous_prior
    return goal, mllm_result


def _is_local_projectable(goal, bev_map, args) -> bool:
    projected = _project_full_goal_to_local(goal, bev_map)
    if projected is None:
        return False
    try:
        return (
            0 <= projected[0] < int(args.local_width)
            and 0 <= projected[1] < int(args.local_height)
        )
    except Exception:
        return False


def _candidate_relevance_passes(args, direct_candidates) -> bool:
    if not direct_candidates:
        return True
    # When no text goal is available in tests/legacy modes, score defaults may be
    # zero. Do not block those paths unless the caller explicitly configured a
    # text-goal threshold and the candidate has target metadata.
    best = direct_candidates[0]
    if not best.get("target_goal_text_present", False):
        return True
    score = best.get("target_relevance")
    reason = best.get("target_match_reason", "")
    if score is None or reason == "no_match":
        return not bool(getattr(args, "goal_type", "") == "text")
    threshold = float(
        getattr(args, "graph_text_goal_direct_relevance_threshold", 0.75)
    )
    return float(score) >= threshold


def _should_use_direct_object_goal(strategy, args, *, direct_goal=None, bev_map=None,
                                   direct_candidates=None):
    """Gate direct object execution.

    Text-goal object evidence normally anchors frontier search. However, when a
    planner has selected an observed target object and its center already
    projects into the active local planning window, exploiting that local target
    preserves the clean positive cases while still using frontier anchoring for
    non-local or uncertain object evidence.
    """

    if not bool(getattr(args, "graph_enable_direct_object_goal", False)):
        goal_type = str(getattr(args, "goal_type", "") or "")
        if goal_type != "text":
            return False
        if not bool(
            getattr(args, "graph_text_goal_allow_local_direct_object_goal", True)
        ):
            return False
        if direct_goal is None or bev_map is None:
            return False
        return (
            _is_local_projectable(direct_goal, bev_map, args)
            and _candidate_relevance_passes(args, direct_candidates)
        )
    goal_type = str(getattr(args, "goal_type", "") or "")
    if goal_type == "text" and bool(
        getattr(args, "graph_text_goal_use_frontier_anchor", True)
    ):
        return bool(
            getattr(args, "graph_text_goal_allow_local_direct_object_goal", True)
        ) and direct_goal is not None and bev_map is not None and _is_local_projectable(
            direct_goal, bev_map, args
        ) and _candidate_relevance_passes(
            args, direct_candidates
        )
    return str(getattr(strategy, "target_region", "") or "").startswith("object:")


def apply_strategy(
    strategy,
    graph,
    bev_map,
    args,
    global_goals,
    *,
    mllm_frontier_planner=None,
    goal_description: str = "",
    trigger: str = "",
    episode_id: Optional[int] = None,
    step_idx: Optional[int] = None,
    trace_writer=None,
):
    """Ground a semantic strategy into a local-map goal update."""
    graph.set_full_map(bev_map.full_map)
    graph.set_full_pose(bev_map.full_pose)
    graph.local_map_boundary = getattr(bev_map, "local_map_boundary", None)
    # Make the current semantic mode visible to the lower-level frontier scorer.
    # A non-local target anchor needs a persistent target-progress value term,
    # whereas generic room/direction search should keep the legacy frontier value.
    try:
        graph.active_target_region = str(getattr(strategy, "target_region", "") or "")
        graph.active_anchor_object = str(getattr(strategy, "anchor_object", "") or "")
    except Exception:
        pass

    goal_before = list(global_goals)
    bias = strategy.bias_position if strategy else None
    direct_goal, direct_candidates = _resolve_direct_object_goal(strategy, graph)
    use_direct_object_goal = (
        direct_goal is not None
        and _should_use_direct_object_goal(
            strategy,
            args,
            direct_goal=direct_goal,
            bev_map=bev_map,
            direct_candidates=direct_candidates,
        )
    )
    object_anchor_relevance_ok = _candidate_relevance_passes(args, direct_candidates)
    if direct_goal is not None and not use_direct_object_goal and object_anchor_relevance_ok:
        # Use the object as semantic evidence/bias, but keep the executable target
        # on a map-valid frontier/value candidate.
        bias = tuple(direct_goal)
    elif direct_goal is not None and not object_anchor_relevance_ok:
        # A planner may still choose contextual objects from the menu. If the
        # caption is not a plausible text-goal target, do not let its center
        # bias frontier selection.
        bias = None
    mllm_frontier_result = {}
    if use_direct_object_goal:
        goal = direct_goal
    else:
        goal, mllm_frontier_result = _ground_with_optional_mllm_frontier_prior(
            strategy=strategy,
            graph=graph,
            bev_map=bev_map,
            args=args,
            bias=bias,
            mllm_frontier_planner=mllm_frontier_planner,
            goal_description=goal_description,
            trigger=trigger,
            episode_id=episode_id,
            step_idx=step_idx,
            trace_writer=trace_writer,
        )
    graph_debug = dict(getattr(graph, "last_goal_debug", {}) or {})
    if mllm_frontier_result:
        graph_debug["mllm_frontier_result"] = {
            "status": mllm_frontier_result.get("status", ""),
            "reason": mllm_frontier_result.get("reason", ""),
            "prompt_hash": mllm_frontier_result.get("prompt_hash", ""),
            "verdict": mllm_frontier_result.get("verdict", {}),
        }
    selected_frontier = graph_debug.get("selected_frontier")
    if selected_frontier is not None:
        selected_frontier = tuple(selected_frontier)
    if use_direct_object_goal:
        graph_debug = {
            "direct_object_goal": list(direct_goal),
            "direct_object_candidates": list(direct_candidates),
            "selected_frontier": None,
            "selected_frontier_same_as_prev": False,
            "selected_frontier_same_as_previous": False,
            "selected_frontier_score": None,
            "selected_frontier_score_breakdown": {},
            "topk_frontiers": [],
            "top1_top2_gap": None,
            "base_score_std": None,
            "bias_score_std": None,
            "candidate_frontier_count_after_bias_filter": len(direct_candidates),
            "selected_from_bias_filtered_subset": False,
            "raw_frontier_count": None,
            "filtered_frontier_count": None,
            "used_raw_frontier_fallback": False,
            "used_relaxed_distance_fallback": False,
            "no_goal_reason": "",
            "frontier_filter_fallback_mode": "",
            "candidate_distance_fallback_mode": "",
        }
        selected_frontier = None
    elif direct_goal is not None and object_anchor_relevance_ok:
        graph_debug = {
            **graph_debug,
            "object_anchor_goal": list(direct_goal),
            "object_anchor_candidates": list(direct_candidates),
            "direct_object_goal_used": False,
        }
    elif direct_goal is not None:
        graph_debug = {
            **graph_debug,
            "object_anchor_goal": None,
            "object_anchor_candidates": list(direct_candidates),
            "direct_object_goal_used": False,
            "direct_object_goal_blocked_reason": "low_target_relevance",
        }

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
    if use_direct_object_goal:
        candidates = [
            {
                "rank": idx + 1,
                "coord": item["center"],
                "score": item["num_detections"],
                "distance_to_bias": item["distance_to_bias"],
                "caption": item["caption"],
                "target_relevance": item.get("target_relevance"),
                "target_match_reason": item.get("target_match_reason"),
                "family": "search",
            }
            for idx, item in enumerate(direct_candidates[:5])
        ]
    else:
        candidates = [
            {
                "rank": item.get("rank", idx + 1),
                "coord": item.get("coord") or item.get("frontier"),
                "score": item.get("score"),
                "final_score": item.get("final_score"),
                "bias_score": item.get("bias_score"),
                "novelty_score": item.get("novelty_score"),
                "actionability_score": item.get("actionability_score"),
                "repeat_penalty": item.get("repeat_penalty"),
                "recent_penalty": item.get("recent_penalty"),
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
        projected_goal = _project_full_goal_to_local(goal, bev_map)
        if projected_goal is None:
            raise ValueError("projection_invalid")
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
