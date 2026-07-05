"""BEV/MLLM frontier-branch planner for SmoothNav.

This module keeps the multimodal planner at the semantic-to-frontier boundary:
SmoothNav still owns frontier extraction, branch IDs, scoring, and executable
coordinates; the MLLM is only allowed to choose/rank branch IDs from the
annotated BEV view.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional

from smoothnav.frontier_branching import branch_ids, render_branch_bev_image
from smoothnav.mllm_frontier_contract import SCHEMA_VERSION, validate_mllm_frontier_plan
from smoothnav.planner import serialize_for_planner
from smoothnav.semantic_bev import (
    build_semantic_bev_frame,
    render_semantic_annotated_bev,
    semantic_bev_summary_for_prompt,
)
from smoothnav.tracing import hash_text


PLANNER_SCHEMA_VERSION = "smoothnav.bev_frontier_mllm_planner.v1"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def _compact_branch(branch: Mapping[str, Any]) -> Dict[str, Any]:
    terms = dict(branch.get("score_terms_aggregate") or {})
    return {
        "id": str(branch.get("id", "")),
        "canonical_direction": str(branch.get("canonical_direction_from_agent") or branch.get("map_direction_from_agent") or branch.get("direction_from_agent", "")),
        "canonical_coord_rc": _jsonable(branch.get("canonical_representative_coord_rc") or branch.get("representative_coord")),
        "centroid": _jsonable(branch.get("centroid")),
        "representative_coord": _jsonable(branch.get("representative_coord")),
        "candidate_point_count": int(branch.get("candidate_point_count", 0) or 0),
        "pixel_count": int(branch.get("pixel_count", 0) or 0),
        "score_terms": {
            "distance_mean": terms.get("distance_mean"),
            "base_score_max": terms.get("base_score_max"),
            "bias_score_max": terms.get("bias_score_max"),
            "novelty_score_max": terms.get("novelty_score_max"),
            "actionability_score_max": terms.get("actionability_score_max"),
            "target_progress_score_max": terms.get("target_progress_score_max"),
            "repeat_penalty_max": terms.get("repeat_penalty_max"),
            "recent_penalty_max": terms.get("recent_penalty_max"),
            "final_score_max": terms.get("final_score_max"),
        },
    }


def compact_branch_list(frontier_branches: Mapping[str, Any], *, max_branches: int = 8) -> list[Dict[str, Any]]:
    branches = [_compact_branch(item) for item in frontier_branches.get("branches", []) or []]
    branches.sort(
        key=lambda item: (
            -float((item.get("score_terms") or {}).get("final_score_max") or 0.0),
            -int(item.get("candidate_point_count", 0) or 0),
            str(item.get("id", "")),
        )
    )
    return branches[: max(1, int(max_branches or 8))]


def _allowed_branch_ids(frontier_branches: Mapping[str, Any], summarized_ids: Iterable[str]) -> list[str]:
    """Return branch IDs exposed to the MLLM.

    This intentionally uses the branch IDs in the summarized BEV view rather
    than only ``candidate_branch_ids``.  ``candidate_branch_ids`` has already
    passed through the text-planner bias filter inside ``Graph.get_goal``; using
    it here would prevent the BEV planner from correcting a bad text direction.
    The final geometric scorer still filters for local actionability before
    executing a concrete frontier point.
    """

    ids = [str(item) for item in summarized_ids if str(item)]
    return ids or branch_ids(frontier_branches)


def build_bev_frontier_prompt(
    *,
    goal_description: str,
    strategy: Any,
    graph: Any,
    frontier_branches: Mapping[str, Any],
    trigger: str = "",
    step_idx: Optional[int] = None,
    max_branches: int = 8,
    semantic_bev_summary: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build the text half of the BEV branch-selection prompt."""

    try:
        scene_text = serialize_for_planner(graph, getattr(strategy, "explored_regions", []) or [])
    except Exception:
        scene_text = "Scene graph unavailable."
    decision_table = list((semantic_bev_summary or {}).get("branch_decision_table") or [])
    if decision_table:
        branch_summary = [
            {
                "id": str(item.get("id") or ""),
                "canonical_direction": item.get("canonical_direction"),
                "canonical_coord_rc": item.get("canonical_coord_rc"),
                "candidate_count": item.get("candidate_count"),
                "target_value_score": item.get("target_value_score"),
                "target_value_rank": item.get("target_value_rank"),
                "executable_decision_score": item.get("executable_decision_score"),
                "executable_decision_rank": item.get("executable_decision_rank"),
                "executable_decision_source": item.get("executable_decision_source"),
                "frontier_final_score": item.get("frontier_final_score"),
                "frontier_actionability_score": item.get("frontier_actionability_score"),
                "evidence_level": item.get("evidence_level"),
                "target_visible": item.get("target_visible"),
                "semantic_support": item.get("semantic_support"),
                "info_gain_score": item.get("info_gain_score"),
                "revisit_penalty": item.get("revisit_penalty"),
                "dead_end_risk": item.get("dead_end_risk"),
            }
            for item in decision_table[: max(1, int(max_branches or 8))]
            if str(item.get("id") or "")
        ]
    else:
        branch_summary = compact_branch_list(frontier_branches, max_branches=max_branches)
    summarized_ids = [str(item.get("id", "")) for item in branch_summary if str(item.get("id", ""))]
    candidate_ids = _allowed_branch_ids(frontier_branches, summarized_ids)
    current_region = str(getattr(strategy, "target_region", "") or "")
    anchor_object = str(getattr(strategy, "anchor_object", "") or "")
    explored = list(getattr(strategy, "explored_regions", []) or [])
    semantic_summary_text = (
        json.dumps(_jsonable(semantic_bev_summary), ensure_ascii=False, indent=2)
        if semantic_bev_summary
        else "{}"
    )

    return (
        "You are SmoothNav's BEV frontier-branch planner.\n"
        "You will see a multi-panel bird's-eye-view decision state, not a raw debug map:\n"
        "- Panel A Geometry: occupancy/free/unknown, robot A, trajectory, and B# frontier candidates.\n"
        "- Panel B Semantic: dense semantic-map regions S#, localized objects O#, and only spatialized room regions R#.\n"
        "- Panel C Target-conditioned value: where it is worth searching for the target; color intensity is value.\n"
        "- Panel D Branch table: branch ID, canonical direction/coordinate, target value, executable decision score/rank, evidence level, support, info gain, revisit/dead-end risk.\n"
        "Unlocalized room priors may appear only as text in the summary/table; do not treat them as spatial evidence.\n"
        "Use exactly the canonical full-map row/col frame exposed in the summary. Ignore raw/repaired/debug coordinate fields.\n\n"
        f"TARGET DESCRIPTION:\n{goal_description or ''}\n\n"
        f"CURRENT TEXT STRATEGY:\n- target_region: {current_region}\n"
        f"- anchor_object: {anchor_object}\n"
        f"- trigger: {trigger}\n"
        f"- step_idx: {step_idx}\n"
        f"- already_explored: {json.dumps(explored, ensure_ascii=False)}\n\n"
        f"SCENE GRAPH TEXT:\n{scene_text}\n\n"
        "SEMANTIC BEV SUMMARY:\n"
        f"{semantic_summary_text}\n\n"
        "BEV BRANCH IDS YOU MAY CHOOSE:\n"
        f"{json.dumps(candidate_ids, ensure_ascii=False)}\n\n"
        "BRANCH DECISION TABLE / SUMMARY:\n"
        f"{json.dumps(branch_summary, ensure_ascii=False, indent=2)}\n\n"
        "Task:\n"
        "1. Choose the branch with the best executable target-conditioned utility: use executable_decision_rank/score when present, and use target_value_score as the semantic/search component.\n"
        "2. Evidence levels mean: direct = target visible/localized; anchor = target-related object/semantic anchor; room_prior = localized room/function-area prior; geometry_only = exploration topology only.\n"
        "3. If `direct_target_pixel_count` is 0, target value source is prior-only/none, or the selected row evidence_level is not `direct`, set target_visible=false.\n"
        "4. If evidence is `geometry_only`, do not claim semantic grounding; describe the exploration intent only.\n"
        "5. If you choose a branch that is not the top target_value branch but is top executable_decision, explain that the executable scorer/actionability overrode semantic value; otherwise explain the risk/ambiguity briefly.\n"
        "6. Choose a branch ID from the allowed list, not a raw coordinate. SmoothNav will convert the branch into executable frontier points.\n"
        "7. Prefer high value + enough evidence + low revisit/dead-end risk. If quality warnings mention coordinate mismatch, be conservative and explain uncertainty.\n\n"
        "Image legend reminder: A=robot; B#=executable frontier branch; O#=observed object; "
        "S#=dense semantic-map region; R#=localized room/function-area region. You must choose a branch ID from the allowed list.\n\n"
        "Output JSON only with exactly this schema shape:\n"
        "{\n"
        f"  \"schema_version\": \"{SCHEMA_VERSION}\",\n"
        "  \"decision_type\": \"direct_target\" or \"explore_frontier\" or \"target_anchor\" or \"hold_or_recover\",\n"
        "  \"evidence_level\": \"direct\" or \"anchor\" or \"room_prior\" or \"geometry_only\",\n"
        "  \"target_visible\": true or false,\n"
        "  \"selected_branch_id\": \"B#\",\n"
        "  \"ranked_branches\": [\n"
        "    {\"id\": \"B#\", \"score\": 0.0 to 1.0, \"evidence_level\": \"direct|anchor|room_prior|geometry_only\", \"rationale\": \"short reason\"}\n"
        "  ],\n"
        "  \"supporting_evidence_ids\": [\"O#/S#/R# if any\"],\n"
        "  \"risk_flags\": [\"short risk tags if any\"],\n"
        "  \"semantic_search_intent\": \"short sentence\"\n"
        "}\n"
    )


class BEVFrontierMLLMPlanner:
    """Online BEV branch-ranker backed by a VLM-compatible callable."""

    def __init__(self, vlm_fn, *, max_branches: int = 8, image_scale: int = 1):
        self.vlm = vlm_fn
        self.max_branches = int(max_branches or 8)
        self.image_scale = max(1, int(image_scale or 1))
        self.call_count = 0
        self.last_result: Dict[str, Any] = {}

    def plan(
        self,
        *,
        goal_description: str,
        strategy: Any,
        graph: Any,
        snapshot: Mapping[str, Any],
        full_map: Any,
        trigger: str = "",
        step_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        frontier_branches = dict(snapshot.get("frontier_branches") or {})
        summarized_ids = [
            str(item.get("id", ""))
            for item in compact_branch_list(
                frontier_branches,
                max_branches=self.max_branches,
            )
            if str(item.get("id", ""))
        ]
        allowed_ids = _allowed_branch_ids(frontier_branches, summarized_ids)
        result_base = {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "step_idx": step_idx,
            "trigger": trigger,
            "allowed_branch_ids": allowed_ids,
            "branch_count": int(frontier_branches.get("branch_count", 0) or 0),
        }
        if not frontier_branches or not allowed_ids:
            result = {
                **result_base,
                "status": "skipped",
                "reason": "no_frontier_branches",
                "prompt": "",
                "prompt_hash": "",
                "raw_response": "",
                "verdict": {"valid": False, "invalid_reason": "no_frontier_branches"},
            }
            self.last_result = result
            return result

        semantic_frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=frontier_branches,
            graph=graph,
            scoring_input=snapshot,
            strategy=strategy,
            step_idx=step_idx,
            goal_description=goal_description,
            source="bev_frontier_mllm_planner",
        )
        semantic_summary = semantic_bev_summary_for_prompt(semantic_frame)
        prompt = build_bev_frontier_prompt(
            goal_description=goal_description,
            strategy=strategy,
            graph=graph,
            frontier_branches=frontier_branches,
            trigger=trigger,
            step_idx=step_idx,
            max_branches=self.max_branches,
            semantic_bev_summary=semantic_summary,
        )
        image = render_semantic_annotated_bev(
            semantic_frame,
            full_map,
            mode="decision",
            scale=self.image_scale,
        )
        image_source = "decision_state_bev"
        if image is None:
            image = render_branch_bev_image(
                full_map,
                frontier_branches,
                agent_coord=snapshot.get("agent_coord"),
                scale=self.image_scale,
            )
            image_source = "legacy_branch_bev"
        if image is None:
            result = {
                **result_base,
                "status": "skipped",
                "reason": "bev_render_unavailable",
                "prompt": prompt,
                "prompt_hash": hash_text(prompt),
                "raw_response": "",
                "semantic_bev_summary": semantic_summary,
                "semantic_bev_quality": semantic_frame.get("quality_checks", {}),
                "image_source": "none",
                "verdict": {"valid": False, "invalid_reason": "bev_render_unavailable"},
            }
            self.last_result = result
            return result
        raw_response = ""
        error = ""
        try:
            self.call_count += 1
            raw_response = self.vlm(prompt, image)
        except Exception as exc:  # pragma: no cover - online safety path
            error = str(exc)
        verdict = validate_mllm_frontier_plan(
            raw_response,
            branch_ids=allowed_ids,
            frontier_branches=frontier_branches,
        )
        status = "valid" if verdict.get("valid") else "invalid"
        reason = "" if verdict.get("valid") else (error or verdict.get("invalid_reason", "invalid_plan"))
        result = {
            **result_base,
            "status": status,
            "reason": reason,
            "prompt": prompt,
            "prompt_hash": hash_text(prompt),
            "raw_response": raw_response,
            "semantic_bev_summary": semantic_summary,
            "semantic_bev_quality": semantic_frame.get("quality_checks", {}),
            "image_source": image_source,
            "verdict": verdict,
        }
        self.last_result = result
        return result
