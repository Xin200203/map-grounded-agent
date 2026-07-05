"""Strict validation for BEV/MLLM frontier-branch planner outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from smoothnav.frontier_branching import build_branch_prior_scores

SCHEMA_VERSION = "smoothnav.mllm_frontier_plan.v1"
ALLOWED_DECISION_TYPES = {
    "direct_target",
    "target_anchor",
    "explore_frontier",
    "hold_or_recover",
}
ALLOWED_EVIDENCE_LEVELS = {
    "direct",
    "anchor",
    "room_prior",
    "geometry_only",
}
_BRANCH_REQUIRED_DECISIONS = {"direct_target", "target_anchor", "explore_frontier"}
_FORBIDDEN_PRIMARY_COORD_KEYS = {
    "selected_coord",
    "selected_coordinate",
    "selected_point",
    "target_coord",
    "target_coordinate",
    "x",
    "y",
}


def _extract_json_object(response: Any) -> Dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)
    text = str(response or "").strip()
    if not text:
        raise ValueError("empty_response")
    try:
        loaded = json.loads(text)
        if isinstance(loaded, Mapping):
            return dict(loaded)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no_json_object")
    loaded = json.loads(match.group(0))
    if not isinstance(loaded, Mapping):
        raise ValueError("json_not_object")
    return dict(loaded)


def _coerce_id_set(branch_ids: Iterable[str]) -> set[str]:
    return {str(item) for item in branch_ids if str(item)}


def _ranked_branches(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    ranked = payload.get("ranked_branches")
    if ranked is None:
        # Backward-compatible alias for the initial frontier-ID wording.  The
        # validator still treats these as branch IDs after Section 88 refinement.
        ranked = payload.get("ranked_frontiers")
    if ranked is None:
        return []
    if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
        raise ValueError("ranked_branches_not_list")
    result = []
    for item in ranked:
        if not isinstance(item, Mapping):
            raise ValueError("ranked_branch_not_object")
        bid = str(item.get("id") or item.get("branch_id") or "")
        if not bid:
            raise ValueError("ranked_branch_missing_id")
        score = item.get("score", 0.0)
        try:
            score = float(score)
        except Exception:
            raise ValueError("ranked_branch_invalid_score")
        if score < 0.0 or score > 1.0:
            raise ValueError("ranked_branch_score_out_of_range")
        result.append({**dict(item), "id": bid, "score": score})
    return result


def validate_mllm_frontier_plan(
    response: Any,
    *,
    branch_ids: Iterable[str],
    frontier_branches: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Parse and validate an MLLM branch-selection response.

    Returns a verdict dictionary rather than raising so online code can log a
    structured fallback reason and continue safely.
    """

    id_set = _coerce_id_set(branch_ids)
    try:
        payload = _extract_json_object(response)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("schema_version_mismatch")
        decision_type = str(payload.get("decision_type") or "")
        if decision_type not in ALLOWED_DECISION_TYPES:
            raise ValueError("invalid_decision_type")
        evidence_level = str(payload.get("evidence_level") or "")
        if evidence_level not in ALLOWED_EVIDENCE_LEVELS:
            raise ValueError("missing_or_invalid_evidence_level")
        target_visible = bool(payload.get("target_visible", False))
        if target_visible and evidence_level != "direct":
            raise ValueError("target_visible_requires_direct_evidence")
        if decision_type == "direct_target" and evidence_level != "direct":
            raise ValueError("direct_target_decision_requires_direct_evidence")
        if decision_type == "target_anchor" and evidence_level not in {"direct", "anchor", "room_prior"}:
            raise ValueError("target_anchor_decision_requires_semantic_or_room_evidence")
        forbidden = sorted(_FORBIDDEN_PRIMARY_COORD_KEYS.intersection(payload.keys()))
        if forbidden:
            raise ValueError(f"raw_coordinate_primary_output:{','.join(forbidden)}")
        selected_branch_id = str(
            payload.get("selected_branch_id")
            or payload.get("selected_frontier_id")
            or ""
        )
        ranked = _ranked_branches(payload)
        for item in ranked:
            if item["id"] not in id_set:
                raise ValueError(f"unknown_ranked_branch_id:{item['id']}")
        if decision_type in _BRANCH_REQUIRED_DECISIONS:
            if not selected_branch_id:
                raise ValueError("missing_selected_branch_id")
            if selected_branch_id not in id_set:
                raise ValueError(f"unknown_selected_branch_id:{selected_branch_id}")
            if not ranked:
                raise ValueError("missing_ranked_branches")
        prior_scores = []
        if frontier_branches is not None:
            prior_scores = build_branch_prior_scores(
                frontier_branches,
                selected_branch_id=selected_branch_id,
                ranked_branches=ranked,
            ).tolist()
        return {
            "schema_version": "smoothnav.mllm_frontier_plan.verdict.v1",
            "valid": True,
            "invalid_reason": "",
            "decision_type": decision_type,
            "evidence_level": evidence_level,
            "target_visible": target_visible,
            "selected_branch_id": selected_branch_id,
            "ranked_branch_ids": [item["id"] for item in ranked],
            "prior_scores": prior_scores,
            "plan": payload,
        }
    except Exception as exc:
        return {
            "schema_version": "smoothnav.mllm_frontier_plan.verdict.v1",
            "valid": False,
            "invalid_reason": str(exc),
            "decision_type": "",
            "evidence_level": "",
            "target_visible": False,
            "selected_branch_id": "",
            "ranked_branch_ids": [],
            "prior_scores": [],
            "plan": None,
        }
