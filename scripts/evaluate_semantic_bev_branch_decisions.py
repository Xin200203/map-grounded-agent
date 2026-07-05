#!/usr/bin/env python3
"""Evaluate semantic-BEV replay branch decisions without Habitat or LLM calls.

The evaluator is intentionally conservative: it does not decide task success.  It
checks whether the saved BEV evidence was strong enough to support the recorded
branch decision and whether the chosen branch agrees with the target/query
heatmap ranking when such a heatmap exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _branch_rank(ranking: List[Mapping[str, Any]], branch_id: str) -> Optional[int]:
    for idx, item in enumerate(ranking, start=1):
        if str(item.get("id", "")) == str(branch_id or ""):
            return idx
    return None


def _table_row(table: List[Mapping[str, Any]], branch_id: str) -> Dict[str, Any]:
    for item in table:
        if str(item.get("id", "")) == str(branch_id or ""):
            return dict(item)
    return {}


def evaluate_frame(frame: Mapping[str, Any], *, frame_path: Path) -> Dict[str, Any]:
    decision = dict(frame.get("decision_overlay") or {})
    qc = dict(frame.get("quality_checks") or {})
    sem = dict(frame.get("semantic_layers") or {})
    heat = dict(frame.get("target_query_heatmap") or {})
    value = dict(frame.get("target_value_state") or {})
    selection_contract = dict(value.get("selection_contract") or frame.get("selection_contract") or {})
    agent = dict(frame.get("agent") or {})
    ranking = list(heat.get("branch_ranking") or [])
    value_ranking = list(value.get("branch_ranking") or [])
    executable_ranking = list(value.get("executable_decision_ranking") or [])
    branch_table = list(frame.get("branch_decision_table") or [])
    final_branch = str(decision.get("final_selected_branch_id") or "")
    mllm_branch = str(decision.get("mllm_selected_branch_id") or "")
    dry_branch = str(decision.get("dry_selected_branch_id") or "")
    top_branch = str(ranking[0].get("id", "")) if ranking else ""
    top_value_branch = str(value_ranking[0].get("id", "")) if value_ranking else ""
    top_executable_branch = str(executable_ranking[0].get("id", "")) if executable_ranking else str(selection_contract.get("top_executable_decision_branch_id") or "")
    final_rank = _branch_rank(ranking, final_branch)
    mllm_rank = _branch_rank(ranking, mllm_branch)
    final_value_rank = _branch_rank(value_ranking, final_branch)
    mllm_value_rank = _branch_rank(value_ranking, mllm_branch)
    final_executable_rank = _branch_rank(executable_ranking, final_branch)
    final_row = _table_row(branch_table, final_branch)
    mllm_row = _table_row(branch_table, mllm_branch)
    issues: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []

    direct_pixels = int(sem.get("direct_target_pixel_count", 0) or 0)
    active_categories = int(sem.get("active_category_count", 0) or 0)
    heat_source = str(heat.get("source") or "none")
    heat_pixels = int(heat.get("positive_pixel_count", 0) or 0)
    value_peak = value.get("peak")
    if value_peak is None:
        value_peak = (value.get("combined_value") or {}).get("peak")
    final_evidence_level = str(final_row.get("evidence_level") or "")
    mllm_evidence_level = str(mllm_row.get("evidence_level") or "")

    if agent.get("display_repaired_from_map_current_channel"):
        issues.append(
            {
                "severity": "warning",
                "code": "agent_display_repaired",
                "detail": f"raw agent coord differs from map-current by {agent.get('agent_channel_distance')}",
            }
        )
    if active_categories == 0:
        issues.append(
            {
                "severity": "blocker_for_semantic_reasoning",
                "code": "no_active_semantic_channels",
                "detail": "MLLM image cannot use dense BEV semantics in this frame.",
            }
        )
    if direct_pixels == 0:
        issues.append(
            {
                "severity": "target_not_visible",
                "code": "direct_target_channel_empty",
                "detail": "Target visibility must be false; any target choice is exploration/prior only.",
            }
        )
    if heat_source == "none" or heat_pixels == 0:
        issues.append(
            {
                "severity": "weak_support",
                "code": "no_target_heatmap",
                "detail": "No spatial target/query heatmap supports a branch preference.",
            }
        )
    if not branch_table:
        issues.append(
            {
                "severity": "blocker_for_decision_state",
                "code": "missing_branch_decision_table",
                "detail": "Planner replay lacks a structured branch decision table.",
            }
        )
    elif any(not str(item.get("evidence_level") or "") for item in branch_table):
        issues.append(
            {
                "severity": "blocker_for_decision_state",
                "code": "branch_missing_evidence_level",
                "detail": "Every branch row must expose direct/anchor/room_prior/geometry_only.",
            }
        )
    if final_branch and final_row and final_evidence_level == "geometry_only" and decision.get("planner_prior_applied"):
        issues.append(
            {
                "severity": "gate_mismatch",
                "code": "geometry_only_branch_hard_gated",
                "detail": f"final={final_branch} has geometry-only evidence but planner prior was applied.",
            }
        )
    if heat_source == "semantic_prior_only":
        issues.append(
            {
                "severity": "prior_only",
                "code": "target_heatmap_prior_only",
                "detail": "Branch preference can be used for exploration, not target grounding.",
            }
        )
    if ranking and final_branch and top_branch and final_branch != top_branch:
        top_score = float(ranking[0].get("target_heatmap_score", 0.0) or 0.0)
        final_item = next((item for item in ranking if str(item.get("id", "")) == final_branch), {})
        final_score = float(final_item.get("target_heatmap_score", 0.0) or 0.0)
        heat_drift_explained = bool(selection_contract.get("drift_is_explained")) and heat_source in {
            "semantic_prior_only",
            "none",
        }
        if top_score - final_score > 0.05:
            if heat_drift_explained:
                notes.append(
                    {
                        "severity": "info",
                        "code": "final_heatmap_override_explained",
                        "detail": (
                            f"final={final_branch} heatmap rank={final_rank}, top={top_branch}, "
                            f"score_gap={top_score - final_score:.3f}, reason={selection_contract.get('override_reason')}"
                        ),
                    }
                )
            else:
                issues.append(
                    {
                        "severity": "branch_value_disagreement",
                        "code": "final_branch_not_top_heatmap_branch",
                        "detail": f"final={final_branch} rank={final_rank}, top={top_branch}, score_gap={top_score - final_score:.3f}",
                    }
                )
    if ranking and mllm_branch and top_branch and mllm_branch != top_branch:
        top_score = float(ranking[0].get("target_heatmap_score", 0.0) or 0.0)
        mllm_item = next((item for item in ranking if str(item.get("id", "")) == mllm_branch), {})
        mllm_score = float(mllm_item.get("target_heatmap_score", 0.0) or 0.0)
        mllm_heat_drift_explained = (
            bool(selection_contract.get("drift_is_explained"))
            and heat_source in {"semantic_prior_only", "none"}
            and (not final_branch or mllm_branch == final_branch)
        )
        if top_score - mllm_score > 0.05:
            if mllm_heat_drift_explained:
                notes.append(
                    {
                        "severity": "info",
                        "code": "mllm_heatmap_override_explained",
                        "detail": (
                            f"mllm={mllm_branch} heatmap rank={mllm_rank}, top={top_branch}, "
                            f"score_gap={top_score - mllm_score:.3f}, reason={selection_contract.get('override_reason')}"
                        ),
                    }
                )
            else:
                issues.append(
                    {
                        "severity": "mllm_value_disagreement",
                        "code": "mllm_branch_not_top_heatmap_branch",
                        "detail": f"mllm={mllm_branch} rank={mllm_rank}, top={top_branch}, score_gap={top_score - mllm_score:.3f}",
                    }
                )

    if value_ranking and final_branch and top_value_branch and final_branch != top_value_branch:
        top_score = float(value_ranking[0].get("target_value_score", 0.0) or 0.0)
        final_item = next((item for item in value_ranking if str(item.get("id", "")) == final_branch), {})
        final_score = float(final_item.get("target_value_score", 0.0) or 0.0)
        drift_explained = bool(selection_contract.get("drift_is_explained"))
        if top_score - final_score > 0.08 and not drift_explained:
            issues.append(
                {
                    "severity": "branch_value_disagreement",
                    "code": "final_branch_not_top_target_value_branch",
                    "detail": f"final={final_branch} rank={final_value_rank}, top={top_value_branch}, value_gap={top_score - final_score:.3f}",
                }
            )
        elif top_score - final_score > 0.08 and drift_explained:
            notes.append(
                {
                    "severity": "info",
                    "code": "final_target_value_override_explained",
                    "detail": (
                        f"final={final_branch} target-value rank={final_value_rank}, top={top_value_branch}, "
                        f"value_gap={top_score - final_score:.3f}, reason={selection_contract.get('override_reason')}"
                    ),
                }
            )

    if executable_ranking and final_branch and top_executable_branch and final_branch != top_executable_branch:
        issues.append(
            {
                "severity": "decision_contract_mismatch",
                "code": "final_branch_not_top_executable_decision_branch",
                "detail": f"final={final_branch} executable_rank={final_executable_rank}, top={top_executable_branch}",
            }
        )

    if final_evidence_level == "direct":
        support = "direct_target_supported"
    elif final_evidence_level == "anchor":
        support = "anchor_supported_exploration"
    elif final_evidence_level == "room_prior":
        support = "room_prior_supported_exploration"
    elif active_categories > 0 and (heat_source != "none") and final_rank == 1 and direct_pixels > 0:
        support = "direct_target_supported"
    elif active_categories > 0 and (heat_source != "none") and final_rank in (1, 2, 3):
        support = "prior_supported_exploration"
    elif active_categories > 0:
        support = "semantic_context_only"
    else:
        support = "geometry_only_or_graph_only"

    return {
        "schema_version": "smoothnav.semantic_bev_branch_decision_eval.v1",
        "frame_path": str(frame_path),
        "step_idx": frame.get("step_idx"),
        "dry_branch": dry_branch,
        "mllm_branch": mllm_branch,
        "final_branch": final_branch,
        "top_heatmap_branch": top_branch,
        "top_target_value_branch": top_value_branch,
        "top_executable_decision_branch": top_executable_branch,
        "final_heatmap_rank": final_rank,
        "mllm_heatmap_rank": mllm_rank,
        "final_target_value_rank": final_value_rank,
        "mllm_target_value_rank": mllm_value_rank,
        "final_executable_decision_rank": final_executable_rank,
        "final_evidence_level": final_evidence_level,
        "mllm_evidence_level": mllm_evidence_level,
        "branch_evidence_counts": value.get("evidence_counts", {}),
        "active_semantic_category_count": active_categories,
        "semantic_region_count": qc.get("semantic_region_count"),
        "target_anchor_semantic_region_count": qc.get("target_anchor_semantic_region_count"),
        "direct_target_pixel_count": direct_pixels,
        "target_heatmap_source": heat_source,
        "target_heatmap_positive_pixel_count": heat_pixels,
        "target_value_available": value.get("available"),
        "target_value_peak": value_peak,
        "selection_contract": selection_contract,
        "support_class": support,
        "issues": issues,
        "notes": notes,
    }


def write_markdown(path: Path, evaluations: List[Mapping[str, Any]]) -> None:
    lines = [
        "# Semantic BEV Branch Decision Evaluation",
        "",
        "Offline evidence audit only: no Habitat and no LLM calls.",
        "",
        "| step | final | top heat/value/exec | final rank | sem cats | sem regions | heat source | support | issues |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in evaluations:
        issues = ", ".join(str(issue.get("code")) for issue in item.get("issues", []) or [])
        lines.append(
        "| {step} | {final} | {top} | {rank} | {cats} | {regions} | {heat} | {support} | {issues} |".format(
                step=item.get("step_idx"),
                final=item.get("final_branch"),
                top=f"{item.get('top_heatmap_branch')}/{item.get('top_target_value_branch')}/{item.get('top_executable_decision_branch')}",
                rank=f"{item.get('final_heatmap_rank')}/{item.get('final_target_value_rank')}/{item.get('final_executable_decision_rank')}",
                cats=item.get("active_semantic_category_count"),
                regions=item.get("semantic_region_count"),
                heat=f"{item.get('target_heatmap_source')} Vpeak={float(item.get('target_value_peak') or 0):.2f}",
                support=f"{item.get('support_class')} ({item.get('final_evidence_level')})",
                issues=issues,
            )
        )
    lines.extend(["", "## Interpretation guide", ""])
    lines.append("- `direct_target_channel_empty`: target must be treated as not visible.")
    lines.append("- `semantic_prior_only`: target heatmap can guide exploration but is not target grounding.")
    lines.append("- `final_branch_not_top_heatmap_branch`: recorded final branch disagrees with target-heat branch value.")
    lines.append("- `selection_contract`: explains target-value vs Graph.get_goal final-score drift when executable scorer/actionability overrides semantic value.")
    lines.append("- `geometry_only_branch_hard_gated`: MLLM/scorer coupling used a planner prior despite geometry-only evidence.")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out or (args.replay_dir / "branch_decision_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for frame_path in sorted(args.replay_dir.glob("step_*/semantic_bev_frame.json")):
        evaluations.append(evaluate_frame(_load_json(frame_path), frame_path=frame_path))
    result = {
        "schema_version": "smoothnav.semantic_bev_branch_decision_eval.summary.v1",
        "replay_dir": str(args.replay_dir),
        "out_dir": str(out_dir),
        "evaluations": evaluations,
    }
    _write_json(out_dir / "summary.json", result)
    write_markdown(out_dir / "review.md", evaluations)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
