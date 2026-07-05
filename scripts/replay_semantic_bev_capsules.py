#!/usr/bin/env python3
"""Regenerate semantic BEV frames/images from saved task-frame capsules.

This is the offline test harness for the semantic annotated BEV contract: it does
not call Habitat or any LLM API.  It reads existing capsule JSON/NPZ artifacts,
rebuilds `SemanticBEVFrame`, renders planner/debug images, and writes a compact
review report for manual branch-decision auditing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoothnav.semantic_bev import (  # noqa: E402
    build_dense_semantic_raster,
    build_pseudo_semantic_support_raster,
    build_semantic_bev_frame,
    hash_file,
    render_pseudo_semantic_support_bev,
    save_dense_semantic_bev_artifacts,
    save_decision_state_bev,
    save_decision_state_field_panel,
    save_semantic_annotated_bev,
    semantic_bev_summary_for_prompt,
)
from smoothnav.bev_decision_state import (  # noqa: E402
    build_decision_state_from_frame,
    save_decision_state_fields_npz,
)


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _find_capsules(capsule_root: Path, steps: Iterable[int]) -> Dict[int, Path]:
    wanted = {int(step) for step in steps}
    result: Dict[int, Path] = {}
    for path in sorted(capsule_root.glob("episode_*_step_*_*")):
        if not path.is_dir():
            continue
        parts = path.name.split("_step_")
        if len(parts) < 2:
            continue
        try:
            step = int(parts[1].split("_", 1)[0])
        except Exception:
            continue
        if step in wanted and step not in result:
            result[step] = path
    return result


def _all_capsule_object_sets(capsule_root: Path) -> Dict[int, set[str]]:
    out: Dict[int, set[str]] = {}
    for path in sorted(capsule_root.glob("episode_*_step_*_*")):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("_step_", 1)[1].split("_", 1)[0])
        except Exception:
            continue
        world = _load_json(path / "world_state.json", {}) or {}
        captions = {
            str(item.get("caption", "")).strip().lower()
            for item in (world.get("object_summary") or [])
            if str(item.get("caption", "")).strip()
        }
        out[step] = captions
    return out


def _load_mllm_calls(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    call_dir = run_dir / "mllm_frontier_calls"
    for path in sorted(call_dir.glob("episode_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            step = rec.get("step_idx")
            if step is None:
                continue
            out[int(step)] = rec
    return out


def _future_delta(step: int, object_sets: Mapping[int, set[str]]) -> Dict[str, Any]:
    later_steps = sorted(s for s in object_sets if s > step)
    current = set(object_sets.get(step, set()))
    if not later_steps:
        return {
            "next_capsule_step": None,
            "new_object_captions_next_capsule": [],
            "new_object_count_next_capsule": 0,
        }
    next_step = later_steps[0]
    new = sorted(set(object_sets.get(next_step, set())) - current)
    return {
        "next_capsule_step": next_step,
        "new_object_captions_next_capsule": new,
        "new_object_count_next_capsule": len(new),
    }


def _selected_fields(frame: Mapping[str, Any]) -> Dict[str, Any]:
    decision = dict(frame.get("decision_overlay") or {})
    qc = dict(frame.get("quality_checks") or {})
    base = dict(frame.get("base_layers") or {})
    heat = dict(frame.get("target_query_heatmap") or {})
    semantic_layers = dict(frame.get("semantic_layers") or {})
    return {
        "agent_alignment_ok": qc.get("agent_on_known_or_near_known"),
        "agent_display_repaired": (frame.get("agent") or {}).get("display_repaired_from_map_current_channel"),
        "raw_agent_to_map_current_distance": (frame.get("agent") or {}).get("agent_channel_distance"),
        "object_count_localized": qc.get("localized_object_count"),
        "target_relevant_object_count": sum(
            1 for item in frame.get("objects", []) or [] if float(item.get("target_relevance", 0.0) or 0.0) > 0.25
        ),
        "room_hypothesis_count": qc.get("localized_room_hypothesis_count"),
        "has_semantic_channels": base.get("has_semantic_channels"),
        "active_semantic_category_count": base.get("active_semantic_category_count"),
        "semantic_region_count": qc.get("semantic_region_count"),
        "target_anchor_semantic_region_count": qc.get("target_anchor_semantic_region_count"),
        "direct_target_pixel_count": semantic_layers.get("direct_target_pixel_count"),
        "has_language_query_heatmap": base.get("has_language_query_heatmap"),
        "target_heatmap_source": heat.get("source"),
        "target_heatmap_positive_pixel_count": heat.get("positive_pixel_count"),
        "target_heatmap_branch_ranking": heat.get("branch_ranking", []),
        "target_value_branch_ranking": (frame.get("target_value_state") or {}).get("branch_ranking", []),
        "executable_decision_branch_ranking": (frame.get("target_value_state") or {}).get("executable_decision_ranking", []),
        "selection_contract": (frame.get("target_value_state") or {}).get("selection_contract", frame.get("selection_contract", {})),
        "branch_evidence_counts": (frame.get("target_value_state") or {}).get("evidence_counts", {}),
        "branch_count": qc.get("branch_count"),
        "branch_semantic_hint_coverage": qc.get("branches_with_semantic_hints"),
        "dry_branch": decision.get("dry_selected_branch_id", ""),
        "mllm_branch": decision.get("mllm_selected_branch_id", ""),
        "final_branch": decision.get("final_selected_branch_id", ""),
        "planner_prior_applied": decision.get("planner_prior_applied", False),
        "quality_warnings": qc.get("warnings", []),
    }


def _dense_metrics(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Replay-summary fields that prove dense BEV is not a label overlay."""

    return {
        "semantic_channel_active_count": int(metadata.get("semantic_channel_active_count", 0) or 0),
        "semantic_coverage_ratio": float(metadata.get("semantic_coverage_ratio", 0.0) or 0.0),
        "per_category_pixel_count": dict(metadata.get("per_category_pixel_count") or {}),
        "true_semantic_pixel_count": int(metadata.get("true_semantic_pixel_count", 0) or 0),
        "object_footprint_pixel_count": int(metadata.get("object_footprint_pixel_count", 0) or 0),
        "object_footprint_source": metadata.get("object_footprint_source", "none"),
        "object_footprint_per_category_pixel_count": dict(metadata.get("object_footprint_per_category_pixel_count") or {}),
        "rejected_object_footprint_count": len(list(metadata.get("rejected_object_footprints") or [])),
        "semantic_bbox_footprint_pixel_count": int(metadata.get("semantic_bbox_footprint_pixel_count", 0) or 0),
        "semantic_bbox_footprint_source": metadata.get("semantic_bbox_footprint_source", "none"),
        "semantic_bbox_footprint_per_category_pixel_count": dict(metadata.get("semantic_bbox_footprint_per_category_pixel_count") or {}),
        "display_semantic_pixel_count": int(metadata.get("display_semantic_pixel_count", 0) or 0),
        "display_semantic_coverage_ratio": float(metadata.get("display_semantic_coverage_ratio", 0.0) or 0.0),
        "display_semantic_source": metadata.get("display_semantic_source"),
        "pseudo_support_pixel_count": int(metadata.get("pseudo_support_pixel_count", 0) or 0),
        "unlocalized_rooms_drawn_count": int(metadata.get("unlocalized_rooms_drawn_count", 0) or 0),
        "object_labels_drawn_on_dense_map_count": int(metadata.get("object_labels_drawn_on_dense_map_count", 0) or 0),
        "branch_labels_drawn_on_dense_map_count": int(metadata.get("branch_labels_drawn_on_dense_map_count", 0) or 0),
        "semantic_source": metadata.get("semantic_source"),
        "dense_semantic_warnings": list(metadata.get("warnings") or []),
    }


def _pseudo_support_metadata(pseudo_support: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": pseudo_support.get("schema_version", "smoothnav.pseudo_semantic_support_raster.v1"),
        "pseudo_support_enabled": bool(pseudo_support.get("pseudo_support_enabled", False)),
        "pseudo_support_source": str(pseudo_support.get("pseudo_support_source", "")),
        "pseudo_support_pixel_count": int(pseudo_support.get("pseudo_support_pixel_count", 0) or 0),
        "pseudo_support_objects": list(pseudo_support.get("pseudo_support_objects") or []),
        "notes": [
            "This artifact is object-center prior support only.",
            "It is not a true dense semantic map and does not affect semantic_coverage_ratio.",
        ],
    }


def replay_one(step: int, capsule_dir: Path, out_dir: Path, *, mllm_call: Optional[Mapping[str, Any]], object_sets: Mapping[int, set[str]]) -> Dict[str, Any]:
    out_step = out_dir / f"step_{step:06d}"
    out_step.mkdir(parents=True, exist_ok=True)
    world_state = _load_json(capsule_dir / "world_state.json", {}) or {}
    frontier_branches = _load_json(capsule_dir / "frontier_branches.json", {}) or {}
    scoring_input = _load_json(capsule_dir / "scoring_input.json", {}) or {}
    grounding_result = _load_json(capsule_dir / "grounding_result.json", {}) or {}
    planner_result = dict(mllm_call or {})
    manifest = _load_json(capsule_dir / "manifest.json", {}) or {}
    maps_path = capsule_dir / "maps.npz"
    if not maps_path.exists():
        raise FileNotFoundError(f"missing maps.npz in {capsule_dir}")
    maps = np.load(maps_path)
    if "full_map" not in maps:
        raise KeyError(f"full_map missing in {maps_path}")
    full_map = maps["full_map"]
    frame = build_semantic_bev_frame(
        full_map=full_map,
        frontier_branches=frontier_branches,
        world_state=world_state,
        scoring_input=scoring_input,
        grounding_result=grounding_result,
        planner_result=planner_result,
        episode_id=manifest.get("episode_id"),
        step_idx=step,
        goal_description=str(manifest.get("target") or manifest.get("goal") or ""),
        source=str(capsule_dir),
    )
    frame["replay_outcome"] = _future_delta(step, object_sets)
    decision_state_fields = {}
    try:
        rebuilt_decision_state, decision_state_fields = build_decision_state_from_frame(full_map, frame)
        # Keep the JSON state aligned with the field artifact while preserving
        # quality flags already attached by build_semantic_bev_frame.
        rebuilt_decision_state["quality_flags"] = list((frame.get("decision_state") or {}).get("quality_flags") or [])
        frame["decision_state"] = rebuilt_decision_state
        frame["target_value_state"] = rebuilt_decision_state.get("target_value_field", frame.get("target_value_state", {}))
        frame["branch_decision_table"] = list(rebuilt_decision_state.get("branch_table") or frame.get("branch_decision_table") or [])
        selection_contract = dict(
            frame.get("selection_contract")
            or ((frame.get("target_value_state") or {}).get("selection_contract") or {})
        )
        if selection_contract:
            frame["selection_contract"] = selection_contract
            frame["target_value_state"]["selection_contract"] = selection_contract
            frame["decision_state"]["selection_contract"] = selection_contract
            if isinstance(frame["decision_state"].get("target_value_field"), dict):
                frame["decision_state"]["target_value_field"]["selection_contract"] = selection_contract
            if isinstance(frame["decision_state"].get("target_value_state"), dict):
                frame["decision_state"]["target_value_state"]["selection_contract"] = selection_contract
    except Exception:
        decision_state_fields = {}
    _write_json(out_step / "semantic_bev_frame.json", frame)
    _write_json(out_step / "semantic_bev_prompt_summary.json", semantic_bev_summary_for_prompt(frame))
    _write_json(out_step / "planner_decision_state.json", frame.get("decision_state", {}))
    fields_npz = ""
    if decision_state_fields:
        fields_npz = save_decision_state_fields_npz(out_step / "planner_decision_state_fields.npz", decision_state_fields)
    planner_img = save_semantic_annotated_bev(out_step / "semantic_annotated_bev_planner.png", frame, full_map, mode="planner")
    debug_img = save_semantic_annotated_bev(out_step / "semantic_annotated_bev_debug.png", frame, full_map, mode="debug")
    debug_overlay_full = out_step / "debug_overlay_full.png"
    if debug_img and Path(debug_img).exists():
        shutil.copy2(debug_img, debug_overlay_full)

    dense_raster = build_dense_semantic_raster(
        full_map,
        objects=frame.get("objects", []) or [],
        semantic_bbox_fill=False,
    )
    footprint_raster = build_dense_semantic_raster(
        full_map,
        objects=frame.get("objects", []) or [],
        semantic_bbox_fill=False,
    )
    pseudo_support = build_pseudo_semantic_support_raster(full_map, frame.get("objects", []) or [])
    dense_artifact = save_dense_semantic_bev_artifacts(
        out_step / "dense_semantic_bev.png",
        out_step / "dense_semantic_bev.json",
        dense_raster,
        agent_pose=(frame.get("agent") or {}).get("coord_rc"),
        trajectory=frame.get("trajectory", []) or [],
        frontier_mask=None,
        unlocalized_room_hypotheses=(frame.get("room_state") or {}).get("unlocalized_room_priors", []) or [],
        pseudo_support=pseudo_support,
        scale=2,
        crop_to_content=True,
        crop_margin=32,
    )
    footprint_artifact = save_dense_semantic_bev_artifacts(
        out_step / "semantic_footprint_bev.png",
        out_step / "semantic_footprint_bev.json",
        footprint_raster,
        agent_pose=(frame.get("agent") or {}).get("coord_rc"),
        trajectory=frame.get("trajectory", []) or [],
        frontier_mask=None,
        unlocalized_room_hypotheses=(frame.get("room_state") or {}).get("unlocalized_room_priors", []) or [],
        pseudo_support=pseudo_support,
        scale=2,
        crop_to_content=True,
        crop_margin=32,
    )
    pseudo_img = ""
    pseudo_json = ""
    if pseudo_support.get("pseudo_support_enabled"):
        pseudo_img_path = out_step / "pseudo_semantic_support_bev.png"
        rendered_pseudo = render_pseudo_semantic_support_bev(
            pseudo_support,
            full_map,
            agent_pose=(frame.get("agent") or {}).get("coord_rc"),
            save_path=pseudo_img_path,
            scale=1,
        )
        pseudo_img = str(pseudo_img_path) if rendered_pseudo is not None and pseudo_img_path.exists() else ""
        pseudo_json = out_step / "pseudo_semantic_support_bev.json"
        _write_json(pseudo_json, _pseudo_support_metadata(pseudo_support))

    decision_img = save_decision_state_bev(out_step / "decision_state_bev_planner.png", frame, full_map, scale=1)
    semantic_panel = save_decision_state_field_panel(out_step / "semantic_field_panel.png", frame, full_map, panel="semantic", scale=1)
    value_panel = save_decision_state_field_panel(out_step / "target_value_field_panel.png", frame, full_map, panel="value", scale=1)
    legacy = capsule_dir / "bev_annotated_branches.png"
    if legacy.exists():
        shutil.copy2(legacy, out_step / "legacy_bev_annotated_branches.png")
    result = {
        "step_idx": step,
        "capsule_dir": str(capsule_dir),
        "out_dir": str(out_step),
        **_selected_fields(frame),
        **_dense_metrics(dense_artifact.get("metadata", {})),
        "semantic_footprint_metrics": _dense_metrics(footprint_artifact.get("metadata", {})),
        "replay_outcome": frame["replay_outcome"],
        "files": {
            "semantic_bev_frame": str(out_step / "semantic_bev_frame.json"),
            "planner_decision_state": str(out_step / "planner_decision_state.json"),
            "planner_decision_state_fields": str(fields_npz) if fields_npz else "",
            "dense_semantic_bev": str(out_step / "dense_semantic_bev.png"),
            "dense_semantic_bev_json": str(out_step / "dense_semantic_bev.json"),
            "semantic_footprint_bev": str(out_step / "semantic_footprint_bev.png"),
            "semantic_footprint_bev_json": str(out_step / "semantic_footprint_bev.json"),
            "debug_overlay_full": str(debug_overlay_full) if debug_overlay_full.exists() else "",
            "pseudo_semantic_support_bev": str(pseudo_img) if pseudo_img else "",
            "pseudo_semantic_support_bev_json": str(pseudo_json) if pseudo_json else "",
            "planner_image": str(planner_img) if planner_img else "",
            "debug_image": str(debug_img) if debug_img else "",
            "decision_state_image": str(decision_img) if decision_img else "",
            "semantic_field_panel": str(semantic_panel) if semantic_panel else "",
            "target_value_field_panel": str(value_panel) if value_panel else "",
            "legacy_image": str(out_step / "legacy_bev_annotated_branches.png") if legacy.exists() else "",
        },
        "hashes": {},
    }
    for key, value in result["files"].items():
        if value and Path(value).exists():
            result["hashes"][key] = hash_file(value)
    _write_json(out_step / "review_result.json", result)
    return result


def write_review(out_dir: Path, results: List[Mapping[str, Any]]) -> None:
    lines = [
        "# s23 Semantic BEV Capsule Replay Review",
        "",
        "Offline replay only: no Habitat, no LLM API call.",
        "",
        "| Step | dry | MLLM | final | agent ok | dense source | dense cov | true sem px | sem cats | sem regions | target heat | branch hints | new objs next | warnings |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            "| {step} | {dry} | {mllm} | {final} | {agent} | {dense_source} | {dense_cov:.4f} | {true_px} | {semcats} | {semregions} | {heat} | {hints} | {new} | {warn} |".format(
                step=item.get("step_idx"),
                dry=item.get("dry_branch", ""),
                mllm=item.get("mllm_branch", ""),
                final=item.get("final_branch", ""),
                agent=item.get("agent_alignment_ok"),
                dense_source=item.get("semantic_source"),
                dense_cov=float(item.get("semantic_coverage_ratio", 0.0) or 0.0),
                true_px=item.get("true_semantic_pixel_count", 0),
                semcats=item.get("active_semantic_category_count"),
                semregions=item.get("semantic_region_count"),
                heat=f"{item.get('target_heatmap_source')}:{item.get('target_heatmap_positive_pixel_count')}",
                hints=item.get("branch_semantic_hint_coverage"),
                new=(item.get("replay_outcome") or {}).get("new_object_count_next_capsule"),
                warn=", ".join(item.get("quality_warnings") or []),
            )
        )
    lines.extend(["", "## Target heatmap branch rankings", ""])
    for item in results:
        ranking = item.get("target_heatmap_branch_ranking") or []
        if not ranking:
            continue
        compact = ", ".join(
            f"{rank.get('id')}:H={float(rank.get('target_heatmap_score', 0) or 0):.2f}/A={float(rank.get('semantic_target_affinity', 0) or 0):.2f}"
            for rank in ranking
        )
        lines.append(f"- Step {item.get('step_idx')}: {compact}")
    lines.extend(["", "## Target-conditioned value branch rankings", ""])
    for item in results:
        ranking = item.get("target_value_branch_ranking") or []
        if not ranking:
            continue
        compact = ", ".join(
            f"{rank.get('id')}:V={float(rank.get('target_value_score', 0) or 0):.2f}/{rank.get('evidence_level')}"
            for rank in ranking
        )
        lines.append(f"- Step {item.get('step_idx')}: {compact}; counts={item.get('branch_evidence_counts')}")
    lines.extend(["", "## Per-step artifacts", ""])
    for item in results:
        step = int(item.get("step_idx"))
        rel = Path(item.get("out_dir", ".")).name
        lines.extend(
            [
                f"### Step {step}",
                "",
                f"- Frame JSON: `{rel}/semantic_bev_frame.json`",
                f"- Dense semantic BEV: `{rel}/dense_semantic_bev.png`",
                f"- Dense semantic BEV JSON: `{rel}/dense_semantic_bev.json`",
                f"- Debug overlay full: `{rel}/debug_overlay_full.png`",
                f"- Decision state JSON: `{rel}/planner_decision_state.json`",
                f"- Decision state fields: `{rel}/planner_decision_state_fields.npz`",
                f"- Planner image: `{rel}/semantic_annotated_bev_planner.png`",
                f"- Decision-state image: `{rel}/decision_state_bev_planner.png`",
                f"- Semantic field panel: `{rel}/semantic_field_panel.png`",
                f"- Target value field panel: `{rel}/target_value_field_panel.png`",
                f"- Debug image: `{rel}/semantic_annotated_bev_debug.png`",
                f"- Legacy image: `{rel}/legacy_bev_annotated_branches.png`",
                "",
            ]
        )
    (out_dir / "review.md").write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=[31, 43, 98, 145])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    capsule_root = args.capsule_root
    run_dir = capsule_root.parent
    capsules = _find_capsules(capsule_root, args.steps)
    missing = sorted(set(args.steps) - set(capsules))
    if missing:
        raise SystemExit(f"missing capsules for steps: {missing}")
    object_sets = _all_capsule_object_sets(capsule_root)
    mllm_calls = _load_mllm_calls(run_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for step in args.steps:
        results.append(
            replay_one(
                int(step),
                capsules[int(step)],
                args.out,
                mllm_call=mllm_calls.get(int(step)),
                object_sets=object_sets,
            )
        )
    summary = {
        "schema_version": "smoothnav.semantic_bev_capsule_replay.summary.v1",
        "capsule_root": str(capsule_root),
        "out_dir": str(args.out),
        "steps": list(args.steps),
        "results": results,
    }
    _write_json(args.out / "summary.json", summary)
    write_review(args.out, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
