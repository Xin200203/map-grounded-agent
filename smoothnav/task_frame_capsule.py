"""Task-frame capsule writer for replayable SmoothNav decision slices."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from smoothnav.frontier_branching import save_branch_bev_image
from smoothnav.semantic_bev import (
    build_dense_semantic_raster,
    build_semantic_bev_frame,
    hash_file,
    save_dense_semantic_bev_artifacts,
    save_semantic_annotated_bev,
)


SCHEMA_VERSION = "smoothnav.task_frame_capsule.v1"


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().numpy().tolist()
        except Exception:
            pass
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


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label or "frame")).strip("_")
    return cleaned[:80] or "frame"


def _array_payload(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object:
        return None
    return arr


def write_task_frame_capsule(
    run_dir: str | Path,
    *,
    episode_id: int,
    step_idx: int,
    label: str,
    artifacts: Mapping[str, Any],
    maps: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a directory of JSON/NPZ/PNG artifacts for one decision frame."""

    capsule_dir = (
        Path(run_dir)
        / "task_frame_capsules"
        / f"episode_{int(episode_id):06d}_step_{int(step_idx):06d}_{_sanitize_label(label)}"
    )
    capsule_dir.mkdir(parents=True, exist_ok=True)

    manifest = dict(artifacts.get("manifest") or {})
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "episode_id": int(episode_id),
            "step_idx": int(step_idx),
            "label": str(label or "frame"),
            "created_local": datetime.now().isoformat(timespec="milliseconds"),
        }
    )
    written_files = []

    def write_json(filename: str, payload: Any) -> None:
        path = capsule_dir / filename
        path.write_text(
            json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        written_files.append(filename)

    write_json("manifest.json", manifest)
    for key, filename in (
        ("world_state", "world_state.json"),
        ("graph_snapshot", "graph_snapshot.json"),
        ("frontier_raw", "frontier_raw.json"),
        ("frontier_branches", "frontier_branches.json"),
        ("planner_prompt", "planner_prompt.json"),
        ("planner_raw_response", "planner_raw_response.json"),
        ("planner_parsed_response", "planner_parsed_response.json"),
        ("planner_contract_verdict", "planner_contract_verdict.json"),
        ("scoring_input", "scoring_input.json"),
        ("scoring_replay", "scoring_replay.json"),
        ("grounding_result", "grounding_result.json"),
        ("executor_followup", "executor_followup.json"),
    ):
        if key in artifacts and artifacts.get(key) is not None:
            write_json(filename, artifacts.get(key))

    map_arrays: Dict[str, np.ndarray] = {}
    for key, value in (maps or {}).items():
        arr = _array_payload(value)
        if arr is not None:
            map_arrays[str(key)] = arr
    if map_arrays:
        np.savez_compressed(capsule_dir / "maps.npz", **map_arrays)
        written_files.append("maps.npz")

    branch_payload = artifacts.get("frontier_branches") or {}
    full_map = map_arrays.get("full_map") if map_arrays else None
    agent_coord = None
    scoring_input = artifacts.get("scoring_input") or {}
    if isinstance(scoring_input, Mapping):
        agent_coord = scoring_input.get("agent_coord")
    if full_map is not None and branch_payload:
        image_path = save_branch_bev_image(
            capsule_dir / "bev_annotated_branches.png",
            full_map,
            branch_payload,
            agent_coord=agent_coord,
        )
        if image_path:
            written_files.append("bev_annotated_branches.png")

    semantic_hashes: Dict[str, str] = {}
    if full_map is not None and branch_payload:
        semantic_frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branch_payload,
            world_state=artifacts.get("world_state"),
            scoring_input=scoring_input,
            grounding_result=artifacts.get("grounding_result"),
            planner_result=(
                artifacts.get("planner_contract_verdict")
                or artifacts.get("planner_parsed_response")
                or artifacts.get("planner_raw_response")
            ),
            episode_id=episode_id,
            step_idx=step_idx,
            goal_description=str(manifest.get("target") or manifest.get("goal") or ""),
            source="task_frame_capsule",
        )
        write_json("semantic_bev_frame.json", semantic_frame)
        world_state_payload = artifacts.get("world_state") or {}
        if isinstance(world_state_payload, Mapping):
            semantic_projection_summary = world_state_payload.get("semantic_projection_summary")
            write_json(
                "semantic_projection_summary.json",
                {
                    "schema_version": "smoothnav.semantic_projection_summary.artifact.v1",
                    "present_in_world_state": semantic_projection_summary is not None,
                    "summary": semantic_projection_summary or {},
                },
            )
            semantic_instance_footprints = world_state_payload.get("semantic_instance_footprints") or []
            write_json(
                "semantic_instance_footprints.json",
                {
                    "schema_version": "smoothnav.semantic_instance_footprints.artifact.v1",
                    "count": int(len(semantic_instance_footprints)),
                    "present_in_world_state": "semantic_instance_footprints" in world_state_payload,
                    "projection_summary_present": bool(world_state_payload.get("semantic_projection_summary")),
                    "instances": semantic_instance_footprints,
                },
            )
        object_footprints = [
            {
                "id": obj.get("id"),
                "caption": obj.get("caption"),
                "bbox_rc": obj.get("bbox_rc"),
                "footprint_source": obj.get("footprint_source"),
                "footprint_pixel_count": obj.get("footprint_pixel_count"),
                "footprint_confidence": obj.get("footprint_confidence"),
                "footprint_encoding": obj.get("footprint_encoding"),
                "footprint_seed_cell_count": obj.get("footprint_seed_cell_count"),
                "pcd_point_count": obj.get("pcd_point_count"),
                "footprint_rejected_reason": obj.get("footprint_rejected_reason"),
                "footprint_rc_indices": obj.get("footprint_rc_indices") or [],
            }
            for obj in (semantic_frame.get("objects") or [])
            if obj.get("footprint_pixel_count") or obj.get("footprint_rejected_reason") or obj.get("footprint_rc_indices")
        ]
        write_json(
            "object_footprints.json",
            {
                "schema_version": "smoothnav.object_footprints.v1",
                "count": int(len(object_footprints)),
                "objects": object_footprints,
            },
        )
        dense_raster = build_dense_semantic_raster(
            full_map,
            objects=semantic_frame.get("objects") or [],
            semantic_bbox_fill=False,
        )
        footprint_raster = build_dense_semantic_raster(
            full_map,
            objects=semantic_frame.get("objects") or [],
            semantic_bbox_fill=False,
        )
        dense_artifact = save_dense_semantic_bev_artifacts(
            capsule_dir / "dense_semantic_bev.png",
            capsule_dir / "dense_semantic_bev.json",
            dense_raster,
            agent_pose=(semantic_frame.get("agent") or {}).get("coord_rc"),
            trajectory=semantic_frame.get("trajectory") or [],
            unlocalized_room_hypotheses=(semantic_frame.get("room_state") or {}).get("unlocalized_room_priors") or [],
            scale=2,
            crop_to_content=True,
            crop_margin=32,
        )
        if dense_artifact.get("image"):
            written_files.append("dense_semantic_bev.png")
        if dense_artifact.get("json"):
            written_files.append("dense_semantic_bev.json")
        footprint_artifact = save_dense_semantic_bev_artifacts(
            capsule_dir / "semantic_footprint_bev.png",
            capsule_dir / "semantic_footprint_bev.json",
            footprint_raster,
            agent_pose=(semantic_frame.get("agent") or {}).get("coord_rc"),
            trajectory=semantic_frame.get("trajectory") or [],
            unlocalized_room_hypotheses=(semantic_frame.get("room_state") or {}).get("unlocalized_room_priors") or [],
            scale=2,
            crop_to_content=True,
            crop_margin=32,
        )
        if footprint_artifact.get("image"):
            written_files.append("semantic_footprint_bev.png")
        if footprint_artifact.get("json"):
            written_files.append("semantic_footprint_bev.json")
        for mode, filename in (
            ("planner", "semantic_annotated_bev_planner.png"),
            ("debug", "semantic_annotated_bev_debug.png"),
        ):
            image_path = save_semantic_annotated_bev(
                capsule_dir / filename,
                semantic_frame,
                full_map,
                mode=mode,
            )
            if image_path:
                written_files.append(filename)
                try:
                    semantic_hashes[filename] = hash_file(image_path)
                except Exception:
                    pass

    write_json(
        "capsule_index.json",
        {
            "schema_version": SCHEMA_VERSION + ".index",
            "capsule_dir": str(capsule_dir),
            "files": sorted(set(written_files)),
            "artifact_hashes": semantic_hashes,
        },
    )
    return {
        "schema_version": SCHEMA_VERSION + ".result",
        "capsule_dir": str(capsule_dir),
        "files": sorted(set(written_files + ["capsule_index.json"])),
    }
