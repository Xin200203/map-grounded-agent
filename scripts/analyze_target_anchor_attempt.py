#!/usr/bin/env python3
"""Analyze target-anchor attempt progress and stalled-decommit counterfactuals.

This is a trace-only tool built on top of `target_branch_capture.json`. It
extracts the target-anchor update sequence, measures frontier-to-anchor
improvement over time, and sweeps simple stalled-decommit policies to identify
which thresholds would prematurely drop a branch versus preserve later evidence.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.capture_target_branch import capture_run_dir  # noqa: E402


TARGET_UPDATE_TRIGGERS = {
    "target_candidate_detected",
    "target_anchor_local_map_refresh",
    "frontier_reached",
}


def _resolve_capture_path(path: Path) -> Path:
    if path.is_dir():
        capture = path / "target_branch_capture.json"
        if capture.exists():
            return capture
        capture_obj = capture_run_dir(path)
        capture.write_text(
            json.dumps(capture_obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return capture
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def _resolve_trace_path(capture_path: Path, capture: Dict[str, Any]) -> Path:
    trace = Path(capture["trace_path"])
    if trace.is_absolute() and trace.exists():
        return trace
    cwd_candidate = Path.cwd() / trace
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root_candidate = capture_path.parents[4] / trace if len(capture_path.parents) >= 5 else None
    if repo_root_candidate is not None and repo_root_candidate.exists():
        return repo_root_candidate
    raise FileNotFoundError(f"Could not resolve trace path {trace} from {capture_path}")


def _is_target_search_anchor(target_region: str) -> bool:
    return bool(target_region and str(target_region).startswith("unexplored target:"))


def _distance(anchor, frontier) -> Optional[float]:
    if anchor is None or frontier is None:
        return None
    return float(
        math.dist(
            [float(anchor[0]), float(anchor[1])],
            [float(frontier[0]), float(frontier[1])],
        )
    )


def _extract_attempts(capture: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments = list(capture.get("target_branch_segments") or [])
    attempts: List[Dict[str, Any]] = []

    for idx, segment in enumerate(segments):
        start = int(segment["start_step"])
        end = int(segment["end_step"])
        updates = []
        for step in steps:
            step_idx = int(step.get("step_idx", -1))
            if step_idx < start or step_idx > end:
                continue
            current_region = str((step.get("current_strategy") or {}).get("target_region") or "")
            if not _is_target_search_anchor(current_region):
                continue
            for event in step.get("grounding_events") or []:
                trigger = event.get("trigger")
                if trigger not in TARGET_UPDATE_TRIGGERS:
                    continue
                bias = event.get("bias_input")
                frontier = event.get("selected_frontier")
                updates.append(
                    {
                        "step_idx": step_idx,
                        "trigger": trigger,
                        "target_region": current_region,
                        "bias_input": bias,
                        "selected_frontier": frontier,
                        "projected_goal": event.get("projected_goal"),
                        "reason": event.get("reason"),
                        "frontier_to_anchor_distance": _distance(bias, frontier),
                    }
                )
        attempts.append(
            {
                "attempt_id": idx + 1,
                "start_step": start,
                "end_step": end,
                "updates": updates,
            }
        )
    return attempts


def _simulate_stall_policy(
    updates: List[Dict[str, Any]],
    *,
    patience_updates: int,
    min_improvement: float,
    evidence_reset: bool = True,
) -> Dict[str, Any]:
    best_distance: Optional[float] = None
    best_step: Optional[int] = None
    stale_count = 0
    decommit_step = None
    decommit_trigger = None
    history = []

    for item in updates:
        step_idx = int(item["step_idx"])
        trigger = str(item.get("trigger") or "")
        distance = item.get("frontier_to_anchor_distance")
        improved = False
        improvement = None
        evidence_reset_applied = False

        if distance is not None:
            if best_distance is None:
                best_distance = float(distance)
                best_step = step_idx
                improved = True
                stale_count = 0
            else:
                improvement = float(best_distance) - float(distance)
                if improvement >= float(min_improvement):
                    best_distance = float(distance)
                    best_step = step_idx
                    improved = True
                    stale_count = 0
                else:
                    if evidence_reset and trigger == "target_candidate_detected":
                        stale_count = 0
                        evidence_reset_applied = True
                    else:
                        stale_count += 1

        history.append(
            {
                "step_idx": step_idx,
                "trigger": trigger,
                "distance": distance,
                "best_distance": best_distance,
                "best_step": best_step,
                "improved": improved,
                "improvement": improvement,
                "stale_count": stale_count,
                "evidence_reset": evidence_reset_applied,
            }
        )

        if stale_count >= int(patience_updates):
            decommit_step = step_idx
            decommit_trigger = trigger
            break

    later_better_evidence = None
    if decommit_step is not None and best_distance is not None:
        for item in updates:
            step_idx = int(item["step_idx"])
            if step_idx <= decommit_step:
                continue
            distance = item.get("frontier_to_anchor_distance")
            trigger = str(item.get("trigger") or "")
            if trigger == "target_candidate_detected" and distance is not None:
                if float(best_distance) - float(distance) >= float(min_improvement):
                    later_better_evidence = {
                        "step_idx": step_idx,
                        "trigger": trigger,
                        "distance": distance,
                    }
                    break

    return {
        "patience_updates": int(patience_updates),
        "min_improvement": float(min_improvement),
        "evidence_reset": bool(evidence_reset),
        "decommit_step": decommit_step,
        "decommit_trigger": decommit_trigger,
        "best_distance_before_decommit": best_distance,
        "later_better_evidence": later_better_evidence,
        "would_false_early_decommit": later_better_evidence is not None,
        "history": history,
    }


def analyze_capture(capture_path: Path) -> Dict[str, Any]:
    capture_path = _resolve_capture_path(Path(capture_path))
    capture = _load_json(capture_path)
    steps = _load_jsonl(_resolve_trace_path(capture_path, capture))
    attempts = _extract_attempts(capture, steps)

    sweep_policies = []
    for patience in (3, 4, 5):
        for min_improvement in (1.0, 5.0):
            sweep_policies.append((patience, min_improvement))

    analyzed_attempts = []
    for attempt in attempts:
        updates = attempt["updates"]
        distances = [
            float(item["frontier_to_anchor_distance"])
            for item in updates
            if item.get("frontier_to_anchor_distance") is not None
        ]
        policy_results = [
            _simulate_stall_policy(
                updates,
                patience_updates=patience,
                min_improvement=min_improvement,
                evidence_reset=True,
            )
            for patience, min_improvement in sweep_policies
        ]
        analyzed_attempts.append(
            {
                **attempt,
                "update_count": len(updates),
                "initial_distance": distances[0] if distances else None,
                "best_distance": min(distances) if distances else None,
                "final_distance": distances[-1] if distances else None,
                "net_improvement": (
                    distances[0] - min(distances) if distances else None
                ),
                "policy_sweep": policy_results,
            }
        )

    return {
        "source_capture": str(capture_path),
        "first_target_step": capture.get("first_target_step"),
        "target_branch_segments": capture.get("target_branch_segments"),
        "branch_drop_events": capture.get("branch_drop_events"),
        "refresh_steps": capture.get("refresh_steps"),
        "attempts": analyzed_attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture",
        help="Path to target_branch_capture.json or a run directory containing it",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write result to <capture_dir>/target_anchor_attempt_analysis.json",
    )
    args = parser.parse_args()

    result = analyze_capture(Path(args.capture))
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.write:
        capture_path = _resolve_capture_path(Path(args.capture))
        output = capture_path.parent / "target_anchor_attempt_analysis.json"
        output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
