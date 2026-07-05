#!/usr/bin/env python3
"""Capture the target-anchor branch from an existing SmoothNav run directory.

This is a trace-only utility. It extracts the first target-evidence branch,
records the key steps where the controller entered / refreshed / lost the target
anchor, and snapshots the associated planner / monitor calls for later replay or
counterfactual analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def _trace_path(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("step_traces/episode_*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No step trace found under {run_dir}")
    return matches[0]


def _planner_path(run_dir: Path) -> Optional[Path]:
    matches = sorted(run_dir.glob("planner_calls/episode_*.jsonl"))
    return matches[0] if matches else None


def _monitor_path(run_dir: Path) -> Optional[Path]:
    matches = sorted(run_dir.glob("monitor_calls/episode_*.jsonl"))
    return matches[0] if matches else None


def _is_target_search_anchor(target_region: str) -> bool:
    return bool(target_region and str(target_region).startswith("unexplored target:"))


def _target_event(step: Dict[str, Any]) -> bool:
    graph_delta = step.get("graph_delta") or {}
    return "target_candidate_detected" in set(graph_delta.get("event_types") or [])


def _step_target_region(step: Dict[str, Any], key: str = "current_strategy") -> str:
    strategy = step.get(key) or {}
    if isinstance(strategy, dict):
        return str(strategy.get("target_region") or "")
    return ""


def _interesting_step(step: Dict[str, Any], prev_step: Optional[Dict[str, Any]]) -> bool:
    triggers = [event.get("trigger") for event in (step.get("grounding_events") or [])]
    current_region = _step_target_region(step)
    pending_region = _step_target_region(step, "pending_strategy")
    prev_region = _step_target_region(prev_step) if prev_step else ""
    return any(
        (
            _target_event(step),
            bool(step.get("planner_called")),
            bool(step.get("monitor_called")),
            bool(triggers),
            _is_target_search_anchor(pending_region),
            current_region != prev_region,
        )
    )


def _snapshot_step(step: Dict[str, Any], prev_step: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    triggers = [event.get("trigger") for event in (step.get("grounding_events") or [])]
    current_region = _step_target_region(step)
    pending_region = _step_target_region(step, "pending_strategy")
    prev_region = _step_target_region(prev_step) if prev_step else ""
    return {
        "step_idx": step.get("step_idx"),
        "events": list((step.get("graph_delta") or {}).get("event_types") or []),
        "planner_called": bool(step.get("planner_called")),
        "planner_reasons": list(step.get("planner_reasons") or []),
        "monitor_called": bool(step.get("monitor_called")),
        "monitor_decision": step.get("monitor_decision"),
        "monitor_reason": step.get("monitor_reason"),
        "current_target_region": current_region,
        "pending_target_region": pending_region,
        "previous_target_region": prev_region,
        "goal_after": step.get("goal_after"),
        "selected_frontier": step.get("selected_frontier"),
        "bias_input": step.get("bias_input"),
        "dist_to_goal": (step.get("graph_delta") or {}).get("dist_to_goal"),
        "pose_after": step.get("pose_after"),
        "grounding_triggers": triggers,
        "grounding_events": [
            {
                "trigger": event.get("trigger"),
                "changed": event.get("changed"),
                "reason": event.get("reason"),
                "selected_frontier": event.get("selected_frontier"),
                "projected_goal": event.get("projected_goal"),
            }
            for event in (step.get("grounding_events") or [])
        ],
    }


def _target_branch_segments(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    start: Optional[int] = None
    last: Optional[int] = None
    for step in steps:
        idx = int(step.get("step_idx", 0))
        current_region = _step_target_region(step)
        if _is_target_search_anchor(current_region):
            if start is None:
                start = idx
            last = idx
        elif start is not None:
            segments.append({"start_step": start, "end_step": last})
            start = None
            last = None
    if start is not None:
        segments.append({"start_step": start, "end_step": last})
    return segments


def _branch_drop_events(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    drops = []
    for prev_step, step in zip(steps, steps[1:]):
        prev_region = _step_target_region(prev_step)
        current_region = _step_target_region(step)
        if _is_target_search_anchor(prev_region) and not _is_target_search_anchor(current_region):
            drops.append(
                {
                    "step_idx": step.get("step_idx"),
                    "from_region": prev_region,
                    "to_region": current_region,
                    "planner_reasons": list(step.get("planner_reasons") or []),
                    "events": list((step.get("graph_delta") or {}).get("event_types") or []),
                    "pending_region_previous_step": _step_target_region(
                        prev_step, "pending_strategy"
                    ),
                    "goal_after": step.get("goal_after"),
                }
            )
    return drops


def _refresh_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refresh = []
    for step in steps:
        for event in step.get("grounding_events") or []:
            if event.get("trigger") == "target_anchor_local_map_refresh":
                refresh.append(
                    {
                        "step_idx": step.get("step_idx"),
                        "selected_frontier": event.get("selected_frontier"),
                        "projected_goal": event.get("projected_goal"),
                        "changed": event.get("changed"),
                        "reason": event.get("reason"),
                    }
                )
    return refresh


def capture_run_dir(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    steps = _load_jsonl(_trace_path(run_dir))
    planner_calls = _load_jsonl(_planner_path(run_dir)) if _planner_path(run_dir) else []
    monitor_calls = _load_jsonl(_monitor_path(run_dir)) if _monitor_path(run_dir) else []

    first_target_step: Optional[int] = None
    first_target_captions: List[str] = []
    first_anchor_step: Optional[int] = None
    first_anchor_region: Optional[str] = None
    interesting_steps: List[Dict[str, Any]] = []

    for idx, step in enumerate(steps):
        prev_step = steps[idx - 1] if idx > 0 else None
        if _target_event(step) and first_target_step is None:
            first_target_step = int(step.get("step_idx", 0))
            details = (step.get("graph_delta") or {}).get("target_candidate_details") or []
            first_target_captions = [str(item.get("caption", "")) for item in details if item.get("caption")]
        current_region = _step_target_region(step)
        if _is_target_search_anchor(current_region) and first_anchor_step is None:
            first_anchor_step = int(step.get("step_idx", 0))
            first_anchor_region = current_region
        if _interesting_step(step, prev_step):
            interesting_steps.append(_snapshot_step(step, prev_step))

    segments = _target_branch_segments(steps)
    refresh = _refresh_steps(steps)
    drops = _branch_drop_events(steps)

    if first_target_step is None:
        window_start = None
        window_end = None
    else:
        related_steps = [
            int(step.get("step_idx", 0))
            for step in steps
            if (
                _target_event(step)
                or _is_target_search_anchor(_step_target_region(step))
                or _is_target_search_anchor(_step_target_region(step, "pending_strategy"))
                or any(
                    event.get("trigger") == "target_anchor_local_map_refresh"
                    for event in (step.get("grounding_events") or [])
                )
            )
        ]
        related_steps.extend(int(item["step_idx"]) for item in drops)
        window_start = min(related_steps) if related_steps else first_target_step
        window_end = max(related_steps) if related_steps else first_target_step

    planner_subset = [
        item for item in planner_calls
        if window_start is not None
        and window_end is not None
        and window_start <= int(item.get("step_idx", -1)) <= window_end
    ]
    monitor_subset = [
        item for item in monitor_calls
        if window_start is not None
        and window_end is not None
        and window_start <= int(item.get("step_idx", -1)) <= window_end
    ]

    terminal_outcome = None
    if steps:
        terminal_outcome = (
            (steps[-1].get("terminal_decision") or {}).get("outcome")
            or steps[-1].get("terminal_outcome")
        )

    interesting_subset = [
        item
        for item in interesting_steps
        if window_start is not None
        and window_end is not None
        and window_start <= int(item.get("step_idx", -1)) <= window_end
    ]

    return {
        "source_run_dir": str(run_dir),
        "trace_path": str(_trace_path(run_dir)),
        "first_target_step": first_target_step,
        "first_target_captions": first_target_captions,
        "first_target_anchor_step": first_anchor_step,
        "first_target_anchor_region": first_anchor_region,
        "capture_window": {
            "start_step": window_start,
            "end_step": window_end,
        },
        "target_branch_segments": segments,
        "refresh_steps": refresh,
        "branch_drop_events": drops,
        "planner_calls": planner_subset,
        "monitor_calls": monitor_subset,
        "interesting_steps": interesting_subset,
        "terminal_outcome": terminal_outcome,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="SmoothNav run directory containing step_traces/")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write capture JSON to <run_dir>/target_branch_capture.json",
    )
    args = parser.parse_args()

    capture = capture_run_dir(Path(args.run_dir))
    text = json.dumps(capture, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.write:
        output = Path(args.run_dir) / "target_branch_capture.json"
        output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
