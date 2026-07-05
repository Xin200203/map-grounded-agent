#!/usr/bin/env python3
"""Summarize target-evidence uptake from existing SmoothNav step traces.

This is a trace-only diagnostic: it does not run Habitat or change results. It
answers when target-like captions first appeared, when the planner next ran, and
whether a target-anchor, direct-object, or direct-goal commitment followed.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _iter_trace_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(path.glob("**/step_traces/episode_*.jsonl"))


def _target_details(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph_delta = step.get("graph_delta") or {}
    details = graph_delta.get("target_candidate_details") or []
    if details:
        return list(details)
    # Backward compatibility for traces written before this diagnostic field.
    world = step.get("world_state_summary") or {}
    objects = world.get("object_summary") or []
    return [
        obj for obj in objects
        if float(obj.get("target_relevance") or 0.0) >= 0.75
    ]


def summarize_trace(path: Path) -> Dict[str, Any]:
    first_target_step: Optional[int] = None
    first_target_captions: List[str] = []
    first_planner_after_target: Optional[int] = None
    first_target_anchor_after_target: Optional[int] = None
    first_object_commit_after_target: Optional[int] = None
    first_direct_goal_after_target: Optional[int] = None
    target_events = 0
    planner_calls = 0
    monitor_skips_after_target = 0
    terminal = None
    last_step = None

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        step = json.loads(line)
        step_idx = int(step.get("step_idx", 0))
        last_step = step_idx
        terminal = (step.get("terminal_decision") or {}).get("outcome") or terminal

        details = _target_details(step)
        if details:
            target_events += 1
            if first_target_step is None:
                first_target_step = step_idx
                first_target_captions = [str(item.get("caption", "")) for item in details]

        if step.get("planner_called"):
            planner_calls += int(step.get("planner_calls_this_step") or 1)
            if first_target_step is not None and first_planner_after_target is None and step_idx >= first_target_step:
                first_planner_after_target = step_idx

        current = step.get("current_strategy") or {}
        target_region = str(current.get("target_region") or "")
        if (
            first_target_step is not None
            and first_target_anchor_after_target is None
            and target_region.startswith("unexplored target:")
            and step_idx >= first_target_step
        ):
            first_target_anchor_after_target = step_idx
        if (
            first_target_step is not None
            and first_object_commit_after_target is None
            and target_region.startswith("object:")
            and step_idx >= first_target_step
        ):
            first_object_commit_after_target = step_idx

        for event in step.get("grounding_events") or []:
            graph_debug = event.get("graph_debug") or {}
            if graph_debug.get("direct_object_goal"):
                if first_target_step is not None and first_direct_goal_after_target is None and step_idx >= first_target_step:
                    first_direct_goal_after_target = step_idx

        if first_target_step is not None and step_idx >= first_target_step:
            graph_delta = step.get("graph_delta") or {}
            if (
                "target_candidate_detected" in set(graph_delta.get("event_types") or [])
                and not step.get("planner_called")
                and not step.get("monitor_called")
            ):
                monitor_skips_after_target += 1

    return {
        "trace": str(path),
        "last_step": last_step,
        "terminal_outcome": terminal,
        "first_target_step": first_target_step,
        "first_target_captions": first_target_captions,
        "target_event_count": target_events,
        "planner_calls": planner_calls,
        "first_planner_after_target": first_planner_after_target,
        "evidence_to_planner_latency": (
            None if first_target_step is None or first_planner_after_target is None
            else first_planner_after_target - first_target_step
        ),
        "first_object_commit_after_target": first_object_commit_after_target,
        "first_target_anchor_after_target": first_target_anchor_after_target,
        "evidence_to_target_anchor_latency": (
            None if first_target_step is None or first_target_anchor_after_target is None
            else first_target_anchor_after_target - first_target_step
        ),
        "evidence_to_object_commit_latency": (
            None if first_target_step is None or first_object_commit_after_target is None
            else first_object_commit_after_target - first_target_step
        ),
        "first_direct_goal_after_target": first_direct_goal_after_target,
        "evidence_to_direct_goal_latency": (
            None if first_target_step is None or first_direct_goal_after_target is None
            else first_direct_goal_after_target - first_target_step
        ),
        "unhandled_target_candidate_steps": monitor_skips_after_target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run dirs, results roots, or step trace jsonl files")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines instead of a compact table")
    args = parser.parse_args()

    summaries = [summarize_trace(path) for path in _iter_trace_files(args.paths)]
    if args.json:
        for item in summaries:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return 0

    header = (
        "trace first_target captions planner_after anchor_after commit_after direct_after "
        "unhandled terminal"
    )
    print(header)
    for item in summaries:
        print(
            f"{item['trace']} "
            f"{item['first_target_step']} "
            f"{','.join(item['first_target_captions']) or '-'} "
            f"{item['first_planner_after_target']} "
            f"{item['first_target_anchor_after_target']} "
            f"{item['first_object_commit_after_target']} "
            f"{item['first_direct_goal_after_target']} "
            f"{item['unhandled_target_candidate_steps']} "
            f"{item['terminal_outcome']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
