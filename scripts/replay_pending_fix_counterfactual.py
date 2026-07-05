#!/usr/bin/env python3
"""Replay the captured s12 pending-promotion bug as a counterfactual test.

This is a trace-only controller experiment. It does not rerun Habitat. Instead,
it loads a `target_branch_capture.json`, recovers the historical step where a
target-anchor branch dropped back to a stale pending direction, and replays that
decision boundary through the current `handle_frontier_reached()` logic.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_events import GraphDelta
from smoothnav.controller_logic import handle_frontier_reached
from smoothnav.controller_state import ControllerState
from smoothnav.planner import Strategy


class _DummyGraph:
    nodes: List[Any] = []
    room_nodes: List[Any] = []

    def get_edges(self):
        return []


class _NoPlanPlanner:
    def __init__(self):
        self.call_count = 0

    def plan(self, **kwargs):
        self.call_count += 1
        raise AssertionError("Counterfactual should not need a fresh planner call")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def _resolve_capture_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "target_branch_capture.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No target_branch_capture.json found under {path}")
    return path


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


def _strategy_from_step(step: Dict[str, Any], key: str) -> Optional[Strategy]:
    payload = step.get(key) or {}
    if not isinstance(payload, dict) or not payload.get("target_region"):
        return None
    bias = payload.get("bias_position")
    if isinstance(bias, list):
        bias = tuple(bias)
    return Strategy(
        target_region=str(payload.get("target_region") or ""),
        bias_position=bias,
        reasoning=str(payload.get("reasoning") or ""),
        explored_regions=list(payload.get("explored_regions") or []),
        anchor_object=str(payload.get("anchor_object") or ""),
    )


def replay_drop_event(capture_path: Path, drop_index: int = 0) -> Dict[str, Any]:
    capture_path = _resolve_capture_path(Path(capture_path))
    capture = _load_json(capture_path)
    drops = list(capture.get("branch_drop_events") or [])
    if not drops:
        raise ValueError(f"No branch_drop_events recorded in {capture_path}")
    if drop_index < 0 or drop_index >= len(drops):
        raise IndexError(f"drop_index {drop_index} out of range for {len(drops)} drop events")

    drop = drops[drop_index]
    drop_step_idx = int(drop["step_idx"])
    trace_steps = _load_jsonl(_resolve_trace_path(capture_path, capture))
    prev_step = None
    drop_step = None
    for idx, step in enumerate(trace_steps):
        if int(step.get("step_idx", -1)) == drop_step_idx:
            drop_step = step
            prev_step = trace_steps[idx - 1] if idx > 0 else None
            break
    if prev_step is None or drop_step is None:
        raise ValueError(f"Could not locate drop step {drop_step_idx} in trace")

    current_strategy = _strategy_from_step(prev_step, "current_strategy")
    pending_strategy = _strategy_from_step(prev_step, "pending_strategy")
    if current_strategy is None:
        raise ValueError("Previous step missing current_strategy")

    state = ControllerState(
        current_strategy=current_strategy,
        pending_strategy=pending_strategy,
        needs_initial_plan=False,
        direction_reuse_count=int(prev_step.get("direction_reuse_count") or 0),
        same_goal_hold_count=int(prev_step.get("same_goal_hold_count") or 0),
        explored_regions=list((prev_step.get("current_strategy") or {}).get("explored_regions") or []),
    )

    delta = GraphDelta(frontier_reached=True)
    applied: List[str] = []

    def _apply(strategy, graph, bev_map, args, global_goals):
        applied.append(strategy.target_region)
        return SimpleNamespace(
            changed=True,
            noop_reason="",
            reason="goal_updated",
            selected_frontier=(0, 0),
        )

    planner = _NoPlanPlanner()
    outcome = handle_frontier_reached(
        controller_state=state,
        graph_delta=delta,
        graph=_DummyGraph(),
        bev_map=SimpleNamespace(),
        args=SimpleNamespace(
            map_size=720,
            controller_direction_reuse_limit=1,
            controller_object_same_goal_replan_threshold=3,
            controller_same_frontier_reuse_threshold=2,
        ),
        global_goals=[0, 0],
        high_planner=planner,
        goal_description="",
        agent_pos=(0, 0),
        apply_strategy_fn=_apply,
        episode_id=0,
        step_idx=drop_step_idx,
        trace_writer=None,
    )

    current_after = (
        state.current_strategy.target_region if state.current_strategy is not None else None
    )
    pending_after = (
        state.pending_strategy.target_region if state.pending_strategy is not None else None
    )

    return {
        "source_capture": str(capture_path),
        "historical_drop_event": drop,
        "historical_before": {
            "step_idx": prev_step.get("step_idx"),
            "current_target_region": current_strategy.target_region,
            "pending_target_region": (
                pending_strategy.target_region if pending_strategy is not None else None
            ),
        },
        "historical_after": {
            "step_idx": drop_step.get("step_idx"),
            "current_target_region": _strategy_from_step(drop_step, "current_strategy").target_region
            if _strategy_from_step(drop_step, "current_strategy") is not None
            else None,
        },
        "counterfactual_outcome": {
            "handled": bool(outcome.get("handled")),
            "pending_promoted": bool(outcome.get("pending_promoted")),
            "pending_promotion_reason": outcome.get("pending_promotion_reason"),
            "forced_replan_due_to_direction_reuse": bool(
                outcome.get("forced_replan_due_to_direction_reuse")
            ),
            "planner_call_count": int(planner.call_count),
            "applied_target_regions": list(applied),
            "current_target_region_after": current_after,
            "pending_target_region_after": pending_after,
        },
        "verdict": {
            "kept_target_anchor": current_after == current_strategy.target_region,
            "cleared_stale_pending": pending_after is None,
            "avoided_historical_drop": current_after != drop.get("to_region"),
            "bug_fixed_on_captured_context": (
                current_after == current_strategy.target_region
                and pending_after is None
                and current_after != drop.get("to_region")
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture",
        help="Path to target_branch_capture.json or a run directory containing it",
    )
    parser.add_argument("--drop-index", type=int, default=0)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write result to <capture_dir>/pending_fix_counterfactual.json",
    )
    args = parser.parse_args()

    result = replay_drop_event(Path(args.capture), drop_index=args.drop_index)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.write:
        capture_path = _resolve_capture_path(Path(args.capture))
        output = capture_path.parent / "pending_fix_counterfactual.json"
        output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
