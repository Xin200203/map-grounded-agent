#!/usr/bin/env python3
"""Replay target-anchor attempt state transitions on a captured branch.

This uses the current runtime helper (`update_target_anchor_attempt`) and checks
the exact frontier-reached decision points where a stalled target-anchor branch
would trigger a replan under the current patience rule.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyze_target_anchor_attempt import analyze_capture  # noqa: E402
from smoothnav.controller_logic import (  # noqa: E402
    target_anchor_attempt_summary,
    update_target_anchor_attempt,
)
from smoothnav.controller_state import ControllerState  # noqa: E402
from smoothnav.planner import Strategy  # noqa: E402


def replay_capture(capture_path: Path) -> Dict[str, Any]:
    analysis = analyze_capture(Path(capture_path))
    args = SimpleNamespace(
        controller_target_anchor_min_improvement=1.0,
        controller_target_anchor_stall_patience_updates=4,
    )

    attempts = []
    for attempt in analysis.get("attempts", []):
        state = ControllerState()
        transitions = []
        stall_replan_steps = []
        updates = list(attempt.get("updates") or [])
        for update in updates:
            strategy = Strategy(
                target_region=str(update.get("target_region") or ""),
                bias_position=tuple(update.get("bias_input") or []) or None,
                reasoning="replay",
            )
            pre_summary = target_anchor_attempt_summary(state)
            if (
                str(update.get("trigger") or "") == "frontier_reached"
                and pre_summary["active"]
                and pre_summary["stall_updates"]
                >= int(getattr(args, "controller_target_anchor_stall_patience_updates", 4) or 4)
            ):
                stall_replan_steps.append(
                    {
                        "step_idx": int(update["step_idx"]),
                        "pre_summary": pre_summary,
                    }
                )
                break

            result = SimpleNamespace(
                bias_input=tuple(update.get("bias_input") or []) or None,
                selected_frontier=tuple(update.get("selected_frontier") or []) or None,
            )
            post_summary = update_target_anchor_attempt(
                state,
                strategy,
                result,
                str(update.get("trigger") or ""),
                int(update["step_idx"]),
                args,
            )
            transitions.append(
                {
                    "step_idx": int(update["step_idx"]),
                    "trigger": str(update.get("trigger") or ""),
                    "pre_summary": pre_summary,
                    "post_summary": post_summary,
                    "distance": update.get("frontier_to_anchor_distance"),
                }
            )

        attempts.append(
            {
                "attempt_id": attempt.get("attempt_id"),
                "window": [attempt.get("start_step"), attempt.get("end_step")],
                "transition_count": len(transitions),
                "stall_replan_steps": stall_replan_steps,
                "final_summary": target_anchor_attempt_summary(state),
                "transitions": transitions,
            }
        )

    return {
        "source_capture": str(capture_path),
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="Path to target_branch_capture.json or containing run dir")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write result to <capture_dir>/target_anchor_attempt_replay.json",
    )
    args = parser.parse_args()

    result = replay_capture(Path(args.capture))
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.write:
        capture = Path(args.capture)
        if capture.is_dir():
            capture = capture / "target_branch_capture.json"
        output = capture.parent / "target_anchor_attempt_replay.json"
        output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
