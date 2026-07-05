"""Tests for replaying target-anchor runtime policy on captured attempts."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.replay_target_anchor_attempt_policy import replay_capture  # noqa: E402


class ReplayTargetAnchorAttemptPolicyTests(unittest.TestCase):
    def test_replay_does_not_trigger_stall_replan_before_patience_four(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "step_traces").mkdir()

            def step(step_idx, trigger, distance):
                return {
                    "step_idx": step_idx,
                    "current_strategy": {"target_region": "unexplored target:tv"},
                    "pending_strategy": {},
                    "grounding_events": [
                        {
                            "trigger": trigger,
                            "bias_input": [0, 0],
                            "selected_frontier": [distance, 0],
                            "projected_goal": [distance, 0],
                            "reason": "goal_updated",
                        }
                    ],
                }

            steps = [
                step(10, "frontier_reached", 100),
                step(20, "target_anchor_local_map_refresh", 101),
                step(30, "target_anchor_local_map_refresh", 102),
                step(40, "target_anchor_local_map_refresh", 103),
                step(50, "frontier_reached", 90),
            ]

            trace_path = run_dir / "step_traces" / "episode_000000.jsonl"
            with trace_path.open("w") as f:
                for item in steps:
                    f.write(json.dumps(item) + "\n")

            capture = {
                "trace_path": str(trace_path),
                "first_target_step": 10,
                "target_branch_segments": [{"start_step": 10, "end_step": 50}],
                "branch_drop_events": [],
                "refresh_steps": [],
            }
            capture_path = run_dir / "target_branch_capture.json"
            capture_path.write_text(json.dumps(capture) + "\n")

            replay = replay_capture(capture_path)
            attempt = replay["attempts"][0]

            self.assertEqual(attempt["stall_replan_steps"], [])
            self.assertEqual(attempt["final_summary"]["stall_updates"], 0)
            self.assertEqual(attempt["final_summary"]["best_distance"], 90.0)


if __name__ == "__main__":
    unittest.main()
