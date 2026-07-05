"""Tests for pending-fix counterfactual replay on captured target branches."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.replay_pending_fix_counterfactual import replay_drop_event  # noqa: E402


class ReplayPendingFixCounterfactualTests(unittest.TestCase):
    def test_replay_keeps_target_anchor_and_clears_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "step_traces").mkdir()

            steps = [
                {
                    "step_idx": 609,
                    "current_strategy": {
                        "target_region": "unexplored target:tv",
                        "bias_position": [180, 299],
                        "reasoning": "target anchor active",
                        "explored_regions": [],
                        "anchor_object": "tv",
                    },
                    "pending_strategy": {
                        "target_region": "unexplored east",
                        "bias_position": [30, 40],
                        "reasoning": "stale direction prefetch",
                        "explored_regions": [],
                        "anchor_object": "",
                    },
                    "direction_reuse_count": 0,
                },
                {
                    "step_idx": 610,
                    "current_strategy": {
                        "target_region": "unexplored east",
                        "bias_position": [30, 40],
                        "reasoning": "historical buggy promotion",
                        "explored_regions": [],
                        "anchor_object": "",
                    },
                    "pending_strategy": {},
                },
            ]
            with (run_dir / "step_traces" / "episode_000000.jsonl").open("w") as f:
                for item in steps:
                    f.write(json.dumps(item) + "\n")

            capture = {
                "trace_path": str(run_dir / "step_traces" / "episode_000000.jsonl"),
                "branch_drop_events": [
                    {
                        "step_idx": 610,
                        "from_region": "unexplored target:tv",
                        "to_region": "unexplored east",
                        "planner_reasons": ["frontier_reached"],
                        "events": ["frontier_near", "frontier_reached"],
                        "pending_region_previous_step": "unexplored east",
                        "goal_after": [56, 76],
                    }
                ],
            }
            capture_path = run_dir / "target_branch_capture.json"
            capture_path.write_text(json.dumps(capture) + "\n")

            result = replay_drop_event(capture_path)

            self.assertEqual(
                result["historical_before"]["current_target_region"],
                "unexplored target:tv",
            )
            self.assertEqual(
                result["historical_after"]["current_target_region"],
                "unexplored east",
            )
            self.assertEqual(
                result["counterfactual_outcome"]["current_target_region_after"],
                "unexplored target:tv",
            )
            self.assertIsNone(
                result["counterfactual_outcome"]["pending_target_region_after"]
            )
            self.assertFalse(result["counterfactual_outcome"]["pending_promoted"])
            self.assertEqual(result["counterfactual_outcome"]["planner_call_count"], 0)
            self.assertTrue(result["verdict"]["bug_fixed_on_captured_context"])


if __name__ == "__main__":
    unittest.main()
