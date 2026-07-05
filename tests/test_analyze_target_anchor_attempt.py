"""Tests for target-anchor attempt analysis and stall-policy sweep."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyze_target_anchor_attempt import analyze_capture  # noqa: E402


class AnalyzeTargetAnchorAttemptTests(unittest.TestCase):
    def test_policy_sweep_flags_false_early_decommit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "step_traces").mkdir()

            def target_step(step_idx, trigger, distance, current_region="unexplored target:tv"):
                return {
                    "step_idx": step_idx,
                    "current_strategy": {"target_region": current_region},
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
                target_step(10, "target_candidate_detected", 100),
                target_step(20, "target_anchor_local_map_refresh", 80),
                target_step(30, "target_anchor_local_map_refresh", 81),
                target_step(40, "target_anchor_local_map_refresh", 82),
                target_step(50, "target_anchor_local_map_refresh", 83),
                target_step(60, "target_candidate_detected", 70),
                target_step(70, "target_anchor_local_map_refresh", 74),
                target_step(80, "target_anchor_local_map_refresh", 75),
                target_step(90, "target_anchor_local_map_refresh", 76),
                target_step(100, "target_anchor_local_map_refresh", 77),
                {
                    "step_idx": 101,
                    "current_strategy": {"target_region": "unexplored east"},
                    "pending_strategy": {},
                    "grounding_events": [],
                },
            ]

            trace_path = run_dir / "step_traces" / "episode_000000.jsonl"
            with trace_path.open("w") as f:
                for item in steps:
                    f.write(json.dumps(item) + "\n")

            capture = {
                "trace_path": str(trace_path),
                "first_target_step": 10,
                "target_branch_segments": [{"start_step": 10, "end_step": 100}],
                "branch_drop_events": [],
                "refresh_steps": [],
            }
            capture_path = run_dir / "target_branch_capture.json"
            capture_path.write_text(json.dumps(capture) + "\n")

            result = analyze_capture(capture_path)
            attempt = result["attempts"][0]
            sweep = {
                (item["patience_updates"], item["min_improvement"]): item
                for item in attempt["policy_sweep"]
            }

            self.assertEqual(attempt["initial_distance"], 100.0)
            self.assertEqual(attempt["best_distance"], 70.0)
            self.assertEqual(attempt["final_distance"], 77.0)

            patience3 = sweep[(3, 1.0)]
            self.assertEqual(patience3["decommit_step"], 50)
            self.assertTrue(patience3["would_false_early_decommit"])
            self.assertEqual(
                patience3["later_better_evidence"]["step_idx"],
                60,
            )

            patience4 = sweep[(4, 1.0)]
            self.assertEqual(patience4["decommit_step"], 100)
            self.assertFalse(patience4["would_false_early_decommit"])


if __name__ == "__main__":
    unittest.main()
