"""Tests for target-branch capture utility."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.capture_target_branch import capture_run_dir  # noqa: E402


class CaptureTargetBranchTests(unittest.TestCase):
    def test_capture_extracts_segments_refresh_and_drop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "step_traces").mkdir()
            (run_dir / "planner_calls").mkdir()
            (run_dir / "monitor_calls").mkdir()

            steps = [
                {
                    "step_idx": 560,
                    "graph_delta": {"event_types": ["no_progress"], "dist_to_goal": 20.0},
                    "current_strategy": {"target_region": "unexplored east"},
                    "pending_strategy": {"target_region": "unexplored east"},
                    "goal_after": [1, 2],
                    "planner_called": False,
                    "monitor_called": False,
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 564,
                    "graph_delta": {
                        "event_types": ["target_candidate_detected"],
                        "dist_to_goal": 75.0,
                        "target_candidate_details": [{"caption": "tv"}],
                    },
                    "current_strategy": {"target_region": "unexplored target:tv"},
                    "pending_strategy": {"target_region": "unexplored east"},
                    "goal_after": [195, 237],
                    "planner_called": True,
                    "planner_reasons": ["target_candidate_detected"],
                    "monitor_called": True,
                    "monitor_decision": "CONTINUE",
                    "grounding_events": [{"trigger": "target_candidate_detected", "changed": True}],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 599,
                    "graph_delta": {"event_types": ["no_progress"], "dist_to_goal": 12.0},
                    "current_strategy": {"target_region": "unexplored target:tv"},
                    "pending_strategy": {"target_region": "unexplored east"},
                    "goal_after": [97, 127],
                    "planner_called": False,
                    "monitor_called": False,
                    "grounding_events": [
                        {
                            "trigger": "target_anchor_local_map_refresh",
                            "changed": True,
                            "reason": "goal_updated",
                            "selected_frontier": [462, 288],
                            "projected_goal": [97, 127],
                        }
                    ],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 610,
                    "graph_delta": {"event_types": ["frontier_reached"], "dist_to_goal": 5.0},
                    "current_strategy": {"target_region": "unexplored east"},
                    "pending_strategy": {},
                    "goal_after": [56, 76],
                    "planner_called": True,
                    "planner_reasons": ["frontier_reached"],
                    "monitor_called": False,
                    "grounding_events": [{"trigger": "frontier_reached", "changed": True}],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 620,
                    "graph_delta": {"event_types": [], "dist_to_goal": 30.0},
                    "current_strategy": {"target_region": "unexplored east"},
                    "pending_strategy": {},
                    "goal_after": [56, 76],
                    "planner_called": False,
                    "monitor_called": False,
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"},
                },
            ]
            planner_calls = [
                {"step_idx": 564, "choice_type": "object", "choice_id": "tv"},
                {"step_idx": 610, "choice_type": "direction", "choice_id": "east"},
            ]
            monitor_calls = [
                {"step_idx": 564, "action": "CONTINUE", "reason": "target seen"},
            ]

            with (run_dir / "step_traces" / "episode_000000.jsonl").open("w") as f:
                for item in steps:
                    f.write(json.dumps(item) + "\n")
            with (run_dir / "planner_calls" / "episode_000000.jsonl").open("w") as f:
                for item in planner_calls:
                    f.write(json.dumps(item) + "\n")
            with (run_dir / "monitor_calls" / "episode_000000.jsonl").open("w") as f:
                for item in monitor_calls:
                    f.write(json.dumps(item) + "\n")

            capture = capture_run_dir(run_dir)

            self.assertEqual(capture["first_target_step"], 564)
            self.assertEqual(capture["first_target_captions"], ["tv"])
            self.assertEqual(capture["first_target_anchor_step"], 564)
            self.assertEqual(capture["first_target_anchor_region"], "unexplored target:tv")
            self.assertEqual(
                capture["target_branch_segments"],
                [{"start_step": 564, "end_step": 599}],
            )
            self.assertEqual(capture["refresh_steps"][0]["step_idx"], 599)
            self.assertEqual(capture["branch_drop_events"][0]["step_idx"], 610)
            self.assertEqual(
                capture["branch_drop_events"][0]["pending_region_previous_step"],
                "unexplored east",
            )
            self.assertEqual(len(capture["planner_calls"]), 2)
            self.assertEqual(len(capture["monitor_calls"]), 1)
            self.assertEqual(capture["terminal_outcome"], "FAILURE_NO_PROGRESS_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
