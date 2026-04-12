"""Phase 2 tests for trace-derived control metrics."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.control_metrics import (
    compute_episode_control_metrics,
    compute_run_control_metrics,
)


class ControlMetricsTests(unittest.TestCase):
    def test_compute_episode_metrics_tracks_delays_and_promotions(self):
        steps = [
            {
                "step_idx": 0,
                "new_node_count": 1,
                "graph_delta": {"new_rooms": []},
                "planner_called": False,
                "monitor_decision": None,
                "strategy_switched": False,
                "pending_created": False,
                "pending_promoted": False,
                "goal_updated": False,
            },
            {
                "step_idx": 1,
                "new_node_count": 0,
                "graph_delta": {"new_rooms": []},
                "planner_called": False,
                "monitor_decision": None,
                "strategy_switched": False,
                "pending_created": False,
                "pending_promoted": False,
                "goal_updated": False,
            },
            {
                "step_idx": 2,
                "new_node_count": 0,
                "graph_delta": {"new_rooms": []},
                "planner_called": True,
                "monitor_decision": "ESCALATE",
                "strategy_switched": True,
                "pending_created": False,
                "pending_promoted": False,
                "goal_updated": True,
            },
            {
                "step_idx": 3,
                "new_node_count": 1,
                "graph_delta": {"new_rooms": ["kitchen"]},
                "planner_called": True,
                "monitor_decision": "PREFETCH",
                "strategy_switched": False,
                "pending_created": True,
                "pending_promoted": False,
                "goal_updated": False,
            },
            {
                "step_idx": 4,
                "new_node_count": 0,
                "graph_delta": {"new_rooms": []},
                "planner_called": False,
                "monitor_decision": None,
                "strategy_switched": True,
                "pending_created": False,
                "pending_promoted": True,
                "goal_updated": True,
            },
        ]

        metrics = compute_episode_control_metrics(steps)

        self.assertEqual(metrics["strategy_switch_count"], 2)
        self.assertEqual(metrics["decision_delay_steps"], 1.0)
        self.assertEqual(metrics["goal_update_delay_steps"], 0.5)
        self.assertEqual(metrics["pending_promotion_rate"], 1.0)

    def test_compute_run_metrics_reads_trace_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = os.path.join(tmpdir, "step_traces")
            os.makedirs(trace_dir, exist_ok=True)
            payload = [
                {
                    "step_idx": 0,
                    "new_node_count": 1,
                    "graph_delta": {"new_rooms": []},
                    "planner_called": True,
                    "monitor_decision": "ESCALATE",
                    "strategy_switched": True,
                    "pending_created": False,
                    "pending_promoted": False,
                    "goal_updated": True,
                }
            ]
            with open(
                os.path.join(trace_dir, "episode_000001.jsonl"),
                "w",
                encoding="utf-8",
            ) as handle:
                for row in payload:
                    handle.write(json.dumps(row) + "\n")

            metrics = compute_run_control_metrics(tmpdir)

            self.assertEqual(metrics["strategy_switch_count"], 1.0)
            self.assertEqual(metrics["decision_delay_steps"], 0.0)
            self.assertEqual(metrics["goal_update_delay_steps"], 0.0)


if __name__ == "__main__":
    unittest.main()
