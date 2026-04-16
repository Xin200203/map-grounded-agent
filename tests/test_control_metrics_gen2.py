"""Generation-2 tests for controller/adoption metrics."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.control_metrics import compute_episode_control_metrics


class ControlMetricsGen2Tests(unittest.TestCase):
    def test_metrics_include_grounding_adoption_and_direction_reuse(self):
        steps = [
            {
                "step_idx": 0,
                "new_node_count": 1,
                "graph_delta": {"new_rooms": []},
                "planner_called": False,
                "monitor_decision": None,
                "pending_created": False,
                "pending_promoted": False,
                "pending_created_and_promoted_same_step": False,
                "goal_updated": False,
                "grounding_events": [],
                "executor_adoption_changed": False,
                "temp_goal_override": False,
                "stuck_goal_override": False,
                "global_goal_override": False,
                "visible_target_override": False,
                "direction_reuse_count": 0,
            },
            {
                "step_idx": 1,
                "new_node_count": 0,
                "graph_delta": {"new_rooms": ["kitchen"]},
                "planner_called": True,
                "monitor_decision": "PREFETCH",
                "pending_created": True,
                "pending_promoted": False,
                "pending_created_and_promoted_same_step": False,
                "goal_updated": True,
                "grounding_events": [
                    {
                        "changed": False,
                        "noop_type": "get_goal_none",
                        "graph_no_goal_reason": "no_frontiers",
                        "frontier_filter_fallback_mode": "raw_frontier_fallback",
                        "candidate_distance_fallback_mode": "",
                        "selected_frontier_same_as_prev": True,
                    }
                ],
                "executor_adoption_changed": False,
                "temp_goal_override": True,
                "stuck_goal_override": False,
                "global_goal_override": False,
                "visible_target_override": False,
                "direction_reuse_count": 1,
            },
            {
                "step_idx": 2,
                "new_node_count": 0,
                "graph_delta": {"new_rooms": []},
                "planner_called": False,
                "monitor_decision": None,
                "pending_created": True,
                "pending_promoted": True,
                "pending_created_and_promoted_same_step": True,
                "goal_updated": False,
                "grounding_events": [],
                "executor_adoption_changed": True,
                "temp_goal_override": False,
                "stuck_goal_override": False,
                "global_goal_override": True,
                "visible_target_override": False,
                "direction_reuse_count": 1,
            },
        ]

        metrics = compute_episode_control_metrics(steps)

        self.assertGreater(metrics["grounding_noop_rate"], 0.0)
        self.assertEqual(metrics["pending_created_count"], 2)
        self.assertEqual(metrics["pending_promoted_count"], 1)
        self.assertEqual(metrics["pending_created_and_promoted_count"], 1)
        self.assertEqual(metrics["direction_reuse_count"], 1)
        self.assertEqual(metrics["executor_adoption_delay_steps"], 1.0)
        self.assertEqual(metrics["control_ack_delay_steps"], 1.0)
        self.assertEqual(metrics["grounding_noop_reason_counts"]["get_goal_none"], 1)
        self.assertEqual(metrics["grounding_no_goal_reason_counts"]["no_frontiers"], 1)
        self.assertEqual(
            metrics["grounding_frontier_fallback_mode_counts"]["raw_frontier_fallback"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
