"""Strategy grounding tests for Phase 1."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.strategy_grounding import apply_strategy


class MockGraph:
    def __init__(self, goal, last_goal_debug=None):
        self.goal = goal
        self.full_map = None
        self.full_pose = None
        self.received_bias = None
        self.last_goal_debug = last_goal_debug or {}

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_goal(self, goal=None):
        self.received_bias = goal
        return self.goal


class MockBoundary:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        row_idx, col_idx = key
        return self.values[row_idx][col_idx]


class ApplyStrategyTests(unittest.TestCase):
    def setUp(self):
        self.bev_map = SimpleNamespace(
            full_map="full-map",
            full_pose="full-pose",
            local_map_boundary=MockBoundary([[10, 0, 20, 0]]),
        )
        self.args = SimpleNamespace(local_width=100, local_height=100)
        self.strategy = SimpleNamespace(bias_position=(33, 44))

    def test_updates_local_goal_when_grounded_goal_is_in_bounds(self):
        graph = MockGraph(goal=(18, 29))
        global_goals = [0, 0]

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, global_goals)

        self.assertEqual(graph.full_map, "full-map")
        self.assertEqual(graph.full_pose, "full-pose")
        self.assertEqual(graph.received_bias, (33, 44))
        self.assertEqual(global_goals, [8, 9])
        self.assertTrue(result.success)
        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "goal_updated")
        self.assertEqual(result.projected_goal, (8, 9))
        self.assertTrue(result.local_projection_valid)

    def test_ignores_missing_goal(self):
        graph = MockGraph(goal=None)
        global_goals = [7, 11]

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, global_goals)

        self.assertEqual(global_goals, [7, 11])
        self.assertFalse(result.success)
        self.assertEqual(result.noop_reason, "get_goal_none")
        self.assertFalse(result.local_projection_valid)

    def test_ignores_grounded_goal_outside_local_map(self):
        graph = MockGraph(goal=(500, 600))
        global_goals = [3, 4]

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, global_goals)

        self.assertEqual(global_goals, [3, 4])
        self.assertFalse(result.success)
        self.assertEqual(result.noop_reason, "out_of_local_window")
        self.assertFalse(result.local_projection_valid)

    def test_marks_same_frontier_as_grounding_noop(self):
        graph = MockGraph(
            goal=(13, 24),
            last_goal_debug={
                "selected_frontier": [13, 24],
                "selected_frontier_same_as_previous": True,
                "selected_frontier_same_as_prev": True,
                "selected_frontier_score": 9.7,
                "topk_frontiers": [],
            },
        )
        global_goals = [3, 4]

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, global_goals)

        self.assertEqual(global_goals, [3, 4])
        self.assertTrue(result.success)
        self.assertFalse(result.changed)
        self.assertEqual(result.noop_reason, "same_frontier_as_prev")


if __name__ == "__main__":
    unittest.main()
