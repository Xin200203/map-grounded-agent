"""Generation-2 tests for structured strategy grounding outcomes."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.strategy_grounding import apply_strategy


class MockGraph:
    def __init__(self, goal, last_goal_debug=None):
        self.goal = goal
        self.last_goal_debug = last_goal_debug or {}

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_goal(self, goal=None):
        self.received_bias = goal
        return self.goal


class MockBoundary:
    def __getitem__(self, key):
        row_idx, col_idx = key
        assert row_idx == 0
        return [[10, 0, 20, 0]][row_idx][col_idx]


class StrategyGroundingGen2Tests(unittest.TestCase):
    def setUp(self):
        self.bev_map = SimpleNamespace(
            full_map="full-map",
            full_pose="full-pose",
            local_map_boundary=MockBoundary(),
        )
        self.args = SimpleNamespace(local_width=100, local_height=100)
        self.strategy = SimpleNamespace(bias_position=(33, 44))

    def test_successful_grounding_updates_goal(self):
        graph = MockGraph(goal=(18, 29))
        goals = [0, 0]

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, goals)

        self.assertTrue(result.success)
        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "goal_updated")
        self.assertEqual(result.projected_goal, (8, 9))
        self.assertEqual(goals, [8, 9])

    def test_missing_graph_goal_returns_structured_noop(self):
        graph = MockGraph(goal=None)

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, [7, 11])

        self.assertFalse(result.success)
        self.assertFalse(result.changed)
        self.assertEqual(result.noop_reason, "get_goal_none")

    def test_same_frontier_noop_is_reported(self):
        graph = MockGraph(
            goal=(13, 24),
            last_goal_debug={
                "selected_frontier": [13, 24],
                "selected_frontier_same_as_prev": True,
            },
        )
        goals = [3, 4]

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, goals)

        self.assertTrue(result.success)
        self.assertFalse(result.changed)
        self.assertEqual(result.noop_reason, "same_frontier_as_prev")
        self.assertEqual(goals, [3, 4])

    def test_projection_invalid_is_reported(self):
        graph = MockGraph(goal=(float("inf"), 20))

        result = apply_strategy(self.strategy, graph, self.bev_map, self.args, [1, 2])

        self.assertFalse(result.success)
        self.assertEqual(result.noop_reason, "projection_invalid")


if __name__ == "__main__":
    unittest.main()
