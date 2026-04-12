"""Strategy grounding tests for Phase 1."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.strategy_grounding import apply_strategy


class MockGraph:
    def __init__(self, goal):
        self.goal = goal
        self.full_map = None
        self.full_pose = None
        self.received_bias = None

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

        apply_strategy(self.strategy, graph, self.bev_map, self.args, global_goals)

        self.assertEqual(graph.full_map, "full-map")
        self.assertEqual(graph.full_pose, "full-pose")
        self.assertEqual(graph.received_bias, (33, 44))
        self.assertEqual(global_goals, [8, 9])

    def test_ignores_missing_goal(self):
        graph = MockGraph(goal=None)
        global_goals = [7, 11]

        apply_strategy(self.strategy, graph, self.bev_map, self.args, global_goals)

        self.assertEqual(global_goals, [7, 11])

    def test_ignores_grounded_goal_outside_local_map(self):
        graph = MockGraph(goal=(500, 600))
        global_goals = [3, 4]

        apply_strategy(self.strategy, graph, self.bev_map, self.args, global_goals)

        self.assertEqual(global_goals, [3, 4])


if __name__ == "__main__":
    unittest.main()
