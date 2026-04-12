"""Controller helper tests for Phase 1 orchestration."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_events import GraphDelta
from smoothnav.controller_logic import (
    handle_frontier_reached,
    handle_stuck_replan,
    maybe_promote_pending,
)
from smoothnav.controller_state import ControllerState
from smoothnav.planner import Strategy


class MockGraph:
    def __init__(self):
        self.nodes = []
        self.room_nodes = []

    def get_edges(self):
        return []


class MockPlanner:
    def __init__(self, next_strategies):
        self.next_strategies = list(next_strategies)
        self.call_count = 0

    def plan(self, **kwargs):
        self.call_count += 1
        return self.next_strategies.pop(0)


class ControllerLogicTests(unittest.TestCase):
    def setUp(self):
        self.graph = MockGraph()
        self.bev_map = SimpleNamespace()
        self.args = SimpleNamespace(map_size=720)
        self.goal_description = "mug"
        self.applied = []

    def _apply(self, strategy, graph, bev_map, args, global_goals):
        self.applied.append((strategy.target_region, list(global_goals)))

    def test_pending_room_strategy_can_be_promoted_early(self):
        state = ControllerState(
            current_strategy=Strategy("unexplored east", (1, 2), "explore"),
            pending_strategy=Strategy("kitchen", (5, 6), "specific room"),
        )
        global_goals = [10, 20]

        promoted = maybe_promote_pending(
            controller_state=state,
            graph=self.graph,
            bev_map=self.bev_map,
            args=self.args,
            global_goals=global_goals,
            apply_strategy_fn=self._apply,
        )

        self.assertTrue(promoted)
        self.assertEqual(state.current_strategy.target_region, "kitchen")
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(self.applied[0][0], "kitchen")

    def test_frontier_reached_uses_pending_strategy_and_marks_room_explored(self):
        state = ControllerState(
            current_strategy=Strategy("bathroom", (1, 2), "current room"),
            pending_strategy=Strategy("kitchen", (5, 6), "next room"),
            needs_initial_plan=False,
        )
        delta = GraphDelta(frontier_reached=True)

        handled = handle_frontier_reached(
            controller_state=state,
            graph_delta=delta,
            graph=self.graph,
            bev_map=self.bev_map,
            args=self.args,
            global_goals=[0, 0],
            high_planner=MockPlanner([]),
            goal_description=self.goal_description,
            agent_pos=(0, 0),
            apply_strategy_fn=self._apply,
            episode_id=1,
            step_idx=2,
            trace_writer=None,
        )

        self.assertTrue(handled)
        self.assertEqual(state.current_strategy.target_region, "kitchen")
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(state.explored_regions, ["bathroom"])
        self.assertEqual(self.applied[0][0], "kitchen")

    def test_stuck_replan_clears_pending_and_tracks_stuck_room(self):
        next_strategy = Strategy("living room", (9, 9), "alternative route")
        planner = MockPlanner([next_strategy])
        state = ControllerState(
            current_strategy=Strategy("kitchen", (1, 2), "search kitchen"),
            pending_strategy=Strategy("bathroom", (3, 4), "prefetched"),
            needs_initial_plan=False,
            no_progress_steps=8,
        )
        delta = GraphDelta(stuck=True)

        handled = handle_stuck_replan(
            controller_state=state,
            graph_delta=delta,
            graph=self.graph,
            bev_map=self.bev_map,
            args=self.args,
            global_goals=[0, 0],
            high_planner=planner,
            goal_description=self.goal_description,
            agent_pos=(12, 34),
            apply_strategy_fn=self._apply,
            episode_id=4,
            step_idx=10,
            trace_writer=None,
        )

        self.assertTrue(handled)
        self.assertEqual(state.current_strategy.target_region, "living room")
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(state.no_progress_steps, 0)
        self.assertEqual(state.explored_regions, ["kitchen (stuck)"])
        self.assertEqual(self.applied[0][0], "living room")


if __name__ == "__main__":
    unittest.main()
