"""Generation-2 tests for pending creation, promotion, and direction reuse."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_events import GraphDelta
from smoothnav.controller_logic import handle_frontier_reached, maybe_promote_pending
from smoothnav.controller_state import ControllerState
from smoothnav.planner import Strategy


class MockPlanner:
    def __init__(self, next_strategies):
        self.next_strategies = list(next_strategies)
        self.call_count = 0

    def plan(self, **kwargs):
        self.call_count += 1
        return self.next_strategies.pop(0)


class MockGraph:
    def __init__(self):
        self.nodes = []
        self.room_nodes = []

    def get_edges(self):
        return []


class PendingPromotionGen2Tests(unittest.TestCase):
    def setUp(self):
        self.applied = []
        self.graph = MockGraph()
        self.bev_map = SimpleNamespace()
        self.args = SimpleNamespace(map_size=720, controller_direction_reuse_limit=1)

    def _apply(self, strategy, graph, bev_map, args, global_goals):
        self.applied.append(strategy.target_region)
        return SimpleNamespace(changed=True, reason="goal_updated", selected_frontier=(5, 6))

    def test_more_specific_pending_strategy_promotes(self):
        state = ControllerState(
            current_strategy=Strategy("unexplored east", (1, 2), "explore"),
            pending_strategy=Strategy("object: chair", (5, 6), "specific object"),
        )

        promoted = maybe_promote_pending(
            controller_state=state,
            graph=self.graph,
            bev_map=self.bev_map,
            args=self.args,
            global_goals=[0, 0],
            apply_strategy_fn=self._apply,
        )

        self.assertTrue(promoted["promoted"])
        self.assertEqual(promoted["reason"], "pending_more_specific")
        self.assertEqual(state.current_strategy.target_region, "object: chair")

    def test_frontier_reached_promotes_pending_room(self):
        state = ControllerState(
            current_strategy=Strategy("kitchen", (1, 2), "current"),
            pending_strategy=Strategy("bedroom", (8, 9), "next room"),
            needs_initial_plan=False,
        )
        planner = MockPlanner([])

        outcome = handle_frontier_reached(
            controller_state=state,
            graph_delta=GraphDelta(frontier_reached=True),
            graph=self.graph,
            bev_map=self.bev_map,
            args=self.args,
            global_goals=[0, 0],
            high_planner=planner,
            goal_description="mug",
            agent_pos=(0, 0),
            apply_strategy_fn=self._apply,
            episode_id=0,
            step_idx=10,
            trace_writer=None,
        )

        self.assertTrue(outcome["handled"])
        self.assertTrue(outcome["pending_promoted"])
        self.assertEqual(outcome["pending_promotion_reason"], "frontier_reached_pending")
        self.assertEqual(state.current_strategy.target_region, "bedroom")

    def test_direction_reuse_limit_forces_replan(self):
        state = ControllerState(
            current_strategy=Strategy("unexplored north", (1, 2), "direction"),
            needs_initial_plan=False,
            direction_reuse_count=1,
        )
        planner = MockPlanner([Strategy("living room", (9, 9), "forced replan")])

        outcome = handle_frontier_reached(
            controller_state=state,
            graph_delta=GraphDelta(frontier_reached=True),
            graph=self.graph,
            bev_map=self.bev_map,
            args=self.args,
            global_goals=[0, 0],
            high_planner=planner,
            goal_description="mug",
            agent_pos=(0, 0),
            apply_strategy_fn=self._apply,
            episode_id=1,
            step_idx=20,
            trace_writer=None,
        )

        self.assertTrue(outcome["forced_replan_due_to_direction_reuse"])
        self.assertEqual(state.current_strategy.target_region, "living room")


if __name__ == "__main__":
    unittest.main()
