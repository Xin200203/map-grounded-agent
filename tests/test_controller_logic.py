"""Controller helper tests for Phase 1 orchestration."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_events import GraphDelta
from smoothnav.controller_logic import (
    build_graph_delta,
    handle_frontier_reached,
    handle_grounding_failure,
    handle_stuck_replan,
    maybe_call_monitor,
    maybe_promote_pending,
    update_grounding_failure_state,
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


class MockMonitor:
    def __init__(self):
        self.call_count = 0

    def evaluate(self, **kwargs):
        self.call_count += 1
        return SimpleNamespace(action=None, reason="called")


class ControllerLogicTests(unittest.TestCase):
    def setUp(self):
        self.graph = MockGraph()
        self.bev_map = SimpleNamespace()
        self.args = SimpleNamespace(map_size=720)
        self.goal_description = "mug"
        self.applied = []

    def _apply(self, strategy, graph, bev_map, args, global_goals):
        self.applied.append((strategy.target_region, list(global_goals)))
        return SimpleNamespace(
            changed=False,
            noop_reason="same_frontier_as_prev",
            reason="same_frontier_as_prev",
            selected_frontier=(5, 6),
        )

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

        self.assertTrue(promoted["promoted"])
        self.assertEqual(promoted["reason"], "pending_more_specific")
        self.assertEqual(state.current_strategy.target_region, "kitchen")
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(self.applied[0][0], "kitchen")

    def test_pending_object_strategy_can_promote_over_room(self):
        state = ControllerState(
            current_strategy=Strategy("kitchen", (1, 2), "search room"),
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
        self.assertEqual(state.current_strategy.target_region, "object: chair")

    def test_monitor_can_trigger_on_frontier_near_without_new_nodes(self):
        state = ControllerState(
            current_strategy=Strategy("unexplored north", (1, 2), "explore"),
            needs_initial_plan=False,
        )
        delta = GraphDelta(frontier_near=True, dist_to_goal=7.0)
        monitor = MockMonitor()

        called, _, trigger_event_types = maybe_call_monitor(
            low_agent=monitor,
            controller_state=state,
            graph_delta=delta,
            graph=self.graph,
            episode_id=1,
            step_idx=3,
            trace_writer=None,
        )

        self.assertTrue(called)
        self.assertEqual(monitor.call_count, 1)
        self.assertEqual(trigger_event_types, ["frontier_near"])

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

        self.assertTrue(handled["handled"])
        self.assertTrue(handled["pending_promoted"])
        self.assertEqual(handled["pending_promotion_reason"], "frontier_reached_pending")
        self.assertEqual(state.current_strategy.target_region, "kitchen")
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(state.explored_regions, ["bathroom"])
        self.assertEqual(self.applied[0][0], "kitchen")

    def test_direction_reuse_limit_forces_replan(self):
        planner = MockPlanner(
            [Strategy("living room", (9, 9), "forced replan after reuse limit")]
        )
        state = ControllerState(
            current_strategy=Strategy("unexplored north", (1, 2), "direction"),
            needs_initial_plan=False,
            direction_reuse_count=1,
        )
        delta = GraphDelta(frontier_reached=True)
        args = SimpleNamespace(map_size=720, controller_direction_reuse_limit=1)

        handled = handle_frontier_reached(
            controller_state=state,
            graph_delta=delta,
            graph=self.graph,
            bev_map=self.bev_map,
            args=args,
            global_goals=[0, 0],
            high_planner=planner,
            goal_description=self.goal_description,
            agent_pos=(0, 0),
            apply_strategy_fn=self._apply,
            episode_id=2,
            step_idx=7,
            trace_writer=None,
        )

        self.assertTrue(handled["handled"])
        self.assertTrue(handled["forced_replan_due_to_direction_reuse"])
        self.assertEqual(state.current_strategy.target_region, "living room")
        self.assertEqual(state.direction_reuse_count, 0)
        self.assertEqual(self.applied[-1][0], "living room")

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

    def test_build_graph_delta_detects_caption_change(self):
        node = SimpleNamespace(caption="chair")
        self.graph.nodes = [node]
        state = ControllerState(prev_node_captions={0: "table"})

        delta = build_graph_delta(
            graph=self.graph,
            controller_state=state,
            frontier_near=False,
            frontier_reached=False,
            no_progress=False,
            stuck=False,
            dist_to_goal=5.0,
        )

        self.assertTrue(delta.node_caption_changed)
        self.assertIn("node_caption_changed", delta.event_types)

    def test_grounding_failure_replan_triggers_after_two_noops(self):
        state = ControllerState(
            current_strategy=Strategy("kitchen", (1, 2), "search room"),
            needs_initial_plan=False,
        )
        update_grounding_failure_state(
            state,
            SimpleNamespace(changed=False, selected_frontier=(5, 6)),
        )
        update_grounding_failure_state(
            state,
            SimpleNamespace(changed=False, selected_frontier=(5, 6)),
        )
        planner = MockPlanner([Strategy("living room", (9, 9), "retry")])

        outcome = handle_grounding_failure(
            controller_state=state,
            last_grounding_result=SimpleNamespace(
                changed=False,
                noop_reason="same_frontier_as_prev",
                reason="same_frontier_as_prev",
            ),
            graph=self.graph,
            bev_map=self.bev_map,
            args=SimpleNamespace(
                map_size=720,
                controller_grounding_noop_replan_threshold=2,
                controller_same_frontier_reuse_threshold=2,
            ),
            global_goals=[0, 0],
            high_planner=planner,
            goal_description=self.goal_description,
            agent_pos=(0, 0),
            apply_strategy_fn=self._apply,
            episode_id=1,
            step_idx=5,
            trace_writer=None,
        )

        self.assertTrue(outcome["replanned"])
        self.assertTrue(outcome["forced_replan_due_to_grounding_failure"])
        self.assertEqual(state.current_strategy.target_region, "living room")


if __name__ == "__main__":
    unittest.main()
