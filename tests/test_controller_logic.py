"""Controller helper tests for Phase 1 orchestration."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_events import GraphDelta
from smoothnav.controller_logic import (
    build_graph_delta,
    empty_response_plan_is_more_specific,
    handle_frontier_reached,
    handle_grounding_failure,
    handle_out_of_local_window,
    handle_stuck_replan,
    is_target_search_anchor,
    maybe_call_monitor,
    maybe_promote_pending,
    target_anchor_attempt_summary,
    update_target_anchor_attempt,
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


class MockBevMap:
    def __init__(self):
        self.move_local_map_calls = 0

    def move_local_map(self):
        self.move_local_map_calls += 1


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

    def test_target_search_anchor_is_not_plain_direction(self):
        self.assertTrue(is_target_search_anchor("unexplored target:tv"))
        self.assertFalse(is_target_search_anchor("unexplored north"))
        self.assertFalse(is_target_search_anchor("object: tv"))

    def test_empty_response_target_fallback_can_replace_generic_direction(self):
        candidate = Strategy("unexplored target:tv", (10, 20), "target fallback")
        fallback = Strategy("unexplored south", (30, 40), "old direction")

        self.assertTrue(empty_response_plan_is_more_specific(candidate, fallback))

    def test_empty_response_generic_direction_does_not_churn_current_plan(self):
        candidate = Strategy("unexplored west", (10, 20), "generic fallback")
        fallback = Strategy("unexplored south", (30, 40), "old direction")

        self.assertFalse(empty_response_plan_is_more_specific(candidate, fallback))

    def test_target_anchor_attempt_tracks_improvement_and_evidence_reset(self):
        state = ControllerState()
        strategy = Strategy("unexplored target:tv", (10, 10), "target anchor", anchor_object="tv")
        args = SimpleNamespace(controller_target_anchor_min_improvement=1.0)

        update_target_anchor_attempt(
            state,
            strategy,
            SimpleNamespace(bias_input=(0, 0), selected_frontier=(100, 0)),
            "target_candidate_detected",
            10,
            args,
        )
        summary = target_anchor_attempt_summary(state)
        self.assertEqual(summary["attempt_id"], 1)
        self.assertEqual(summary["best_distance"], 100.0)
        self.assertEqual(summary["stall_updates"], 0)

        update_target_anchor_attempt(
            state,
            strategy,
            SimpleNamespace(bias_input=(0, 0), selected_frontier=(101, 0)),
            "target_anchor_local_map_refresh",
            20,
            args,
        )
        self.assertEqual(state.target_anchor_stall_updates, 1)

        update_target_anchor_attempt(
            state,
            strategy,
            SimpleNamespace(bias_input=(0, 0), selected_frontier=(99.5, 0)),
            "target_candidate_detected",
            30,
            args,
        )
        self.assertEqual(state.target_anchor_stall_updates, 0)
        self.assertEqual(state.target_anchor_best_distance, 100.0)

        update_target_anchor_attempt(
            state,
            strategy,
            SimpleNamespace(bias_input=(0, 0), selected_frontier=(95, 0)),
            "target_anchor_local_map_refresh",
            40,
            args,
        )
        self.assertEqual(state.target_anchor_best_distance, 95.0)
        self.assertEqual(state.target_anchor_last_improvement_step, 40)
        self.assertEqual(state.target_anchor_stall_updates, 0)

    def test_pending_target_anchor_promotes_over_direction(self):
        state = ControllerState(
            current_strategy=Strategy("unexplored east", (1, 2), "search"),
            pending_strategy=Strategy("unexplored target:tv", (5, 6), "target anchor"),
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
        self.assertEqual(state.current_strategy.target_region, "unexplored target:tv")
        self.assertIsNone(state.pending_strategy)

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

    def test_frontier_reached_holds_object_target_without_replan(self):
        planner = MockPlanner([Strategy("unused", (9, 9), "should not run")])
        state = ControllerState(
            current_strategy=Strategy("object: cabinet", (3, 4), "specific object"),
            needs_initial_plan=False,
        )
        delta = GraphDelta(frontier_reached=True)

        handled = handle_frontier_reached(
            controller_state=state,
            graph_delta=delta,
            graph=self.graph,
            bev_map=self.bev_map,
            args=SimpleNamespace(map_size=720, controller_direction_reuse_limit=1),
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
        self.assertFalse(handled["forced_replan_due_to_direction_reuse"])
        self.assertEqual(planner.call_count, 0)
        self.assertEqual(state.direction_reuse_count, 0)
        self.assertEqual(self.applied[-1][0], "object: cabinet")

    def test_frontier_reached_replans_stagnant_object_target(self):
        planner = MockPlanner([Strategy("bedroom", (9, 9), "switch away from cabinet")])
        state = ControllerState(
            current_strategy=Strategy("object: cabinet", (3, 4), "specific object"),
            needs_initial_plan=False,
            same_goal_hold_count=3,
        )
        delta = GraphDelta(frontier_reached=True)

        handled = handle_frontier_reached(
            controller_state=state,
            graph_delta=delta,
            graph=self.graph,
            bev_map=self.bev_map,
            args=SimpleNamespace(
                map_size=720,
                controller_direction_reuse_limit=1,
                controller_object_same_goal_replan_threshold=3,
            ),
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
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(state.current_strategy.target_region, "bedroom")
        self.assertIn("object: cabinet (stagnant)", state.explored_regions)

    def test_frontier_reached_discards_stale_pending_behind_target_anchor(self):
        planner = MockPlanner([Strategy("unused", (9, 9), "should not run")])
        state = ControllerState(
            current_strategy=Strategy("unexplored target:tv", (30, 40), "target anchor"),
            pending_strategy=Strategy("unexplored east", (5, 6), "old prefetch"),
            needs_initial_plan=False,
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
        self.assertFalse(handled["pending_promoted"])
        self.assertEqual(state.current_strategy.target_region, "unexplored target:tv")
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(self.applied[-1][0], "unexplored target:tv")

    def test_frontier_reached_replans_stalled_target_anchor(self):
        planner = MockPlanner([Strategy("living room", (9, 9), "alternative search")])
        state = ControllerState(
            current_strategy=Strategy("unexplored target:tv", (30, 40), "target anchor"),
            needs_initial_plan=False,
            target_anchor_attempt_id=2,
            target_anchor_target_region="unexplored target:tv",
            target_anchor_best_distance=270.0,
            target_anchor_last_distance=274.0,
            target_anchor_stall_updates=4,
            target_anchor_update_count=6,
            target_anchor_last_update_step=799,
            target_anchor_last_improvement_step=679,
            target_anchor_last_trigger="target_anchor_local_map_refresh",
        )
        delta = GraphDelta(frontier_reached=True)

        handled = handle_frontier_reached(
            controller_state=state,
            graph_delta=delta,
            graph=self.graph,
            bev_map=self.bev_map,
            args=SimpleNamespace(
                map_size=720,
                controller_direction_reuse_limit=1,
                controller_target_anchor_stall_patience_updates=4,
            ),
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
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(state.current_strategy.target_region, "living room")
        self.assertIn("unexplored target:tv (stalled)", state.explored_regions)
        self.assertEqual(state.target_anchor_target_region, "")

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

    def test_stuck_replan_decommits_object_target(self):
        planner = MockPlanner([Strategy("living room", (9, 9), "alternative")])
        state = ControllerState(
            current_strategy=Strategy("object: plant", (1, 2), "stay on object"),
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
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(state.current_strategy.target_region, "living room")
        self.assertIn("object: plant (stuck)", state.explored_regions)

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

    def test_same_goal_noop_does_not_trigger_grounding_failure_replan(self):
        state = ControllerState(
            current_strategy=Strategy("object: plant", (10, 20), "search plant"),
            needs_initial_plan=False,
            consecutive_grounding_noops=10,
            same_frontier_reuse_count=10,
        )
        planner = MockPlanner([Strategy("living room", (9, 9), "should not run")])

        outcome = handle_grounding_failure(
            controller_state=state,
            last_grounding_result=SimpleNamespace(
                changed=False,
                noop_reason="same_goal_as_prev",
                reason="same_goal_as_prev",
                noop_type="same_goal_as_prev",
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

        self.assertFalse(outcome["replanned"])
        self.assertEqual(planner.call_count, 0)

    def test_out_of_local_window_moves_local_map_and_retries(self):
        state = ControllerState(
            current_strategy=Strategy("unexplored target:plant", (10, 20), "search plant", anchor_object="plant"),
            needs_initial_plan=False,
        )
        bev_map = MockBevMap()
        results = iter([
            SimpleNamespace(
                local_projection_valid=True,
                changed=True,
                failure_code=None,
            )
        ])
        applied = []

        def apply_fn(strategy, graph, bev_map, args, global_goals):
            applied.append(strategy.target_region)
            global_goals[:] = [5, 6]
            return next(results)

        outcome = handle_out_of_local_window(
            controller_state=state,
            last_grounding_result=SimpleNamespace(
                failure_code="out_of_local_window",
                noop_type="out_of_local_window",
                primary_goal={"full_map_coord": [100, 200], "local_map_coord": [-1, 9]},
                selected_frontier=(100, 200),
            ),
            graph=self.graph,
            bev_map=bev_map,
            args=self.args,
            global_goals=[0, 0],
            apply_strategy_fn=apply_fn,
        )

        self.assertTrue(outcome["handled"])
        self.assertTrue(outcome["resolved"])
        self.assertFalse(outcome["deferred"])
        self.assertEqual(bev_map.move_local_map_calls, 1)
        self.assertEqual(applied, ["unexplored target:plant"])
        self.assertFalse(state.grounding_deferred)
        self.assertTrue(outcome["stale_local_goal_invalidated"])
        self.assertTrue(outcome["target_anchor_reprojected"])
        self.assertFalse(outcome["target_anchor_reprojection_failed"])

    def test_out_of_local_window_deferred_when_retry_stays_invalid(self):
        state = ControllerState(
            current_strategy=Strategy("unexplored target:plant", (10, 20), "search plant", anchor_object="plant"),
            needs_initial_plan=False,
        )
        bev_map = MockBevMap()

        def apply_fn(strategy, graph, bev_map, args, global_goals):
            return SimpleNamespace(
                local_projection_valid=False,
                changed=False,
                failure_code="out_of_local_window",
            )

        outcome = handle_out_of_local_window(
            controller_state=state,
            last_grounding_result=SimpleNamespace(
                failure_code="out_of_local_window",
                noop_type="out_of_local_window",
                primary_goal={"full_map_coord": [327, 304], "local_map_coord": [-3, 13]},
                selected_frontier=(327, 304),
            ),
            graph=self.graph,
            bev_map=bev_map,
            args=self.args,
            global_goals=[30, 82],
            apply_strategy_fn=apply_fn,
        )

        self.assertTrue(outcome["handled"])
        self.assertFalse(outcome["resolved"])
        self.assertTrue(outcome["deferred"])
        self.assertEqual(bev_map.move_local_map_calls, 1)
        self.assertTrue(state.grounding_deferred)
        self.assertEqual(state.grounding_deferred_reason, "out_of_local_window")
        self.assertEqual(state.grounding_deferred_full_goal, [327, 304])
        self.assertTrue(outcome["stale_local_goal_invalidated"])
        self.assertFalse(outcome["target_anchor_reprojected"])
        self.assertTrue(outcome["target_anchor_reprojection_failed"])


if __name__ == "__main__":
    unittest.main()
