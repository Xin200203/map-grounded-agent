"""Target-commitment persistence: path-level failures must not evict targets."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_events import GraphDelta
from smoothnav.controller_logic import handle_frontier_reached, handle_stuck_replan
from smoothnav.controller_state import ControllerState
from smoothnav.planner import Strategy


class MockPlanner:
    def __init__(self, strategy):
        self.strategy = strategy
        self.call_count = 0

    def plan(self, **kwargs):
        self.call_count += 1
        return self.strategy


def make_args(persistence: bool):
    return SimpleNamespace(
        controller_target_commitment_persistence=persistence,
        controller_target_commit_max_approach_retries=3,
        controller_target_commit_max_stuck_strikes=2,
        controller_direction_reuse_limit=1,
        controller_object_same_goal_replan_threshold=3,
        controller_same_frontier_reuse_threshold=2,
        controller_target_anchor_stall_patience_updates=4,
        map_size=720,
    )


def make_delta(**overrides):
    delta = GraphDelta()
    delta.frontier_reached = overrides.get("frontier_reached", False)
    delta.stuck = overrides.get("stuck", False)
    return delta


def run_frontier(state, args, planner):
    applied = []
    outcome = handle_frontier_reached(
        controller_state=state,
        graph_delta=make_delta(frontier_reached=True),
        graph=SimpleNamespace(nodes=[], room_nodes=[]),
        bev_map=SimpleNamespace(),
        args=args,
        global_goals=[0, 0],
        high_planner=planner,
        goal_description="a black tv",
        agent_pos=(0, 0),
        apply_strategy_fn=lambda strategy, *rest: applied.append(strategy),
        episode_id=0,
        step_idx=10,
    )
    return outcome, applied


class PendingGuardTests(unittest.TestCase):
    def _state_with_pending(self):
        state = ControllerState()
        state.needs_initial_plan = False
        state.current_strategy = Strategy(
            target_region="object: tv", bias_position=(5, 5), reasoning="commit"
        )
        state.pending_strategy = Strategy(
            target_region="bedroom", bias_position=(2, 2), reasoning="stale"
        )
        return state

    def test_persistence_discards_stale_pending_behind_object_commit(self):
        state = self._state_with_pending()
        outcome, _ = run_frontier(state, make_args(True), MockPlanner(None))
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(state.current_strategy.target_region, "object: tv")
        self.assertFalse(outcome["pending_promoted"])

    def test_legacy_applies_pending_over_object_commit(self):
        state = self._state_with_pending()
        outcome, _ = run_frontier(state, make_args(False), MockPlanner(None))
        self.assertEqual(state.current_strategy.target_region, "bedroom")
        self.assertTrue(outcome["pending_promoted"])


class StagnationRetryTests(unittest.TestCase):
    def _stagnant_state(self):
        state = ControllerState()
        state.needs_initial_plan = False
        state.current_strategy = Strategy(
            target_region="object: tv", bias_position=(5, 5), reasoning="commit"
        )
        state.same_goal_hold_count = 3
        return state

    def test_persistence_converts_to_search_anchor_without_poisoning(self):
        state = self._stagnant_state()
        planner = MockPlanner(None)
        outcome, applied = run_frontier(state, make_args(True), planner)
        self.assertEqual(
            state.current_strategy.target_region, "unexplored target:tv"
        )
        self.assertEqual(state.target_commit_approach_retries, 1)
        self.assertEqual(planner.call_count, 0)
        self.assertEqual(state.explored_regions, [])
        self.assertTrue(outcome["handled"])
        self.assertEqual(len(applied), 1)

    def test_retry_budget_exhaustion_falls_back_to_legacy_replan(self):
        state = self._stagnant_state()
        state.target_commit_label = "tv"
        state.target_commit_approach_retries = 3
        planner = MockPlanner(
            Strategy(target_region="unexplored east", bias_position=None, reasoning="alt")
        )
        run_frontier(state, make_args(True), planner)
        self.assertEqual(planner.call_count, 1)
        self.assertIn("object: tv (stagnant)", state.explored_regions)


class StuckStrikeTests(unittest.TestCase):
    def _stuck_call(self, state, args, planner):
        applied = []
        replanned = handle_stuck_replan(
            controller_state=state,
            graph_delta=make_delta(stuck=True),
            graph=SimpleNamespace(nodes=[], room_nodes=[]),
            bev_map=SimpleNamespace(),
            args=args,
            global_goals=[0, 0],
            high_planner=planner,
            goal_description="a black tv",
            agent_pos=(0, 0),
            apply_strategy_fn=lambda strategy, *rest: applied.append(strategy),
            episode_id=0,
            step_idx=20,
        )
        return replanned, applied

    def test_stuck_keeps_target_and_counts_strike(self):
        state = ControllerState()
        state.needs_initial_plan = False
        state.current_strategy = Strategy(
            target_region="unexplored target:tv",
            bias_position=(5, 5),
            reasoning="anchor",
            anchor_object="tv",
        )
        planner = MockPlanner(None)
        replanned, applied = self._stuck_call(state, make_args(True), planner)
        self.assertTrue(replanned)
        self.assertEqual(planner.call_count, 0)
        self.assertEqual(
            state.current_strategy.target_region, "unexplored target:tv"
        )
        self.assertEqual(state.target_commit_stuck_strikes, 1)
        self.assertEqual(state.explored_regions, [])
        self.assertEqual(len(applied), 1)

    def test_stuck_strikes_exhausted_falls_back_to_replan(self):
        state = ControllerState()
        state.needs_initial_plan = False
        state.current_strategy = Strategy(
            target_region="object: tv", bias_position=(5, 5), reasoning="commit"
        )
        state.target_commit_label = "tv"
        state.target_commit_stuck_strikes = 2
        planner = MockPlanner(
            Strategy(target_region="unexplored east", bias_position=None, reasoning="alt")
        )
        replanned, _ = self._stuck_call(state, make_args(True), planner)
        self.assertTrue(replanned)
        self.assertEqual(planner.call_count, 1)
        self.assertIn("object: tv (stuck)", state.explored_regions)


if __name__ == "__main__":
    unittest.main()


class AutoCommitTests(unittest.TestCase):
    def _graph_with_tv(self, detections):
        node = SimpleNamespace(
            caption="tv", center=(50, 60), object={"num_detections": detections}
        )
        return SimpleNamespace(nodes=[node], room_nodes=[], text_goal="a black tv")

    def _args(self, min_det=2):
        args = make_args(True)
        args.controller_target_auto_commit = True
        args.controller_auto_commit_relevance = 0.95
        args.graph_anchor_min_detections = min_det
        args.graph_text_goal_direct_relevance_threshold = 0.75
        return args

    def _state(self):
        state = ControllerState()
        state.needs_initial_plan = False
        state.current_strategy = Strategy(
            target_region="bedroom", bias_position=None, reasoning="explore"
        )
        return state

    def test_graph_scan_fallback_commits_at_anchor_tier(self):
        from smoothnav.controller_logic import maybe_auto_commit_target

        strategy = maybe_auto_commit_target(
            self._state(), make_delta(), self._args(), 5, graph=self._graph_with_tv(2)
        )
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.target_region, "unexplored target:tv")
        self.assertEqual(strategy.bias_position, (50, 60))

    def test_single_glimpse_node_below_anchor_tier_defers(self):
        from smoothnav.controller_logic import maybe_auto_commit_target

        strategy = maybe_auto_commit_target(
            self._state(), make_delta(), self._args(), 5, graph=self._graph_with_tv(1)
        )
        self.assertIsNone(strategy)

    def test_dedupes_existing_commitment(self):
        from smoothnav.controller_logic import maybe_auto_commit_target

        state = self._state()
        state.current_strategy = Strategy(
            target_region="unexplored target:tv", bias_position=(1, 1), reasoning="x"
        )
        strategy = maybe_auto_commit_target(
            state, make_delta(), self._args(), 5, graph=self._graph_with_tv(3)
        )
        self.assertIsNone(strategy)

    def test_flag_off_never_commits(self):
        from smoothnav.controller_logic import maybe_auto_commit_target

        args = self._args()
        args.controller_target_auto_commit = False
        strategy = maybe_auto_commit_target(
            self._state(), make_delta(), args, 5, graph=self._graph_with_tv(3)
        )
        self.assertIsNone(strategy)


class VisibleTakeoverGateTests(unittest.TestCase):
    def test_suppressed_by_default_even_under_commitment(self):
        from smoothnav.executor_adoption import should_allow_text_visible_temp_goal

        self.assertFalse(should_allow_text_visible_temp_goal(
            goal_type="text", current_target_region="unexplored target:tv"))

    def test_c6_allows_takeover_only_under_commitment(self):
        from smoothnav.executor_adoption import should_allow_text_visible_temp_goal

        self.assertTrue(should_allow_text_visible_temp_goal(
            goal_type="text", current_target_region="unexplored target:tv",
            takeover_under_commitment=True))
        self.assertTrue(should_allow_text_visible_temp_goal(
            goal_type="text", current_target_region="object: tv",
            takeover_under_commitment=True))
        self.assertFalse(should_allow_text_visible_temp_goal(
            goal_type="text", current_target_region="bedroom",
            takeover_under_commitment=True))
        self.assertFalse(should_allow_text_visible_temp_goal(
            goal_type="text", current_target_region="unexplored north",
            takeover_under_commitment=True))

    def test_non_text_goals_unaffected(self):
        from smoothnav.executor_adoption import should_allow_text_visible_temp_goal

        self.assertTrue(should_allow_text_visible_temp_goal(
            goal_type="ins-image", current_target_region=""))
