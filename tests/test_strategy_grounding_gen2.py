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
        self.nodes = []
        self.local_map_boundary = None
        self.received_bias = None

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

    def test_direct_object_goal_produces_structured_candidate_trace(self):
        graph = MockGraph(
            goal=(19, 30),
            last_goal_debug={
                "selected_frontier": [19, 30],
                "selected_frontier_score": 4.2,
                "topk_frontiers": [
                    {
                        "rank": 1,
                        "frontier": [19, 30],
                        "final_score": 4.2,
                        "bias_score": 0.9,
                        "novelty_score": 0.7,
                        "actionability_score": 1.0,
                        "repeat_penalty": 0.0,
                        "recent_penalty": 0.0,
                    }
                ],
            },
        )
        graph.nodes = [
            SimpleNamespace(
                caption="plant",
                center=[18, 29],
                object={"num_detections": 5},
            )
        ]
        strategy = SimpleNamespace(
            bias_position=(18, 29),
            target_region="object: plant",
        )
        goals = [0, 0]

        result = apply_strategy(strategy, graph, self.bev_map, self.args, goals)

        self.assertTrue(result.success)
        self.assertEqual(goals, [9, 10])
        self.assertEqual(graph.received_bias, (18, 29))
        self.assertEqual(graph.local_map_boundary, self.bev_map.local_map_boundary)
        self.assertEqual(result.graph_debug["object_anchor_goal"], [18, 29])
        self.assertFalse(result.graph_debug["direct_object_goal_used"])
        self.assertEqual(result.candidates[0]["coord"], [19, 30])
        self.assertEqual(result.candidates[0]["actionability_score"], 1.0)

    def test_direct_object_goal_can_be_enabled_for_legacy_image_goal_mode(self):
        graph = MockGraph(goal=None)
        graph.nodes = [
            SimpleNamespace(
                caption="plant",
                center=[18, 29],
                object={"num_detections": 5},
            )
        ]
        strategy = SimpleNamespace(
            bias_position=(18, 29),
            target_region="object: plant",
        )
        args = SimpleNamespace(
            local_width=100,
            local_height=100,
            graph_enable_direct_object_goal=True,
            graph_text_goal_use_frontier_anchor=False,
            goal_type="image",
        )
        goals = [0, 0]

        result = apply_strategy(strategy, graph, self.bev_map, args, goals)

        self.assertTrue(result.success)
        self.assertEqual(goals, [8, 9])
        self.assertEqual(result.candidates[0]["coord"], [18, 29])
        self.assertEqual(result.candidates[0]["caption"], "plant")

    def test_text_object_goal_uses_local_direct_target_when_projectable(self):
        graph = MockGraph(goal=(99, 99))
        graph.nodes = [
            SimpleNamespace(
                caption="plant",
                center=[18, 29],
                object={"num_detections": 5},
            )
        ]
        strategy = SimpleNamespace(
            bias_position=(18, 29),
            target_region="object: plant",
        )
        args = SimpleNamespace(
            local_width=100,
            local_height=100,
            goal_type="text",
            graph_enable_direct_object_goal=False,
            graph_text_goal_use_frontier_anchor=True,
            graph_text_goal_allow_local_direct_object_goal=True,
        )
        goals = [0, 0]

        result = apply_strategy(strategy, graph, self.bev_map, args, goals)

        self.assertTrue(result.success)
        self.assertEqual(goals, [8, 9])
        self.assertEqual(graph.received_bias, None)
        self.assertEqual(result.graph_debug["direct_object_goal"], [18, 29])

    def test_text_object_goal_falls_back_to_frontier_anchor_when_not_projectable(self):
        graph = MockGraph(
            goal=(19, 30),
            last_goal_debug={
                "selected_frontier": [19, 30],
                "topk_frontiers": [{"rank": 1, "frontier": [19, 30]}],
            },
        )
        graph.nodes = [
            SimpleNamespace(
                caption="plant",
                center=[500, 500],
                object={"num_detections": 5},
            )
        ]
        strategy = SimpleNamespace(
            bias_position=(500, 500),
            target_region="object: plant",
        )
        args = SimpleNamespace(
            local_width=100,
            local_height=100,
            goal_type="text",
            graph_enable_direct_object_goal=False,
            graph_text_goal_use_frontier_anchor=True,
            graph_text_goal_allow_local_direct_object_goal=True,
        )
        goals = [0, 0]

        result = apply_strategy(strategy, graph, self.bev_map, args, goals)

        self.assertTrue(result.success)
        self.assertEqual(goals, [9, 10])
        self.assertEqual(graph.received_bias, (500, 500))
        self.assertEqual(result.graph_debug["object_anchor_goal"], [500, 500])
        self.assertEqual(graph.active_target_region, "object: plant")

    def test_target_search_anchor_marks_graph_for_target_progress_scoring(self):
        graph = MockGraph(
            goal=(19, 30),
            last_goal_debug={
                "selected_frontier": [19, 30],
                "topk_frontiers": [{"rank": 1, "frontier": [19, 30]}],
            },
        )
        strategy = SimpleNamespace(
            bias_position=(500, 500),
            target_region="unexplored target:tv",
            anchor_object="tv",
        )
        args = SimpleNamespace(local_width=100, local_height=100)

        result = apply_strategy(strategy, graph, self.bev_map, args, [0, 0])

        self.assertTrue(result.success)
        self.assertEqual(graph.active_target_region, "unexplored target:tv")
        self.assertEqual(graph.active_anchor_object, "tv")

    def test_geometry_only_mllm_frontier_prior_is_not_applied(self):
        class PriorGraph(MockGraph):
            def __init__(self):
                super().__init__(goal=(18, 29))
                self.last_goal_replay_snapshot = {
                    "frontier_branches": {"branches": [{"id": "B1"}]},
                }
                self.priors_seen = []

            def get_goal(self, goal=None, record_selection_history=True):
                self.received_bias = goal
                self.priors_seen.append(self.planner_frontier_prior)
                self.last_goal_debug = {
                    "selected_frontier": [18, 29],
                    "selected_frontier_score_breakdown": {"branch_id": "B1"},
                    "selected_frontier_branch_id": "B1",
                }
                return self.goal

        class GeometryOnlyPlanner:
            def plan(self, **_kwargs):
                return {
                    "status": "ok",
                    "verdict": {
                        "valid": True,
                        "selected_branch_id": "B1",
                        "evidence_level": "geometry_only",
                        "prior_scores": [1.0],
                        "ranked_branch_ids": ["B1"],
                    },
                }

        graph = PriorGraph()
        args = SimpleNamespace(
            local_width=100,
            local_height=100,
            mllm_frontier_planner_mode="online",
        )

        result = apply_strategy(
            self.strategy,
            graph,
            self.bev_map,
            args,
            [0, 0],
            mllm_frontier_planner=GeometryOnlyPlanner(),
            goal_description="find the tv",
        )

        self.assertTrue(result.success)
        self.assertEqual(graph.priors_seen, [None, None])
        mllm_debug = result.graph_debug["mllm_frontier"]
        self.assertFalse(mllm_debug["applied"])
        self.assertEqual(mllm_debug["evidence_level"], "geometry_only")
        self.assertEqual(mllm_debug["evidence_gate_strength"], 0.0)


if __name__ == "__main__":
    unittest.main()
