"""Tests for text-goal target evidence uptake and direct commit guards."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_logic import build_graph_delta
from smoothnav.controller_state import ControllerState
from smoothnav.planner import HighLevelPlanner, Strategy, build_choices_text
from smoothnav.strategy_grounding import apply_strategy
from smoothnav.tactical_arbiter import TacticalArbiter
from smoothnav.target_matching import score_caption_against_goal
from smoothnav.types import TacticalMode


class MockNode:
    def __init__(self, caption, center=(10, 10), detections=1):
        self.caption = caption
        self.center = center
        self.object = {"num_detections": detections}


class MockGraph:
    def __init__(
        self,
        nodes=None,
        room_nodes=None,
        text_goal="",
        args=None,
        local_map_boundary=None,
    ):
        self.nodes = nodes or []
        self.room_nodes = room_nodes or []
        self.text_goal = text_goal
        self.obj_goal = ""
        self.args = args
        self.goal = None
        self.last_goal_debug = {}
        self.local_map_boundary = local_map_boundary
        self.received_bias = None

    def get_edges(self):
        return []

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_goal(self, goal=None):
        self.received_bias = goal
        return self.goal


class MockBoundary:
    def __getitem__(self, key):
        return [[0, 0, 0, 0]][key[0]][key[1]]


class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, prompt="", **kwargs):
        if self.responses:
            return self.responses.pop(0)
        return "{}"


class TargetEvidenceUptakeTests(unittest.TestCase):
    def test_tv_alias_scores_as_primary_television_target(self):
        match = score_caption_against_goal(
            "tv",
            "The screen of the television in this image is gray and has a silver frame.",
        )

        self.assertGreaterEqual(match["score"], 0.75)
        self.assertIn("television", match["primary_categories"])

    def test_context_object_scores_below_target_candidate_threshold(self):
        match = score_caption_against_goal(
            "painting picture",
            "The screen of the television is gray. This image has an abstract painting near the sofa.",
        )

        self.assertLess(match["score"], 0.75)
        self.assertEqual(match["reason"], "context_category_alias")

    def test_graph_delta_marks_target_candidate_event(self):
        graph = MockGraph(
            nodes=[MockNode("tv", (12, 14))],
            text_goal="The screen of the television is gray.",
        )
        state = ControllerState(current_strategy=Strategy("unexplored east", (1, 2), "search"))

        delta = build_graph_delta(
            graph=graph,
            controller_state=state,
            frontier_near=False,
            frontier_reached=False,
            no_progress=False,
            stuck=False,
            dist_to_goal=9.0,
        )

        self.assertTrue(delta.has_target_candidates)
        self.assertIn("target_candidate_detected", delta.event_types)
        self.assertEqual(delta.target_candidate_captions, ["tv"])

    def test_graph_delta_marks_caption_changed_target_candidate(self):
        graph = MockGraph(
            nodes=[MockNode("tv", (12, 14))],
            text_goal="The screen of the television is gray.",
        )
        state = ControllerState(
            current_strategy=Strategy("unexplored east", (1, 2), "search"),
            prev_node_count=1,
            prev_node_captions={0: "picture"},
        )

        delta = build_graph_delta(
            graph=graph,
            controller_state=state,
            frontier_near=False,
            frontier_reached=False,
            no_progress=False,
            stuck=False,
            dist_to_goal=9.0,
        )

        self.assertTrue(delta.node_caption_changed)
        self.assertTrue(delta.has_target_candidates)
        self.assertEqual(delta.target_candidate_captions, ["tv"])

    def test_tactical_arbiter_requests_replan_for_target_candidate_on_direction(self):
        graph = MockGraph(
            nodes=[MockNode("tv", (12, 14))],
            text_goal="The screen of the television is gray.",
        )
        state = ControllerState(current_strategy=Strategy("unexplored east", (1, 2), "search"))
        delta = build_graph_delta(
            graph=graph,
            controller_state=state,
            frontier_near=False,
            frontier_reached=False,
            no_progress=False,
            stuck=False,
            dist_to_goal=9.0,
        )
        world_state = SimpleNamespace(graph_delta=delta, world_epoch=1)
        mission_state = SimpleNamespace()

        decision = TacticalArbiter().decide(
            world_state=world_state,
            mission_state=mission_state,
            current_strategy=state.current_strategy,
            pending_strategy=None,
            needs_initial_plan=False,
        )

        self.assertEqual(decision.mode, TacticalMode.REPLAN_REQUIRED)
        self.assertEqual(decision.reason, "target_candidate_detected")

    def test_text_direct_object_gate_blocks_context_object(self):
        graph = MockGraph(
            nodes=[MockNode("painting picture", (18, 29), detections=5)],
            text_goal="The material of the chair is wood and leather.",
        )
        graph.goal = (40, 50)
        graph.last_goal_debug = {"selected_frontier": [40, 50]}
        strategy = SimpleNamespace(
            bias_position=(18, 29),
            target_region="object: painting picture",
        )
        bev_map = SimpleNamespace(
            full_map="full",
            full_pose="pose",
            local_map_boundary=MockBoundary(),
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

        result = apply_strategy(strategy, graph, bev_map, args, goals)

        self.assertTrue(result.success)
        self.assertEqual(goals, [40, 50])
        self.assertIsNone(graph.received_bias)
        self.assertIsNone(result.graph_debug["object_anchor_goal"])
        self.assertEqual(
            result.graph_debug["direct_object_goal_blocked_reason"],
            "low_target_relevance",
        )

    def test_nonlocal_text_target_object_becomes_search_anchor(self):
        graph = MockGraph(
            nodes=[MockNode("tv", (180, 299), detections=3)],
            text_goal="The screen of the television is gray.",
            args=SimpleNamespace(
                goal_type="text",
                local_width=100,
                local_height=100,
                graph_text_goal_nonlocal_object_as_search_anchor=True,
                graph_text_goal_direct_relevance_threshold=0.75,
            ),
            local_map_boundary=MockBoundary(),
        )
        planner = HighLevelPlanner(
            llm_fn=MockLLM(
                ['{"choice_type":"object","choice_id":"tv","reasoning":"target seen"}']
            )
        )

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  living room: tv",
            goal_description="The screen of the television is gray.",
            explored_regions=[],
            graph=graph,
            agent_pos=(10, 10),
            map_size=720,
        )

        self.assertEqual(strategy.target_region, "unexplored target:tv")
        self.assertEqual(strategy.anchor_object, "tv")
        self.assertEqual(strategy.bias_position, (180, 299))

    def test_local_text_target_object_remains_object_stage(self):
        graph = MockGraph(
            nodes=[MockNode("chair", (20, 30), detections=3)],
            text_goal="The material of the chair is wood and leather.",
            args=SimpleNamespace(
                goal_type="text",
                local_width=100,
                local_height=100,
                graph_text_goal_nonlocal_object_as_search_anchor=True,
                graph_text_goal_direct_relevance_threshold=0.75,
            ),
            local_map_boundary=MockBoundary(),
        )
        planner = HighLevelPlanner(
            llm_fn=MockLLM(
                ['{"choice_type":"object","choice_id":"chair","reasoning":"target is local"}']
            )
        )

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  bedroom: chair",
            goal_description="The material of the chair is wood and leather.",
            explored_regions=[],
            graph=graph,
            agent_pos=(10, 10),
            map_size=720,
        )

        self.assertEqual(strategy.target_region, "object: chair")
        self.assertEqual(strategy.anchor_object, "chair")
        self.assertEqual(strategy.bias_position, (20, 30))


if __name__ == "__main__":
    unittest.main()
