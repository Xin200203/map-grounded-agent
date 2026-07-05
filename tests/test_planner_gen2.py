"""Generation-2 planner tests for Phase 1."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.planner import HighLevelPlanner, build_choices_text


class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, prompt="", **kwargs):
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "{}"


class MockTraceWriter:
    def __init__(self):
        self.planner_calls = []

    def record_planner_call(self, episode_id, payload):
        self.planner_calls.append((episode_id, payload))


class MockNode:
    def __init__(self, caption, center):
        self.caption = caption
        self.center = center


class MockRoomNode:
    def __init__(self, caption, nodes):
        self.caption = caption
        self.nodes = nodes


class MockBoundary:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        row, col = key
        return self.values[row][col]


class MockGraph:
    def __init__(
        self,
        nodes=None,
        room_nodes=None,
        frontier_locations_16=None,
        args=None,
        local_map_boundary=None,
        text_goal="",
    ):
        self.nodes = nodes or []
        self.room_nodes = room_nodes or []
        self.frontier_locations_16 = frontier_locations_16 or []
        self.args = args
        self.local_map_boundary = local_map_boundary
        self.text_goal = text_goal

    def get_edges(self):
        return []


class PlannerGen2Tests(unittest.TestCase):
    def test_build_choices_offers_only_primary_target_objects(self):
        nodes = [
            MockNode("cabinet", (1, 1)),
            MockNode("cabinet", (2, 2)),
            MockNode("bed", (3, 3)),
            MockNode("plant", (4, 4)),
        ]
        graph = MockGraph(nodes=nodes)

        choices = build_choices_text(
            graph,
            explored_regions=[],
            goal_description="find the plant in a white vase near a cabinet",
        )

        self.assertIn("  [object] plant", choices)
        self.assertNotIn("  [object] cabinet", choices)
        self.assertNotIn("  [object] bed", choices)

    def test_build_choices_room_lines_are_room_ids_only(self):
        bedroom_node = MockNode("painting picture", (1, 1))
        graph = MockGraph(room_nodes=[MockRoomNode("bedroom", [bedroom_node])])

        choices = build_choices_text(
            graph,
            explored_regions=[],
            goal_description="find the chair",
        )

        self.assertIn("  [room] bedroom", choices)
        self.assertNotIn("  [room] bedroom:", choices)

    def test_build_choices_promotes_room_like_object_caption_to_room_choice(self):
        graph = MockGraph(
            nodes=[MockNode("living room", (1, 1)), MockNode("windows", (2, 2))],
            room_nodes=[MockRoomNode("bedroom", [MockNode("windows", (2, 2))])],
        )

        choices = build_choices_text(
            graph,
            explored_regions=[],
            goal_description="find the television near a sofa",
        )

        self.assertIn("  [room] living room", choices)
        self.assertNotIn("  [object] living room", choices)

    def test_build_choices_excludes_explored_object_anchors(self):
        graph = MockGraph(
            nodes=[
                MockNode("table", (1, 1)),
                MockNode("chair", (2, 2)),
            ]
        )

        choices = build_choices_text(
            graph,
            explored_regions=["object: table (stagnant)"],
            goal_description="find the chair",
        )

        self.assertNotIn("  [object] table", choices)
        self.assertIn("  [object] chair", choices)

    def test_build_choices_omits_empty_object_section_after_filtering(self):
        graph = MockGraph(nodes=[MockNode("table", (1, 1))])

        choices = build_choices_text(
            graph,
            explored_regions=["object: table (stagnant)"],
            goal_description="find the chair",
        )

        self.assertNotIn("OBJECTS (pick if you recognize the target):", choices)
        self.assertIn("  [direction] east", choices)

    def test_build_choices_does_not_offer_unobserved_room_taxonomy(self):
        graph = MockGraph(
            nodes=[],
            room_nodes=[
                MockRoomNode("bedroom", []),
                MockRoomNode("living room", []),
                MockRoomNode("kitchen", []),
            ],
        )

        choices = build_choices_text(
            graph,
            explored_regions=[],
            goal_description="find the chair",
        )

        self.assertNotIn("  [room] bedroom", choices)
        self.assertNotIn("  [room] living room", choices)
        self.assertIn("  [direction] north", choices)

    def test_parse_extracts_json_object(self):
        planner = HighLevelPlanner(llm_fn=MockLLM([]))
        parsed = planner._parse('prefix {"choice_type":"direction","choice_id":"east"} suffix')
        self.assertEqual(parsed["choice_type"], "direction")
        self.assertEqual(parsed["choice_id"], "east")

    def test_parse_normalizes_room_choice_with_scene_context_suffix(self):
        planner = HighLevelPlanner(llm_fn=MockLLM([]))
        parsed = planner._parse(
            '{"choice_type":"room","choice_id":"bedroom: painting picture, windows"}'
        )
        self.assertEqual(parsed["choice_type"], "room")
        self.assertEqual(parsed["choice_id"], "bedroom")

    def test_parse_coerces_room_like_object_choice_to_room(self):
        planner = HighLevelPlanner(llm_fn=MockLLM([]))
        parsed = planner._parse(
            '{"choice_type":"object","choice_id":"living room","reasoning":"tv likely there"}'
        )
        self.assertEqual(parsed["choice_type"], "room")
        self.assertEqual(parsed["choice_id"], "living room")

    def test_plan_falls_back_to_frontier_weighted_direction_when_parse_fails(self):
        planner = HighLevelPlanner(llm_fn=MockLLM(["not-json"]))
        trace = MockTraceWriter()
        graph = MockGraph(
            frontier_locations_16=[
                [40, 70],
                [42, 74],
                [41, 76],
                [10, 40],
            ]
        )

        strategy = planner.plan(
            scene_text="No objects observed yet.",
            goal_description="mug",
            explored_regions=[],
            graph=graph,
            agent_pos=(40, 40),
            map_size=90,
            episode_id=3,
            step_idx=12,
            trace_writer=trace,
        )

        self.assertEqual(strategy.target_region, "unexplored east")
        self.assertEqual(strategy.bias_position, (41, 73))
        self.assertEqual(planner.call_count, 1)
        self.assertTrue(trace.planner_calls[0][1]["fallback_triggered"])
        self.assertEqual(
            trace.planner_calls[0][1]["resolved_bias_position"],
            (41, 73),
        )

    def test_empty_response_retries_before_fallback(self):
        llm = MockLLM(
            [
                "",
                '{"choice_type":"direction","choice_id":"east","reasoning":"retry succeeded"}',
            ]
        )
        planner = HighLevelPlanner(llm_fn=llm)

        strategy = planner.plan(
            scene_text="No objects observed yet.",
            goal_description="mug",
            explored_regions=[],
            graph=MockGraph(),
            agent_pos=(40, 40),
            map_size=90,
        )

        self.assertEqual(strategy.target_region, "unexplored east")
        self.assertEqual(len(llm.calls), 2)

    def test_empty_response_fallback_uses_observed_primary_target_object(self):
        llm = MockLLM(["", "", ""])
        planner = HighLevelPlanner(llm_fn=llm)
        tv = MockNode("tv", (50, 50))
        graph = MockGraph(
            nodes=[tv, MockNode("cabinet", (10, 10))],
            room_nodes=[MockRoomNode("living room", [tv])],
            args=SimpleNamespace(
                goal_type="text",
                graph_text_goal_nonlocal_object_as_search_anchor=True,
                graph_text_goal_direct_relevance_threshold=0.75,
                local_width=20,
                local_height=20,
                map_size=90,
            ),
            local_map_boundary=MockBoundary([[0, 20, 0, 20]]),
            text_goal="The screen of the television is gray.",
        )
        trace = MockTraceWriter()

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  living room: tv, cabinet",
            goal_description="The screen of the television is gray.",
            explored_regions=[],
            graph=graph,
            agent_pos=(10, 10),
            map_size=90,
            episode_id=3,
            step_idx=668,
            trace_writer=trace,
        )

        self.assertEqual(strategy.target_region, "unexplored target:tv")
        self.assertEqual(strategy.anchor_object, "tv")
        self.assertEqual(strategy.bias_position, (50, 50))
        self.assertEqual(len(llm.calls), 3)
        payload = trace.planner_calls[0][1]
        self.assertTrue(payload["fallback_triggered"])
        self.assertEqual(payload["error_message"], "empty_response")
        self.assertEqual(payload["choice_type"], "object")
        self.assertEqual(payload["choice_id"], "tv")
        self.assertIn(
            "target_aware_parse_failure_fallback",
            payload["parsed_result"]["reasoning"],
        )

    def test_empty_response_fallback_ignores_explored_target_object(self):
        llm = MockLLM(["", "", ""])
        planner = HighLevelPlanner(llm_fn=llm)
        graph = MockGraph(
            nodes=[MockNode("tv", (50, 50))],
            frontier_locations_16=[
                [40, 72],
                [42, 75],
                [41, 76],
                [5, 40],
            ],
        )

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  living room: tv",
            goal_description="The screen of the television is gray.",
            explored_regions=["object: tv (stagnant)"],
            graph=graph,
            agent_pos=(40, 40),
            map_size=90,
            current_target_region="unexplored north",
        )

        self.assertEqual(strategy.target_region, "unexplored east")
        self.assertEqual(len(llm.calls), 3)

    def test_parse_failure_fallback_avoids_reusing_current_direction(self):
        planner = HighLevelPlanner(llm_fn=MockLLM(["not-json"]))
        graph = MockGraph(
            frontier_locations_16=[
                [38, 68],
                [41, 74],
                [39, 75],
                [8, 40],
            ]
        )

        strategy = planner.plan(
            scene_text="No objects observed yet.",
            goal_description="chair",
            explored_regions=[],
            graph=graph,
            agent_pos=(40, 40),
            map_size=90,
            current_target_region="unexplored north",
        )

        self.assertEqual(strategy.target_region, "unexplored east")

    def test_semantic_stagnation_uses_direction_fallback_without_llm(self):
        llm = MockLLM(['{"choice_type":"direction","choice_id":"north","reasoning":"unused"}'])
        planner = HighLevelPlanner(llm_fn=llm)
        graph = MockGraph(
            nodes=[
                MockNode("painting picture", (10, 10)),
                MockNode("windows", (12, 12)),
            ],
            room_nodes=[MockRoomNode("bedroom", [MockNode("painting picture", (10, 10))])],
            frontier_locations_16=[
                [40, 72],
                [42, 75],
                [41, 76],
                [5, 40],
            ],
        )

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\\n  bedroom: painting picture, windows",
            goal_description="find the chair with brownish-yellow seat cushion",
            explored_regions=["bedroom (stuck)", "object: painting picture (stagnant)"],
            escalate_reason="Object target stagnated at same local goal, need alternative target",
            graph=graph,
            agent_pos=(40, 40),
            map_size=90,
            current_target_region="unexplored north",
        )

        self.assertEqual(strategy.target_region, "unexplored east")
        self.assertEqual(len(llm.calls), 0)

    def test_plan_resolves_object_choice_to_object_center(self):
        llm = MockLLM(
            ['{"choice_type":"object","choice_id":"kitchen sink","reasoning":"target likely nearby"}']
        )
        planner = HighLevelPlanner(llm_fn=llm)
        node = MockNode("kitchen sink", (123, 456))
        graph = MockGraph(nodes=[node], room_nodes=[MockRoomNode("kitchen", [node])])

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  kitchen: kitchen sink",
            goal_description="mug",
            explored_regions=["bedroom"],
            graph=graph,
            agent_pos=(300, 300),
            map_size=720,
        )

        self.assertEqual(strategy.target_region, "object: kitchen sink")
        self.assertEqual(strategy.anchor_object, "kitchen sink")
        self.assertEqual(strategy.bias_position, (123, 456))

    def test_stuck_planner_prefers_relevant_unsearched_object_when_sparse(self):
        llm = MockLLM([])
        planner = HighLevelPlanner(llm_fn=llm)
        table = MockNode("table", (120, 120))
        chair = MockNode("chair", (130, 130))
        graph = MockGraph(nodes=[table, chair], room_nodes=[MockRoomNode("bedroom", [table, chair])])

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  bedroom: table, chair",
            goal_description="find the chair with brownish-yellow seat cushion",
            explored_regions=["bedroom: table (stuck)"],
            graph=graph,
            agent_pos=(300, 300),
            map_size=720,
            escalate_reason="Agent stuck, need alternative route",
        )

        self.assertEqual(strategy.target_region, "object: chair")
        self.assertEqual(strategy.bias_position, (130, 130))
        self.assertEqual(len(llm.calls), 0)

    def test_stuck_planner_ignores_irrelevant_sparse_objects_for_direction_fallback(self):
        llm = MockLLM([])
        planner = HighLevelPlanner(llm_fn=llm)
        table = MockNode("table", (120, 120))
        curtain = MockNode("curtain", (130, 130))
        graph = MockGraph(
            nodes=[table, curtain],
            room_nodes=[MockRoomNode("bedroom", [table, curtain])],
            frontier_locations_16=[
                [40, 72],
                [42, 75],
                [41, 76],
                [5, 40],
            ],
        )

        strategy = planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  bedroom: table, curtain",
            goal_description="plant in a white vase near a cabinet",
            explored_regions=["bedroom: table (stuck)"],
            graph=graph,
            agent_pos=(40, 40),
            map_size=90,
            current_target_region="unexplored north",
            escalate_reason="Agent stuck, need alternative route",
        )

        self.assertEqual(strategy.target_region, "unexplored east")
        self.assertEqual(len(llm.calls), 0)

    def test_direction_choice_uses_frontier_cluster_when_available(self):
        llm = MockLLM(
            ['{"choice_type":"direction","choice_id":"north","reasoning":"explore north frontiers"}']
        )
        planner = HighLevelPlanner(llm_fn=llm)
        graph = MockGraph(
            frontier_locations_16=[
                [10, 40],
                [12, 42],
                [15, 39],
                [60, 60],
            ]
        )

        strategy = planner.plan(
            scene_text="No objects observed yet.",
            goal_description="mug",
            explored_regions=[],
            graph=graph,
            agent_pos=(40, 40),
            map_size=90,
        )

        self.assertEqual(strategy.target_region, "unexplored north")
        self.assertEqual(strategy.bias_position, (12, 40))

    def test_planner_trace_includes_resolution_debug_fields(self):
        llm = MockLLM(
            ['{"choice_type":"object","choice_id":"kitchen sink","reasoning":"target likely nearby"}']
        )
        planner = HighLevelPlanner(llm_fn=llm)
        trace = MockTraceWriter()
        node = MockNode("kitchen sink", (123, 456))
        graph = MockGraph(nodes=[node], room_nodes=[MockRoomNode("kitchen", [node])])

        planner.plan(
            scene_text="ROOMS AND OBJECTS:\n  kitchen: kitchen sink",
            goal_description="mug",
            explored_regions=[],
            graph=graph,
            agent_pos=(300, 300),
            map_size=720,
            episode_id=1,
            step_idx=4,
            trace_writer=trace,
        )

        payload = trace.planner_calls[0][1]
        self.assertEqual(payload["trace_kind"], "planner_call")
        self.assertEqual(payload["choice_type"], "object")
        self.assertEqual(payload["resolved_bias_position"], (123, 456))
        self.assertEqual(payload["planner_choice_distribution"]["object"], 1)
        self.assertEqual(
            payload["resolved_object_candidates"][0]["center"],
            [123, 456],
        )

    def test_plan_stage_goal_uses_controller_explored_override(self):
        llm = MockLLM(
            ['{"choice_type":"direction","choice_id":"east","reasoning":"search elsewhere"}']
        )
        planner = HighLevelPlanner(llm_fn=llm)
        graph = MockGraph(nodes=[MockNode("tv", (123, 456))])
        mission_state = SimpleNamespace(
            mission_text="The screen of the television is gray.",
            required_evidence=["unexplored north"],
            replan_reason="",
        )
        world_state = SimpleNamespace(
            graph=graph,
            explored_regions=[],
            pose={"map_x": 40, "map_y": 40},
            bev_map=SimpleNamespace(args=SimpleNamespace(map_size=90)),
        )

        stage_goal = planner.plan_stage_goal(
            mission_state=mission_state,
            world_state=world_state,
            reason="Agent stuck, need alternative route",
            explored_regions_override=["object: tv (stuck)"],
            current_target_region="unexplored target:tv",
        )

        self.assertEqual(stage_goal.target_region, "unexplored east")
        self.assertNotIn("  [object] tv", llm.calls[0])


if __name__ == "__main__":
    unittest.main()
