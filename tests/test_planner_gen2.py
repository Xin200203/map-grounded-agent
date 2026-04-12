"""Generation-2 planner tests for Phase 1."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.planner import HighLevelPlanner


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


class MockGraph:
    def __init__(self, nodes=None, room_nodes=None):
        self.nodes = nodes or []
        self.room_nodes = room_nodes or []

    def get_edges(self):
        return []


class PlannerGen2Tests(unittest.TestCase):
    def test_parse_extracts_json_object(self):
        planner = HighLevelPlanner(llm_fn=MockLLM([]))
        parsed = planner._parse('prefix {"choice_type":"direction","choice_id":"east"} suffix')
        self.assertEqual(parsed["choice_type"], "direction")
        self.assertEqual(parsed["choice_id"], "east")

    def test_plan_falls_back_to_north_direction_when_parse_fails(self):
        planner = HighLevelPlanner(llm_fn=MockLLM(["not-json"]))
        trace = MockTraceWriter()
        graph = MockGraph()

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

        self.assertEqual(strategy.target_region, "unexplored north")
        self.assertEqual(strategy.bias_position, (10, 40))
        self.assertEqual(planner.call_count, 1)
        self.assertTrue(trace.planner_calls[0][1]["fallback_triggered"])

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


if __name__ == "__main__":
    unittest.main()
