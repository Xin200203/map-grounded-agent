"""Generation-2 monitor tests for Phase 1."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.low_level_agent import (
    EscalationOnlyMonitor,
    LowLevelAction,
    LowLevelAgent,
    RuleBasedMonitor,
)


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
        self.monitor_calls = []

    def record_monitor_call(self, episode_id, payload):
        self.monitor_calls.append((episode_id, payload))


class MockNode:
    def __init__(self, caption, center):
        self.caption = caption
        self.center = center


class MockGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class MonitorGen2Tests(unittest.TestCase):
    def setUp(self):
        self.strategy = SimpleNamespace(
            target_region="kitchen",
            reasoning="mugs are common near sinks",
        )
        self.direction_strategy = SimpleNamespace(
            target_region="unexplored north",
            reasoning="explore new space",
        )

    def test_parse_maps_string_action_to_enum(self):
        agent = LowLevelAgent(llm_fn=MockLLM([]))
        result = agent._parse('{"action":"ESCALATE","reason":"wrong room"}')
        self.assertEqual(result.action, LowLevelAction.ESCALATE)
        self.assertEqual(result.reason, "wrong room")

    def test_no_new_nodes_skips_llm_and_records_fast_path(self):
        llm = MockLLM([])
        agent = LowLevelAgent(llm_fn=llm)
        trace = MockTraceWriter()

        result = agent.evaluate(
            strategy=self.strategy,
            new_nodes=[],
            dist_to_goal=22.0,
            total_nodes=4,
            graph=MockGraph([]),
            graph_delta=SimpleNamespace(event_types=[],
                                        new_rooms=[],
                                        frontier_near=False),
            episode_id=9,
            step_idx=5,
            trace_writer=trace,
        )

        self.assertEqual(result.action, LowLevelAction.CONTINUE)
        self.assertEqual(result.reason, "no_new_objects")
        self.assertEqual(len(llm.calls), 0)
        self.assertFalse(trace.monitor_calls[0][1]["llm_called"])

    def test_adjust_action_resolves_anchor_bias(self):
        llm = MockLLM(['{"action":"ADJUST","reason":"target likely near sink","adjust_anchor":"sink"}'])
        agent = LowLevelAgent(llm_fn=llm)
        graph = MockGraph([MockNode("kitchen sink", (33, 44))])

        result = agent.evaluate(
            strategy=self.strategy,
            new_nodes=[MockNode("kitchen sink", (33, 44))],
            dist_to_goal=8.0,
            total_nodes=7,
            graph=graph,
            graph_delta=SimpleNamespace(event_types=["new_nodes"],
                                        new_rooms=[],
                                        frontier_near=False),
        )

        self.assertEqual(result.action, LowLevelAction.ADJUST)
        self.assertEqual(result.adjust_anchor, "sink")
        self.assertEqual(result.adjust_bias, (33, 44))

    def test_invalid_response_falls_back_to_continue(self):
        llm = MockLLM(["not-json"])
        agent = LowLevelAgent(llm_fn=llm)
        trace = MockTraceWriter()

        result = agent.evaluate(
            strategy=self.strategy,
            new_nodes=[MockNode("chair", (1, 2))],
            dist_to_goal=5.0,
            total_nodes=1,
            graph=MockGraph([]),
            graph_delta=SimpleNamespace(event_types=["new_nodes"],
                                        new_rooms=[],
                                        frontier_near=False),
            episode_id=1,
            step_idx=2,
            trace_writer=trace,
        )

        self.assertEqual(result.action, LowLevelAction.CONTINUE)
        self.assertEqual(result.reason, "parse_failure")
        self.assertTrue(trace.monitor_calls[0][1]["fallback_triggered"])

    def test_empty_response_short_circuits_extra_monitor_retries(self):
        llm = MockLLM([""])
        agent = LowLevelAgent(llm_fn=llm)

        result = agent.evaluate(
            strategy=self.strategy,
            new_nodes=[MockNode("chair", (1, 2))],
            dist_to_goal=5.0,
            total_nodes=1,
            graph=MockGraph([]),
            graph_delta=SimpleNamespace(event_types=["new_nodes"],
                                        new_rooms=[],
                                        frontier_near=False),
        )

        self.assertEqual(result.action, LowLevelAction.CONTINUE)
        self.assertEqual(result.reason, "parse_failure")
        self.assertEqual(len(llm.calls), 1)

    def test_rules_monitor_prefetches_when_near_frontier(self):
        monitor = RuleBasedMonitor(prefetch_near_threshold=12.0)

        result = monitor.evaluate(
            strategy=self.strategy,
            new_nodes=[MockNode("chair", (1, 2))],
            dist_to_goal=5.0,
            total_nodes=3,
            graph=MockGraph([]),
            graph_delta=SimpleNamespace(event_types=["frontier_near"],
                                        new_rooms=[],
                                        frontier_near=True,
                                        no_progress=False,
                                        stuck=False),
        )

        self.assertEqual(result.action, LowLevelAction.PREFETCH)
        self.assertEqual(result.reason, "rules_prefetch_near_frontier")

    def test_rules_monitor_escalates_direction_when_new_room_appears(self):
        monitor = RuleBasedMonitor(prefetch_near_threshold=12.0)

        result = monitor.evaluate(
            strategy=self.direction_strategy,
            new_nodes=[MockNode("bed", (3, 4))],
            dist_to_goal=20.0,
            total_nodes=3,
            graph=MockGraph([]),
            graph_delta=SimpleNamespace(event_types=["new_rooms"],
                                        new_rooms=["bedroom"],
                                        frontier_near=False,
                                        no_progress=False,
                                        stuck=False),
        )

        self.assertEqual(result.action, LowLevelAction.ESCALATE)
        self.assertEqual(result.reason, "rules_new_room_escalate")

    def test_rules_monitor_adjusts_room_to_first_new_object(self):
        monitor = RuleBasedMonitor(prefetch_near_threshold=12.0)
        graph = MockGraph([MockNode("kitchen sink", (11, 22))])

        result = monitor.evaluate(
            strategy=self.strategy,
            new_nodes=[MockNode("kitchen sink", (11, 22))],
            dist_to_goal=20.0,
            total_nodes=3,
            graph=graph,
            graph_delta=SimpleNamespace(event_types=["room_object_count_increase"],
                                        new_rooms=[],
                                        frontier_near=False,
                                        no_progress=False,
                                        stuck=False),
        )

        self.assertEqual(result.action, LowLevelAction.ADJUST)
        self.assertEqual(result.adjust_anchor, "kitchen sink")
        self.assertEqual(result.adjust_bias, (11, 22))

    def test_escalation_only_monitor_skips_frontier_near_events(self):
        monitor = EscalationOnlyMonitor(llm_fn=MockLLM([]), prefetch_near_threshold=12.0)

        should_evaluate = monitor.should_evaluate(
            strategy=self.strategy,
            graph_delta=SimpleNamespace(
                event_types=["frontier_near"],
                new_rooms=[],
                frontier_near=True,
                no_progress=False,
                stuck=False,
            ),
            no_progress_steps=0,
            dist_to_goal=5.0,
        )

        self.assertFalse(should_evaluate)

    def test_escalation_only_monitor_calls_llm_for_semantic_conflict(self):
        llm = MockLLM(['{"action":"ESCALATE","reason":"new room contradicts current strategy"}'])
        monitor = EscalationOnlyMonitor(llm_fn=llm, prefetch_near_threshold=12.0)
        trace = MockTraceWriter()

        result = monitor.evaluate(
            strategy=self.strategy,
            new_nodes=[MockNode("sofa", (1, 2))],
            dist_to_goal=25.0,
            total_nodes=3,
            graph=MockGraph([]),
            graph_delta=SimpleNamespace(
                event_types=["new_rooms"],
                new_rooms=["living room"],
                room_object_count_increase_rooms=[],
                frontier_near=False,
                no_progress=False,
                stuck=False,
            ),
            no_progress_steps=0,
            episode_id=3,
            step_idx=9,
            trace_writer=trace,
        )

        self.assertEqual(result.action, LowLevelAction.ESCALATE)
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(trace.monitor_calls[0][1]["llm_called"])


if __name__ == "__main__":
    unittest.main()
