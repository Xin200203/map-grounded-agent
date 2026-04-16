"""Generation-2 tests for richer GraphDelta construction."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_logic import build_graph_delta, maybe_call_monitor
from smoothnav.controller_state import ControllerState
from smoothnav.planner import Strategy


class MockNode:
    def __init__(self, caption, center=(0, 0)):
        self.caption = caption
        self.center = center
        self.edges = set()


class MockGraph:
    def __init__(self, nodes=None, room_nodes=None):
        self.nodes = nodes or []
        self.room_nodes = room_nodes or []

    def get_edges(self):
        return []


class MockMonitor:
    def __init__(self):
        self.calls = []
        self.call_count = 0

    def evaluate(self, **kwargs):
        self.call_count += 1
        self.calls.append(kwargs)
        return SimpleNamespace(action=None, reason="called")


class GraphDeltaGen2Tests(unittest.TestCase):
    def test_build_graph_delta_tracks_caption_changes_and_strategy_type(self):
        graph = MockGraph(nodes=[MockNode("kitchen sink")])
        state = ControllerState(
            current_strategy=Strategy("kitchen", (1, 2), "search room"),
            prev_node_captions={0: "chair"},
        )

        delta = build_graph_delta(
            graph=graph,
            controller_state=state,
            frontier_near=False,
            frontier_reached=False,
            no_progress=False,
            stuck=False,
            dist_to_goal=7.0,
        )

        self.assertTrue(delta.node_caption_changed)
        self.assertEqual(delta.current_strategy_type, "room")
        self.assertIn("node_caption_changed", delta.event_types)

    def test_monitor_can_trigger_without_new_nodes(self):
        graph = MockGraph(nodes=[])
        state = ControllerState(
            current_strategy=Strategy("unexplored north", (1, 2), "explore"),
            needs_initial_plan=False,
        )
        delta = build_graph_delta(
            graph=graph,
            controller_state=state,
            frontier_near=True,
            frontier_reached=False,
            no_progress=True,
            stuck=False,
            dist_to_goal=5.0,
        )
        monitor = MockMonitor()

        called, _, trigger_event_types = maybe_call_monitor(
            low_agent=monitor,
            controller_state=state,
            graph_delta=delta,
            graph=graph,
            episode_id=1,
            step_idx=5,
            trace_writer=None,
        )

        self.assertTrue(called)
        self.assertIn("frontier_near", trigger_event_types)
        self.assertIn("no_progress", trigger_event_types)


if __name__ == "__main__":
    unittest.main()
