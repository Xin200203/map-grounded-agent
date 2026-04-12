"""Controller state and event tests for Phase 1."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.controller_logic import build_graph_delta
from smoothnav.controller_state import ControllerState


class MockNode:
    def __init__(self, caption, center=None):
        self.caption = caption
        self.center = center


class MockRoomNode:
    def __init__(self, caption, nodes):
        self.caption = caption
        self.nodes = nodes


class MockGraph:
    def __init__(self, nodes, room_nodes):
        self.nodes = nodes
        self.room_nodes = room_nodes


class ControllerStateTests(unittest.TestCase):
    def test_controller_state_defaults_are_explicit(self):
        state = ControllerState()

        self.assertIsNone(state.current_strategy)
        self.assertIsNone(state.pending_strategy)
        self.assertEqual(state.explored_regions, [])
        self.assertEqual(state.prev_node_count, 0)
        self.assertEqual(state.prev_room_object_counts, {})
        self.assertTrue(state.needs_initial_plan)

    def test_build_graph_delta_tracks_new_nodes_and_room_changes(self):
        chair = MockNode("chair", (1, 2))
        table = MockNode("table", (3, 4))
        graph = MockGraph(
            nodes=[chair, table],
            room_nodes=[
                MockRoomNode("living room", [chair, table]),
                MockRoomNode("kitchen", []),
            ],
        )
        state = ControllerState(
            prev_node_count=1,
            prev_room_object_counts={"living room": 1},
        )

        delta = build_graph_delta(
            graph=graph,
            controller_state=state,
            frontier_near=True,
            frontier_reached=False,
            no_progress=True,
            stuck=False,
            dist_to_goal=6.5,
        )

        self.assertEqual(delta.new_nodes, [table])
        self.assertEqual(delta.new_node_captions, ["table"])
        self.assertEqual(delta.new_rooms, [])
        self.assertEqual(
            delta.room_object_count_changes["living room"],
            {"before": 1, "after": 2},
        )
        self.assertTrue(delta.frontier_near)
        self.assertTrue(delta.no_progress)
        self.assertFalse(delta.stuck)


if __name__ == "__main__":
    unittest.main()
