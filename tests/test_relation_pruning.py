import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "base_UniGoal"))

from src.graph.relation_pruning import (  # noqa: E402
    prune_relation_edges,
    select_relation_node_pairs,
)


class FakeEdge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.deleted = False

    def delete(self):
        self.deleted = True


class RelationPruningTests(unittest.TestCase):
    def test_select_relation_node_pairs_prunes_before_edge_creation(self):
        room = object()
        new_node = SimpleNamespace(center=[0, 0], room_node=room)
        near = SimpleNamespace(center=[1, 0], room_node=room)
        far = SimpleNamespace(center=[20, 0], room_node=None)

        selected, dropped, deferred, stats = select_relation_node_pairs(
            [new_node],
            [near, far],
            same_room_only=True,
            topk_per_new_node=1,
            max_pairs=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1].center, [1, 0])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(len(deferred), 0)
        self.assertEqual(stats["selected_edge_count"], 1)

    def test_same_room_only_prunes_cross_room_edges(self):
        room_a = object()
        room_b = object()
        new_node = SimpleNamespace(center=[0, 0], room_node=room_a)
        same_room = SimpleNamespace(center=[1, 0], room_node=room_a)
        cross_room = SimpleNamespace(center=[2, 0], room_node=room_b)

        selected, dropped, deferred, stats = prune_relation_edges(
            [FakeEdge(new_node, same_room), FakeEdge(new_node, cross_room)],
            new_nodes=[new_node],
            same_room_only=True,
            topk_per_new_node=0,
            max_pairs=0,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(len(deferred), 0)
        self.assertEqual(stats["dropped_edge_count"], 1)

    def test_topk_per_new_node_keeps_nearest_edges(self):
        room = object()
        new_node = SimpleNamespace(center=[0, 0], room_node=room)
        near = SimpleNamespace(center=[1, 0], room_node=room)
        mid = SimpleNamespace(center=[2, 0], room_node=room)
        far = SimpleNamespace(center=[3, 0], room_node=room)

        selected, dropped, deferred, stats = prune_relation_edges(
            [FakeEdge(new_node, far), FakeEdge(new_node, near), FakeEdge(new_node, mid)],
            new_nodes=[new_node],
            same_room_only=False,
            topk_per_new_node=2,
            max_pairs=0,
        )

        kept_centers = [edge.node2.center for edge in selected]
        self.assertEqual(kept_centers, [[1, 0], [2, 0]])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(stats["selected_edge_count"], 2)

    def test_max_pairs_defers_over_budget_edges(self):
        room = object()
        new_node = SimpleNamespace(center=[0, 0], room_node=room)
        nodes = [
            SimpleNamespace(center=[i + 1, 0], room_node=room)
            for i in range(4)
        ]
        edges = [FakeEdge(new_node, node) for node in nodes]

        selected, dropped, deferred, stats = prune_relation_edges(
            edges,
            new_nodes=[new_node],
            same_room_only=False,
            topk_per_new_node=0,
            max_pairs=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(dropped), 0)
        self.assertEqual(len(deferred), 2)
        self.assertEqual(stats["deferred_edge_count"], 2)


if __name__ == "__main__":
    unittest.main()
