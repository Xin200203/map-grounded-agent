"""Generation-2 tests for frontier scoring used by Graph.get_goal()."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from smoothnav.frontier_scoring import (
        choose_frontier_locations,
        select_distance_candidate_indices,
        select_bias_candidate_indices,
        summarize_frontier_selection,
    )
except ImportError:  # pragma: no cover - local system Python may lack numpy
    np = None
    choose_frontier_locations = None
    select_distance_candidate_indices = None
    select_bias_candidate_indices = None
    summarize_frontier_selection = None


@unittest.skipIf(np is None, "numpy is required for frontier scoring tests")
class GraphGoalScoringGen2Tests(unittest.TestCase):
    def test_base_only_scoring_keeps_highest_base_frontier(self):
        summary = summarize_frontier_selection(
            frontier_locations_16=np.array([[11, 11], [21, 21], [31, 31]]),
            distances_16=np.array([1.5, 3.0, 5.0]),
            base_scores=np.array([8.0, 6.0, 4.0]),
            bias_distances_16=None,
            bias_scores=np.array([0.0, 0.0, 0.0]),
            final_scores=np.array([8.0, 6.0, 4.0]),
            candidate_indices=np.array([0, 1, 2]),
            topk_limit=3,
            semantic_bias_weight=4.0,
            last_selected_frontier=None,
        )

        self.assertEqual(summary["selected_frontier"], [10, 10])
        self.assertAlmostEqual(summary["selected_frontier_score_breakdown"]["base_score"], 8.0)
        self.assertIsNone(summary["selected_frontier_score_breakdown"]["bias_distance"])

    def test_bias_can_flip_top_frontier_inside_candidate_subset(self):
        summary = summarize_frontier_selection(
            frontier_locations_16=np.array([[11, 11], [21, 21], [31, 31]]),
            distances_16=np.array([1.5, 2.5, 3.5]),
            base_scores=np.array([8.0, 7.0, 6.0]),
            bias_distances_16=np.array([8.0, 1.0, 9.0]),
            bias_scores=np.array([0.0, 0.9, 0.0]),
            final_scores=np.array([8.0, 10.6, 6.0]),
            candidate_indices=np.array([0, 1, 2]),
            topk_limit=3,
            semantic_bias_weight=4.0,
            last_selected_frontier=[10, 10],
        )

        self.assertEqual(summary["selected_frontier"], [20, 20])
        self.assertFalse(summary["selected_same"])
        self.assertGreater(summary["top1_top2_gap"], 0.0)

    def test_two_stage_bias_candidate_subset_is_applied(self):
        candidate_indices, selected_from_subset = select_bias_candidate_indices(
            4,
            np.array([10.0, 1.0, 2.0, 3.0]),
            bias_candidate_topk=2,
            bias_candidate_radius=2.5,
        )

        self.assertTrue(selected_from_subset)
        self.assertEqual(candidate_indices.tolist(), [1, 2])

    def test_uses_raw_frontiers_when_filtered_frontiers_are_empty(self):
        choice = choose_frontier_locations(
            raw_frontier_locations=np.array([[10, 10], [11, 11]]),
            filtered_frontier_locations=np.empty((0, 2), dtype=int),
            allow_raw_frontier_fallback=True,
        )

        self.assertEqual(choice["frontier_fallback_mode"], "raw_frontier_fallback")
        self.assertTrue(choice["used_raw_frontier_fallback"])
        self.assertEqual(choice["raw_frontier_count"], 2)
        self.assertEqual(choice["filtered_frontier_count"], 0)
        self.assertEqual(choice["frontier_locations"].tolist(), [[10, 10], [11, 11]])

    def test_relaxes_distance_threshold_when_no_far_frontiers_exist(self):
        selection = select_distance_candidate_indices(
            np.array([0.2, 0.5, 1.0]),
            distance_threshold=1.2,
            allow_relaxed_distance_fallback=True,
        )

        self.assertEqual(selection["candidate_fallback_mode"], "relaxed_distance_threshold")
        self.assertTrue(selection["used_relaxed_distance_fallback"])
        self.assertEqual(selection["distance_threshold_used"], 0.0)
        self.assertEqual(selection["candidate_indices"].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
