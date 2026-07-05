"""Generation-2 tests for frontier scoring used by Graph.get_goal()."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from smoothnav.frontier_scoring import (
        choose_frontier_locations,
        compose_frontier_value_scores,
        compute_frontier_repeat_penalties,
        compute_target_progress_scores,
        compute_frontier_unknown_novelty_scores,
        compute_local_actionability_scores,
        filter_candidate_indices_for_actionability,
        select_distance_candidate_indices,
        select_bias_candidate_indices,
        summarize_frontier_selection,
    )
except ImportError:  # pragma: no cover - local system Python may lack numpy
    np = None
    choose_frontier_locations = None
    compose_frontier_value_scores = None
    compute_frontier_repeat_penalties = None
    compute_target_progress_scores = None
    compute_frontier_unknown_novelty_scores = None
    compute_local_actionability_scores = None
    filter_candidate_indices_for_actionability = None
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

    def test_value_scores_can_prefer_actionable_novel_frontier_over_bias_only(self):
        scores = compose_frontier_value_scores(
            base_scores=np.array([5.0, 5.1]),
            bias_scores=np.array([2.0, 0.0]),
            semantic_bias_weight=1.0,
            novelty_scores=np.array([0.0, 0.8]),
            actionability_scores=np.array([0.0, 1.0]),
            repeat_penalties=np.array([0.0, 0.0]),
            recent_penalties=np.array([0.0, 0.0]),
            frontier_novelty_weight=1.0,
            local_actionability_weight=3.0,
        )

        self.assertLess(scores[0], scores[1])

    def test_target_progress_keeps_nonlocal_anchor_signal_nonzero(self):
        progress, mode = compute_target_progress_scores(
            frontier_locations=np.array([[40, 0], [200, 0]]),
            target_coord=[400, 0],
            agent_coord=[0, 0],
            target_distances=np.array([30.0, 10.0]),
            agent_target_distance=40.0,
            distance_scale=20.0,
        )

        self.assertEqual(mode, "fmm")
        self.assertGreater(progress[0], 0.0)
        self.assertGreater(progress[1], progress[0])

    def test_target_progress_can_beat_generic_novelty_when_bias_is_zero(self):
        scores = compose_frontier_value_scores(
            base_scores=np.array([5.0, 5.0]),
            bias_scores=np.array([0.0, 0.0]),
            novelty_scores=np.array([1.0, 0.0]),
            actionability_scores=np.array([1.0, 1.0]),
            target_progress_scores=np.array([0.0, 0.8]),
            frontier_novelty_weight=1.0,
            local_actionability_weight=1.0,
            target_progress_weight=4.0,
        )

        self.assertLess(scores[0], scores[1])

    def test_planner_prior_can_flip_frontier_when_weighted(self):
        scores = compose_frontier_value_scores(
            base_scores=np.array([5.0, 5.1]),
            bias_scores=np.array([0.0, 0.0]),
            planner_prior_scores=np.array([1.0, 0.0]),
            planner_prior_weight=1.0,
        )

        self.assertGreater(scores[0], scores[1])

    def test_actionability_filter_relaxes_invalid_bias_subset_to_local_frontier(self):
        candidate_indices, mode = filter_candidate_indices_for_actionability(
            np.array([0]),
            np.array([False, True, False]),
            relax_to_any_valid=True,
        )

        self.assertEqual(candidate_indices.tolist(), [1])
        self.assertEqual(mode, "relaxed_bias_to_local_actionable")

    def test_local_actionability_uses_planning_window_boundary(self):
        valid, scores = compute_local_actionability_scores(
            np.array([[15, 25], [5, 25], [15, 125]]),
            np.array([[10, 0, 20, 0]]),
            local_width=100,
            local_height=100,
        )

        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertEqual(scores.tolist(), [1.0, 0.0, 0.0])

    def test_novelty_and_repeat_penalties_are_reported_in_summary(self):
        unknown_map = np.zeros((50, 50), dtype=bool)
        unknown_map[28:35, 28:35] = True
        novelty = compute_frontier_unknown_novelty_scores(
            np.array([[10, 10], [30, 30]]),
            unknown_map,
            radius=3,
        )
        repeat_penalties, recent_penalties = compute_frontier_repeat_penalties(
            np.array([[10, 10], [30, 30]]),
            last_selected_frontier=[10, 10],
            recent_selected_frontiers=[[30, 30]],
        )
        scores = compose_frontier_value_scores(
            base_scores=np.array([5.0, 5.0]),
            bias_scores=np.array([0.0, 0.0]),
            novelty_scores=novelty,
            repeat_penalties=repeat_penalties,
            recent_penalties=recent_penalties,
            frontier_novelty_weight=1.0,
            frontier_repeat_penalty=2.0,
            frontier_recent_penalty=0.5,
        )
        summary = summarize_frontier_selection(
            frontier_locations_16=np.array([[11, 11], [31, 31]]),
            distances_16=np.array([2.0, 3.0]),
            base_scores=np.array([5.0, 5.0]),
            bias_distances_16=None,
            bias_scores=np.array([0.0, 0.0]),
            final_scores=scores,
            candidate_indices=np.array([0, 1]),
            novelty_scores=novelty,
            repeat_penalties=repeat_penalties,
            recent_penalties=recent_penalties,
            frontier_novelty_weight=1.0,
            frontier_repeat_penalty=2.0,
            frontier_recent_penalty=0.5,
            planner_prior_scores=np.array([0.0, 0.7]),
            planner_prior_weight=1.0,
            local_index_to_branch_id={"0": "B1", "1": "B2"},
        )

        self.assertEqual(summary["selected_frontier"], [30, 30])
        self.assertIn(
            "novelty_score",
            summary["selected_frontier_score_breakdown"],
        )
        self.assertEqual(summary["topk_frontiers"][1]["repeat_penalty"], 1.0)
        self.assertEqual(
            summary["selected_frontier_score_breakdown"]["branch_id"], "B2"
        )
        self.assertEqual(summary["topk_frontiers"][0]["planner_prior_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
