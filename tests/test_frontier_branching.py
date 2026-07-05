import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from smoothnav.frontier_branching import (
        branch_for_local_idx,
        build_branch_prior_scores,
        build_frontier_branches,
        save_branch_bev_image,
    )
except Exception:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy unavailable")
class FrontierBranchingTests(unittest.TestCase):
    def test_groups_adjacent_frontier_points_into_branch_ids(self):
        coords = np.array(
            [
                [10, 10],
                [10, 11],
                [11, 11],
                [50, 50],
                [51, 50],
            ]
        )
        branches = build_frontier_branches(
            coords,
            candidate_indices=np.array([0, 1, 3, 4]),
            distances=np.array([5.0, 4.0, 6.0, 3.0, 2.0]),
            final_scores=np.array([1.0, 2.0, 0.5, 4.0, 3.0]),
            actionability_scores=np.array([1.0, 1.0, 0.0, 1.0, 1.0]),
            agent_coord=[0, 0],
        )

        self.assertEqual(branches["schema_version"], "smoothnav.frontier_branches.v1")
        self.assertEqual(branches["branch_count"], 2)
        self.assertEqual(branches["branches"][0]["id"], "B1")
        self.assertEqual(branches["branches"][0]["representative_local_idx"], 1)
        self.assertEqual(branches["branches"][1]["representative_local_idx"], 3)
        self.assertEqual(branch_for_local_idx(branches, 4), "B2")
        self.assertEqual(branches["candidate_branch_ids"], ["B1", "B2"])

    def test_branch_prior_scores_expand_ranked_branch_intent_to_points(self):
        branches = build_frontier_branches(
            np.array([[10, 10], [10, 11], [50, 50]]),
            candidate_indices=np.array([0, 1, 2]),
        )
        scores = build_branch_prior_scores(
            branches,
            selected_branch_id="B2",
            ranked_branches=[{"id": "B1", "score": 0.25}, {"id": "B2", "score": 0.9}],
        )

        self.assertEqual(scores.tolist(), [0.25, 0.25, 1.0])

    def test_branch_overlay_renderer_writes_png_when_pillow_available(self):
        full_map = np.zeros((2, 32, 32), dtype=float)
        full_map[1, 4:24, 4:24] = 1.0
        branches = build_frontier_branches(np.array([[8, 8], [20, 20]]))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_branch_bev_image(Path(tmpdir) / "bev.png", full_map, branches)
            self.assertTrue(path is None or Path(path).exists())


if __name__ == "__main__":
    unittest.main()
