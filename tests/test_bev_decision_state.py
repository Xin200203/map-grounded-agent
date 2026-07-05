import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from smoothnav.bev_decision_state import (
        build_bev_decision_state,
        save_decision_state_fields_npz,
    )
    from smoothnav.frontier_branching import build_frontier_branches
except Exception:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy unavailable")
class BEVDecisionStateTests(unittest.TestCase):
    def test_empty_semantic_channels_remain_empty_with_object_support_only(self):
        full_map = np.zeros((20, 64, 64), dtype=float)
        full_map[1, 8:56, 8:56] = 1.0
        branches = build_frontier_branches(np.array([[12, 50], [50, 12]]), agent_coord=[32, 32])
        state, fields = build_bev_decision_state(
            full_map=full_map,
            step_idx=31,
            target_description="find the tv",
            agent={"coord_rc": [32, 32]},
            branches=branches["branches"],
            objects=[{"id": "O1", "caption": "mirror", "center_rc": [18, 48], "confidence": 0.5}],
            rooms=[{"id": "R1", "caption": "living room", "geometry_type": "unlocalized_hypothesis", "localized": False}],
        )

        self.assertEqual(state["semantic_field"]["semantic_reliability"], "empty")
        self.assertEqual(state["semantic_field"]["active_category_count"], 0)
        self.assertEqual(len(state["semantic_field"]["object_supports"]), 1)
        self.assertEqual(len(state["semantic_field"]["localized_rooms"]), 0)
        self.assertEqual(len(state["semantic_field"]["unlocalized_room_hypotheses"]), 1)
        self.assertTrue(np.any(fields["semantic_object_support"] > 0))
        self.assertTrue(np.any(fields["target_geometry_fallback_value"] > 0))
        self.assertTrue(all(row["evidence_level"] == "geometry_only" for row in state["branch_table"]))

    def test_true_semantic_anchor_creates_anchor_value_and_branch_table_scores(self):
        full_map = np.zeros((20, 64, 64), dtype=float)
        full_map[1, 8:56, 8:56] = 1.0
        # semantic category index 1 => couch, a TV anchor prior
        full_map[5, 45:50, 45:50] = 1.0
        branches = build_frontier_branches(np.array([[46, 46], [12, 12]]), agent_coord=[32, 32])
        state, fields = build_bev_decision_state(
            full_map=full_map,
            step_idx=145,
            target_description="find the tv",
            agent={"coord_rc": [32, 32]},
            branches=branches["branches"],
            objects=[],
            rooms=[],
        )

        self.assertEqual(state["semantic_field"]["active_category_count"], 1)
        self.assertGreater(fields["target_anchor_value"].max(), 0)
        top = state["branch_table"][0]
        self.assertIn(top["evidence_level"], {"anchor", "geometry_only"})
        self.assertIn("target_value_mean", top)
        self.assertIn("target_value_max", top)

    def test_branch_table_preserves_graph_frontier_score_rank(self):
        full_map = np.zeros((20, 64, 64), dtype=float)
        full_map[1, 8:56, 8:56] = 1.0
        branches = [
            {
                "id": "Bsemantic",
                "representative_coord": [12, 12],
                "candidate_points": [{"coord": [12, 12]}],
                "candidate_point_count": 1,
                "score_terms_aggregate": {
                    "final_score_max": 0.2,
                    "base_score_max": 0.1,
                    "bias_score_max": 0.4,
                    "novelty_score_max": 0.2,
                    "actionability_score_max": 0.2,
                },
            },
            {
                "id": "Bexec",
                "representative_coord": [45, 45],
                "candidate_points": [{"coord": [45, 45]}],
                "candidate_point_count": 1,
                "score_terms_aggregate": {
                    "final_score_max": 0.9,
                    "base_score_max": 0.2,
                    "bias_score_max": 0.1,
                    "novelty_score_max": 0.8,
                    "actionability_score_max": 1.0,
                },
            },
        ]

        state, _fields = build_bev_decision_state(
            full_map=full_map,
            step_idx=12,
            target_description="find the tv",
            agent={"coord_rc": [32, 32]},
            branches=branches,
            objects=[],
            rooms=[],
        )

        by_id = {row["id"]: row for row in state["branch_table"]}
        self.assertEqual(by_id["Bsemantic"]["canonical_coord_rc"], [12, 12])
        self.assertEqual(by_id["Bexec"]["frontier_final_score_rank"], 1)
        self.assertEqual(by_id["Bexec"]["executable_decision_rank"], 1)
        self.assertEqual(by_id["Bsemantic"]["frontier_final_score_rank"], 2)
        self.assertEqual(by_id["Bexec"]["frontier_final_score"], 0.9)
        self.assertEqual(by_id["Bexec"]["frontier_actionability_score"], 1.0)
        self.assertIn("frontier_final_score", state["target_value_field"]["branch_ranking"][0])
        self.assertEqual(state["target_value_field"]["executable_decision_ranking"][0]["id"], "Bexec")

    def test_sparse_category_bbox_does_not_become_anchor_rectangle(self):
        full_map = np.zeros((20, 80, 80), dtype=float)
        full_map[1, 4:76, 4:76] = 1.0
        # Two separate couch fragments.  The old bug used one category-wide
        # bbox, which made the empty middle of the map look like TV-anchor
        # evidence.
        full_map[5, 10:13, 10:13] = 1.0
        full_map[5, 60:63, 60:63] = 1.0
        branches = [
            {
                "id": "Bmid",
                "representative_coord": [35, 35],
                "candidate_points": [{"coord": [35, 35]}],
                "candidate_point_count": 1,
                "score_terms_aggregate": {"novelty_score_max": 0.3, "actionability_score_max": 0.3},
            },
            {
                "id": "Bnear",
                "representative_coord": [11, 11],
                "candidate_points": [{"coord": [11, 11]}],
                "candidate_point_count": 1,
                "score_terms_aggregate": {"novelty_score_max": 0.3, "actionability_score_max": 0.3},
            },
        ]

        state, fields = build_bev_decision_state(
            full_map=full_map,
            step_idx=145,
            target_description="find the tv",
            agent={"coord_rc": [40, 40]},
            branches=branches,
            objects=[],
            rooms=[],
        )

        self.assertEqual(state["semantic_field"]["category_masks"][0]["component_count"], 2)
        self.assertEqual(float(fields["target_anchor_value"][35, 35]), 0.0)
        by_id = {row["id"]: row for row in state["branch_table"]}
        self.assertEqual(by_id["Bmid"]["anchor_value"], 0.0)
        self.assertEqual(by_id["Bmid"]["evidence_level"], "geometry_only")
        self.assertGreater(by_id["Bnear"]["anchor_value"], 0.0)
        self.assertEqual(by_id["Bnear"]["evidence_level"], "anchor")

    def test_fields_npz_is_saved_for_replay_artifact(self):
        full_map = np.zeros((2, 32, 32), dtype=float)
        full_map[1, 4:28, 4:28] = 1.0
        branches = build_frontier_branches(np.array([[8, 8]]), agent_coord=[16, 16])
        _state, fields = build_bev_decision_state(
            full_map=full_map,
            step_idx=1,
            target_description="find the tv",
            agent={"coord_rc": [16, 16]},
            branches=branches["branches"],
            objects=[],
            rooms=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_decision_state_fields_npz(Path(tmpdir) / "fields.npz", fields)
            self.assertTrue(Path(path).exists())
            loaded = np.load(path)
            self.assertIn("target_combined_value", loaded.files)


if __name__ == "__main__":
    unittest.main()
