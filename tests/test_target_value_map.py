import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from smoothnav.target_value_map import build_target_value_state
except Exception:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy unavailable")
class TargetValueMapTests(unittest.TestCase):
    def _branches(self):
        return [
            {
                "id": "B1",
                "canonical_representative_coord_rc": [10, 10],
                "canonical_direction_from_agent": "north",
                "candidate_point_count": 3,
                "score_terms_aggregate": {"novelty_score_max": 1.0, "actionability_score_max": 1.0},
            },
            {
                "id": "B2",
                "canonical_representative_coord_rc": [45, 45],
                "canonical_direction_from_agent": "south",
                "candidate_point_count": 3,
                "score_terms_aggregate": {"novelty_score_max": 1.0, "actionability_score_max": 1.0},
            },
        ]

    def test_direct_heat_produces_direct_evidence(self):
        heat = np.zeros((64, 64), dtype=float)
        heat[44:47, 44:47] = 1.0
        _, summary, table = build_target_value_state(
            map_shape=(64, 64),
            target_heatmap=heat,
            target_heatmap_summary={"source": "direct_target"},
            branches=self._branches(),
            semantic_regions=[],
            objects=[],
            rooms=[],
            goal_description="find the tv",
        )

        self.assertTrue(summary["available"])
        b2 = next(row for row in table if row["id"] == "B2")
        self.assertEqual(b2["evidence_level"], "direct")
        self.assertTrue(b2["target_visible"])

    def test_unlocalized_room_does_not_create_room_prior_evidence(self):
        _, summary, table = build_target_value_state(
            map_shape=(64, 64),
            target_heatmap=None,
            target_heatmap_summary={"source": "none"},
            branches=self._branches(),
            semantic_regions=[],
            objects=[],
            rooms=[
                {
                    "id": "R1",
                    "caption": "living room",
                    "localized": False,
                    "center_rc": None,
                }
            ],
            goal_description="find the tv",
        )

        self.assertTrue(summary["available"])
        self.assertEqual(summary["evidence_counts"]["room_prior"], 0)
        self.assertTrue(all(row["evidence_level"] == "geometry_only" for row in table))

    def test_localized_room_creates_room_prior_only(self):
        _, summary, table = build_target_value_state(
            map_shape=(64, 64),
            target_heatmap=None,
            target_heatmap_summary={"source": "none"},
            branches=self._branches(),
            semantic_regions=[],
            objects=[],
            rooms=[
                {
                    "id": "R1",
                    "caption": "living room",
                    "localized": True,
                    "center_rc": [45, 45],
                    "bbox_rc": [35, 35, 55, 55],
                }
            ],
            goal_description="find the tv",
        )

        self.assertTrue(summary["available"])
        b2 = next(row for row in table if row["id"] == "B2")
        self.assertEqual(b2["evidence_level"], "room_prior")
        self.assertFalse(b2["target_visible"])


if __name__ == "__main__":
    unittest.main()

