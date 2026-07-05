import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from smoothnav.frontier_branching import build_frontier_branches
    from smoothnav.mllm_frontier_contract import validate_mllm_frontier_plan
except Exception:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy unavailable")
class MLLMFrontierContractTests(unittest.TestCase):
    def _branches(self):
        return build_frontier_branches(np.array([[10, 10], [10, 11], [50, 50]]))

    def test_valid_branch_plan_is_accepted_and_expands_prior(self):
        branches = self._branches()
        verdict = validate_mllm_frontier_plan(
            json.dumps(
                {
                    "schema_version": "smoothnav.mllm_frontier_plan.v1",
                    "decision_type": "explore_frontier",
                    "evidence_level": "anchor",
                    "target_visible": False,
                    "selected_branch_id": "B2",
                    "ranked_branches": [
                        {"id": "B2", "score": 0.8},
                        {"id": "B1", "score": 0.3},
                    ],
                }
            ),
            branch_ids=["B1", "B2"],
            frontier_branches=branches,
        )

        self.assertTrue(verdict["valid"], verdict)
        self.assertEqual(verdict["selected_branch_id"], "B2")
        self.assertEqual(verdict["prior_scores"], [0.3, 0.3, 1.0])

    def test_unknown_branch_is_rejected(self):
        verdict = validate_mllm_frontier_plan(
            {
                "schema_version": "smoothnav.mllm_frontier_plan.v1",
                "decision_type": "explore_frontier",
                "evidence_level": "geometry_only",
                "target_visible": False,
                "selected_branch_id": "B9",
                "ranked_branches": [{"id": "B9", "score": 0.8}],
            },
            branch_ids=["B1", "B2"],
        )

        self.assertFalse(verdict["valid"])
        self.assertIn("unknown_ranked_branch_id", verdict["invalid_reason"])

    def test_raw_coordinate_primary_output_is_rejected(self):
        verdict = validate_mllm_frontier_plan(
            {
                "schema_version": "smoothnav.mllm_frontier_plan.v1",
                "decision_type": "explore_frontier",
                "evidence_level": "anchor",
                "target_visible": False,
                "selected_branch_id": "B1",
                "selected_coord": [10, 10],
                "ranked_branches": [{"id": "B1", "score": 0.8}],
            },
            branch_ids=["B1"],
        )

        self.assertFalse(verdict["valid"])
        self.assertIn("raw_coordinate_primary_output", verdict["invalid_reason"])

    def test_evidence_level_is_required_and_target_visible_requires_direct(self):
        missing = validate_mllm_frontier_plan(
            {
                "schema_version": "smoothnav.mllm_frontier_plan.v1",
                "decision_type": "explore_frontier",
                "target_visible": False,
                "selected_branch_id": "B1",
                "ranked_branches": [{"id": "B1", "score": 0.8}],
            },
            branch_ids=["B1"],
        )
        self.assertFalse(missing["valid"])
        self.assertIn("missing_or_invalid_evidence_level", missing["invalid_reason"])

        visible_without_direct = validate_mllm_frontier_plan(
            {
                "schema_version": "smoothnav.mllm_frontier_plan.v1",
                "decision_type": "explore_frontier",
                "evidence_level": "anchor",
                "target_visible": True,
                "selected_branch_id": "B1",
                "ranked_branches": [{"id": "B1", "score": 0.8}],
            },
            branch_ids=["B1"],
        )
        self.assertFalse(visible_without_direct["valid"])
        self.assertIn("target_visible_requires_direct_evidence", visible_without_direct["invalid_reason"])


if __name__ == "__main__":
    unittest.main()
