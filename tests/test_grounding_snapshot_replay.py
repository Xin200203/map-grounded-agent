import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - local lightweight env may omit numpy
    np = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if np is not None:
    from smoothnav.frontier_scoring import (  # noqa: E402
        build_frontier_value_replay_snapshot,
        replay_frontier_value_snapshot,
    )
    from scripts.replay_grounding_snapshot import replay_records  # noqa: E402


@unittest.skipIf(np is None, "numpy unavailable")
class GroundingSnapshotReplayTests(unittest.TestCase):
    def _snapshot(self):
        frontier_locations = np.array([[100, 100], [130, 100], [180, 100]], dtype=float)
        target = [120, 100]
        # Legacy semantic bias is intentionally zero for all candidates; target
        # progress must provide the non-zero target-anchor signal.
        return build_frontier_value_replay_snapshot(
            frontier_locations_16=frontier_locations + 1,
            distances_16=np.array([5.0, 6.0, 7.0]),
            base_scores=np.array([0.0, 0.0, 0.0]),
            bias_distances_16=np.array([99.0, 99.0, 99.0]),
            bias_scores=np.array([0.0, 0.0, 0.0]),
            novelty_scores=np.array([0.1, 0.1, 0.1]),
            actionability_scores=np.array([1.0, 1.0, 1.0]),
            repeat_penalties=np.zeros(3),
            recent_penalties=np.zeros(3),
            target_progress_scores=None,
            candidate_indices=np.array([0, 1, 2]),
            final_scores=np.zeros(3),
            bias_input=target,
            agent_coord=[200, 100],
            active_target_region="unexplored target:tv",
            active_anchor_object="tv",
            target_progress_active=True,
            semantic_bias_weight=4.0,
            frontier_novelty_weight=0.0,
            local_actionability_weight=0.0,
            target_progress_weight=4.0,
            target_progress_distance_scale=20.0,
        )

    def test_replay_recomputes_nonzero_target_progress_for_target_anchor(self):
        result = replay_frontier_value_snapshot(self._snapshot())

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["checks"]["target_progress_nonzero"])
        self.assertGreater(result["semantic_term"], 0.0)
        self.assertGreater(
            result["selected_score_breakdown"]["target_progress_score"], 0.0
        )

    def test_replay_records_summarizes_target_stage_checks(self):
        summary = replay_records(
            [
                {
                    "step_idx": 330,
                    "trigger": "target_candidate_detected",
                    "snapshot": self._snapshot(),
                }
            ]
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(summary["target_record_count"], 1)
        self.assertEqual(summary["target_progress_zero_count"], 0)
        self.assertEqual(summary["target_semantic_term_zero_count"], 0)

    def test_replay_preserves_frontier_branch_and_planner_prior_terms(self):
        snapshot = build_frontier_value_replay_snapshot(
            frontier_locations_16=np.array([[11, 11], [21, 21]]),
            distances_16=np.array([2.0, 2.0]),
            base_scores=np.array([5.0, 5.1]),
            bias_scores=np.array([0.0, 0.0]),
            planner_prior_scores=np.array([1.0, 0.0]),
            candidate_indices=np.array([0, 1]),
            final_scores=np.array([6.0, 5.1]),
            planner_prior_weight=1.0,
            planner_prior_source="unit",
            planner_selected_branch_id="B1",
            frontier_branches={
                "schema_version": "smoothnav.frontier_branches.v1",
                "local_index_to_branch_id": {"0": "B1", "1": "B2"},
                "branches": [{"id": "B1"}, {"id": "B2"}],
            },
            selected_frontier=[10, 10],
        )

        result = replay_frontier_value_snapshot(snapshot)

        self.assertTrue(result["checks"]["has_candidates"])
        self.assertTrue(result["checks"]["selected_frontier_matches_snapshot"])
        self.assertEqual(result["selected_branch_id"], "B1")
        self.assertEqual(
            result["selected_score_breakdown"]["planner_prior_score"], 1.0
        )

    def test_cli_replays_jsonl_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode_000000.jsonl"
            path.write_text(
                json.dumps({"step_idx": 330, "snapshot": self._snapshot()}) + "\n"
            )
            proc = subprocess.run(
                [sys.executable, "scripts/replay_grounding_snapshot.py", str(path)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("status=pass", proc.stdout)
        self.assertIn("target_records=1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
