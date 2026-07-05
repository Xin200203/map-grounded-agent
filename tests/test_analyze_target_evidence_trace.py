"""Tests for target-evidence uptake trace summaries."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyze_target_evidence_trace import summarize_trace  # noqa: E402


class AnalyzeTargetEvidenceTraceTests(unittest.TestCase):
    def test_summarize_trace_reports_target_anchor_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "episode_000000.jsonl"
            steps = [
                {
                    "step_idx": 10,
                    "graph_delta": {
                        "target_candidate_details": [{"caption": "tv", "score": 1.0}],
                        "event_types": ["target_candidate_detected"],
                    },
                    "planner_called": True,
                    "planner_calls_this_step": 1,
                    "current_strategy": {"target_region": "unexplored target:tv"},
                },
                {
                    "step_idx": 11,
                    "graph_delta": {},
                    "current_strategy": {"target_region": "unexplored target:tv"},
                },
            ]
            with trace_path.open("w") as handle:
                for item in steps:
                    handle.write(json.dumps(item) + "\n")

            summary = summarize_trace(trace_path)

        self.assertEqual(summary["first_target_step"], 10)
        self.assertEqual(summary["first_planner_after_target"], 10)
        self.assertEqual(summary["first_target_anchor_after_target"], 10)
        self.assertEqual(summary["evidence_to_target_anchor_latency"], 0)
        self.assertIsNone(summary["first_object_commit_after_target"])


if __name__ == "__main__":
    unittest.main()
