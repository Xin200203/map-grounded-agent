"""Phase 0 tests for JSONL tracing helpers."""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.tracing import RunTracer, hash_text, strategy_to_dict


@dataclass
class FakePayload:
    step_idx: int
    reason: str


class TracingTests(unittest.TestCase):
    def test_hash_text_is_stable(self):
        self.assertEqual(hash_text("same-text"), hash_text("same-text"))
        self.assertNotEqual(hash_text("same-text"), hash_text("different-text"))

    def test_strategy_to_dict_handles_none_and_objects(self):
        self.assertIsNone(strategy_to_dict(None))

        strategy = SimpleNamespace(
            target_region="kitchen",
            bias_position=(12, 18),
            reasoning="objects suggest kitchen",
            explored_regions=["bedroom"],
            anchor_object="sink",
        )
        payload = strategy_to_dict(strategy)
        self.assertEqual(payload["target_region"], "kitchen")
        self.assertEqual(payload["bias_position"], [12, 18])
        self.assertEqual(payload["anchor_object"], "sink")

    def test_run_tracer_writes_jsonl_per_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = RunTracer(tmpdir)
            tracer.record_step(7, {"payload": FakePayload(step_idx=3, reason="trace")})
            tracer.record_planner_call(7, {"planner": {"called": True}})
            tracer.record_monitor_call(7, {"monitor": {"action": "CONTINUE"}})
            tracer.close()

            step_path = os.path.join(tmpdir, "step_traces", "episode_000007.jsonl")
            planner_path = os.path.join(tmpdir, "planner_calls", "episode_000007.jsonl")
            monitor_path = os.path.join(tmpdir, "monitor_calls", "episode_000007.jsonl")

            self.assertTrue(os.path.exists(step_path))
            self.assertTrue(os.path.exists(planner_path))
            self.assertTrue(os.path.exists(monitor_path))

            with open(step_path, "r", encoding="utf-8") as handle:
                step_record = json.loads(handle.readline())
            self.assertIn("timestamp_local", step_record)
            self.assertEqual(step_record["payload"]["step_idx"], 3)
            self.assertEqual(step_record["payload"]["reason"], "trace")


if __name__ == "__main__":
    unittest.main()
