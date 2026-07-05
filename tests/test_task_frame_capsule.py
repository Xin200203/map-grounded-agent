import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from smoothnav.frontier_branching import build_frontier_branches
    from smoothnav.task_frame_capsule import write_task_frame_capsule
    from smoothnav.tracing import RunTracer
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        Image = None
except Exception:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy unavailable")
class TaskFrameCapsuleTests(unittest.TestCase):
    def test_writer_saves_structured_capsule_files(self):
        branches = build_frontier_branches(np.array([[10, 10], [10, 11], [20, 20]]))
        full_map = np.zeros((2, 32, 32), dtype=float)
        full_map[1, 4:24, 4:24] = 1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_task_frame_capsule(
                tmpdir,
                episode_id=228,
                step_idx=330,
                label="tv_unknown",
                artifacts={
                    "manifest": {"target": "tv"},
                    "world_state": {"visible_targets": []},
                    "frontier_branches": branches,
                    "scoring_input": {"agent_coord": [5, 5]},
                    "grounding_result": {"local_projection_valid": True},
                },
                maps={"full_map": full_map},
            )
            capsule_dir = Path(result["capsule_dir"])
            self.assertTrue((capsule_dir / "manifest.json").exists())
            self.assertTrue((capsule_dir / "frontier_branches.json").exists())
            self.assertTrue((capsule_dir / "maps.npz").exists())
            self.assertTrue((capsule_dir / "semantic_bev_frame.json").exists())
            if Image is not None:
                self.assertTrue((capsule_dir / "semantic_annotated_bev_planner.png").exists())
                self.assertTrue((capsule_dir / "semantic_annotated_bev_debug.png").exists())
            self.assertTrue((capsule_dir / "semantic_projection_summary.json").exists())
            self.assertTrue((capsule_dir / "semantic_instance_footprints.json").exists())
            manifest = json.loads((capsule_dir / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "smoothnav.task_frame_capsule.v1")
            self.assertEqual(manifest["episode_id"], 228)
            semantic_frame = json.loads((capsule_dir / "semantic_bev_frame.json").read_text())
            self.assertEqual(semantic_frame["schema_version"], "smoothnav.semantic_bev_frame.v1")
            footprint_payload = json.loads((capsule_dir / "semantic_instance_footprints.json").read_text())
            self.assertEqual(footprint_payload["count"], 0)
            self.assertFalse(footprint_payload["present_in_world_state"])
            projection_payload = json.loads((capsule_dir / "semantic_projection_summary.json").read_text())
            self.assertFalse(projection_payload["present_in_world_state"])

    def test_tracer_can_write_capsule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = RunTracer(tmpdir)
            result = tracer.record_task_frame_capsule(
                1,
                step_idx=2,
                label="probe",
                artifacts={"manifest": {"kind": "unit"}},
            )
            tracer.close()
            self.assertIsNotNone(result)
            self.assertTrue(Path(result["capsule_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
