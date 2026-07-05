import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
    from PIL import Image  # noqa: F401
    from smoothnav.frontier_branching import build_frontier_branches
    from smoothnav.semantic_bev import (
        DENSE_SEMANTIC_CLASS_OFFSET,
        build_dense_semantic_raster,
        build_pseudo_semantic_support_raster,
        build_semantic_bev_frame,
        render_decision_state_bev,
        render_dense_semantic_bev,
        render_semantic_annotated_bev,
        save_dense_semantic_bev_artifacts,
        save_semantic_annotated_bev,
        semantic_bev_summary_for_prompt,
    )
    from smoothnav.world_state import summarize_objects
except Exception:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy/Pillow unavailable")
class SemanticBEVRendererTests(unittest.TestCase):
    def _full_map(self):
        full_map = np.zeros((2, 64, 64), dtype=float)
        full_map[1, 8:56, 8:56] = 1.0
        full_map[0, 24:30, 24:40] = 1.0
        return full_map

    def test_frame_contains_objects_rooms_branches_and_quality(self):
        full_map = self._full_map()
        branches = build_frontier_branches(
            np.array([[12, 50], [13, 50], [50, 12], [51, 12]]),
            agent_coord=[32, 32],
        )
        world_state = {
            "step_idx": 7,
            "object_summary": [
                {
                    "caption": "television",
                    "center": [18, 48],
                    "num_detections": 3,
                    "target_relevance": 1.0,
                },
                {"caption": "cabinet", "center": [44, 18], "num_detections": 2},
            ],
            "room_summary": [
                {"caption": "living room", "object_count": 1, "objects": ["television"]},
                {"caption": "kitchen", "object_count": 1, "objects": ["cabinet"]},
            ],
        }
        frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branches,
            world_state=world_state,
            scoring_input={"agent_coord": [32, 32]},
            grounding_result={
                "selected_frontier_score_breakdown": {"branch_id": "B1"},
                "graph_debug": {"mllm_frontier": {"dry_selected_branch_id": "B2", "selected_branch_id": "B1", "applied": True}},
            },
            goal_description="find the tv",
            episode_id=1,
            step_idx=7,
            source="unit_test",
        )

        self.assertEqual(frame["schema_version"], "smoothnav.semantic_bev_frame.v1")
        self.assertEqual(frame["quality_checks"]["localized_object_count"], 2)
        self.assertEqual(frame["quality_checks"]["localized_room_hypothesis_count"], 2)
        self.assertTrue(frame["quality_checks"]["agent_on_known_or_near_known"])
        self.assertTrue(any(obj["caption"] == "television" for obj in frame["objects"]))
        self.assertTrue(any(room["geometry_type"] == "member_object_hypothesis" for room in frame["rooms"]))
        self.assertTrue(any(branch.get("decision_roles") for branch in frame["branches"]))
        self.assertIn("decision_state", frame)
        self.assertIn("branch_decision_table", frame)
        self.assertTrue(frame["branch_decision_table"])
        self.assertIn(frame["branch_decision_table"][0]["evidence_level"], {"direct", "anchor", "room_prior", "geometry_only"})
        summary = semantic_bev_summary_for_prompt(frame)
        self.assertIn("object_labels", summary)
        self.assertIn("branch_semantic_hints", summary)
        self.assertIn("branch_decision_table", summary)
        self.assertNotIn("raw_coord_rc", summary["agent"])

    def test_selection_contract_explains_graph_frontier_override(self):
        full_map = np.zeros((20, 64, 64), dtype=float)
        full_map[1, 8:56, 8:56] = 1.0
        # channel 9 = semantic category index 5 = tv; target-value should favor B1.
        full_map[9, 10:15, 10:15] = 1.0
        branches = {
            "branches": [
                {
                    "id": "B1",
                    "representative_coord": [12, 12],
                    "candidate_points": [{"coord": [12, 12]}],
                    "candidate_point_count": 1,
                    "score_terms_aggregate": {
                        "final_score_max": 0.1,
                        "novelty_score_max": 0.1,
                        "actionability_score_max": 0.1,
                    },
                },
                {
                    "id": "B2",
                    "representative_coord": [50, 50],
                    "candidate_points": [{"coord": [50, 50]}],
                    "candidate_point_count": 1,
                    "score_terms_aggregate": {
                        "final_score_max": 0.9,
                        "novelty_score_max": 0.9,
                        "actionability_score_max": 1.0,
                    },
                },
            ]
        }

        frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branches,
            scoring_input={"agent_coord": [32, 32]},
            grounding_result={"selected_frontier_score_breakdown": {"branch_id": "B2"}},
            goal_description="find the tv",
        )

        contract = frame["target_value_state"]["selection_contract"]
        self.assertEqual(contract["top_target_value_branch_id"], "B1")
        self.assertEqual(contract["final_branch_id"], "B2")
        self.assertEqual(contract["top_frontier_score_branch_id"], "B2")
        self.assertEqual(contract["top_executable_decision_branch_id"], "B2")
        self.assertEqual(contract["final_executable_decision_rank"], 1)
        self.assertTrue(contract["drift_is_explained"])
        self.assertEqual(contract["override_reason"], "graph_frontier_score_contract")
        self.assertNotIn("target_value_final_branch_unexplained_drift", frame["quality_checks"]["warnings"])
        summary = semantic_bev_summary_for_prompt(frame)
        self.assertEqual(
            summary["target_value_state"]["selection_contract"]["override_reason"],
            "graph_frontier_score_contract",
        )
        self.assertEqual(summary["target_value_state"]["executable_decision_ranking"][0]["id"], "B2")
        summary_rows = {row["id"]: row for row in summary["branch_decision_table"]}
        self.assertEqual(summary_rows["B2"]["executable_decision_rank"], 1)

    def test_render_and_save_semantic_bev_images(self):
        full_map = self._full_map()
        branches = build_frontier_branches(np.array([[12, 50], [50, 12]]), agent_coord=[32, 32])
        frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branches,
            world_state={
                "object_summary": [{"caption": "mirror", "center": [18, 48], "num_detections": 1}],
                "room_summary": [{"caption": "bedroom", "object_count": 1, "objects": ["mirror"]}],
            },
            scoring_input={"agent_coord": [32, 32]},
            step_idx=9,
        )
        image = render_semantic_annotated_bev(frame, full_map, mode="debug", scale=1)
        self.assertIsNotNone(image)
        self.assertGreater(image.size[0], 0)
        self.assertGreater(image.size[1], 0)
        decision_image = render_decision_state_bev(frame, full_map, scale=1)
        self.assertIsNotNone(decision_image)
        self.assertGreater(decision_image.size[0], image.size[0])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_semantic_annotated_bev(Path(tmpdir) / "semantic.png", frame, full_map, mode="planner", scale=1)
            self.assertTrue(Path(out).exists())
            decision_out = save_semantic_annotated_bev(Path(tmpdir) / "decision.png", frame, full_map, mode="decision", scale=1)
            self.assertTrue(Path(decision_out).exists())

    def test_agent_row_flip_quality_guard(self):
        full_map = np.zeros((2, 100, 100), dtype=float)
        full_map[1, 10:25, 10:25] = 1.0
        branches = build_frontier_branches(np.array([[15, 25]]), agent_coord=[15, 15])
        good = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branches,
            scoring_input={"agent_coord": [15, 15]},
        )
        flipped = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branches,
            scoring_input={"agent_coord": [84, 15]},
        )
        self.assertTrue(good["quality_checks"]["agent_on_known_or_near_known"])
        self.assertFalse(flipped["quality_checks"]["agent_on_known_or_near_known"])
        self.assertIn("agent_not_on_known_or_near_known", flipped["quality_checks"]["warnings"])

    def test_semantic_channels_target_prior_and_agent_repair(self):
        full_map = np.zeros((20, 80, 80), dtype=float)
        full_map[1, 20:65, 15:65] = 1.0
        full_map[2, 30:35, 25:30] = 1.0
        full_map[3, 25:45, 25:35] = 1.0
        # channel 5 = semantic category index 1 = couch; TV target should use
        # it as a weak co-occurrence prior, not direct target evidence.
        full_map[5, 52:58, 24:30] = 0.8
        branches = build_frontier_branches(
            np.array([[55, 28], [30, 60]]),
            agent_coord=[74, 27],
        )
        frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branches,
            world_state={
                "object_summary": [
                    {
                        "caption": "mirror",
                        "center": [40, 42],
                        "target_relevance": 0.0,
                        "target_primary_categories": ["television"],
                    }
                ]
            },
            scoring_input={"agent_coord": [74, 27]},
            goal_description="find the tv",
        )
        self.assertTrue(frame["base_layers"]["has_semantic_channels"])
        self.assertTrue(frame["base_layers"]["has_language_query_heatmap"])
        self.assertEqual(frame["semantic_layers"]["target_category_labels"], ["tv"])
        self.assertEqual(frame["semantic_layers"]["direct_target_pixel_count"], 0)
        self.assertEqual(frame["target_query_heatmap"]["source"], "semantic_prior_only")
        self.assertTrue(frame["semantic_regions"])
        self.assertEqual(frame["semantic_regions"][0]["label"], "couch")
        self.assertGreater(max(b["target_heatmap_score"] for b in frame["branches"]), 0.0)
        self.assertTrue(frame["target_value_state"]["available"])
        self.assertTrue(any(row["evidence_level"] == "anchor" for row in frame["branch_decision_table"]))
        self.assertEqual(frame["agent"]["coord_rc"], [32, 27])
        self.assertTrue(frame["agent"]["display_repaired_from_map_current_channel"])
        summary = semantic_bev_summary_for_prompt(frame)
        self.assertIn("semantic_regions", summary)
        self.assertIn("target_query_heatmap", summary)
        self.assertIn("target_value_state", summary)
        self.assertGreater(summary["quality_checks"].get("debug_coordinate_warnings_hidden_from_prompt", 0), 0)
        self.assertFalse(
            any(
                str(item).startswith("raw_agent_coord_disagrees")
                or str(item).startswith("agent_display_coord_repaired")
                for item in summary["quality_checks"].get("warnings", [])
            )
        )

    def test_unlocalized_rooms_stay_weak_priors_not_spatial_regions(self):
        full_map = self._full_map()
        branches = build_frontier_branches(np.array([[12, 50], [50, 12]]), agent_coord=[32, 32])
        frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches=branches,
            world_state={
                "object_summary": [{"caption": "mirror", "center": [18, 48], "num_detections": 1}],
                "room_summary": [{"caption": "living room", "object_count": 0, "objects": ["unknown sofa"]}],
            },
            scoring_input={"agent_coord": [32, 32]},
            goal_description="find the tv",
        )

        self.assertEqual(frame["quality_checks"]["localized_room_hypothesis_count"], 0)
        self.assertEqual(frame["quality_checks"]["unlocalized_room_prior_count"], 1)
        self.assertEqual(len(frame["room_state"]["localized_room_regions"]), 0)
        self.assertEqual(len(frame["room_state"]["unlocalized_room_priors"]), 1)
        self.assertTrue(all(row["evidence_level"] != "room_prior" for row in frame["branch_decision_table"]))

    def test_dense_semantic_raster_reports_empty_semantic_channels(self):
        full_map = np.zeros((20, 32, 32), dtype=float)
        full_map[1, 4:28, 4:28] = 1.0
        full_map[0, 12:14, 8:20] = 1.0

        dense = build_dense_semantic_raster(full_map)

        self.assertEqual(dense["semantic_source"], "empty_map_channels")
        self.assertEqual(dense["semantic_coverage_ratio"], 0.0)
        self.assertEqual(dense["true_semantic_pixel_count"], 0)
        self.assertIn("semantic_channels_empty", dense["warnings"])
        self.assertIn("dense_semantic_bev_cannot_show_object_regions_without_semantic_projection", dense["warnings"])

        pseudo = build_pseudo_semantic_support_raster(
            full_map,
            [{"id": "O001", "caption": "chair", "center_rc": [16, 16], "confidence": 0.8}],
        )
        self.assertGreater(pseudo["pseudo_support_pixel_count"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_dense_semantic_bev_artifacts(
                Path(tmpdir) / "dense_semantic_bev.png",
                Path(tmpdir) / "dense_semantic_bev.json",
                dense,
                agent_pose=[10, 10],
                trajectory=[[5, 5], [6, 6], [7, 7]],
                unlocalized_room_hypotheses=[{"caption": "living room", "geometry_type": "unlocalized_hypothesis"}],
                pseudo_support=pseudo,
            )
            self.assertTrue(Path(out["image"]).exists())
            payload = json.loads(Path(out["json"]).read_text())
            self.assertEqual(payload["unlocalized_rooms_drawn_count"], 0)
            self.assertEqual(payload["object_labels_drawn_on_dense_map_count"], 0)
            self.assertEqual(payload["branch_labels_drawn_on_dense_map_count"], 0)
            self.assertEqual(payload["pseudo_support_source"], "object_center_prior")

    def test_dense_semantic_raster_uses_cell_level_semantic_channels(self):
        full_map = np.zeros((20, 48, 48), dtype=float)
        full_map[1, 5:43, 5:43] = 1.0
        # category index 0 = chair, category index 1 = couch
        full_map[4, 10:16, 12:19] = 0.9
        full_map[5, 25:34, 26:38] = 0.8

        dense = build_dense_semantic_raster(full_map, semantic_threshold=0.5)

        self.assertEqual(dense["semantic_source"], "map_channels")
        self.assertEqual(dense["semantic_channel_active_count"], 2)
        self.assertEqual(dense["per_category_pixel_count"]["chair"], 6 * 7)
        self.assertEqual(dense["per_category_pixel_count"]["couch"], 9 * 12)
        self.assertEqual(dense["semantic_bbox_footprint_pixel_count"], 0)
        self.assertEqual(dense["semantic_bbox_footprint_source"], "none")
        self.assertEqual(dense["class_id_map"][11, 13], DENSE_SEMANTIC_CLASS_OFFSET)
        self.assertEqual(dense["class_id_map"][30, 30], DENSE_SEMANTIC_CLASS_OFFSET + 1)

        image = render_dense_semantic_bev(dense, agent_pose=[20, 20], trajectory=[[8, 8], [9, 9]])
        self.assertIsNotNone(image)
        self.assertGreater(image.size[0], 0)
        self.assertGreater(image.size[1], 0)

    def test_sparse_semantic_channel_bbox_fill_makes_visible_object_extent(self):
        full_map = np.zeros((20, 40, 40), dtype=float)
        full_map[1, 4:36, 4:36] = 1.0
        full_map[4, 10, 10:21] = 0.1
        full_map[4, 18, 10:21] = 0.1
        full_map[4, 10:19, 10] = 0.1
        full_map[4, 10:19, 20] = 0.1

        dense = build_dense_semantic_raster(
            full_map,
            semantic_threshold=0.5,
            semantic_bbox_fill=True,
            semantic_bbox_fill_threshold=0.05,
        )

        self.assertEqual(dense["true_semantic_pixel_count"], 0)
        self.assertGreater(dense["semantic_bbox_footprint_pixel_count"], 4)
        self.assertEqual(dense["semantic_bbox_footprint_source"], "semantic_channel_bbox_fill")
        self.assertEqual(dense["display_semantic_source"], "semantic_channel_bbox_fill")
        self.assertEqual(dense["class_id_map"][14, 15], DENSE_SEMANTIC_CLASS_OFFSET)

    def test_dense_semantic_raster_fills_graph_object_footprint_cells(self):
        full_map = np.zeros((20, 48, 48), dtype=float)
        full_map[1, 5:43, 5:43] = 1.0
        obj = {
            "id": "O001",
            "caption": "cabinet",
            "bbox_rc": [10, 12, 18, 25],
            "footprint_rc_indices": [[r, c] for r in range(10, 19) for c in range(12, 26)],
            "footprint_source": "graph_pcd_bbox_cells",
            "footprint_confidence": 0.75,
        }

        dense = build_dense_semantic_raster(full_map, objects=[obj], semantic_threshold=0.5)

        self.assertEqual(dense["semantic_source"], "empty_map_channels")
        self.assertEqual(dense["true_semantic_pixel_count"], 0)
        self.assertEqual(dense["object_footprint_source"], "graph_object_footprints")
        self.assertEqual(dense["object_footprint_pixel_count"], 9 * 14)
        self.assertEqual(dense["object_footprint_per_category_pixel_count"]["cabinet"], 9 * 14)
        self.assertEqual(dense["display_semantic_source"], "graph_object_footprints")
        self.assertGreater(dense["display_semantic_coverage_ratio"], 0.0)
        self.assertNotEqual(dense["class_id_map"][12, 14], 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_dense_semantic_bev_artifacts(
                Path(tmpdir) / "dense_semantic_bev.png",
                Path(tmpdir) / "dense_semantic_bev.json",
                dense,
            )
            payload = json.loads(Path(out["json"]).read_text())
            self.assertEqual(payload["object_footprint_pixel_count"], 9 * 14)
            self.assertEqual(payload["object_labels_drawn_on_dense_map_count"], 0)
            self.assertEqual(payload["branch_labels_drawn_on_dense_map_count"], 0)

    def test_dense_semantic_raster_rejects_footprints_mostly_outside_explored_map(self):
        full_map = np.zeros((20, 48, 48), dtype=float)
        full_map[1, 5:20, 5:20] = 1.0
        obj = {
            "id": "O_bad",
            "caption": "cabinet",
            "bbox_rc": [30, 30, 38, 38],
            "footprint_rc_indices": [[r, c] for r in range(30, 39) for c in range(30, 39)],
            "footprint_source": "graph_pcd_bbox_cells",
            "footprint_confidence": 0.75,
        }

        dense = build_dense_semantic_raster(full_map, objects=[obj], semantic_threshold=0.5)

        self.assertEqual(dense["object_footprint_pixel_count"], 0)
        self.assertEqual(dense["object_footprint_source"], "none")
        self.assertEqual(dense["rejected_object_footprints"][0]["reason"], "footprint_mostly_unknown_unexplored")
        self.assertEqual(dense["class_id_map"][34, 34], 0)

    def test_semantic_instance_footprints_enter_frame_and_dense_raster(self):
        full_map = np.zeros((20, 64, 64), dtype=float)
        full_map[1, 6:58, 6:58] = 1.0
        footprint_cells = [[r, c] for r in range(20, 30) for c in range(22, 40)]
        world_state = {
            "object_summary": [],
            "semantic_instance_footprints": [
                {
                    "id": "SI001",
                    "caption": "couch",
                    "center_rc": [25, 31],
                    "bbox_rc": [20, 22, 29, 39],
                    "footprint_source": "semantic_instance_depth_bbox_cells",
                    "footprint_pixel_count": len(footprint_cells),
                    "footprint_confidence": 0.65,
                    "footprint_rc_indices": footprint_cells,
                    "footprint_encoding": "rc_indices_from_projected_instance_bbox",
                    "source": "bev_map.semantic_instance_footprints",
                }
            ],
        }

        frame = build_semantic_bev_frame(
            full_map=full_map,
            frontier_branches={},
            world_state=world_state,
            scoring_input={"agent_coord": [32, 32]},
            goal_description="find the tv",
        )
        self.assertTrue(any(obj["source"] == "bev_map.semantic_instance_footprints" for obj in frame["objects"]))

        dense = build_dense_semantic_raster(full_map, objects=frame["objects"])
        self.assertEqual(dense["object_footprint_source"], "graph_object_footprints")
        self.assertEqual(dense["object_footprint_pixel_count"], len(footprint_cells))
        self.assertEqual(dense["object_footprint_per_category_pixel_count"]["couch"], len(footprint_cells))

    def test_world_state_summarize_objects_exports_graph_pcd_bbox_cells(self):
        class FakePoints:
            def __init__(self, points):
                self.points = points

        node = SimpleNamespace(
            center=[10, 20],
            caption="cabinet",
            object={
                "num_detections": 2,
                "pcd": FakePoints(np.array([[1.0, 2.0, 0.4], [1.2, 2.4, 0.6]], dtype=float)),
                "xyxy": [np.array([10, 20, 30, 40])],
                "pixel_area": [400],
            },
        )
        graph = SimpleNamespace(nodes=[node], text_goal="find the tv", map_size=100, map_resolution=10)

        summary = summarize_objects(graph)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["footprint_source"], "graph_pcd_bbox_cells")
        self.assertIsNotNone(summary[0]["bbox_rc"])
        self.assertGreater(summary[0]["footprint_pixel_count"], 0)
        self.assertTrue(summary[0]["footprint_rc_indices"])
        self.assertEqual(summary[0]["detection_xyxy_count"], 1)


if __name__ == "__main__":
    unittest.main()
