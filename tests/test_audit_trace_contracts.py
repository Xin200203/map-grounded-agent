"""Tests for layered SmoothNav trace contract auditor."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.audit_trace_contracts import audit_paths, audit_run_dir  # noqa: E402


class AuditTraceContractsTests(unittest.TestCase):
    def _make_run(self, run_dir: Path, steps, *, summary=None, config=None, episode_results=None, planner_calls=None):
        (run_dir / "step_traces").mkdir(parents=True)
        (run_dir / "planner_calls").mkdir(parents=True)
        (run_dir / "monitor_calls").mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                summary
                or {
                    "run_id": run_dir.name,
                    "controller_profile": "smoothnav-full",
                    "SR": 0.0,
                    "SPL": 0.0,
                    "terminal_outcome_counts": {"FAILURE_NO_PROGRESS_TIMEOUT": 1},
                }
            )
        )
        (run_dir / "effective_config.json").write_text(
            json.dumps(
                config
                or {
                    "run_id": run_dir.name,
                    "controller_profile": "smoothnav-full",
                    "episode_id": 228,
                    "goal_type": "text",
                    "split": "val",
                    "text_goal_dataset": "data/datasets/textnav/val/val_text.json.gz",
                    "success_dist": 1.0,
                    "max_episode_length": 1000,
                    "seed": 1,
                    "llm_model": "test-model",
                }
            )
        )
        (run_dir / "episode_results.json").write_text(
            json.dumps(
                episode_results
                if episode_results is not None
                else [
                    {
                        "habitat_episode_no": 228,
                        "success": 0.0,
                        "spl": 0.0,
                        "terminal_outcome": "FAILURE_NO_PROGRESS_TIMEOUT",
                        "total_steps": steps[-1].get("step_idx", 0) + 1,
                    }
                ]
            )
        )
        with (run_dir / "step_traces" / "episode_000000.jsonl").open("w") as handle:
            for step in steps:
                handle.write(json.dumps(step) + "\n")
        with (run_dir / "planner_calls" / "episode_000000.jsonl").open("w") as handle:
            for call in planner_calls or []:
                handle.write(json.dumps(call) + "\n")
        (run_dir / "monitor_calls" / "episode_000000.jsonl").write_text("")

    def test_target_value_collapse_is_dominant_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_value_collapse"
            steps = [
                {
                    "step_idx": 0,
                    "task_spec": {"primary_goal": "find the gray television screen"},
                    "current_strategy": {"target_region": "unexplored north"},
                    "graph_delta": {"event_types": []},
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 10,
                    "current_strategy": {"target_region": "unexplored target:tv", "anchor_object": "tv"},
                    "graph_delta": {
                        "event_types": ["target_candidate_detected"],
                        "target_candidate_details": [{"caption": "tv", "center": [10, 10], "target_relevance": 1.0}],
                    },
                    "planner_called": True,
                    "planner_reasons": ["target_candidate_detected"],
                    "grounding_events": [
                        {
                            "trigger": "target_candidate_detected",
                            "bias_input": [10, 10],
                            "selected_frontier": [100, 100],
                            "projected_goal": [20, 20],
                            "local_projection_valid": True,
                            "changed": True,
                            "selected_frontier_score_breakdown": {
                                "bias_score": 0.0,
                                "base_score": 0.0,
                                "novelty_score": 0.8,
                                "actionability_score": 1.0,
                                "final_score": 2.8,
                                "semantic_bias_weight": 4.0,
                                "bias_distance": 25.0,
                            },
                            "top1_top2_gap": 0.0,
                        }
                    ],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 20,
                    "current_strategy": {"target_region": "unexplored target:tv", "anchor_object": "tv"},
                    "graph_delta": {"event_types": []},
                    "grounding_events": [
                        {
                            "trigger": "target_anchor_local_map_refresh",
                            "bias_input": [10, 10],
                            "selected_frontier": [101, 100],
                            "projected_goal": [21, 20],
                            "local_projection_valid": True,
                            "changed": True,
                            "selected_frontier_score_breakdown": {
                                "bias_score": 0.0,
                                "base_score": 0.0,
                                "novelty_score": 0.7,
                                "actionability_score": 1.0,
                                "final_score": 2.7,
                                "semantic_bias_weight": 4.0,
                                "bias_distance": 26.0,
                            },
                            "top1_top2_gap": 0.0,
                        }
                    ],
                    "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"},
                },
            ]
            self._make_run(
                run_dir,
                steps,
                planner_calls=[
                    {
                        "step_idx": 10,
                        "prompt_hash": "abc",
                        "raw_prompt": "[object] tv",
                        "raw_response": '{"choice_type":"object","choice_id":"tv"}',
                        "choice_id": "tv",
                        "strategy": {"target_region": "unexplored target:tv"},
                    }
                ],
            )
            report = audit_run_dir(run_dir)
            self.assertEqual(report["dominant_failure_layer"], "grounding_frontier_value")
            grounding = report["layers"]["grounding_frontier_value"]["metrics"]
            self.assertEqual(grounding["target_bias_score_zero_rate"], 1.0)
            self.assertEqual(grounding["target_value_flatline_rate"], 1.0)
            self.assertEqual(set(report["layers"]), {
                "suite_artifact_config",
                "dataset_scene_episode",
                "determinism_branch_reproducibility",
                "perception_graph_target_evidence",
                "planner_menu_choice",
                "strategy_contract",
                "grounding_frontier_value",
                "grounding_stage_snapshot_replay",
                "local_map_lifecycle",
                "controller_pending_prefetch",
                "target_anchor_attempt",
                "monitor_trigger_coverage",
                "executor_low_level_control",
                "terminal_metric_arbitration",
                "replay_counterfactual_coverage",
            })

    def test_target_progress_term_prevents_false_semantic_collapse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_target_progress"
            steps = [
                {
                    "step_idx": 10,
                    "task_spec": {"primary_goal": "find the television"},
                    "current_strategy": {"target_region": "unexplored target:tv", "anchor_object": "tv"},
                    "graph_delta": {
                        "event_types": ["target_candidate_detected"],
                        "target_candidate_details": [{"caption": "tv", "center": [10, 10]}],
                    },
                    "planner_called": True,
                    "grounding_events": [
                        {
                            "trigger": "target_candidate_detected",
                            "bias_input": [10, 10],
                            "selected_frontier": [100, 100],
                            "projected_goal": [20, 20],
                            "local_projection_valid": True,
                            "changed": True,
                            "selected_frontier_score_breakdown": {
                                "bias_score": 0.0,
                                "target_progress_score": 0.8,
                                "target_progress_weight": 4.0,
                                "base_score": 0.0,
                                "novelty_score": 0.2,
                                "actionability_score": 1.0,
                                "final_score": 5.2,
                                "semantic_bias_weight": 4.0,
                                "bias_distance": 25.0,
                            },
                            "top1_top2_gap": 0.5,
                        }
                    ],
                    "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"},
                }
            ]
            self._make_run(
                run_dir,
                steps,
                planner_calls=[
                    {
                        "step_idx": 10,
                        "raw_prompt": "[object] tv",
                        "raw_response": '{"choice_type":"object","choice_id":"tv"}',
                        "choice_id": "tv",
                        "strategy": {"target_region": "unexplored target:tv"},
                    }
                ],
            )

            report = audit_run_dir(run_dir)
            grounding = report["layers"]["grounding_frontier_value"]
            self.assertNotEqual(grounding["status"], "fail")
            self.assertGreater(
                grounding["metrics"]["target_semantic_term_share_median"],
                0.05,
            )

    def test_no_target_branch_is_branch_coverage_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_no_target"
            steps = [
                {
                    "step_idx": 0,
                    "task_spec": {"primary_goal": "find the television"},
                    "current_strategy": {"target_region": "unexplored north"},
                    "graph_delta": {"event_types": ["frontier_near"]},
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 998,
                    "current_strategy": {"target_region": "unexplored east"},
                    "graph_delta": {"event_types": []},
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"},
                },
            ]
            self._make_run(run_dir, steps)
            report = audit_run_dir(run_dir)
            self.assertEqual(report["dominant_failure_layer"], "determinism_branch_reproducibility")
            self.assertEqual(
                report["layers"]["perception_graph_target_evidence"]["metrics"]["target_event_count"],
                0,
            )

    def test_harmful_pending_promotion_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_pending_drop"
            steps = [
                {
                    "step_idx": 10,
                    "task_spec": {"primary_goal": "find the television"},
                    "current_strategy": {"target_region": "unexplored target:tv"},
                    "pending_strategy": {"target_region": "unexplored east"},
                    "graph_delta": {
                        "event_types": ["target_candidate_detected"],
                        "target_candidate_details": [{"caption": "tv", "center": [1, 1]}],
                    },
                    "planner_called": True,
                    "grounding_events": [
                        {
                            "trigger": "target_candidate_detected",
                            "bias_input": [1, 1],
                            "selected_frontier": [2, 2],
                            "selected_frontier_score_breakdown": {"bias_score": 1.0, "base_score": 1.0, "final_score": 5.0, "semantic_bias_weight": 4.0},
                        }
                    ],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 11,
                    "current_strategy": {"target_region": "unexplored east"},
                    "pending_strategy": {},
                    "graph_delta": {"event_types": ["frontier_reached"]},
                    "planner_reasons": ["frontier_reached"],
                    "pending_promoted": True,
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"},
                },
            ]
            self._make_run(
                run_dir,
                steps,
                planner_calls=[
                    {
                        "step_idx": 10,
                        "raw_prompt": "[object] tv",
                        "raw_response": '{"choice_type":"object","choice_id":"tv"}',
                        "choice_id": "tv",
                        "strategy": {"target_region": "unexplored target:tv"},
                    }
                ],
            )
            report = audit_run_dir(run_dir)
            self.assertEqual(report["dominant_failure_layer"], "controller_pending_prefetch")
            pending = report["layers"]["controller_pending_prefetch"]["metrics"]
            self.assertEqual(pending["harmful_pending_promotion_count"], 1)

    def test_target_seen_but_planner_ignores_target_is_planner_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_planner_ignores_target"
            steps = [
                {
                    "step_idx": 0,
                    "task_spec": {"primary_goal": "find the television"},
                    "current_strategy": {"target_region": "unexplored north"},
                    "graph_delta": {"event_types": []},
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "RUNNING"},
                },
                {
                    "step_idx": 100,
                    "current_strategy": {"target_region": "unexplored east"},
                    "graph_delta": {
                        "event_types": ["target_candidate_detected"],
                        "target_candidate_details": [{"caption": "tv", "center": [3, 4]}],
                    },
                    "planner_called": True,
                    "planner_reasons": ["target_candidate_detected"],
                    "grounding_events": [],
                    "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"},
                },
            ]
            self._make_run(
                run_dir,
                steps,
                planner_calls=[
                    {
                        "step_idx": 100,
                        "raw_prompt": "OBJECTS:\n  [object] tv",
                        "raw_response": '{"choice_type":"direction","choice_id":"east"}',
                        "choice_id": "east",
                        "strategy": {"target_region": "unexplored east"},
                    }
                ],
            )
            report = audit_run_dir(run_dir)
            self.assertEqual(report["dominant_failure_layer"], "planner_menu_choice")
            self.assertFalse(report["layers"]["planner_menu_choice"]["metrics"]["selected_target_after_target"])


    def test_aggregate_compares_branch_coverage_across_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            no_target = root / "profile" / "date" / "run_no_target"
            target = root / "profile" / "date" / "run_target"
            self._make_run(
                no_target,
                [
                    {"step_idx": 0, "task_spec": {"primary_goal": "find tv"}, "current_strategy": {"target_region": "unexplored north"}, "graph_delta": {"event_types": []}, "grounding_events": [], "terminal_decision": {"outcome": "RUNNING"}},
                    {"step_idx": 50, "current_strategy": {"target_region": "unexplored north"}, "graph_delta": {"event_types": []}, "grounding_events": [], "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"}},
                ],
            )
            self._make_run(
                target,
                [
                    {"step_idx": 0, "task_spec": {"primary_goal": "find tv"}, "current_strategy": {"target_region": "unexplored north"}, "graph_delta": {"event_types": []}, "grounding_events": [], "terminal_decision": {"outcome": "RUNNING"}},
                    {"step_idx": 20, "current_strategy": {"target_region": "unexplored target:tv"}, "graph_delta": {"event_types": ["target_candidate_detected"], "target_candidate_details": [{"caption": "tv"}]}, "planner_called": True, "grounding_events": [], "terminal_decision": {"outcome": "FAILURE_NO_PROGRESS_TIMEOUT"}},
                ],
            )
            report = audit_paths([str(root)])
            self.assertEqual(report["num_runs"], 2)
            self.assertEqual(report["branch_reproducibility_comparisons"][0]["target_branch_coverage_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
