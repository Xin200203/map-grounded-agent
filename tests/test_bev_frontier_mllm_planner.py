import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from smoothnav.bev_frontier_planner import build_bev_frontier_prompt
    from smoothnav.strategy_grounding import apply_strategy
except Exception:  # pragma: no cover
    build_bev_frontier_prompt = None
    apply_strategy = None


class FakeMLLMPlanner:
    def __init__(self):
        self.calls = []

    def plan(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "schema_version": "smoothnav.bev_frontier_mllm_planner.v1",
            "status": "valid",
            "reason": "",
            "prompt_hash": "abc123",
            "allowed_branch_ids": ["B1", "B2"],
            "verdict": {
                "valid": True,
                "selected_branch_id": "B2",
                "evidence_level": "anchor",
                "target_visible": False,
                "ranked_branch_ids": ["B2", "B1"],
                "prior_scores": [0.2, 1.0],
            },
        }


class Trace:
    def __init__(self):
        self.records = []

    def record_mllm_frontier_call(self, episode_id, payload):
        self.records.append((episode_id, payload))


class MockGraph:
    def __init__(self):
        self.nodes = []
        self.room_nodes = []
        self.planner_frontier_prior = None
        self.calls = []
        self.last_goal_debug = {}
        self.last_goal_replay_snapshot = None

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_edges(self):
        return []

    def get_goal(self, goal=None, *, record_selection_history=True):
        self.calls.append(
            {
                "goal": goal,
                "record_selection_history": record_selection_history,
                "planner_frontier_prior": self.planner_frontier_prior,
            }
        )
        self.last_goal_replay_snapshot = {
            "agent_coord": [5, 5],
            "frontier_branches": {
                "schema_version": "smoothnav.frontier_branches.v1",
                "branch_count": 2,
                "candidate_branch_ids": ["B1", "B2"],
                "local_index_to_branch_id": {"0": "B1", "1": "B2"},
                "branches": [
                    {
                        "id": "B1",
                        "candidate_point_count": 1,
                        "representative_coord": [10, 10],
                        "score_terms_aggregate": {"final_score_max": 3.0},
                    },
                    {
                        "id": "B2",
                        "candidate_point_count": 1,
                        "representative_coord": [20, 20],
                        "score_terms_aggregate": {"final_score_max": 2.0},
                    },
                ],
            },
        }
        if self.planner_frontier_prior:
            self.last_goal_debug = {
                "selected_frontier": [20, 20],
                "selected_frontier_branch_id": "B2",
                "selected_frontier_score_breakdown": {"branch_id": "B2"},
                "planner_prior_weight": 4.0,
                "topk_frontiers": [{"rank": 1, "frontier": [20, 20]}],
            }
            return (20, 20)
        self.last_goal_debug = {
            "selected_frontier": [10, 10],
            "selected_frontier_branch_id": "B1",
            "selected_frontier_score_breakdown": {"branch_id": "B1"},
            "topk_frontiers": [{"rank": 1, "frontier": [10, 10]}],
        }
        return (10, 10)


class MockBoundary:
    def __getitem__(self, key):
        row_idx, col_idx = key
        values = [[0, 0, 0, 0]]
        return values[row_idx][col_idx]


@unittest.skipIf(build_bev_frontier_prompt is None, "BEV frontier planner deps unavailable")
class BEVFrontierMLLMPlannerTests(unittest.TestCase):
    def test_prompt_requires_branch_id_not_coordinate(self):
        prompt = build_bev_frontier_prompt(
            goal_description="find the tv in a living room",
            strategy=SimpleNamespace(target_region="unexplored north", explored_regions=[]),
            graph=SimpleNamespace(room_nodes=[], nodes=[], get_edges=lambda: []),
            frontier_branches={
                "candidate_branch_ids": ["B1"],
                "branches": [
                    {
                        "id": "B1",
                        "candidate_point_count": 2,
                        "representative_coord": [12, 13],
                    }
                ],
            },
        )

        self.assertIn("BEV BRANCH IDS", prompt)
        self.assertIn("selected_branch_id", prompt)
        self.assertIn("not a raw coordinate", prompt)
        self.assertIn("O#=observed object", prompt)
        self.assertIn("R#=localized room/function-area region", prompt)
        self.assertIn("evidence_level", prompt)
        self.assertIn("executable_decision_rank", prompt)

    def test_apply_strategy_recomputes_with_mllm_branch_prior(self):
        graph = MockGraph()
        planner = FakeMLLMPlanner()
        trace = Trace()
        bev_map = SimpleNamespace(
            full_map="full-map",
            full_pose="full-pose",
            local_map_boundary=MockBoundary(),
        )
        args = SimpleNamespace(
            local_width=100,
            local_height=100,
            mllm_frontier_planner_mode="online",
        )
        strategy = SimpleNamespace(
            target_region="unexplored north",
            bias_position=(9, 9),
            explored_regions=[],
            anchor_object="",
        )
        goals = [0, 0]

        result = apply_strategy(
            strategy,
            graph,
            bev_map,
            args,
            goals,
            mllm_frontier_planner=planner,
            goal_description="find the tv",
            trigger="unit_test",
            episode_id=7,
            step_idx=11,
            trace_writer=trace,
        )

        self.assertEqual(len(graph.calls), 2)
        self.assertFalse(graph.calls[0]["record_selection_history"])
        self.assertTrue(graph.calls[1]["record_selection_history"])
        self.assertEqual(graph.calls[1]["planner_frontier_prior"]["selected_branch_id"], "B2")
        self.assertEqual(result.selected_frontier, (20, 20))
        self.assertEqual(result.projected_goal, (20, 20))
        self.assertTrue(result.graph_debug["mllm_frontier"]["applied"])
        self.assertEqual(result.graph_debug["mllm_frontier"]["evidence_level"], "anchor")
        self.assertEqual(trace.records[0][0], 7)
        self.assertTrue(trace.records[0][1]["applied"])


if __name__ == "__main__":
    unittest.main()
