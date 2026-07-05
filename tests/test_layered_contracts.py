import unittest
from types import SimpleNamespace

from smoothnav.controller_runtime import layered_trace_payload
from smoothnav.executor_adapter import (
    ExecutorAdapter,
    null_geometric_goal,
    should_clear_temp_goal_for_command,
    should_disable_recovery_for_command,
)
from smoothnav.graph_delta import GraphDelta
from smoothnav.mission_state import MissionProgressManager
from smoothnav.tactical_arbiter import TacticalArbiter
from smoothnav.types import (
    GeometricGoalType,
    StageGoal,
    TacticalMode,
    stage_goal_from_strategy,
)
from smoothnav.world_state import build_world_state


class DummyAgent:
    def __init__(self):
        self.last_override_info = {
            "adopted_goal_source": "exp_goal",
            "adopted_goal_summary": [1, 2],
        }
        self.seen_input = None
        self.temp_goal = "placeholder"
        self.temp_goal_epoch = 3

    def step(self, agent_input):
        self.seen_input = dict(agent_input)
        return "obs", "rgbd", False, {"ok": True}


class LayeredContractTests(unittest.TestCase):
    def test_stage_goal_from_strategy_keeps_legacy_fields(self):
        strategy = SimpleNamespace(
            target_region="object: chair",
            bias_position=(10, 20),
            reasoning="search chair",
            explored_regions=["bedroom"],
            anchor_object="chair",
        )

        stage_goal = stage_goal_from_strategy(strategy)

        self.assertEqual(stage_goal.stage_type, "object")
        self.assertEqual(stage_goal.target_object, "chair")
        self.assertEqual(stage_goal.reasoning, "search chair")
        self.assertEqual(stage_goal.bias_position, (10, 20))

    def test_mission_progress_records_stage(self):
        manager = MissionProgressManager("find chair", "text")
        manager.note_stage_goal(
            StageGoal(
                stage_type="room",
                target_region="bedroom",
                semantic_intent="search bedroom",
                planner_reason="bedrooms may contain chairs",
            ),
            replan_reason="new_room",
        )

        self.assertEqual(manager.state.stage_status, "active")
        self.assertEqual(manager.state.current_stage_id, 1)
        self.assertEqual(manager.state.replan_reason, "new_room")
        self.assertEqual(manager.state.required_evidence, ["bedroom"])

    def test_tactical_arbiter_promotes_more_specific_pending(self):
        arbiter = TacticalArbiter(direction_reuse_limit=1)
        world = SimpleNamespace(
            graph_delta=GraphDelta(event_types=["new_rooms"], new_rooms=["kitchen"])
        )
        current = SimpleNamespace(
            target_region="unexplored north",
            bias_position=(1, 2),
            reasoning="explore",
            explored_regions=[],
            anchor_object="",
        )
        pending = SimpleNamespace(
            target_region="object: chair",
            bias_position=(3, 4),
            reasoning="specific",
            explored_regions=[],
            anchor_object="chair",
        )

        decision = arbiter.decide(
            world_state=world,
            mission_state=None,
            current_strategy=current,
            pending_strategy=pending,
            needs_initial_plan=False,
        )

        self.assertEqual(decision.mode, TacticalMode.ADOPT_PENDING_STAGE)
        self.assertTrue(decision.should_promote_pending)

    def test_tactical_arbiter_post_grounding_patch_holds_out_of_local_window(self):
        arbiter = TacticalArbiter(direction_reuse_limit=1)

        decision = arbiter.post_grounding_patch(
            SimpleNamespace(failure_code="out_of_local_window")
        )

        self.assertEqual(decision.mode, TacticalMode.HOLD_AND_WAIT_FOR_FRONTIER)
        self.assertEqual(decision.reason, "grounding_hold:out_of_local_window")

    def test_executor_adapter_passes_command_fields(self):
        agent = DummyAgent()
        adapter = ExecutorAdapter(agent)
        command = adapter.build_command(
            geometric_goal=null_geometric_goal(),
            strategy_epoch=7,
            goal_epoch=3,
            allow_target_lock=False,
            allow_recovery=False,
        )

        result = adapter.step({"goal": "map"}, command)

        self.assertEqual(result.obs, "obs")
        self.assertEqual(agent.seen_input["strategy_epoch"], 7)
        self.assertEqual(agent.seen_input["goal_epoch"], 3)
        self.assertFalse(agent.seen_input["allow_target_lock"])
        self.assertFalse(agent.seen_input["allow_recovery"])

    def test_executor_adapter_clears_temp_goal_when_requested(self):
        agent = DummyAgent()
        adapter = ExecutorAdapter(agent)
        command = adapter.build_command(
            geometric_goal=null_geometric_goal(),
            strategy_epoch=1,
            goal_epoch=1,
            clear_temp_goal=True,
        )

        adapter.step({"goal": "map"}, command)

        self.assertIsNone(agent.temp_goal)
        self.assertIsNone(agent.temp_goal_epoch)
        self.assertTrue(agent.seen_input["clear_temp_goal"])

    def test_temp_goal_is_cleared_for_direction_after_long_override(self):
        adoption_trace = {
            "temp_goal_override": True,
        }

        self.assertTrue(
            should_clear_temp_goal_for_command(
                adoption_trace,
                override_duration=9,
                current_target_region="unexplored east",
                threshold=8,
            )
        )

    def test_temp_goal_is_cleared_earlier_for_object_targets(self):
        adoption_trace = {
            "temp_goal_override": True,
        }

        self.assertTrue(
            should_clear_temp_goal_for_command(
                adoption_trace,
                override_duration=4,
                current_target_region="object: table cabinet",
                threshold=8,
            )
        )
        self.assertFalse(
            should_clear_temp_goal_for_command(
                adoption_trace,
                override_duration=3,
                current_target_region="object: table cabinet",
                threshold=8,
            )
        )

    def test_temp_goal_is_cleared_earlier_for_room_targets(self):
        adoption_trace = {
            "temp_goal_override": True,
        }

        self.assertTrue(
            should_clear_temp_goal_for_command(
                adoption_trace,
                override_duration=4,
                current_target_region="bedroom",
                threshold=8,
            )
        )
        self.assertFalse(
            should_clear_temp_goal_for_command(
                adoption_trace,
                override_duration=3,
                current_target_region="bedroom",
                threshold=8,
            )
        )

    def test_direction_recovery_is_disabled_after_long_override(self):
        self.assertTrue(
            should_disable_recovery_for_command(
                {"stuck_goal_override": True},
                override_duration=5,
                current_target_region="unexplored east",
                threshold=4,
            )
        )
        self.assertTrue(
            should_disable_recovery_for_command(
                {"temp_goal_override": True},
                override_duration=5,
                current_target_region="unexplored east",
                threshold=4,
            )
        )
        self.assertFalse(
            should_disable_recovery_for_command(
                {"stuck_goal_override": True},
                override_duration=2,
                current_target_region="unexplored east",
                threshold=4,
            )
        )
        self.assertFalse(
            should_disable_recovery_for_command(
                {"stuck_goal_override": True},
                override_duration=5,
                current_target_region="",
                threshold=4,
            )
        )

    def test_object_recovery_is_disabled_after_shorter_override(self):
        self.assertTrue(
            should_disable_recovery_for_command(
                {"stuck_goal_override": True},
                override_duration=3,
                current_target_region="object: table cabinet",
                threshold=4,
            )
        )

    def test_room_recovery_is_disabled_after_shorter_override(self):
        self.assertTrue(
            should_disable_recovery_for_command(
                {"stuck_goal_override": True},
                override_duration=3,
                current_target_region="bedroom",
                threshold=4,
            )
        )
        self.assertTrue(
            should_disable_recovery_for_command(
                {"temp_goal_override": True},
                override_duration=3,
                current_target_region="bedroom",
                threshold=4,
            )
        )
        self.assertFalse(
            should_disable_recovery_for_command(
                {"stuck_goal_override": True},
                override_duration=2,
                current_target_region="bedroom",
                threshold=4,
            )
        )
        self.assertTrue(
            should_disable_recovery_for_command(
                {"temp_goal_override": True},
                override_duration=3,
                current_target_region="object: table cabinet",
                threshold=4,
            )
        )
        self.assertFalse(
            should_disable_recovery_for_command(
                {"stuck_goal_override": True},
                override_duration=2,
                current_target_region="object: table cabinet",
                threshold=4,
            )
        )

    def test_layered_trace_payload_contains_required_sections(self):
        graph = SimpleNamespace(nodes=[], room_nodes=[])
        controller_state = SimpleNamespace(
            explored_regions=["bedroom"],
            no_progress_steps=2,
        )
        delta = GraphDelta(no_progress=True, event_types=["no_progress"])
        world = build_world_state(
            step_idx=5,
            pose={"x": 1.0},
            local_pose=[0, 0, 0],
            graph=graph,
            bev_map=SimpleNamespace(),
            controller_state=controller_state,
            graph_delta=delta,
        )
        mission = MissionProgressManager("find chair", "text").state
        goal = null_geometric_goal()
        self.assertEqual(goal.goal_type, GeometricGoalType.NONE)

        payload = layered_trace_payload(
            world_state=world,
            mission_state=mission,
            current_strategy=None,
            pending_strategy=None,
            tactical_decision=None,
            geometric_goal=goal,
            executor_command=None,
        )

        self.assertIn("world_state_summary", payload)
        self.assertIn("mission_state_summary", payload)
        self.assertIn("tactical_decision", payload)
        self.assertIn("geometric_goal", payload)


if __name__ == "__main__":
    unittest.main()
