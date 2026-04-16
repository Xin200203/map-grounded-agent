import unittest
from types import SimpleNamespace

from smoothnav.budget_context import BudgetGovernor
from smoothnav.evidence_ledger import EvidenceLedger
from smoothnav.executor_adapter import ExecutorAdapter, null_geometric_goal
from smoothnav.graph_delta import GraphDelta
from smoothnav.pending_proposals import PendingProposalManager
from smoothnav.tactical_arbiter import TacticalArbiter
from smoothnav.task_belief import TaskBeliefUpdater, reject_stale_stage_goal
from smoothnav.task_spec import parse_task_spec
from smoothnav.terminal_arbitration import TerminalArbiter
from smoothnav.types import (
    PendingProposalStatus,
    StageGoal,
    SteadyMode,
    TacticalMode,
    TerminalOutcome,
    TransitionIntent,
)
from smoothnav.world_state import WorldStateBuilder
from smoothnav.writer_guards import WriterToken, require_writer


class DummyAgent:
    def __init__(self):
        self.temp_goal = None
        self.temp_goal_epoch = None
        self.last_override_info = {
            "temp_goal_override": True,
            "adopted_goal_source": "temp_goal",
            "goal_epoch": 2,
        }

    def step(self, agent_input):
        return "obs", "rgbd", False, {}


class MEPContractTests(unittest.TestCase):
    def _world(self, step_idx=1, delta=None):
        builder = WorldStateBuilder()
        return builder.build(
            step_idx=step_idx,
            pose={"x": 0.0},
            local_pose=[0, 0, 0],
            graph=SimpleNamespace(nodes=[], room_nodes=[]),
            bev_map=SimpleNamespace(),
            controller_state=SimpleNamespace(
                explored_regions=[],
                no_progress_steps=0,
            ),
            graph_delta=delta or GraphDelta(event_types=["new_rooms"], new_rooms=["kitchen"]),
        )

    def test_task_spec_has_terminal_constraint_and_stop_condition(self):
        spec = parse_task_spec("find the chair", "text", task_id="ep0")

        self.assertEqual(spec.task_id, "ep0")
        self.assertEqual(spec.composition.value, "all_of")
        self.assertEqual(len(spec.candidate_stop_conditions), 1)
        self.assertEqual(spec.constraints[0].constraint_type.value, "terminal")

    def test_writer_guard_rejects_wrong_writer(self):
        with self.assertRaises(PermissionError):
            require_writer("belief_updater", WriterToken("planner"), "TaskBelief")

    def test_evidence_ledger_tracks_derived_and_invalidates_dependents(self):
        ledger = EvidenceLedger()
        observed = ledger.add_observation(
            "exists(region=kitchen)",
            source="graph_delta",
            scope="region",
            confidence=0.7,
            timestamp=1,
            entity_bindings={"region": "kitchen"},
        )
        derived = ledger.add_derived(
            "constraint_satisfied(constraint_id=c1)",
            source="belief_updater",
            scope="constraint",
            confidence=0.6,
            timestamp=2,
            derived_from=[observed.evidence_id],
        )

        ledger.mark_contradicted(observed.evidence_id, timestamp=3)

        self.assertFalse(ledger.is_valid(observed.evidence_id))
        self.assertFalse(ledger.is_valid(derived.evidence_id))
        self.assertEqual(ledger.get(derived.evidence_id).derived_from, [observed.evidence_id])

    def test_task_belief_update_and_stale_stage_rejection(self):
        spec = parse_task_spec("find the chair", "text")
        ledger = EvidenceLedger()
        updater = TaskBeliefUpdater(spec, ledger)
        world = self._world(step_idx=5)

        belief = updater.update(world)
        stale_stage = StageGoal(
            stage_id="stage_old",
            task_epoch=belief.task_epoch,
            belief_epoch=max(0, belief.belief_epoch - 4),
            world_epoch=world.world_epoch,
            stage_type="search",
        )

        self.assertGreaterEqual(belief.belief_epoch, 1)
        self.assertEqual(
            reject_stale_stage_goal(
                stale_stage,
                belief,
                world,
                ledger,
                belief_lag_tolerance=0,
            ),
            "belief_epoch_stale",
        )

    def test_pending_proposal_k_limit_and_single_adoption(self):
        spec = parse_task_spec("find the chair", "text")
        ledger = EvidenceLedger()
        updater = TaskBeliefUpdater(spec, ledger)
        world = self._world(step_idx=1)
        belief = updater.update(world)
        manager = PendingProposalManager(max_pending=2)

        for idx, target in enumerate(["unexplored north", "bedroom", "object: chair"]):
            manager.create(
                StageGoal(
                    stage_id=f"stage_{idx}",
                    task_epoch=belief.task_epoch,
                    belief_epoch=belief.belief_epoch,
                    world_epoch=world.world_epoch,
                    stage_type="search",
                    target_region=target,
                    stage_selection_confidence=0.3 + idx * 0.1,
                ),
                task_belief=belief,
                world_state=world,
                created_reason="test",
            )

        adopted = manager.adopt_best(task_belief=belief, world_state=world, ledger=ledger)

        self.assertLessEqual(len(manager.pending()), 1)
        self.assertIsNotNone(adopted)
        self.assertEqual(manager.active().status, PendingProposalStatus.ADOPTED)
        self.assertEqual(manager.active().stage_goal.target_region, "object: chair")

    def test_tactical_decision_separates_mode_and_transition_intent(self):
        arbiter = TacticalArbiter()
        decision = arbiter.decide(
            world_state=self._world(delta=GraphDelta(event_types=["frontier_reached"], frontier_reached=True)),
            mission_state=None,
            current_strategy=SimpleNamespace(target_region="unexplored north"),
            pending_strategy=None,
            needs_initial_plan=False,
        )

        self.assertEqual(decision.mode, TacticalMode.REPLAN_REQUIRED)
        self.assertEqual(decision.steady_mode, SteadyMode.FOLLOW_STAGE_GOAL)
        self.assertEqual(decision.transition_intent, TransitionIntent.REQUEST_REPLAN)
        self.assertGreaterEqual(decision.min_dwell_steps, 1)
        self.assertTrue(decision.review_trigger)

    def test_executor_feedback_contract_reports_local_override(self):
        adapter = ExecutorAdapter(DummyAgent())
        command = adapter.build_command(
            geometric_goal=null_geometric_goal(),
            strategy_epoch=1,
            goal_epoch=2,
        )

        result = adapter.step({"step_idx": 9}, command)

        self.assertEqual(result.executor_feedback.goal_epoch, 2)
        self.assertEqual(result.executor_feedback.executor_step_idx, 9)
        self.assertEqual(result.executor_feedback.override_reason, "temp_goal")
        self.assertEqual(result.executor_feedback.adopted_goal_source, "temp_goal")

    def test_terminal_arbiter_closes_failure_outcome(self):
        spec = parse_task_spec("find the chair", "text")
        ledger = EvidenceLedger()
        belief = TaskBeliefUpdater(spec, ledger).belief
        budget = BudgetGovernor(planner_call_budget=1)
        budget.record_planner_call(1)

        decision = TerminalArbiter().decide(
            done=False,
            infos={},
            task_belief=belief,
            budget_state=budget.state,
            controller_state=SimpleNamespace(no_progress_steps=0, consecutive_grounding_noops=0),
        )

        self.assertEqual(decision.outcome, TerminalOutcome.FAILURE_BUDGET_EXHAUSTED)
        self.assertTrue(decision.is_terminal)


if __name__ == "__main__":
    unittest.main()
