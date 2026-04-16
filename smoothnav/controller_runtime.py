"""Composition helpers for the layered SmoothNav runtime."""

from typing import Any, Dict, Optional

from smoothnav.tracing import strategy_to_dict, to_jsonable
from smoothnav.types import (
    GeometricGoal,
    MissionState,
    TacticalDecision,
    WorldState,
    stage_goal_from_strategy,
)


def stage_goal_to_dict(strategy_or_stage_goal) -> Optional[Dict[str, Any]]:
    stage_goal = stage_goal_from_strategy(strategy_or_stage_goal)
    return stage_goal.to_dict() if stage_goal is not None else None


def layered_trace_payload(
    *,
    world_state: Optional[WorldState],
    mission_state: Optional[MissionState],
    task_spec=None,
    task_belief=None,
    evidence_ledger=None,
    pending_proposals=None,
    current_strategy,
    pending_strategy,
    tactical_decision: Optional[TacticalDecision],
    geometric_goal: Optional[GeometricGoal],
    executor_command=None,
    executor_feedback=None,
    budget_state=None,
    terminal_decision=None,
) -> Dict[str, Any]:
    """Return the standardized layered trace block for a step."""

    return {
        "world_state_summary": (
            world_state.summary() if world_state is not None else None
        ),
        "mission_state_summary": (
            mission_state.to_dict() if mission_state is not None else None
        ),
        "task_spec": task_spec.to_dict() if task_spec is not None else None,
        "task_belief": task_belief.to_dict() if task_belief is not None else None,
        "evidence_ledger_summary": (
            evidence_ledger.summary()
            if evidence_ledger is not None and hasattr(evidence_ledger, "summary")
            else None
        ),
        "pending_proposals": (
            pending_proposals.to_dict()
            if pending_proposals is not None and hasattr(pending_proposals, "to_dict")
            else pending_proposals
        ),
        "current_stage_goal": stage_goal_to_dict(current_strategy),
        "pending_stage_goal": stage_goal_to_dict(pending_strategy),
        "tactical_decision": (
            tactical_decision.to_dict() if tactical_decision is not None else None
        ),
        "geometric_goal": (
            geometric_goal.to_dict() if geometric_goal is not None else None
        ),
        "executor_command": (
            executor_command.to_dict() if executor_command is not None else None
        ),
        "executor_feedback": (
            executor_feedback.to_dict() if executor_feedback is not None else None
        ),
        "budget_state": budget_state.to_dict() if budget_state is not None else None,
        "terminal_decision": (
            terminal_decision.to_dict() if terminal_decision is not None else None
        ),
        # Keep legacy strategy fields nearby for migration debugging.
        "current_strategy_legacy": strategy_to_dict(current_strategy),
        "pending_strategy_legacy": strategy_to_dict(pending_strategy),
    }


def compact_layered_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return to_jsonable(payload)
