"""Thin adapter around UniGoal_Agent for the Layer 5 executor boundary."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from smoothnav.types import (
    ExecutorCommand,
    ExecutorFeedback,
    GeometricGoal,
    GeometricGoalType,
)


@dataclass
class ExecutorStepResult:
    obs: Any
    rgbd: Any
    done: bool
    infos: Dict[str, Any]
    adoption_trace: Dict[str, Any] = field(default_factory=dict)
    executor_feedback: ExecutorFeedback = field(default_factory=ExecutorFeedback)


class ExecutorAdapter:
    """Wrap UniGoal_Agent.step() without rewriting the mature executor."""

    def __init__(self, agent):
        self.agent = agent
        self._last_override_key = None
        self._override_duration = 0
        self._detour_steps = 0

    def build_command(
        self,
        *,
        geometric_goal: GeometricGoal,
        strategy_epoch: int,
        goal_epoch: int,
        allow_target_lock: bool = True,
        allow_recovery: bool = True,
        clear_temp_goal: bool = False,
    ) -> ExecutorCommand:
        return ExecutorCommand(
            geometric_goal=geometric_goal,
            allow_target_lock=allow_target_lock,
            allow_recovery=allow_recovery,
            clear_temp_goal=clear_temp_goal,
            strategy_epoch=int(strategy_epoch),
            goal_epoch=int(goal_epoch),
        )

    def step(self, agent_input, command: ExecutorCommand) -> ExecutorStepResult:
        agent_input = dict(agent_input)
        agent_input["strategy_epoch"] = command.strategy_epoch
        agent_input["goal_epoch"] = command.goal_epoch
        agent_input["allow_target_lock"] = bool(command.allow_target_lock)
        agent_input["allow_recovery"] = bool(command.allow_recovery)
        agent_input["clear_temp_goal"] = bool(command.clear_temp_goal)
        if command.clear_temp_goal and hasattr(self.agent, "temp_goal"):
            self.agent.temp_goal = None
            if hasattr(self.agent, "temp_goal_epoch"):
                self.agent.temp_goal_epoch = None

        obs, rgbd, done, infos = self.agent.step(agent_input)
        adoption_trace = dict(getattr(self.agent, "last_override_info", {}) or {})
        executor_feedback = self._build_feedback(
            adoption_trace,
            goal_epoch=command.goal_epoch,
            executor_step_idx=int(agent_input.get("step_idx", 0) or 0),
        )
        return ExecutorStepResult(
            obs=obs,
            rgbd=rgbd,
            done=done,
            infos=infos,
            adoption_trace=adoption_trace,
            executor_feedback=executor_feedback,
        )

    def _build_feedback(
        self,
        adoption_trace: Dict[str, Any],
        *,
        goal_epoch: int,
        executor_step_idx: int,
    ) -> ExecutorFeedback:
        source = adoption_trace.get("adopted_goal_source", "unknown")
        override_reason = None
        if adoption_trace.get("temp_goal_override"):
            override_reason = "temp_goal"
        elif adoption_trace.get("stuck_goal_override"):
            override_reason = "stuck_goal"
        elif adoption_trace.get("visible_target_override"):
            override_reason = "visible_target"
        elif adoption_trace.get("global_goal_override"):
            override_reason = "global_goal"

        key = (int(goal_epoch or 0), override_reason, source)
        if override_reason is None:
            self._override_duration = 0
            self._detour_steps = 0
            self._last_override_key = key
        elif key == self._last_override_key:
            self._override_duration += 1
            self._detour_steps += 1
        else:
            self._override_duration = 1
            self._detour_steps = 1
            self._last_override_key = key

        local_unreachable = bool(
            adoption_trace.get("local_unreachable")
            or adoption_trace.get("stuck_goal_override")
        )
        forced_stop = bool(adoption_trace.get("forced_stop", False))
        escalation_required = bool(
            local_unreachable
            or forced_stop
            or self._override_duration > 8
            or adoption_trace.get("temp_goal_suppressed_by_epoch", False)
        )
        return ExecutorFeedback(
            goal_epoch=int(goal_epoch or 0),
            executor_step_idx=int(executor_step_idx or 0),
            override_reason=override_reason,
            override_duration=int(self._override_duration),
            actual_detour_steps=int(self._detour_steps),
            local_unreachable=local_unreachable,
            forced_stop=forced_stop,
            escalation_required=escalation_required,
            adopted_goal_source=str(source or "unknown"),
        )


def null_geometric_goal() -> GeometricGoal:
    return GeometricGoal(goal_type=GeometricGoalType.NONE)
