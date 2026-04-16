"""Tactical arbitration layer for SmoothNav."""

from typing import List

from smoothnav.graph_delta import strategy_specificity
from smoothnav.transition_table import next_mode
from smoothnav.types import (
    SteadyMode,
    TacticalDecision,
    TacticalMode,
    TransitionIntent,
    stage_goal_from_strategy,
)


class TacticalArbiter:
    """Heuristic-first arbiter that centralizes tactical mode selection."""

    def __init__(self, *, direction_reuse_limit: int = 1, hold_max_steps: int = 5):
        self.direction_reuse_limit = int(direction_reuse_limit or 1)
        self.hold_max_steps = max(1, int(hold_max_steps))
        self.mode_epoch = 0
        self.steady_mode = SteadyMode.FOLLOW_STAGE_GOAL
        self._hold_enter_step = None

    def reset(self) -> None:
        self.mode_epoch = 0
        self.steady_mode = SteadyMode.FOLLOW_STAGE_GOAL
        self._hold_enter_step = None

    def decide(
        self,
        *,
        world_state,
        mission_state,
        current_strategy,
        pending_strategy,
        needs_initial_plan: bool,
        no_frontiers: bool = False,
    ) -> TacticalDecision:
        graph_delta = world_state.graph_delta
        current_stage_goal = stage_goal_from_strategy(
            current_strategy,
            world_epoch=int(getattr(world_state, "world_epoch", 0) or 0),
        )
        pending_stage_goal = stage_goal_from_strategy(
            pending_strategy,
            world_epoch=int(getattr(world_state, "world_epoch", 0) or 0),
        )
        event_types: List[str] = list(getattr(graph_delta, "event_types", []) or [])

        if needs_initial_plan:
            return self._decision(
                mode=TacticalMode.REPLAN_REQUIRED,
                reason="initial_plan_required",
                active_stage_goal=current_stage_goal,
                pending_stage_goal=pending_stage_goal,
                should_call_planner=True,
                trigger_event_types=event_types,
                transition_intent=TransitionIntent.REQUEST_REPLAN,
            )

        if no_frontiers:
            return self._decision(
                mode=TacticalMode.HOLD_AND_WAIT_FOR_FRONTIER,
                reason="no_frontiers",
                active_stage_goal=current_stage_goal,
                pending_stage_goal=pending_stage_goal,
                trigger_event_types=event_types,
                steady_mode=SteadyMode.HOLD_AND_WAIT,
                transition_intent=TransitionIntent.NONE,
                min_dwell_steps=1,
                review_trigger="frontier_summary_changed_or_hold_timeout",
                fallback_mode=SteadyMode.FOLLOW_STAGE_GOAL,
            )

        if getattr(graph_delta, "stuck", False):
            return self._decision(
                mode=TacticalMode.RECOVERY,
                reason="stuck",
                active_stage_goal=current_stage_goal,
                pending_stage_goal=pending_stage_goal,
                recovery_policy="stuck_replan",
                should_call_planner=True,
                trigger_event_types=event_types,
                steady_mode=SteadyMode.MOTION_RECOVERY,
                transition_intent=TransitionIntent.REQUEST_SEMANTIC_RECOVERY,
                preemption_priority=10,
                review_trigger="executor_feedback_or_recovery_timeout",
            )

        if pending_stage_goal is not None:
            current_specificity = strategy_specificity(
                getattr(current_strategy, "target_region", "")
            )
            pending_specificity = strategy_specificity(
                getattr(pending_strategy, "target_region", "")
            )
            if getattr(graph_delta, "frontier_reached", False):
                return self._decision(
                    mode=TacticalMode.ADOPT_PENDING_STAGE,
                    reason="frontier_reached_pending",
                    active_stage_goal=current_stage_goal,
                    pending_stage_goal=pending_stage_goal,
                    should_promote_pending=True,
                    trigger_event_types=event_types,
                    transition_intent=TransitionIntent.ADOPT_PENDING,
                )
            if pending_specificity > current_specificity:
                return self._decision(
                    mode=TacticalMode.ADOPT_PENDING_STAGE,
                    reason="pending_more_specific",
                    active_stage_goal=current_stage_goal,
                    pending_stage_goal=pending_stage_goal,
                    should_promote_pending=True,
                    trigger_event_types=event_types,
                    transition_intent=TransitionIntent.ADOPT_PENDING,
                )

        if getattr(graph_delta, "frontier_reached", False):
            return self._decision(
                mode=TacticalMode.REPLAN_REQUIRED,
                reason="frontier_reached",
                active_stage_goal=current_stage_goal,
                pending_stage_goal=pending_stage_goal,
                should_call_planner=True,
                trigger_event_types=event_types,
                transition_intent=TransitionIntent.REQUEST_REPLAN,
            )

        if getattr(graph_delta, "has_new_rooms", False):
            return self._decision(
                mode=TacticalMode.REPLAN_REQUIRED,
                reason="new_room_discovered",
                active_stage_goal=current_stage_goal,
                pending_stage_goal=pending_stage_goal,
                should_call_planner=True,
                should_call_monitor=True,
                trigger_event_types=event_types,
                transition_intent=TransitionIntent.REQUEST_REPLAN,
            )

        if getattr(graph_delta, "has_new_nodes", False) or getattr(
            graph_delta, "has_room_object_increase", False
        ):
            return self._decision(
                mode=TacticalMode.FOLLOW_STAGE_GOAL,
                reason="semantic_evidence_update",
                active_stage_goal=current_stage_goal,
                pending_stage_goal=pending_stage_goal,
                should_call_monitor=True,
                trigger_event_types=event_types,
                transition_intent=TransitionIntent.NONE,
            )

        return self._decision(
            mode=TacticalMode.FOLLOW_STAGE_GOAL,
            reason="default_follow",
            active_stage_goal=current_stage_goal,
            pending_stage_goal=pending_stage_goal,
            trigger_event_types=event_types,
            transition_intent=TransitionIntent.NONE,
        )

    def post_grounding_patch(self, grounding_result) -> TacticalDecision:
        """Map grounding failures to explicit liveness-preserving decisions."""

        failure_code = getattr(grounding_result, "failure_code", None) or getattr(
            grounding_result, "graph_no_goal_reason", ""
        ) or getattr(grounding_result, "noop_type", "")
        if failure_code in {"no_frontiers", "stage_not_groundable_yet"}:
            return self._decision(
                mode=TacticalMode.HOLD_AND_WAIT_FOR_FRONTIER,
                reason=f"grounding_hold:{failure_code}",
                steady_mode=SteadyMode.HOLD_AND_WAIT,
                transition_intent=TransitionIntent.NONE,
                review_trigger="frontiers_restored_or_hold_timeout",
                fallback_mode=SteadyMode.FOLLOW_STAGE_GOAL,
            )
        if failure_code in {"stage_not_groundable_in_principle", "no_candidate_frontiers"}:
            return self._decision(
                mode=TacticalMode.REPLAN_REQUIRED,
                reason=f"grounding_replan:{failure_code}",
                transition_intent=TransitionIntent.REQUEST_REPLAN,
            )
        return self._decision(
            mode=TacticalMode.FOLLOW_STAGE_GOAL,
            reason=f"grounding_retry:{failure_code or 'unknown'}",
            transition_intent=TransitionIntent.NONE,
        )

    def _decision(self, **kwargs) -> TacticalDecision:
        steady_mode = kwargs.get("steady_mode")
        transition_intent = kwargs.get("transition_intent", TransitionIntent.NONE)
        if steady_mode is None:
            mode = kwargs.get("mode", TacticalMode.FOLLOW_STAGE_GOAL)
            steady_mode = {
                TacticalMode.EXPLOIT_VISIBLE_TARGET: SteadyMode.EXPLOIT_VISIBLE_TARGET,
                TacticalMode.RECOVERY: SteadyMode.MOTION_RECOVERY,
                TacticalMode.HOLD_AND_WAIT_FOR_FRONTIER: SteadyMode.HOLD_AND_WAIT,
            }.get(mode, SteadyMode.FOLLOW_STAGE_GOAL)
        try:
            target_mode = next_mode(self.steady_mode, transition_intent)
        except ValueError:
            target_mode = steady_mode
        if target_mode != self.steady_mode or steady_mode != self.steady_mode:
            self.mode_epoch += 1
        self.steady_mode = steady_mode
        kwargs["steady_mode"] = steady_mode
        kwargs["transition_intent"] = transition_intent
        kwargs["mode_epoch"] = self.mode_epoch
        kwargs.setdefault("enter_guard", "event_guard_satisfied")
        kwargs.setdefault("exit_guard", "review_trigger_or_preemption")
        kwargs.setdefault("review_trigger", "next_event")
        kwargs.setdefault("fallback_mode", SteadyMode.FOLLOW_STAGE_GOAL)
        return TacticalDecision(**kwargs)
