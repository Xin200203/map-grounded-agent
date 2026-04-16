"""Geometric grounding layer for SmoothNav StageGoal/TacticalDecision outputs."""

from typing import Optional, Tuple

from smoothnav.strategy_grounding import GroundingResult, apply_strategy
from smoothnav.types import (
    GeometricGoal,
    GeometricGoalType,
    GroundingCandidateFamily,
    TacticalDecision,
    TacticalMode,
    stage_goal_from_strategy,
)


def geometric_goal_from_grounding(
    grounding_result: GroundingResult,
    *,
    source_mode: Optional[TacticalMode] = None,
    source_stage_type: Optional[str] = None,
) -> GeometricGoal:
    if grounding_result.success and grounding_result.projected_goal is not None:
        goal_type = GeometricGoalType.FRONTIER
    elif grounding_result.reason == "get_goal_none":
        goal_type = GeometricGoalType.NONE
    else:
        goal_type = GeometricGoalType.NONE

    return GeometricGoal(
        goal_type=goal_type,
        full_map_coord=(
            list(grounding_result.selected_frontier)
            if grounding_result.selected_frontier is not None
            else None
        ),
        local_map_coord=(
            list(grounding_result.projected_goal)
            if grounding_result.projected_goal is not None
            else None
        ),
        source_mode=source_mode,
        source_stage_type=source_stage_type,
        selected_frontier=(
            list(grounding_result.selected_frontier)
            if grounding_result.selected_frontier is not None
            else None
        ),
        grounding_success=bool(grounding_result.success),
        grounding_noop_reason=grounding_result.noop_reason,
        graph_no_goal_reason=grounding_result.graph_no_goal_reason,
        changed=bool(grounding_result.changed),
    )


def grounding_family_for_stage(stage_goal, source_mode) -> str:
    if source_mode == TacticalMode.RECOVERY:
        return GroundingCandidateFamily.MOTION_RECOVERY.value
    stage_type = getattr(stage_goal, "stage_type", "") if stage_goal is not None else ""
    if stage_type in {"verify", "stop_candidate"}:
        return GroundingCandidateFamily.VERIFY.value
    if stage_type == "approach":
        return GroundingCandidateFamily.APPROACH.value
    if stage_type == "disambiguate":
        return GroundingCandidateFamily.DISAMBIGUATE.value
    return GroundingCandidateFamily.SEARCH.value


class GeometricGrounder:
    """Adapter around the existing strategy grounding implementation."""

    def __init__(self):
        self.last_grounding_result: Optional[GroundingResult] = None
        self.last_geometric_goal: Optional[GeometricGoal] = None

    def ground_strategy(
        self,
        strategy,
        graph,
        bev_map,
        args,
        global_goals,
        *,
        source_mode: TacticalMode = TacticalMode.FOLLOW_STAGE_GOAL,
        task_belief=None,
        world_state=None,
        goal_epoch: int = 0,
    ) -> Tuple[GroundingResult, GeometricGoal]:
        result = apply_strategy(strategy, graph, bev_map, args, global_goals)
        stage_goal = stage_goal_from_strategy(
            strategy,
            task_epoch=int(getattr(task_belief, "task_epoch", 0) or 0),
            belief_epoch=int(getattr(task_belief, "belief_epoch", 0) or 0),
            world_epoch=int(getattr(world_state, "world_epoch", 0) or 0),
        )
        result.goal_epoch = int(goal_epoch or 0)
        result.task_epoch = int(getattr(stage_goal, "task_epoch", 0) or 0)
        result.belief_epoch = int(getattr(stage_goal, "belief_epoch", 0) or 0)
        result.world_epoch = int(getattr(stage_goal, "world_epoch", 0) or 0)
        result.source_mode = source_mode.value if source_mode is not None else ""
        result.candidate_family = grounding_family_for_stage(stage_goal, source_mode)
        for candidate in result.candidates:
            candidate["family"] = result.candidate_family
        geometric_goal = geometric_goal_from_grounding(
            result,
            source_mode=source_mode,
            source_stage_type=stage_goal.stage_type if stage_goal is not None else None,
        )
        self.last_grounding_result = result
        self.last_geometric_goal = geometric_goal
        return result, geometric_goal

    def ground_decision(
        self,
        decision: TacticalDecision,
        graph,
        bev_map,
        args,
        global_goals,
    ) -> Tuple[GroundingResult, GeometricGoal]:
        strategy = decision.active_stage_goal
        return self.ground_strategy(
            strategy,
            graph,
            bev_map,
            args,
            global_goals,
            source_mode=decision.mode,
        )
