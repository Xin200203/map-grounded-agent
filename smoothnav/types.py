"""Shared dataclasses and enums for the layered SmoothNav controller.

The MEP objects in this file are intentionally plain dataclasses. Runtime
modules own the write paths; traces and prompts should only consume serialized
snapshots of these objects.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TacticalMode(str, Enum):
    FOLLOW_STAGE_GOAL = "follow_stage_goal"
    EXPLOIT_VISIBLE_TARGET = "exploit_visible_target"
    ADOPT_PENDING_STAGE = "adopt_pending_stage"
    RECOVERY = "recovery"
    REPLAN_REQUIRED = "replan_required"
    HOLD_AND_WAIT_FOR_FRONTIER = "hold_and_wait_for_frontier"


class TaskType(str, Enum):
    TEXT_GOAL = "text_goal"
    INSTRUCTION = "instruction"


class ConstraintType(str, Enum):
    TOPOLOGICAL = "topological"
    DIRECTIONAL = "directional"
    RELATIONAL = "relational"
    TEMPORAL = "temporal"
    TERMINAL = "terminal"


class ConstraintHardness(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    PREFERENCE = "preference"


class AnchorResolutionState(str, Enum):
    UNRESOLVED = "unresolved"
    GROUNDED = "grounded"
    INVALIDATED = "invalidated"


class SatisfactionScope(str, Enum):
    ONCE = "once"
    PERSISTENT = "persistent"
    UNTIL_REPLACED = "until_replaced"


class ViolationEffect(str, Enum):
    HARD_FAIL = "hard_fail"
    DEGRADE_CONFIDENCE = "degrade_confidence"
    REPLAN_HINT = "replan_hint"


class TaskComposition(str, Enum):
    ALL_OF = "all_of"
    ANY_OF = "any_of"
    ORDERED_LIST = "ordered_list"
    PREFERENCE_SET = "preference_set"


class VerificationLevel(str, Enum):
    OBSERVED = "observed"
    HYPOTHESIZED = "hypothesized"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"


class StageGoalType(str, Enum):
    SEARCH = "search"
    VERIFY = "verify"
    APPROACH = "approach"
    DISAMBIGUATE = "disambiguate"
    STOP_CANDIDATE = "stop_candidate"
    SEMANTIC_RECOVERY = "semantic_recovery"


class PendingProposalStatus(str, Enum):
    PROPOSED = "proposed"
    ELIGIBLE = "eligible"
    ADOPTED = "adopted"
    SHELVED = "shelved"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


class SteadyMode(str, Enum):
    FOLLOW_STAGE_GOAL = "FOLLOW_STAGE_GOAL"
    EXPLOIT_VISIBLE_TARGET = "EXPLOIT_VISIBLE_TARGET"
    MOTION_RECOVERY = "MOTION_RECOVERY"
    HOLD_AND_WAIT = "HOLD_AND_WAIT"
    FINAL_VERIFY = "FINAL_VERIFY"
    COMMIT_STOP = "COMMIT_STOP"


class TransitionIntent(str, Enum):
    NONE = "NONE"
    REQUEST_REPLAN = "REQUEST_REPLAN"
    ADOPT_PENDING = "ADOPT_PENDING"
    REQUEST_SEMANTIC_RECOVERY = "REQUEST_SEMANTIC_RECOVERY"


class TerminalOutcome(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE_BUDGET_EXHAUSTED = "FAILURE_BUDGET_EXHAUSTED"
    FAILURE_UNGROUNDED = "FAILURE_UNGROUNDED"
    FAILURE_STUCK_PERSISTENT = "FAILURE_STUCK_PERSISTENT"
    FAILURE_CONTRADICTION_UNRESOLVED = "FAILURE_CONTRADICTION_UNRESOLVED"
    FAILURE_NO_PROGRESS_TIMEOUT = "FAILURE_NO_PROGRESS_TIMEOUT"


class GroundingCandidateFamily(str, Enum):
    SEARCH = "search"
    VERIFY = "verify"
    DISAMBIGUATE = "disambiguate"
    APPROACH = "approach"
    MOTION_RECOVERY = "motion_recovery"
    FINAL_VERIFY = "final_verify"


class GeometricGoalType(str, Enum):
    FRONTIER = "frontier"
    VISIBLE_TARGET = "visible_target"
    RECOVERY_POINT = "recovery_point"
    GLOBAL_GOAL = "global_goal"
    NONE = "none"


@dataclass
class TaskConstraint:
    constraint_id: str
    constraint_type: ConstraintType = ConstraintType.TERMINAL
    constraint_hardness: ConstraintHardness = ConstraintHardness.HARD
    anchor_resolution_state: AnchorResolutionState = AnchorResolutionState.UNRESOLVED
    satisfaction_test: str = ""
    violation_test: str = ""
    satisfaction_scope: SatisfactionScope = SatisfactionScope.ONCE
    violation_effect: ViolationEffect = ViolationEffect.REPLAN_HINT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "constraint_hardness": self.constraint_hardness.value,
            "anchor_resolution_state": self.anchor_resolution_state.value,
            "satisfaction_test": self.satisfaction_test,
            "violation_test": self.violation_test,
            "satisfaction_scope": self.satisfaction_scope.value,
            "violation_effect": self.violation_effect.value,
        }


@dataclass
class TaskSpec:
    task_id: str
    task_type: TaskType = TaskType.TEXT_GOAL
    primary_goal: Optional[str] = None
    constraints: List[TaskConstraint] = field(default_factory=list)
    composition: TaskComposition = TaskComposition.ALL_OF
    candidate_stop_conditions: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.composition:
            raise ValueError("TaskSpec.composition must not be empty")
        terminal_ids = {
            constraint.constraint_id
            for constraint in self.constraints
            if constraint.constraint_type == ConstraintType.TERMINAL
        }
        if terminal_ids and not self.candidate_stop_conditions:
            raise ValueError("terminal constraints require candidate_stop_conditions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "primary_goal": self.primary_goal,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "composition": self.composition.value,
            "candidate_stop_conditions": list(self.candidate_stop_conditions),
        }


@dataclass
class Evidence:
    evidence_id: str
    proposition: str
    source: str
    scope: str
    confidence: float
    timestamp: int
    revocable: bool = True
    verification_level: VerificationLevel = VerificationLevel.OBSERVED
    entity_bindings: Dict[str, Any] = field(default_factory=dict)
    supports: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)
    freshness: float = 1.0
    coverage_witness: Optional[Dict[str, Any]] = None
    staleness_policy: str = "decay"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "proposition": self.proposition,
            "source": self.source,
            "scope": self.scope,
            "confidence": float(self.confidence),
            "timestamp": int(self.timestamp),
            "revocable": bool(self.revocable),
            "verification_level": self.verification_level.value,
            "entity_bindings": dict(self.entity_bindings),
            "supports": list(self.supports),
            "derived_from": list(self.derived_from),
            "freshness": float(self.freshness),
            "coverage_witness": (
                dict(self.coverage_witness)
                if self.coverage_witness is not None
                else None
            ),
            "staleness_policy": self.staleness_policy,
        }


@dataclass
class TaskBelief:
    task_epoch: int = 0
    belief_epoch: int = 0
    active_stage_ref: Optional[str] = None
    completed_constraints: List[str] = field(default_factory=list)
    open_constraints: List[str] = field(default_factory=list)
    excluded_regions: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    stop_evidence: List[str] = field(default_factory=list)
    semantic_belief_confidence: float = 0.0
    evidence_ids_used_recently: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_epoch": int(self.task_epoch),
            "belief_epoch": int(self.belief_epoch),
            "active_stage_ref": self.active_stage_ref,
            "completed_constraints": list(self.completed_constraints),
            "open_constraints": list(self.open_constraints),
            "excluded_regions": [dict(item) for item in self.excluded_regions],
            "contradictions": list(self.contradictions),
            "stop_evidence": list(self.stop_evidence),
            "semantic_belief_confidence": float(self.semantic_belief_confidence),
            "evidence_ids_used_recently": list(self.evidence_ids_used_recently),
        }


@dataclass
class MissionState:
    mission_text: str = ""
    mission_type: str = ""
    current_stage_id: int = 0
    current_stage_desc: str = ""
    stage_status: str = "not_started"
    completed_stages: List[str] = field(default_factory=list)
    blocked_stages: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    obtained_evidence: List[str] = field(default_factory=list)
    replan_reason: Optional[str] = None
    stop_condition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_text": self.mission_text,
            "mission_type": self.mission_type,
            "current_stage_id": self.current_stage_id,
            "current_stage_desc": self.current_stage_desc,
            "stage_status": self.stage_status,
            "completed_stages": list(self.completed_stages),
            "blocked_stages": list(self.blocked_stages),
            "required_evidence": list(self.required_evidence),
            "obtained_evidence": list(self.obtained_evidence),
            "replan_reason": self.replan_reason,
            "stop_condition": self.stop_condition,
        }


@dataclass
class StageGoal:
    stage_id: str = "stage_0"
    task_epoch: int = 0
    belief_epoch: int = 0
    world_epoch: int = 0
    stage_epoch: int = 0
    stage_type: str = "direction"
    target_region: Optional[str] = None
    target_object: Optional[str] = None
    anchor_object: Optional[str] = None
    semantic_intent: str = ""
    bias_position: Optional[Tuple[int, int]] = None
    stop_condition: str = "frontier_reached"
    confidence: Optional[float] = None
    planner_reason: str = ""
    stage_selection_confidence: float = 0.0
    explored_regions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.confidence is not None and not self.stage_selection_confidence:
            self.stage_selection_confidence = float(self.confidence)
        if self.confidence is None:
            self.confidence = float(self.stage_selection_confidence)

    def normalized_stage_type(self) -> str:
        if self.stage_type in {item.value for item in StageGoalType}:
            return self.stage_type
        return StageGoalType.SEARCH.value

    @property
    def reasoning(self) -> str:
        """Compatibility property for existing Strategy-based call sites."""

        return self.planner_reason

    @reasoning.setter
    def reasoning(self, value: str) -> None:
        self.planner_reason = value or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "task_epoch": int(self.task_epoch),
            "belief_epoch": int(self.belief_epoch),
            "world_epoch": int(self.world_epoch),
            "stage_epoch": int(self.stage_epoch),
            "stage_type": self.normalized_stage_type(),
            "target_kind": self.stage_type,
            "legacy_stage_type": self.stage_type,
            "target_region": self.target_region,
            "target_object": self.target_object,
            "anchor_object": self.anchor_object,
            "semantic_intent": self.semantic_intent,
            "bias_position": (
                list(self.bias_position) if self.bias_position is not None else None
            ),
            "stop_condition": self.stop_condition,
            "confidence": self.confidence,
            "planner_reason": self.planner_reason,
            "stage_selection_confidence": float(self.stage_selection_confidence),
            "explored_regions": list(self.explored_regions),
        }


@dataclass
class PendingStageProposal:
    proposal_id: str
    task_epoch: int
    belief_epoch: int
    world_epoch: int
    proposal_epoch: int
    stage_goal: StageGoal
    depends_on_evidence_ids: List[str] = field(default_factory=list)
    created_reason: str = ""
    eligibility_guard: str = ""
    expiry_condition: str = ""
    status: PendingProposalStatus = PendingProposalStatus.PROPOSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "task_epoch": int(self.task_epoch),
            "belief_epoch": int(self.belief_epoch),
            "world_epoch": int(self.world_epoch),
            "proposal_epoch": int(self.proposal_epoch),
            "stage_goal": self.stage_goal.to_dict(),
            "depends_on_evidence_ids": list(self.depends_on_evidence_ids),
            "created_reason": self.created_reason,
            "eligibility_guard": self.eligibility_guard,
            "expiry_condition": self.expiry_condition,
            "status": self.status.value,
        }


@dataclass
class TacticalDecision:
    mode: TacticalMode
    reason: str
    active_stage_goal: Optional[StageGoal] = None
    pending_stage_goal: Optional[StageGoal] = None
    chosen_visible_target: Optional[Dict[str, Any]] = None
    recovery_policy: Optional[str] = None
    should_call_planner: bool = False
    should_create_pending: bool = False
    should_promote_pending: bool = False
    should_call_monitor: bool = False
    trigger_event_types: List[str] = field(default_factory=list)
    mode_epoch: int = 0
    steady_mode: Optional[SteadyMode] = None
    transition_intent: TransitionIntent = TransitionIntent.NONE
    justification: str = ""
    enter_guard: str = ""
    exit_guard: str = ""
    min_dwell_steps: int = 1
    review_trigger: str = "next_event"
    preemption_priority: int = 0
    veto_conditions: List[str] = field(default_factory=list)
    fallback_mode: Optional[SteadyMode] = SteadyMode.FOLLOW_STAGE_GOAL

    def __post_init__(self) -> None:
        if not self.justification:
            self.justification = self.reason
        if self.steady_mode is None:
            self.steady_mode = tactical_mode_to_steady_mode(self.mode)
        if self.mode == TacticalMode.REPLAN_REQUIRED:
            self.transition_intent = TransitionIntent.REQUEST_REPLAN
        elif self.mode == TacticalMode.ADOPT_PENDING_STAGE:
            self.transition_intent = TransitionIntent.ADOPT_PENDING
        elif self.mode == TacticalMode.RECOVERY:
            self.transition_intent = TransitionIntent.REQUEST_SEMANTIC_RECOVERY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "active_stage_goal": (
                self.active_stage_goal.to_dict()
                if self.active_stage_goal is not None
                else None
            ),
            "pending_stage_goal": (
                self.pending_stage_goal.to_dict()
                if self.pending_stage_goal is not None
                else None
            ),
            "chosen_visible_target": self.chosen_visible_target,
            "recovery_policy": self.recovery_policy,
            "should_call_planner": bool(self.should_call_planner),
            "should_create_pending": bool(self.should_create_pending),
            "should_promote_pending": bool(self.should_promote_pending),
            "should_call_monitor": bool(self.should_call_monitor),
            "trigger_event_types": list(self.trigger_event_types),
            "mode_epoch": int(self.mode_epoch),
            "steady_mode": self.steady_mode.value if self.steady_mode else None,
            "transition_intent": self.transition_intent.value,
            "justification": self.justification,
            "enter_guard": self.enter_guard,
            "exit_guard": self.exit_guard,
            "min_dwell_steps": int(self.min_dwell_steps),
            "review_trigger": self.review_trigger,
            "preemption_priority": int(self.preemption_priority),
            "veto_conditions": list(self.veto_conditions),
            "fallback_mode": (
                self.fallback_mode.value
                if isinstance(self.fallback_mode, SteadyMode)
                else self.fallback_mode
            ),
        }


@dataclass
class GeometricGoal:
    goal_type: GeometricGoalType = GeometricGoalType.NONE
    full_map_coord: Optional[List[int]] = None
    local_map_coord: Optional[List[int]] = None
    source_mode: Optional[TacticalMode] = None
    source_stage_type: Optional[str] = None
    selected_frontier: Optional[List[int]] = None
    grounding_success: bool = False
    grounding_noop_reason: Optional[str] = None
    graph_no_goal_reason: str = ""
    changed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_type": self.goal_type.value,
            "full_map_coord": self.full_map_coord,
            "local_map_coord": self.local_map_coord,
            "source_mode": (
                self.source_mode.value if self.source_mode is not None else None
            ),
            "source_stage_type": self.source_stage_type,
            "selected_frontier": self.selected_frontier,
            "grounding_success": bool(self.grounding_success),
            "grounding_noop_reason": self.grounding_noop_reason,
            "graph_no_goal_reason": self.graph_no_goal_reason,
            "changed": bool(self.changed),
        }


@dataclass
class ExecutorFeedback:
    goal_epoch: int = 0
    executor_step_idx: int = 0
    override_reason: Optional[str] = None
    override_duration: int = 0
    actual_detour_steps: int = 0
    local_unreachable: bool = False
    forced_stop: bool = False
    escalation_required: bool = False
    adopted_goal_source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_epoch": int(self.goal_epoch),
            "executor_step_idx": int(self.executor_step_idx),
            "override_reason": self.override_reason,
            "override_duration": int(self.override_duration),
            "actual_detour_steps": int(self.actual_detour_steps),
            "local_unreachable": bool(self.local_unreachable),
            "forced_stop": bool(self.forced_stop),
            "escalation_required": bool(self.escalation_required),
            "adopted_goal_source": self.adopted_goal_source,
        }


@dataclass
class BudgetState:
    step_idx: int = 0
    planner_calls_in_window: int = 0
    adjudicator_calls_in_window: int = 0
    planner_budget_exhausted: bool = False
    adjudicator_budget_exhausted: bool = False
    cooldown_steps_remaining: int = 0
    belief_summarization_due: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": int(self.step_idx),
            "planner_calls_in_window": int(self.planner_calls_in_window),
            "adjudicator_calls_in_window": int(self.adjudicator_calls_in_window),
            "planner_budget_exhausted": bool(self.planner_budget_exhausted),
            "adjudicator_budget_exhausted": bool(
                self.adjudicator_budget_exhausted
            ),
            "cooldown_steps_remaining": int(self.cooldown_steps_remaining),
            "belief_summarization_due": bool(self.belief_summarization_due),
        }


@dataclass
class ExecutorCommand:
    geometric_goal: GeometricGoal
    allow_target_lock: bool = True
    allow_recovery: bool = True
    clear_temp_goal: bool = False
    strategy_epoch: int = 0
    goal_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometric_goal": self.geometric_goal.to_dict(),
            "allow_target_lock": bool(self.allow_target_lock),
            "allow_recovery": bool(self.allow_recovery),
            "clear_temp_goal": bool(self.clear_temp_goal),
            "strategy_epoch": int(self.strategy_epoch),
            "goal_epoch": int(self.goal_epoch),
        }


@dataclass
class WorldState:
    step_idx: int
    pose: Any
    local_pose: Any
    graph: Any
    bev_map: Any
    world_epoch: int = 0
    explored_regions: List[str] = field(default_factory=list)
    frontier_count: int = 0
    frontier_summary: List[Dict[str, Any]] = field(default_factory=list)
    room_summary: List[Dict[str, Any]] = field(default_factory=list)
    object_summary: List[Dict[str, Any]] = field(default_factory=list)
    visible_targets: List[Dict[str, Any]] = field(default_factory=list)
    visible_target_summary: List[Dict[str, Any]] = field(default_factory=list)
    stuck_signal: bool = False
    no_progress_steps: int = 0
    graph_delta: Any = None

    def summary(self) -> Dict[str, Any]:
        visible_target_summary = self.visible_target_summary or self.visible_targets
        return {
            "world_epoch": int(self.world_epoch),
            "step_idx": int(self.step_idx),
            "explored_regions": list(self.explored_regions),
            "frontier_count": int(self.frontier_count),
            "frontier_summary": list(self.frontier_summary),
            "room_summary": list(self.room_summary),
            "object_summary": list(self.object_summary),
            "visible_targets": list(visible_target_summary),
            "visible_target_summary": list(visible_target_summary),
            "stuck_signal": bool(self.stuck_signal),
            "no_progress_steps": int(self.no_progress_steps),
            "graph_delta": (
                self.graph_delta.to_dict()
                if hasattr(self.graph_delta, "to_dict")
                else self.graph_delta
            ),
        }


def tactical_mode_to_steady_mode(mode: TacticalMode) -> SteadyMode:
    if mode == TacticalMode.EXPLOIT_VISIBLE_TARGET:
        return SteadyMode.EXPLOIT_VISIBLE_TARGET
    if mode == TacticalMode.RECOVERY:
        return SteadyMode.MOTION_RECOVERY
    if mode == TacticalMode.HOLD_AND_WAIT_FOR_FRONTIER:
        return SteadyMode.HOLD_AND_WAIT
    return SteadyMode.FOLLOW_STAGE_GOAL


def stage_goal_from_strategy(
    strategy: Any,
    *,
    task_epoch: int = 0,
    belief_epoch: int = 0,
    world_epoch: int = 0,
    stage_epoch: int = 0,
) -> Optional[StageGoal]:
    if strategy is None:
        return None
    if isinstance(strategy, StageGoal):
        return strategy
    target_region = getattr(strategy, "target_region", None)
    anchor_object = getattr(strategy, "anchor_object", "") or ""
    if target_region and str(target_region).startswith("object:"):
        stage_type = "object"
        target_object = str(target_region).split("object:", 1)[1].strip()
    elif target_region and not str(target_region).startswith("unexplored"):
        stage_type = "room"
        target_object = None
    else:
        stage_type = "direction"
        target_object = None
    return StageGoal(
        stage_id=f"stage_{int(stage_epoch)}",
        task_epoch=int(task_epoch),
        belief_epoch=int(belief_epoch),
        world_epoch=int(world_epoch),
        stage_epoch=int(stage_epoch),
        stage_type=stage_type,
        target_region=target_region,
        target_object=target_object,
        anchor_object=anchor_object,
        semantic_intent=getattr(strategy, "reasoning", "") or str(target_region or ""),
        bias_position=getattr(strategy, "bias_position", None),
        stop_condition="frontier_reached",
        confidence=getattr(strategy, "confidence", None),
        stage_selection_confidence=float(getattr(strategy, "confidence", 0.0) or 0.0),
        planner_reason=getattr(strategy, "reasoning", ""),
        explored_regions=list(getattr(strategy, "explored_regions", []) or []),
    )
