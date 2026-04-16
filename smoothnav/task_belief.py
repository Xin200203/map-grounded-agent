"""TaskBelief updater and stale-plan rejection helpers."""

from typing import Any, Iterable, List, Optional

from smoothnav.evidence_ledger import EvidenceLedger
from smoothnav.types import (
    ConstraintType,
    PendingStageProposal,
    StageGoal,
    TaskBelief,
    TaskSpec,
    VerificationLevel,
)
from smoothnav.writer_guards import BELIEF_WRITER, WriterToken, require_writer


class TaskBeliefUpdater:
    """Single writer for TaskBelief and EvidenceLedger updates."""

    def __init__(self, task_spec: TaskSpec, ledger: EvidenceLedger):
        self.writer_token = WriterToken(BELIEF_WRITER)
        require_writer(BELIEF_WRITER, self.writer_token, "TaskBelief")
        self.task_spec = task_spec
        self.ledger = ledger
        self.belief = TaskBelief(
            task_epoch=0,
            belief_epoch=0,
            open_constraints=[
                constraint.constraint_id for constraint in task_spec.constraints
            ],
        )
        self._last_event_signature = None

    def reset(self, task_spec: TaskSpec) -> TaskBelief:
        self.task_spec = task_spec
        self.belief = TaskBelief(
            task_epoch=self.belief.task_epoch + 1,
            belief_epoch=0,
            open_constraints=[
                constraint.constraint_id for constraint in task_spec.constraints
            ],
        )
        self._last_event_signature = None
        return self.belief

    def update(self, world_state, *, executor_feedback=None) -> TaskBelief:
        require_writer(BELIEF_WRITER, self.writer_token, "TaskBelief")
        delta = getattr(world_state, "graph_delta", None)
        event_types = tuple(getattr(delta, "event_types", []) or [])
        signature = (
            int(getattr(world_state, "world_epoch", 0) or 0),
            int(getattr(world_state, "step_idx", 0) or 0),
            event_types,
        )
        changed = signature != self._last_event_signature
        if changed:
            self.belief.belief_epoch += 1
            self._last_event_signature = signature
        self.ledger.refresh_freshness(int(getattr(world_state, "step_idx", 0) or 0))

        used: List[str] = []
        if changed:
            for room_name in getattr(delta, "new_rooms", []) or []:
                evidence = self.ledger.add_observation(
                    f"exists(region={_normalize_entity(room_name)})",
                    source="graph_delta.new_rooms",
                    scope="region",
                    confidence=0.75,
                    timestamp=int(world_state.step_idx),
                    entity_bindings={"region": room_name},
                )
                used.append(evidence.evidence_id)
            for room_name in getattr(delta, "room_object_count_increase_rooms", []) or []:
                evidence = self.ledger.add_observation(
                    f"contains(region={_normalize_entity(room_name)}, object=unknown)",
                    source="graph_delta.room_object_count_increase",
                    scope="region_object",
                    confidence=0.55,
                    timestamp=int(world_state.step_idx),
                    entity_bindings={"region": room_name},
                )
                used.append(evidence.evidence_id)
            for caption in getattr(delta, "new_node_captions", []) or []:
                evidence = self.ledger.add_observation(
                    f"candidate_match(object_id={_normalize_entity(caption)}, target={_normalize_entity(self.task_spec.primary_goal or 'goal')})",
                    source="graph_delta.new_nodes",
                    scope="object",
                    confidence=0.45,
                    timestamp=int(world_state.step_idx),
                    entity_bindings={"caption": caption},
                )
                used.append(evidence.evidence_id)
        if executor_feedback is not None and getattr(executor_feedback, "escalation_required", False):
            evidence = self.ledger.add_observation(
                "executor_feedback(type=escalation_required)",
                source="executor_feedback",
                scope="executor",
                confidence=1.0,
                timestamp=int(world_state.step_idx),
                entity_bindings={"goal_epoch": executor_feedback.goal_epoch},
            )
            used.append(evidence.evidence_id)

        if used:
            self.belief.evidence_ids_used_recently = used[-8:]
            self.belief.semantic_belief_confidence = min(
                1.0, self.belief.semantic_belief_confidence + 0.05 * len(used)
            )
        self.belief.open_constraints = [
            constraint.constraint_id
            for constraint in self.task_spec.constraints
            if constraint.constraint_id not in self.belief.completed_constraints
        ]
        self.belief.contradictions = [
            evidence.evidence_id
            for evidence in self.ledger.by_level(VerificationLevel.CONTRADICTED)
        ]
        self.belief.stop_evidence = [
            evidence.evidence_id
            for evidence in self.ledger.by_level(VerificationLevel.VERIFIED)
            if evidence.proposition.startswith("stop_condition_satisfied(")
        ]
        return self.belief

    def note_active_stage(self, stage_goal: Optional[StageGoal]) -> TaskBelief:
        if stage_goal is not None:
            self.belief.active_stage_ref = stage_goal.stage_id
        return self.belief


def reject_stale_stage_goal(
    stage_goal: StageGoal,
    task_belief: TaskBelief,
    world_state,
    ledger: EvidenceLedger,
    evidence_ids: Optional[Iterable[str]] = None,
    *,
    belief_lag_tolerance: int = 1,
    world_lag_tolerance: int = 1,
) -> Optional[str]:
    if stage_goal.task_epoch != task_belief.task_epoch:
        return "task_epoch_mismatch"
    if stage_goal.belief_epoch < task_belief.belief_epoch - belief_lag_tolerance:
        return "belief_epoch_stale"
    if stage_goal.world_epoch < int(getattr(world_state, "world_epoch", 0) or 0) - world_lag_tolerance:
        return "world_epoch_stale"
    invalid = [
        evidence_id
        for evidence_id in (list(evidence_ids or []))
        if not ledger.is_valid(evidence_id)
    ]
    if invalid:
        return "supporting_evidence_invalid"
    return None


def reject_stale_pending_proposal(
    proposal: PendingStageProposal,
    task_belief: TaskBelief,
    world_state,
    ledger: EvidenceLedger,
) -> Optional[str]:
    stage_reason = reject_stale_stage_goal(
        proposal.stage_goal,
        task_belief,
        world_state,
        ledger,
        proposal.depends_on_evidence_ids,
    )
    if stage_reason:
        return stage_reason
    invalid = [
        evidence_id
        for evidence_id in proposal.depends_on_evidence_ids
        if not ledger.is_valid(evidence_id)
    ]
    if invalid:
        return "supporting_evidence_invalid"
    return None


def _normalize_entity(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return "_".join(part for part in text.replace("(", "").replace(")", "").split() if part) or "unknown"
