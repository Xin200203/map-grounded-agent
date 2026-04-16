"""PendingStageProposal manager with K-limit, dominance, and adoption rules."""

from typing import Any, Dict, List, Optional

from smoothnav.graph_delta import strategy_specificity
from smoothnav.task_belief import reject_stale_pending_proposal
from smoothnav.types import (
    PendingProposalStatus,
    PendingStageProposal,
    StageGoal,
    TaskBelief,
)
from smoothnav.writer_guards import PENDING_WRITER, WriterToken, require_writer


class PendingProposalManager:
    """Single writer for pending proposal status transitions."""

    def __init__(self, *, max_pending: int = 2):
        self.writer_token = WriterToken(PENDING_WRITER)
        require_writer(PENDING_WRITER, self.writer_token, "PendingStageProposal")
        self.max_pending = min(2, max(1, int(max_pending)))
        self._proposal_epoch = 0
        self._proposals: List[PendingStageProposal] = []

    @property
    def proposal_epoch(self) -> int:
        return self._proposal_epoch

    def reset(self) -> None:
        self._proposal_epoch = 0
        self._proposals = []

    def create(
        self,
        stage_goal: StageGoal,
        *,
        task_belief: TaskBelief,
        world_state,
        depends_on_evidence_ids: Optional[List[str]] = None,
        created_reason: str = "",
        eligibility_guard: str = "frontier_reached_or_more_specific",
        expiry_condition: str = "stale_epoch_or_invalid_evidence",
    ) -> PendingStageProposal:
        require_writer(PENDING_WRITER, self.writer_token, "PendingStageProposal")
        self._proposal_epoch += 1
        proposal = PendingStageProposal(
            proposal_id=f"pending_{self._proposal_epoch:06d}",
            task_epoch=int(task_belief.task_epoch),
            belief_epoch=int(task_belief.belief_epoch),
            world_epoch=int(getattr(world_state, "world_epoch", 0) or 0),
            proposal_epoch=self._proposal_epoch,
            stage_goal=stage_goal,
            depends_on_evidence_ids=list(depends_on_evidence_ids or task_belief.evidence_ids_used_recently),
            created_reason=created_reason,
            eligibility_guard=eligibility_guard,
            expiry_condition=expiry_condition,
            status=PendingProposalStatus.PROPOSED,
        )
        self._proposals.append(proposal)
        self._enforce_k_limit()
        return proposal

    def adopt_best(
        self,
        *,
        task_belief: TaskBelief,
        world_state,
        ledger,
        reason: str = "",
    ) -> Optional[PendingStageProposal]:
        require_writer(PENDING_WRITER, self.writer_token, "PendingStageProposal")
        self.invalidate_stale(task_belief=task_belief, world_state=world_state, ledger=ledger)
        eligible = [
            proposal
            for proposal in self._proposals
            if proposal.status in {PendingProposalStatus.PROPOSED, PendingProposalStatus.ELIGIBLE}
        ]
        if not eligible:
            return None
        chosen = sorted(eligible, key=self._dominance_key, reverse=True)[0]
        for proposal in self._proposals:
            if proposal is chosen:
                proposal.status = PendingProposalStatus.ADOPTED
            elif proposal.status == PendingProposalStatus.ADOPTED:
                proposal.status = PendingProposalStatus.SHELVED
        return chosen

    def invalidate_stale(self, *, task_belief: TaskBelief, world_state, ledger) -> None:
        for proposal in self._proposals:
            if proposal.status in {
                PendingProposalStatus.ADOPTED,
                PendingProposalStatus.SHELVED,
                PendingProposalStatus.EXPIRED,
                PendingProposalStatus.INVALIDATED,
            }:
                continue
            reason = reject_stale_pending_proposal(proposal, task_belief, world_state, ledger)
            if reason == "supporting_evidence_invalid":
                proposal.status = PendingProposalStatus.INVALIDATED
            elif reason:
                proposal.status = PendingProposalStatus.EXPIRED

    def active(self) -> Optional[PendingStageProposal]:
        adopted = [
            proposal
            for proposal in self._proposals
            if proposal.status == PendingProposalStatus.ADOPTED
        ]
        if len(adopted) > 1:
            raise AssertionError("MEP violation: more than one adopted pending proposal")
        return adopted[0] if adopted else None

    def pending(self) -> List[PendingStageProposal]:
        return [
            proposal
            for proposal in self._proposals
            if proposal.status in {PendingProposalStatus.PROPOSED, PendingProposalStatus.ELIGIBLE}
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_epoch": int(self._proposal_epoch),
            "max_pending": int(self.max_pending),
            "proposals": [proposal.to_dict() for proposal in self._proposals],
        }

    def _enforce_k_limit(self) -> None:
        active = [
            proposal
            for proposal in self._proposals
            if proposal.status in {PendingProposalStatus.PROPOSED, PendingProposalStatus.ELIGIBLE}
        ]
        if len(active) <= self.max_pending:
            return
        keep = set(id(proposal) for proposal in sorted(active, key=self._dominance_key, reverse=True)[: self.max_pending])
        for proposal in active:
            if id(proposal) not in keep:
                proposal.status = PendingProposalStatus.SHELVED

    @staticmethod
    def _dominance_key(proposal: PendingStageProposal):
        stage_goal = proposal.stage_goal
        target_region = stage_goal.target_region or ""
        specificity = strategy_specificity(target_region)
        evidence_count = len(proposal.depends_on_evidence_ids)
        confidence = float(stage_goal.stage_selection_confidence or 0.0)
        return (specificity, evidence_count, confidence, proposal.proposal_epoch)
