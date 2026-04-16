"""Stop, success, and failure arbitration for MEP terminal closure."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from smoothnav.types import TerminalOutcome
from smoothnav.writer_guards import TERMINAL_WRITER, WriterToken, require_writer


@dataclass
class TerminalDecision:
    outcome: TerminalOutcome
    termination_confidence: float
    reason: str
    stop_veto_source: Optional[str] = None
    final_verify_timeout: int = 0
    evidence_ids: Optional[list] = None

    @property
    def is_terminal(self) -> bool:
        return self.outcome != TerminalOutcome.RUNNING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "termination_confidence": float(self.termination_confidence),
            "reason": self.reason,
            "stop_veto_source": self.stop_veto_source,
            "final_verify_timeout": int(self.final_verify_timeout),
            "evidence_ids": list(self.evidence_ids or []),
            "is_terminal": bool(self.is_terminal),
        }


class TerminalArbiter:
    """Single writer for terminal outcomes."""

    def __init__(
        self,
        *,
        max_no_progress_steps: int = 120,
        max_grounding_noops: int = 12,
        max_stuck_steps: int = 60,
    ):
        self.writer_token = WriterToken(TERMINAL_WRITER)
        self.max_no_progress_steps = int(max_no_progress_steps)
        self.max_grounding_noops = int(max_grounding_noops)
        self.max_stuck_steps = int(max_stuck_steps)

    def decide(
        self,
        *,
        done: bool,
        infos: Optional[Dict[str, Any]],
        task_belief,
        budget_state,
        controller_state,
        latest_grounding=None,
        executor_feedback=None,
    ) -> TerminalDecision:
        require_writer(TERMINAL_WRITER, self.writer_token, "TerminalDecision")
        infos = infos or {}
        if done and float(infos.get("success") or 0.0) > 0:
            return TerminalDecision(
                outcome=TerminalOutcome.SUCCESS,
                termination_confidence=1.0,
                reason="habitat_success",
                evidence_ids=list(getattr(task_belief, "stop_evidence", []) or []),
            )
        if getattr(budget_state, "planner_budget_exhausted", False):
            return TerminalDecision(
                outcome=TerminalOutcome.FAILURE_BUDGET_EXHAUSTED,
                termination_confidence=0.9,
                reason="planner_budget_exhausted",
            )
        if len(getattr(task_belief, "contradictions", []) or []) > 0:
            return TerminalDecision(
                outcome=TerminalOutcome.FAILURE_CONTRADICTION_UNRESOLVED,
                termination_confidence=0.8,
                reason="belief_contradiction_unresolved",
                evidence_ids=list(getattr(task_belief, "contradictions", []) or []),
            )
        if int(getattr(controller_state, "no_progress_steps", 0) or 0) >= self.max_no_progress_steps:
            return TerminalDecision(
                outcome=TerminalOutcome.FAILURE_NO_PROGRESS_TIMEOUT,
                termination_confidence=0.85,
                reason="no_progress_timeout",
            )
        if int(getattr(controller_state, "no_progress_steps", 0) or 0) >= self.max_stuck_steps:
            return TerminalDecision(
                outcome=TerminalOutcome.FAILURE_STUCK_PERSISTENT,
                termination_confidence=0.85,
                reason="stuck_persistent",
            )
        if int(getattr(controller_state, "consecutive_grounding_noops", 0) or 0) >= self.max_grounding_noops:
            return TerminalDecision(
                outcome=TerminalOutcome.FAILURE_UNGROUNDED,
                termination_confidence=0.8,
                reason=getattr(latest_grounding, "noop_type", "grounding_noop_limit"),
            )
        if done:
            return TerminalDecision(
                outcome=TerminalOutcome.FAILURE_NO_PROGRESS_TIMEOUT,
                termination_confidence=0.5,
                reason="environment_done_without_success",
            )
        return TerminalDecision(
            outcome=TerminalOutcome.RUNNING,
            termination_confidence=0.0,
            reason="running",
        )
