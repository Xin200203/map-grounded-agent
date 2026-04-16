"""Budget and hot-context governance for SmoothNav MEP."""

from collections import deque
from typing import Any, Deque, Dict, Tuple

from smoothnav.types import BudgetState
from smoothnav.writer_guards import BUDGET_WRITER, WriterToken, require_writer


class BudgetGovernor:
    """Source of truth for planner/adjudicator call budgets."""

    def __init__(
        self,
        *,
        planner_call_budget: int = 8,
        adjudicator_call_budget: int = 8,
        window_steps: int = 100,
        forced_cooldown_after_replan: int = 0,
        belief_summarization_interval: int = 50,
    ):
        self.writer_token = WriterToken(BUDGET_WRITER)
        self.planner_call_budget = int(planner_call_budget)
        self.adjudicator_call_budget = int(adjudicator_call_budget)
        self.window_steps = max(1, int(window_steps))
        self.forced_cooldown_after_replan = max(0, int(forced_cooldown_after_replan))
        self.belief_summarization_interval = max(1, int(belief_summarization_interval))
        self._planner_calls: Deque[int] = deque()
        self._adjudicator_calls: Deque[int] = deque()
        self.state = BudgetState()

    def reset(self) -> BudgetState:
        require_writer(BUDGET_WRITER, self.writer_token, "BudgetState")
        self._planner_calls.clear()
        self._adjudicator_calls.clear()
        self.state = BudgetState()
        return self.state

    def update(self, step_idx: int) -> BudgetState:
        require_writer(BUDGET_WRITER, self.writer_token, "BudgetState")
        self._drop_old(self._planner_calls, step_idx)
        self._drop_old(self._adjudicator_calls, step_idx)
        if self.state.cooldown_steps_remaining > 0:
            self.state.cooldown_steps_remaining -= 1
        self.state.step_idx = int(step_idx)
        self.state.planner_calls_in_window = len(self._planner_calls)
        self.state.adjudicator_calls_in_window = len(self._adjudicator_calls)
        self.state.planner_budget_exhausted = (
            len(self._planner_calls) >= self.planner_call_budget
        )
        self.state.adjudicator_budget_exhausted = (
            len(self._adjudicator_calls) >= self.adjudicator_call_budget
        )
        self.state.belief_summarization_due = (
            int(step_idx) > 0 and int(step_idx) % self.belief_summarization_interval == 0
        )
        return self.state

    def can_call_planner(self, step_idx: int) -> bool:
        state = self.update(step_idx)
        return not state.planner_budget_exhausted and state.cooldown_steps_remaining == 0

    def record_planner_call(self, step_idx: int) -> BudgetState:
        require_writer(BUDGET_WRITER, self.writer_token, "BudgetState")
        self._planner_calls.append(int(step_idx))
        self.state.cooldown_steps_remaining = self.forced_cooldown_after_replan
        return self.update(step_idx)

    def record_adjudicator_call(self, step_idx: int) -> BudgetState:
        require_writer(BUDGET_WRITER, self.writer_token, "BudgetState")
        self._adjudicator_calls.append(int(step_idx))
        return self.update(step_idx)

    def hot_context(self, *, task_belief, world_state, pending_manager=None) -> Dict[str, Any]:
        pending = pending_manager.to_dict() if pending_manager is not None else None
        return {
            "budget_state": self.state.to_dict(),
            "task_belief": task_belief.to_dict() if task_belief is not None else None,
            "world_state": world_state.summary() if world_state is not None else None,
            "pending_proposals": pending,
        }

    def _drop_old(self, calls: Deque[int], step_idx: int) -> None:
        while calls and int(step_idx) - calls[0] >= self.window_steps:
            calls.popleft()
