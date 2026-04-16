"""Runtime writer-boundary helpers for MEP source-of-truth objects."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


TASK_BOOTSTRAP_WRITER = "task_bootstrap"
WORLD_STATE_WRITER = "world_updater"
BELIEF_WRITER = "belief_updater"
PLANNER_WRITER = "strategic_planner"
PENDING_WRITER = "strategic_planner"
ARBITER_WRITER = "tactical_arbiter"
GROUNDER_WRITER = "geometric_grounder"
EXECUTOR_WRITER = "reactive_executor"
BUDGET_WRITER = "budget_governor"
TERMINAL_WRITER = "terminal_arbiter"


@dataclass(frozen=True)
class WriterToken:
    """Explicit token carried by modules that own an object write path."""

    writer: str


def require_writer(expected: str, token: WriterToken, object_name: str) -> None:
    if token.writer != expected:
        raise PermissionError(
            f"{object_name} can only be written by {expected}; got {token.writer}"
        )


@contextmanager
def writer_scope(expected: str, token: WriterToken, object_name: str) -> Iterator[None]:
    require_writer(expected, token, object_name)
    yield
