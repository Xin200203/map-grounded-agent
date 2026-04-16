"""TaskSpec bootstrap and validation.

This module is the only runtime writer for TaskSpec. Downstream modules should
read TaskSpec snapshots instead of mutating task semantics in-place.
"""

from typing import Optional

from smoothnav.types import (
    ConstraintHardness,
    ConstraintType,
    SatisfactionScope,
    TaskComposition,
    TaskConstraint,
    TaskSpec,
    TaskType,
    ViolationEffect,
)
from smoothnav.writer_guards import (
    TASK_BOOTSTRAP_WRITER,
    WriterToken,
    require_writer,
)


def parse_task_spec(
    goal_description: str,
    goal_type: str = "text",
    *,
    task_id: Optional[str] = None,
    writer_token: Optional[WriterToken] = None,
) -> TaskSpec:
    token = writer_token or WriterToken(TASK_BOOTSTRAP_WRITER)
    require_writer(TASK_BOOTSTRAP_WRITER, token, "TaskSpec")

    normalized_goal = (goal_description or "").strip()
    task_type = TaskType.INSTRUCTION if goal_type == "instruction" else TaskType.TEXT_GOAL
    condition_id = "stop_primary_goal_visible"
    spec = TaskSpec(
        task_id=task_id or "task_0",
        task_type=task_type,
        primary_goal=normalized_goal or None,
        constraints=[
            TaskConstraint(
                constraint_id=condition_id,
                constraint_type=ConstraintType.TERMINAL,
                constraint_hardness=ConstraintHardness.HARD,
                satisfaction_test="visible_target_matches_primary_goal",
                violation_test="budget_or_progress_failure",
                satisfaction_scope=SatisfactionScope.ONCE,
                violation_effect=ViolationEffect.REPLAN_HINT,
            )
        ],
        composition=TaskComposition.ALL_OF,
        candidate_stop_conditions=[condition_id],
    )
    spec.validate()
    return spec
