"""Explicit mode/intent transition table and liveness checks."""

from typing import Dict, Iterable, Tuple

from smoothnav.types import SteadyMode, TransitionIntent


ALLOWED_TRANSITIONS = {
    (SteadyMode.FOLLOW_STAGE_GOAL, TransitionIntent.NONE): SteadyMode.FOLLOW_STAGE_GOAL,
    (SteadyMode.FOLLOW_STAGE_GOAL, TransitionIntent.REQUEST_REPLAN): SteadyMode.FOLLOW_STAGE_GOAL,
    (SteadyMode.FOLLOW_STAGE_GOAL, TransitionIntent.ADOPT_PENDING): SteadyMode.FOLLOW_STAGE_GOAL,
    (SteadyMode.FOLLOW_STAGE_GOAL, TransitionIntent.REQUEST_SEMANTIC_RECOVERY): SteadyMode.MOTION_RECOVERY,
    (SteadyMode.EXPLOIT_VISIBLE_TARGET, TransitionIntent.NONE): SteadyMode.EXPLOIT_VISIBLE_TARGET,
    (SteadyMode.MOTION_RECOVERY, TransitionIntent.REQUEST_REPLAN): SteadyMode.FOLLOW_STAGE_GOAL,
    (SteadyMode.MOTION_RECOVERY, TransitionIntent.NONE): SteadyMode.MOTION_RECOVERY,
    (SteadyMode.HOLD_AND_WAIT, TransitionIntent.NONE): SteadyMode.HOLD_AND_WAIT,
    (SteadyMode.HOLD_AND_WAIT, TransitionIntent.REQUEST_REPLAN): SteadyMode.FOLLOW_STAGE_GOAL,
    (SteadyMode.FINAL_VERIFY, TransitionIntent.NONE): SteadyMode.FINAL_VERIFY,
    (SteadyMode.COMMIT_STOP, TransitionIntent.NONE): SteadyMode.COMMIT_STOP,
}


def next_mode(current: SteadyMode, intent: TransitionIntent) -> SteadyMode:
    key = (current, intent)
    if key not in ALLOWED_TRANSITIONS:
        raise ValueError(f"Illegal tactical transition: {current.value} + {intent.value}")
    return ALLOWED_TRANSITIONS[key]


def allowed_transition_summary() -> Dict[str, str]:
    return {
        f"{mode.value}::{intent.value}": target.value
        for (mode, intent), target in ALLOWED_TRANSITIONS.items()
    }
