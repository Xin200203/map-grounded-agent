"""Pure helpers for executor goal adoption bookkeeping."""


def resolve_strategy_epoch_transition(
    *,
    current_strategy_epoch,
    incoming_strategy_epoch,
    has_temp_goal,
    temp_goal_epoch,
):
    incoming = int(incoming_strategy_epoch or 0)
    current = int(current_strategy_epoch or 0)
    stale_temp_goal_cleared = False

    if incoming != current and has_temp_goal and temp_goal_epoch is not None:
        if int(temp_goal_epoch) < incoming:
            stale_temp_goal_cleared = True

    return {
        "next_strategy_epoch": incoming,
        "stale_temp_goal_cleared": bool(stale_temp_goal_cleared),
    }


def compute_adoption_transition(last_snapshot, *, source, goal_summary, goal_epoch):
    """Return before/after snapshots plus a stable changed bit."""

    signature = (source, tuple(goal_summary) if goal_summary else None)
    previous_signature = None
    adopted_before = None
    if last_snapshot is not None:
        previous_signature = (
            last_snapshot.get("source"),
            tuple(last_snapshot.get("goal")) if last_snapshot.get("goal") else None,
        )
        adopted_before = dict(last_snapshot)

    adopted_after = {
        "source": source,
        "goal": goal_summary,
        "goal_epoch": int(goal_epoch or 0),
    }
    return {
        "signature": signature,
        "adopted_before": adopted_before,
        "adopted_after": adopted_after,
        "adopted_changed": signature != previous_signature,
    }


def should_suppress_stuck_override(*, been_stuck, suppress_stuck_override):
    return bool(been_stuck and suppress_stuck_override)
