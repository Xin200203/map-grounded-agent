"""Pure helpers for executor goal adoption bookkeeping."""


def resolve_strategy_epoch_transition(
    *,
    current_strategy_epoch,
    incoming_strategy_epoch,
    has_temp_goal,
    temp_goal_epoch,
    has_stuck_goal=False,
):
    incoming = int(incoming_strategy_epoch or 0)
    current = int(current_strategy_epoch or 0)
    stale_temp_goal_cleared = False
    stale_stuck_goal_cleared = False

    if incoming != current and has_temp_goal and temp_goal_epoch is not None:
        if int(temp_goal_epoch) < incoming:
            stale_temp_goal_cleared = True
    if incoming != current and has_stuck_goal:
        stale_stuck_goal_cleared = True

    return {
        "next_strategy_epoch": incoming,
        "stale_temp_goal_cleared": bool(stale_temp_goal_cleared),
        "stale_stuck_goal_cleared": bool(stale_stuck_goal_cleared),
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


def should_suppress_stuck_override(
    *,
    been_stuck,
    suppress_stuck_override,
    current_target_region: str = "",
):
    target = str(current_target_region or "")
    if target.startswith("object:"):
        return bool(been_stuck)
    return bool(been_stuck and suppress_stuck_override)


def should_allow_text_visible_temp_goal(*, goal_type, current_target_region,
                                        goal_name=None,
                                        takeover_under_commitment=False):
    if str(goal_type or "") != "text":
        return True
    # Historical suppression (April, ep527 era): opportunistic visible-target
    # takeover on arbitrary sightings proved unstable, so the takeover link of
    # the executor's approach funnel (sighting -> temp goal -> close-range
    # re-discrimination -> lock -> stop) was severed for text goals. That left
    # stop unreachable from the anchor path (S8 junction audit, 2026-07-07:
    # anchored_failed 16/40, visible_failed 0).
    #
    # C6 reopens the link under controller evidence only: takeover is allowed
    # exactly when the controller currently vouches for a committed target
    # (object:/search-anchor strategy). Uncommitted exploration keeps the
    # historical suppression.
    if not takeover_under_commitment:
        return False
    target = str(current_target_region or "")
    return target.startswith("unexplored target:") or target.startswith("object:")
