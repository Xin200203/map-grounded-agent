"""Phase 2 control metrics derived from step traces."""

from collections import Counter
import json
import os
from glob import glob


REACTION_DECISIONS = {"ADJUST", "PREFETCH", "ESCALATE"}


def _read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda item: int(item.get("step_idx", 0)))
    return records


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _merge_counters(counter_items):
    merged = Counter()
    for item in counter_items:
        merged.update(item or {})
    return dict(merged)


def _episode_id_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    _, _, suffix = stem.partition("episode_")
    return int(suffix) if suffix else None


def compute_episode_control_metrics(steps):
    strategy_switch_count = 0
    pending_created_count = 0
    pending_promoted_count = 0
    pending_created_and_promoted_count = 0
    grounding_attempt_count = 0
    grounding_noop_count = 0
    same_frontier_as_prev_count = 0
    grounding_noop_reason_counts = Counter()
    grounding_no_goal_reason_counts = Counter()
    grounding_frontier_fallback_mode_counts = Counter()
    grounding_candidate_distance_fallback_mode_counts = Counter()
    temp_goal_override_steps = 0
    stuck_goal_override_steps = 0
    global_goal_override_steps = 0
    visible_target_override_steps = 0
    executor_override_steps = 0
    direction_reuse_events = 0
    decision_delays = []
    goal_update_delays = []
    executor_adoption_delays = []
    terminal_outcome_counts = Counter()
    missing_epoch_trace_steps = 0

    open_decision_step = None
    open_goal_update_step = None
    open_executor_adoption_step = None
    previous_direction_reuse_count = 0

    for step in steps:
        step_idx = int(step.get("step_idx", 0))
        if any(
            step.get(field) is None
            for field in [
                "task_epoch",
                "belief_epoch",
                "stage_epoch",
                "mode_epoch",
                "goal_epoch",
                "pending_proposal_epoch",
            ]
        ):
            missing_epoch_trace_steps += 1
        terminal_outcome = step.get("terminal_outcome")
        if terminal_outcome and terminal_outcome != "RUNNING":
            terminal_outcome_counts[terminal_outcome] += 1
        graph_delta = step.get("graph_delta", {})
        has_semantic_event = any(
            [
                bool(step.get("new_node_count", 0)),
                bool(graph_delta.get("new_rooms", [])),
                bool(graph_delta.get("room_object_count_changes", {})),
                bool(graph_delta.get("node_caption_changed")),
                bool(graph_delta.get("frontier_near")),
                bool(graph_delta.get("no_progress")),
                bool(graph_delta.get("stuck")),
            ]
        )
        reaction_happened = bool(step.get("planner_called")) or (
            step.get("monitor_decision") in REACTION_DECISIONS
        )

        if has_semantic_event and open_decision_step is None:
            open_decision_step = step_idx
        if open_decision_step is not None and reaction_happened:
            decision_delays.append(step_idx - open_decision_step)
            open_decision_step = None

        if step.get("strategy_switched"):
            strategy_switch_count += 1
        if step.get("pending_created"):
            pending_created_count += 1
        if step.get("pending_promoted"):
            pending_promoted_count += 1
        if step.get("pending_created_and_promoted_same_step"):
            pending_created_and_promoted_count += 1

        grounding_events = step.get("grounding_events", [])
        grounding_attempt_count += len(grounding_events)
        grounding_noop_count += sum(
            1 for event in grounding_events if not event.get("changed", False)
        )
        same_frontier_as_prev_count += sum(
            1
            for event in grounding_events
            if event.get("selected_frontier_same_as_prev")
            or event.get("selected_frontier_same_as_previous")
        )
        for event in grounding_events:
            if not event.get("changed", False):
                reason = (
                    event.get("noop_type")
                    or event.get("noop_reason")
                    or event.get("reason")
                    or "unknown_noop"
                )
                grounding_noop_reason_counts[reason] += 1
                no_goal_reason = (
                    event.get("graph_no_goal_reason")
                    or (event.get("graph_debug") or {}).get("no_goal_reason")
                    or ""
                )
                if no_goal_reason:
                    grounding_no_goal_reason_counts[no_goal_reason] += 1
            frontier_fallback_mode = (
                event.get("frontier_filter_fallback_mode")
                or (event.get("graph_debug") or {}).get("frontier_filter_fallback_mode")
                or ""
            )
            if frontier_fallback_mode:
                grounding_frontier_fallback_mode_counts[frontier_fallback_mode] += 1
            candidate_distance_fallback_mode = (
                event.get("candidate_distance_fallback_mode")
                or (event.get("graph_debug") or {}).get("candidate_distance_fallback_mode")
                or ""
            )
            if candidate_distance_fallback_mode:
                grounding_candidate_distance_fallback_mode_counts[
                    candidate_distance_fallback_mode
                ] += 1
        visible_target_override = bool(step.get("visible_target_override"))
        if step.get("temp_goal_override"):
            temp_goal_override_steps += 1
        if step.get("stuck_goal_override"):
            stuck_goal_override_steps += 1
        if step.get("global_goal_override"):
            global_goal_override_steps += 1
        if visible_target_override:
            visible_target_override_steps += 1
        if any(
            [
                visible_target_override,
                bool(step.get("temp_goal_override")),
                bool(step.get("stuck_goal_override")),
                bool(step.get("global_goal_override")),
            ]
        ):
            executor_override_steps += 1

        direction_reuse_count = int(step.get("direction_reuse_count", 0) or 0)
        if direction_reuse_count > previous_direction_reuse_count and direction_reuse_count > 0:
            direction_reuse_events += 1
        previous_direction_reuse_count = direction_reuse_count

        if step.get("planner_called"):
            if step.get("goal_updated"):
                goal_update_delays.append(0)
            elif open_goal_update_step is None:
                open_goal_update_step = step_idx

        if open_goal_update_step is not None and step.get("goal_updated"):
            goal_update_delays.append(step_idx - open_goal_update_step)
            open_goal_update_step = None

        if step.get("goal_updated"):
            if step.get("executor_adoption_changed", step.get("executor_adopted_goal_changed")):
                executor_adoption_delays.append(0)
            elif open_executor_adoption_step is None:
                open_executor_adoption_step = step_idx
        if open_executor_adoption_step is not None and step.get(
            "executor_adoption_changed",
            step.get("executor_adopted_goal_changed"),
        ):
            executor_adoption_delays.append(step_idx - open_executor_adoption_step)
            open_executor_adoption_step = None

    return {
        "strategy_switch_count": strategy_switch_count,
        "decision_delay_steps": _mean(decision_delays),
        "control_ack_delay_steps": _mean(decision_delays),
        "decision_delay_event_count": len(decision_delays),
        "goal_update_delay_steps": _mean(goal_update_delays),
        "controller_goal_update_delay_steps": _mean(goal_update_delays),
        "goal_update_event_count": len(goal_update_delays),
        "executor_adoption_delay_steps": _mean(executor_adoption_delays),
        "executor_adoption_event_count": len(executor_adoption_delays),
        "pending_created_count": pending_created_count,
        "pending_promoted_count": pending_promoted_count,
        "pending_created_and_promoted_count": pending_created_and_promoted_count,
        "pending_promotion_rate": (
            float(pending_promoted_count / pending_created_count)
            if pending_created_count
            else 0.0
        ),
        "grounding_attempt_count": grounding_attempt_count,
        "grounding_noop_count": grounding_noop_count,
        "grounding_noop_rate": (
            float(grounding_noop_count / grounding_attempt_count)
            if grounding_attempt_count
            else 0.0
        ),
        "grounding_noop_reason_counts": dict(grounding_noop_reason_counts),
        "grounding_no_goal_reason_counts": dict(grounding_no_goal_reason_counts),
        "grounding_frontier_fallback_mode_counts": dict(
            grounding_frontier_fallback_mode_counts
        ),
        "grounding_candidate_distance_fallback_mode_counts": dict(
            grounding_candidate_distance_fallback_mode_counts
        ),
        "selected_frontier_same_as_prev_rate": (
            float(same_frontier_as_prev_count / grounding_attempt_count)
            if grounding_attempt_count
            else 0.0
        ),
        "temp_goal_override_ratio": (
            float(temp_goal_override_steps / len(steps)) if steps else 0.0
        ),
        "stuck_goal_override_ratio": (
            float(stuck_goal_override_steps / len(steps)) if steps else 0.0
        ),
        "global_goal_override_ratio": (
            float(global_goal_override_steps / len(steps)) if steps else 0.0
        ),
        "visible_target_override_ratio": (
            float(visible_target_override_steps / len(steps)) if steps else 0.0
        ),
        "executor_override_ratio": (
            float(executor_override_steps / len(steps)) if steps else 0.0
        ),
        "direction_reuse_count": direction_reuse_events,
        "terminal_outcome_counts": dict(terminal_outcome_counts),
        "missing_epoch_trace_steps": missing_epoch_trace_steps,
    }


def compute_run_control_metrics(run_dir, episode_ids=None):
    trace_paths = sorted(glob(os.path.join(run_dir, "step_traces", "episode_*.jsonl")))
    if episode_ids is not None:
        allowed_ids = {int(episode_id) for episode_id in episode_ids}
        trace_paths = [
            path for path in trace_paths if _episode_id_from_path(path) in allowed_ids
        ]
    if not trace_paths:
        return {
            "strategy_switch_count": 0.0,
            "decision_delay_steps": 0.0,
            "control_ack_delay_steps": 0.0,
            "goal_update_delay_steps": 0.0,
            "controller_goal_update_delay_steps": 0.0,
            "executor_adoption_delay_steps": 0.0,
            "pending_promotion_rate": 0.0,
            "decision_delay_event_count": 0,
            "goal_update_event_count": 0,
            "executor_adoption_event_count": 0,
            "pending_created_count": 0,
            "pending_promoted_count": 0,
            "pending_created_and_promoted_count": 0,
            "grounding_attempt_count": 0,
            "grounding_noop_count": 0,
            "grounding_noop_rate": 0.0,
            "grounding_noop_reason_counts": {},
            "grounding_no_goal_reason_counts": {},
            "grounding_frontier_fallback_mode_counts": {},
            "grounding_candidate_distance_fallback_mode_counts": {},
            "selected_frontier_same_as_prev_rate": 0.0,
            "temp_goal_override_ratio": 0.0,
            "stuck_goal_override_ratio": 0.0,
            "global_goal_override_ratio": 0.0,
            "visible_target_override_ratio": 0.0,
            "executor_override_ratio": 0.0,
            "direction_reuse_count": 0.0,
            "terminal_outcome_counts": {},
            "missing_epoch_trace_steps": 0,
        }

    per_episode = [compute_episode_control_metrics(_read_jsonl(path)) for path in trace_paths]

    strategy_switches = [item["strategy_switch_count"] for item in per_episode]
    decision_delay_values = [
        item["decision_delay_steps"]
        for item in per_episode
        if item["decision_delay_event_count"] > 0
    ]
    goal_delay_values = [
        item["goal_update_delay_steps"]
        for item in per_episode
        if item["goal_update_event_count"] > 0
    ]
    adoption_delay_values = [
        item["executor_adoption_delay_steps"]
        for item in per_episode
        if item["executor_adoption_event_count"] > 0
    ]
    pending_created = sum(item["pending_created_count"] for item in per_episode)
    pending_promoted = sum(item["pending_promoted_count"] for item in per_episode)
    pending_created_and_promoted = sum(
        item["pending_created_and_promoted_count"] for item in per_episode
    )
    grounding_attempts = sum(item["grounding_attempt_count"] for item in per_episode)
    grounding_noops = sum(item["grounding_noop_count"] for item in per_episode)
    grounding_noop_reason_counts = _merge_counters(
        item["grounding_noop_reason_counts"] for item in per_episode
    )
    grounding_no_goal_reason_counts = _merge_counters(
        item["grounding_no_goal_reason_counts"] for item in per_episode
    )
    grounding_frontier_fallback_mode_counts = _merge_counters(
        item["grounding_frontier_fallback_mode_counts"] for item in per_episode
    )
    grounding_candidate_distance_fallback_mode_counts = _merge_counters(
        item["grounding_candidate_distance_fallback_mode_counts"] for item in per_episode
    )
    same_frontier_rates = [
        item["selected_frontier_same_as_prev_rate"] for item in per_episode
    ]
    temp_goal_override_ratios = [item["temp_goal_override_ratio"] for item in per_episode]
    stuck_goal_override_ratios = [item["stuck_goal_override_ratio"] for item in per_episode]
    global_goal_override_ratios = [item["global_goal_override_ratio"] for item in per_episode]
    visible_target_override_ratios = [
        item["visible_target_override_ratio"] for item in per_episode
    ]
    executor_override_ratios = [item["executor_override_ratio"] for item in per_episode]
    direction_reuse_counts = [item["direction_reuse_count"] for item in per_episode]
    terminal_outcome_counts = _merge_counters(
        item["terminal_outcome_counts"] for item in per_episode
    )

    return {
        "strategy_switch_count": _mean(strategy_switches),
        "decision_delay_steps": _mean(decision_delay_values),
        "control_ack_delay_steps": _mean(decision_delay_values),
        "goal_update_delay_steps": _mean(goal_delay_values),
        "controller_goal_update_delay_steps": _mean(goal_delay_values),
        "executor_adoption_delay_steps": _mean(adoption_delay_values),
        "pending_promotion_rate": (
            float(pending_promoted / pending_created) if pending_created else 0.0
        ),
        "decision_delay_event_count": sum(
            item["decision_delay_event_count"] for item in per_episode
        ),
        "goal_update_event_count": sum(
            item["goal_update_event_count"] for item in per_episode
        ),
        "executor_adoption_event_count": sum(
            item["executor_adoption_event_count"] for item in per_episode
        ),
        "pending_created_count": pending_created,
        "pending_promoted_count": pending_promoted,
        "pending_created_and_promoted_count": pending_created_and_promoted,
        "grounding_attempt_count": grounding_attempts,
        "grounding_noop_count": grounding_noops,
        "grounding_noop_rate": (
            float(grounding_noops / grounding_attempts) if grounding_attempts else 0.0
        ),
        "grounding_noop_reason_counts": grounding_noop_reason_counts,
        "grounding_no_goal_reason_counts": grounding_no_goal_reason_counts,
        "grounding_frontier_fallback_mode_counts": grounding_frontier_fallback_mode_counts,
        "grounding_candidate_distance_fallback_mode_counts": (
            grounding_candidate_distance_fallback_mode_counts
        ),
        "selected_frontier_same_as_prev_rate": _mean(same_frontier_rates),
        "temp_goal_override_ratio": _mean(temp_goal_override_ratios),
        "stuck_goal_override_ratio": _mean(stuck_goal_override_ratios),
        "global_goal_override_ratio": _mean(global_goal_override_ratios),
        "visible_target_override_ratio": _mean(visible_target_override_ratios),
        "executor_override_ratio": _mean(executor_override_ratios),
        "direction_reuse_count": _mean(direction_reuse_counts),
        "terminal_outcome_counts": terminal_outcome_counts,
        "missing_epoch_trace_steps": sum(
            item["missing_epoch_trace_steps"] for item in per_episode
        ),
    }
