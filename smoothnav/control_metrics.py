"""Phase 2 control metrics derived from step traces."""

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


def compute_episode_control_metrics(steps):
    strategy_switch_count = 0
    pending_created_count = 0
    pending_promoted_count = 0
    decision_delays = []
    goal_update_delays = []

    open_decision_step = None
    open_goal_update_step = None

    for step in steps:
        step_idx = int(step.get("step_idx", 0))
        graph_delta = step.get("graph_delta", {})
        has_semantic_event = bool(step.get("new_node_count", 0)) or bool(
            graph_delta.get("new_rooms", [])
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

        if step.get("planner_called"):
            if step.get("goal_updated"):
                goal_update_delays.append(0)
            elif open_goal_update_step is None:
                open_goal_update_step = step_idx

        if open_goal_update_step is not None and step.get("goal_updated"):
            goal_update_delays.append(step_idx - open_goal_update_step)
            open_goal_update_step = None

    return {
        "strategy_switch_count": strategy_switch_count,
        "decision_delay_steps": _mean(decision_delays),
        "decision_delay_event_count": len(decision_delays),
        "goal_update_delay_steps": _mean(goal_update_delays),
        "goal_update_event_count": len(goal_update_delays),
        "pending_created_count": pending_created_count,
        "pending_promoted_count": pending_promoted_count,
        "pending_promotion_rate": (
            float(pending_promoted_count / pending_created_count)
            if pending_created_count
            else 0.0
        ),
    }


def compute_run_control_metrics(run_dir):
    trace_paths = sorted(glob(os.path.join(run_dir, "step_traces", "episode_*.jsonl")))
    if not trace_paths:
        return {
            "strategy_switch_count": 0.0,
            "decision_delay_steps": 0.0,
            "goal_update_delay_steps": 0.0,
            "pending_promotion_rate": 0.0,
            "decision_delay_event_count": 0,
            "goal_update_event_count": 0,
            "pending_created_count": 0,
            "pending_promoted_count": 0,
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
    pending_created = sum(item["pending_created_count"] for item in per_episode)
    pending_promoted = sum(item["pending_promoted_count"] for item in per_episode)

    return {
        "strategy_switch_count": _mean(strategy_switches),
        "decision_delay_steps": _mean(decision_delay_values),
        "goal_update_delay_steps": _mean(goal_delay_values),
        "pending_promotion_rate": (
            float(pending_promoted / pending_created) if pending_created else 0.0
        ),
        "decision_delay_event_count": sum(
            item["decision_delay_event_count"] for item in per_episode
        ),
        "goal_update_event_count": sum(
            item["goal_update_event_count"] for item in per_episode
        ),
        "pending_created_count": pending_created,
        "pending_promoted_count": pending_promoted,
    }
