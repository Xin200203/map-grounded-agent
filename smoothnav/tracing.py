"""JSONL tracing helpers for SmoothNav runs."""

import dataclasses
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime


def to_jsonable(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def hash_text(text):
    if text is None:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def strategy_to_dict(strategy):
    if strategy is None:
        return None
    return {
        "target_region": getattr(strategy, "target_region", ""),
        "bias_position": to_jsonable(getattr(strategy, "bias_position", None)),
        "reasoning": getattr(strategy, "reasoning", ""),
        "explored_regions": to_jsonable(getattr(strategy, "explored_regions", [])),
        "anchor_object": getattr(strategy, "anchor_object", ""),
    }


_HEAVY_STEP_TRACE_KEYS = {
    "monitor_trigger_reason",
    "monitor_trigger_event_types",
    "goal_epoch",
    "goal_epoch_advanced",
    "pending_create_reason",
    "pending_strategy_type",
    "pending_promotion_reason",
    "bias_input",
    "selected_frontier",
    "selected_frontier_same_as_prev",
    "grounding_success",
    "grounding_changed",
    "grounding_noop_reason",
    "local_projection_valid",
    "topk_frontier_scores",
    "selected_frontier_score_breakdown",
    "top1_top2_gap",
    "base_score_std",
    "bias_score_std",
    "candidate_frontier_count_after_bias_filter",
    "selected_from_bias_filtered_subset",
    "consecutive_grounding_noops",
    "same_frontier_reuse_count",
    "forced_replan_due_to_grounding_failure",
    "grounding_failure_reason",
    "grounding_events",
    "executor_adopted_goal_source",
    "adopted_goal_source",
    "adopted_goal_before",
    "adopted_goal_after",
    "adopted_goal_epoch",
    "executor_adoption_changed",
    "temp_goal_cleared_on_strategy_switch",
    "temp_goal_suppressed_by_epoch",
    "executor_stuck_override_suppressed",
    "controller_stuck_replan_triggered",
    "stuck_suppression_steps_remaining",
    "forced_replan_due_to_direction_reuse",
    "executor_temp_goal_epoch",
}


def _strip_controller_trace(payload):
    stripped = {}
    for key, value in payload.items():
        if key in _HEAVY_STEP_TRACE_KEYS:
            continue
        if key == "graph_delta" and isinstance(value, dict):
            graph_delta = deepcopy(value)
            graph_delta.pop("node_captions_snapshot", None)
            stripped[key] = graph_delta
            continue
        stripped[key] = value
    return stripped


class RunTracer:
    """Append-only JSONL writers keyed by episode id."""

    def __init__(self, run_dir, enable_controller_trace=True):
        self.run_dir = run_dir
        self.enable_controller_trace = bool(enable_controller_trace)
        self._handles = {}

    def _episode_path(self, subdir, episode_id):
        filename = f"episode_{int(episode_id):06d}.jsonl"
        return os.path.join(self.run_dir, subdir, filename)

    def _write_jsonl(self, subdir, episode_id, payload):
        path = self._episode_path(subdir, episode_id)
        handle = self._handles.get(path)
        if handle is None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handle = open(path, "a")
            self._handles[path] = handle

        record = {"timestamp_local": datetime.now().isoformat(timespec="milliseconds")}
        record.update(to_jsonable(payload))
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    def record_step(self, episode_id, payload):
        if not self.enable_controller_trace:
            payload = _strip_controller_trace(payload)
        self._write_jsonl("step_traces", episode_id, payload)

    def record_planner_call(self, episode_id, payload):
        if not self.enable_controller_trace:
            return
        self._write_jsonl("planner_calls", episode_id, payload)

    def record_monitor_call(self, episode_id, payload):
        if not self.enable_controller_trace:
            return
        self._write_jsonl("monitor_calls", episode_id, payload)

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
