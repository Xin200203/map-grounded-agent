"""JSONL tracing helpers for SmoothNav runs."""

import dataclasses
import hashlib
import json
import os
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


class RunTracer:
    """Append-only JSONL writers keyed by episode id."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
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
        self._write_jsonl("step_traces", episode_id, payload)

    def record_planner_call(self, episode_id, payload):
        self._write_jsonl("planner_calls", episode_id, payload)

    def record_monitor_call(self, episode_id, payload):
        self._write_jsonl("monitor_calls", episode_id, payload)

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
