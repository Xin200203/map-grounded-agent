#!/usr/bin/env python3
"""Audit SmoothNav experiment artifacts against layered runtime contracts.

This is a post-hoc monitor: it does not run Habitat and does not mutate results.
It reads completed run directories (or results roots containing run directories),
checks every major SmoothNav contract layer, and assigns a dominant first-failing
layer for failed runs.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TARGET_BRANCH_PREFIX = "unexplored target:"
OBJECT_PREFIX = "object:"
RUNNING_OUTCOMES = {"", "RUNNING", None}
SUCCESS_OUTCOMES = {"SUCCESS"}
EXPECTED_RUN_FILES = (
    "summary.json",
    "effective_config.json",
    "episode_results.json",
)
LAYER_ORDER = [
    "suite_artifact_config",
    "dataset_scene_episode",
    "determinism_branch_reproducibility",
    "perception_graph_target_evidence",
    "planner_menu_choice",
    "strategy_contract",
    "grounding_frontier_value",
    "grounding_stage_snapshot_replay",
    "local_map_lifecycle",
    "controller_pending_prefetch",
    "target_anchor_attempt",
    "monitor_trigger_coverage",
    "executor_low_level_control",
    "terminal_metric_arbitration",
    "replay_counterfactual_coverage",
]


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"_parse_error": line[:200]})
    return items


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _status(status: str, metrics: Optional[Dict[str, Any]] = None,
            observations: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "status": status,
        "metrics": metrics or {},
        "observations": observations or [],
    }


def _region(step: Dict[str, Any], key: str = "current_strategy") -> str:
    strategy = step.get(key) or {}
    if isinstance(strategy, dict):
        return str(strategy.get("target_region") or "")
    return ""


def _is_target_anchor(region: str) -> bool:
    return str(region or "").startswith(TARGET_BRANCH_PREFIX)


def _is_object(region: str) -> bool:
    return str(region or "").startswith(OBJECT_PREFIX)


def _specificity(region: str) -> int:
    region = str(region or "")
    if _is_object(region):
        return 3
    if _is_target_anchor(region):
        return 2
    if region.startswith("unexplored"):
        return 0
    if region:
        return 1
    return -1


def _event_types(step: Dict[str, Any]) -> List[str]:
    return list((step.get("graph_delta") or {}).get("event_types") or [])


def _target_details(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph_delta = step.get("graph_delta") or {}
    details = graph_delta.get("target_candidate_details") or []
    if details:
        return [dict(item) for item in details if isinstance(item, dict)]
    world = step.get("world_state_summary") or {}
    objects = world.get("object_summary") or []
    result = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        relevance = _safe_float(obj.get("target_relevance")) or 0.0
        if relevance >= 0.75:
            result.append(dict(obj))
    return result


def _terminal_outcome(summary: Dict[str, Any], episode_results: Any,
                      steps: Sequence[Dict[str, Any]]) -> str:
    for step in reversed(steps):
        terminal = step.get("terminal_decision") or {}
        outcome = terminal.get("outcome") or step.get("terminal_outcome")
        if outcome not in RUNNING_OUTCOMES:
            return str(outcome)
    if isinstance(episode_results, list) and episode_results:
        outcome = episode_results[-1].get("terminal_outcome")
        if outcome not in RUNNING_OUTCOMES:
            return str(outcome)
    counts = summary.get("terminal_outcome_counts") or summary.get("terminal_outcomes") or {}
    if isinstance(counts, dict) and counts:
        return str(max(counts.items(), key=lambda item: item[1])[0])
    sr = _safe_float(summary.get("SR"))
    if sr is not None and sr > 0:
        return "SUCCESS"
    return ""


def _first_step_with(steps: Sequence[Dict[str, Any]], predicate) -> Optional[int]:
    for step in steps:
        if predicate(step):
            return _safe_int(step.get("step_idx"))
    return None


def _step_trace_files(run_dir: Path) -> List[Path]:
    return sorted((run_dir / "step_traces").glob("episode_*.jsonl"))


def _planner_call_files(run_dir: Path) -> List[Path]:
    return sorted((run_dir / "planner_calls").glob("episode_*.jsonl"))


def _monitor_call_files(run_dir: Path) -> List[Path]:
    return sorted((run_dir / "monitor_calls").glob("episode_*.jsonl"))


def _load_trace(run_dir: Path) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for path in _step_trace_files(run_dir):
        steps.extend(_read_jsonl(path))
    steps.sort(key=lambda item: (_safe_int(item.get("episode_id")) or 0, _safe_int(item.get("step_idx")) or -1))
    return steps


def _load_jsonl_many(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in paths:
        items.extend(_read_jsonl(path))
    return items


def _dist(a: Any, b: Any) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        return float(math.dist([float(a[0]), float(a[1])], [float(b[0]), float(b[1])]))
    except Exception:
        return None


def _grounding_event_trigger(step: Dict[str, Any], event: Dict[str, Any]) -> str:
    trigger = event.get("trigger")
    if trigger:
        return str(trigger)
    reasons = step.get("planner_reasons") or []
    if reasons:
        return str(reasons[0])
    events = _event_types(step)
    if events:
        return str(events[0])
    return ""


def _target_grounding_events(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for step in steps:
        current_region = _region(step)
        if not _is_target_anchor(current_region):
            continue
        for event in step.get("grounding_events") or []:
            if not isinstance(event, dict):
                continue
            br = event.get("selected_frontier_score_breakdown") or {}
            bias = event.get("bias_input")
            frontier = event.get("selected_frontier")
            semantic_term = None
            final_score = _safe_float(br.get("final_score"))
            bias_score = _safe_float(br.get("bias_score"))
            semantic_weight = _safe_float(br.get("semantic_bias_weight"))
            target_progress_score = _safe_float(br.get("target_progress_score"))
            target_progress_weight = _safe_float(br.get("target_progress_weight"))
            if final_score and bias_score is not None and semantic_weight is not None:
                target_term = 0.0
                if (
                    target_progress_score is not None
                    and target_progress_weight is not None
                ):
                    target_term = float(target_progress_weight * target_progress_score)
                semantic_term = float(
                    (semantic_weight * bias_score + target_term) / final_score
                )
            events.append(
                {
                    "step_idx": _safe_int(step.get("step_idx")),
                    "trigger": _grounding_event_trigger(step, event),
                    "target_region": current_region,
                    "bias_input": bias,
                    "selected_frontier": frontier,
                    "anchor_euclidean_distance": _dist(bias, frontier),
                    "projected_goal": event.get("projected_goal"),
                    "local_projection_valid": bool(event.get("local_projection_valid")),
                    "changed": bool(event.get("changed")),
                    "bias_distance": _safe_float(br.get("bias_distance")),
                    "bias_score": bias_score,
                    "target_progress_score": target_progress_score,
                    "target_progress_weight": target_progress_weight,
                    "target_progress_mode": br.get("target_progress_mode"),
                    "base_score": _safe_float(br.get("base_score")),
                    "novelty_score": _safe_float(br.get("novelty_score")),
                    "actionability_score": _safe_float(br.get("actionability_score")),
                    "repeat_penalty": _safe_float(br.get("repeat_penalty")),
                    "recent_penalty": _safe_float(br.get("recent_penalty")),
                    "final_score": final_score,
                    "semantic_term_share": semantic_term,
                    "top1_top2_gap": _safe_float(event.get("top1_top2_gap")),
                    "candidate_frontier_count_after_bias_filter": _safe_int(
                        event.get("candidate_frontier_count_after_bias_filter")
                    ),
                    "selected_from_bias_filtered_subset": bool(
                        event.get("selected_from_bias_filtered_subset")
                    ),
                    "raw_frontier_count": _safe_int(event.get("raw_frontier_count")),
                    "filtered_frontier_count": _safe_int(event.get("filtered_frontier_count")),
                }
            )
    return events


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _values(items: Sequence[Dict[str, Any]], key: str) -> List[float]:
    vals = []
    for item in items:
        val = _safe_float(item.get(key))
        if val is not None:
            vals.append(val)
    return vals


def _audit_artifacts(run_dir: Path, steps: Sequence[Dict[str, Any]],
                     planner_calls: Sequence[Dict[str, Any]],
                     monitor_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    missing = [name for name in EXPECTED_RUN_FILES if not (run_dir / name).is_file()]
    if not _step_trace_files(run_dir):
        missing.append("step_traces/episode_*.jsonl")
    optional_missing = []
    if not _planner_call_files(run_dir):
        optional_missing.append("planner_calls/episode_*.jsonl")
    if not _monitor_call_files(run_dir):
        optional_missing.append("monitor_calls/episode_*.jsonl")
    status = "fail" if missing else "pass"
    if not missing and optional_missing:
        status = "warn"
    return _status(
        status,
        {
            "missing_required_artifacts": missing,
            "missing_optional_artifacts": optional_missing,
            "step_count": len(steps),
            "planner_call_count": len(planner_calls),
            "monitor_call_count": len(monitor_calls),
        },
        ["required artifacts missing" if missing else "required artifacts present"],
    )


def _audit_dataset(config: Dict[str, Any], steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first = steps[0] if steps else {}
    task_spec = first.get("task_spec") or {}
    primary_goal = task_spec.get("primary_goal") or config.get("goal") or ""
    required = ["episode_id", "split", "goal_type", "text_goal_dataset", "success_dist", "max_episode_length"]
    missing = [key for key in required if config.get(key) in (None, "")]
    if not primary_goal and str(config.get("goal_type") or "") == "text":
        missing.append("task_spec.primary_goal")
    status = "warn" if missing else "pass"
    return _status(
        status,
        {
            "episode_id": config.get("episode_id"),
            "split": config.get("split"),
            "goal_type": config.get("goal_type"),
            "text_goal_dataset": config.get("text_goal_dataset"),
            "success_dist": config.get("success_dist"),
            "max_episode_length": config.get("max_episode_length"),
            "primary_goal_present": bool(primary_goal),
            "missing_metadata": missing,
        },
    )


def _audit_branch_repro(config: Dict[str, Any], steps: Sequence[Dict[str, Any]],
                        planner_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first_target_step = _first_step_with(steps, lambda s: bool(_target_details(s)))
    first_target_anchor_step = _first_step_with(steps, lambda s: _is_target_anchor(_region(s)))
    first_regions = []
    last_region = None
    for step in steps:
        region = _region(step)
        if region and region != last_region:
            first_regions.append({"step_idx": _safe_int(step.get("step_idx")), "target_region": region})
            last_region = region
        if len(first_regions) >= 12:
            break
    prompt_hashes = [str(item.get("prompt_hash") or _hash_text(str(item.get("raw_prompt") or ""))) for item in planner_calls[:12]]
    response_hashes = [str(item.get("response_hash") or _hash_text(str(item.get("raw_response") or ""))) for item in planner_calls[:12]]
    target_events = sum(1 for step in steps if _target_details(step))
    status = "pass"
    if first_target_step is None:
        status = "warn"
    elif first_target_anchor_step is None:
        status = "warn"
    return _status(
        status,
        {
            "seed": config.get("seed"),
            "llm_model": config.get("llm_model"),
            "api_provider": config.get("api_provider"),
            "first_target_step": first_target_step,
            "first_target_anchor_step": first_target_anchor_step,
            "target_branch_covered": first_target_anchor_step is not None,
            "target_event_count": target_events,
            "first_strategy_sequence": first_regions,
            "planner_prompt_hashes_head": prompt_hashes,
            "planner_response_hashes_head": response_hashes,
        },
    )


def _audit_perception(steps: Sequence[Dict[str, Any]], terminal_outcome: str,
                      goal_type: str) -> Dict[str, Any]:
    details_by_step = []
    captions = Counter()
    centers = []
    for step in steps:
        details = _target_details(step)
        if not details:
            continue
        step_idx = _safe_int(step.get("step_idx"))
        step_captions = []
        for item in details:
            caption = str(item.get("caption") or item.get("name") or "")
            if caption:
                captions[caption] += 1
                step_captions.append(caption)
            center = item.get("center") or item.get("object_center")
            if center is not None:
                centers.append(center)
        details_by_step.append({"step_idx": step_idx, "captions": step_captions, "count": len(details)})
    first_target_step = details_by_step[0]["step_idx"] if details_by_step else None
    status = "pass" if details_by_step else "warn"
    if str(goal_type) == "text" and terminal_outcome not in SUCCESS_OUTCOMES and not details_by_step:
        status = "fail"
    return _status(
        status,
        {
            "first_target_step": first_target_step,
            "target_event_count": len(details_by_step),
            "target_candidate_captions": dict(captions),
            "target_center_updates": len(centers),
            "target_candidate_flicker_rate": 0.0 if not steps else 1.0 - _rate(len(details_by_step), len(steps)),
            "sample_target_steps": details_by_step[:8],
        },
    )


def _audit_planner(steps: Sequence[Dict[str, Any]], planner_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first_target_step = _first_step_with(steps, lambda s: bool(_target_details(s)))
    first_planner_after_target = None
    for step in steps:
        idx = _safe_int(step.get("step_idx"))
        if first_target_step is not None and idx is not None and idx >= first_target_step and step.get("planner_called"):
            first_planner_after_target = idx
            break
    empty_responses = [item for item in planner_calls if not str(item.get("raw_response") or "").strip()]
    fallbacks = [item for item in planner_calls if item.get("fallback_triggered")]
    invalid_parse = [item for item in planner_calls if item.get("error_message") and not item.get("parsed_result")]
    target_in_prompt_after_target = None
    selected_target_after_target = None
    if first_target_step is not None:
        for call in planner_calls:
            idx = _safe_int(call.get("step_idx"))
            if idx is None or idx < first_target_step:
                continue
            prompt = str(call.get("raw_prompt") or "").lower()
            if target_in_prompt_after_target is None:
                target_in_prompt_after_target = ("[object]" in prompt and ("tv" in prompt or "television" in prompt))
            if selected_target_after_target is None:
                strategy = call.get("strategy") or {}
                target_region = str(strategy.get("target_region") or "")
                choice_id = str(call.get("choice_id") or "")
                selected_target_after_target = target_region.startswith(TARGET_BRANCH_PREFIX) or choice_id.lower() in {"tv", "television"}
    latency = None if first_target_step is None or first_planner_after_target is None else first_planner_after_target - first_target_step
    status = "pass"
    if first_target_step is not None and first_planner_after_target is None:
        status = "fail"
    elif (
        first_target_step is not None
        and target_in_prompt_after_target is True
        and selected_target_after_target is False
    ):
        status = "fail"
    elif empty_responses or invalid_parse:
        status = "warn"
    return _status(
        status,
        {
            "planner_call_count": len(planner_calls),
            "first_planner_after_target": first_planner_after_target,
            "evidence_to_planner_latency": latency,
            "empty_response_count": len(empty_responses),
            "fallback_triggered_count": len(fallbacks),
            "invalid_parse_count": len(invalid_parse),
            "target_in_prompt_after_target": target_in_prompt_after_target,
            "selected_target_after_target": selected_target_after_target,
        },
    )


def _audit_strategy(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first_target_step = _first_step_with(steps, lambda s: bool(_target_details(s)))
    first_object_commit = _first_step_with(
        steps,
        lambda s: first_target_step is not None
        and (_safe_int(s.get("step_idx")) or -1) >= first_target_step
        and _is_object(_region(s)),
    )
    first_target_anchor = _first_step_with(
        steps,
        lambda s: first_target_step is not None
        and (_safe_int(s.get("step_idx")) or -1) >= first_target_step
        and _is_target_anchor(_region(s)),
    )
    direct_goal_steps = []
    blocked_reasons = Counter()
    for step in steps:
        idx = _safe_int(step.get("step_idx"))
        for event in step.get("grounding_events") or []:
            graph_debug = event.get("graph_debug") or {}
            if graph_debug.get("direct_object_goal"):
                direct_goal_steps.append(idx)
            reason = graph_debug.get("direct_object_goal_blocked_reason")
            if reason:
                blocked_reasons[str(reason)] += 1
    status = "pass"
    if first_target_step is not None and first_object_commit is not None and first_target_anchor is None and not direct_goal_steps:
        status = "warn"
    return _status(
        status,
        {
            "first_object_commit_after_target": first_object_commit,
            "first_target_anchor_after_target": first_target_anchor,
            "first_direct_goal_after_target": direct_goal_steps[0] if direct_goal_steps else None,
            "direct_object_goal_step_count": len(direct_goal_steps),
            "direct_object_blocked_reasons": dict(blocked_reasons),
        },
    )


def _audit_grounding(steps: Sequence[Dict[str, Any]], terminal_outcome: str) -> Dict[str, Any]:
    all_events = []
    noop_reasons = Counter()
    for step in steps:
        for event in step.get("grounding_events") or []:
            if not isinstance(event, dict):
                continue
            all_events.append(event)
            reason = event.get("noop_reason") or event.get("failure_code")
            if reason:
                noop_reasons[str(reason)] += 1
    target_events = _target_grounding_events(steps)
    total = len(target_events)
    bias_zero = sum(1 for item in target_events if (_safe_float(item.get("bias_score")) or 0.0) <= 0.0)
    target_progress_zero = sum(
        1
        for item in target_events
        if (_safe_float(item.get("target_progress_score")) or 0.0) <= 0.0
    )
    base_zero = sum(1 for item in target_events if (_safe_float(item.get("base_score")) or 0.0) <= 0.0)
    flat_gap = sum(1 for item in target_events if (_safe_float(item.get("top1_top2_gap")) or 0.0) <= 1e-6)
    semantic_shares = _values(target_events, "semantic_term_share")
    anchor_distances = _values(target_events, "anchor_euclidean_distance")
    bias_distances = _values(target_events, "bias_distance")
    target_bias_zero_rate = _rate(bias_zero, total)
    target_base_zero_rate = _rate(base_zero, total)
    target_flatline_rate = _rate(flat_gap, total)
    semantic_share_median = median(semantic_shares) if semantic_shares else None
    target_anchor_delta = None
    target_best_delta = None
    if anchor_distances:
        target_anchor_delta = anchor_distances[0] - anchor_distances[-1]
        target_best_delta = anchor_distances[0] - min(anchor_distances)
    status = "pass"
    observations = []
    if total and terminal_outcome not in SUCCESS_OUTCOMES:
        if target_bias_zero_rate > 0.5 and (semantic_share_median is None or semantic_share_median < 0.05):
            status = "fail"
            observations.append("target branch semantic value signal collapsed")
        elif target_flatline_rate > 0.3:
            status = "warn"
            observations.append("target branch frontier scores are frequently tied")
    if noop_reasons and status == "pass":
        status = "warn"
    return _status(
        status,
        {
            "grounding_event_count": len(all_events),
            "grounding_noop_reasons": dict(noop_reasons),
            "target_grounding_event_count": total,
            "target_bias_score_zero_count": bias_zero,
            "target_bias_score_zero_rate": target_bias_zero_rate,
            "target_progress_score_zero_count": target_progress_zero,
            "target_progress_score_zero_rate": _rate(target_progress_zero, total),
            "target_base_score_zero_count": base_zero,
            "target_base_score_zero_rate": target_base_zero_rate,
            "target_value_flatline_count": flat_gap,
            "target_value_flatline_rate": target_flatline_rate,
            "target_semantic_term_share_median": semantic_share_median,
            "target_selected_frontier_anchor_distance_delta": target_anchor_delta,
            "target_best_seen_anchor_distance_delta": target_best_delta,
            "target_bias_distance_min": min(bias_distances) if bias_distances else None,
            "target_bias_distance_max": max(bias_distances) if bias_distances else None,
            "sample_target_grounding_events": target_events[:12],
        },
        observations,
    )


def _audit_local_map(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    move_steps = []
    refresh_steps = []
    stale_risks = []
    for i, step in enumerate(steps):
        idx = _safe_int(step.get("step_idx"))
        moved = bool(step.get("grounding_patch_moved_local_map") or step.get("local_map_moved"))
        current_target = _is_target_anchor(_region(step))
        has_refresh = any(
            _grounding_event_trigger(step, event) == "target_anchor_local_map_refresh"
            for event in step.get("grounding_events") or []
            if isinstance(event, dict)
        ) or "target_anchor_local_map_refresh" in (step.get("planner_reasons") or [])
        has_refresh = has_refresh or bool(
            step.get("grounding_patch_target_anchor_reprojected")
        )
        if moved:
            move_steps.append(idx)
        if has_refresh:
            refresh_steps.append(idx)
        if moved and current_target:
            next_has_refresh = has_refresh
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                next_has_refresh = next_has_refresh or any(
                    _grounding_event_trigger(next_step, event) == "target_anchor_local_map_refresh"
                    for event in next_step.get("grounding_events") or []
                    if isinstance(event, dict)
                )
            if not next_has_refresh:
                stale_risks.append(idx)
    status = "fail" if stale_risks else "pass"
    if not stale_risks and move_steps and not refresh_steps:
        status = "warn"
    return _status(
        status,
        {
            "local_map_move_steps": move_steps[:20],
            "target_anchor_refresh_steps": refresh_steps[:40],
            "target_anchor_refresh_count": len(refresh_steps),
            "stale_local_goal_risk_steps": stale_risks,
            "stale_local_goal_risk_count": len(stale_risks),
        },
    )


def _audit_pending(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    harmful = []
    stale_discard_steps = []
    pending_ages: List[int] = []
    pending_created_step: Optional[int] = None
    last_pending_region = ""
    for prev, step in zip([None] + list(steps[:-1]), steps):
        idx = _safe_int(step.get("step_idx"))
        pending_region = _region(step, "pending_strategy")
        current_region = _region(step)
        if pending_region:
            if pending_region != last_pending_region:
                pending_created_step = idx
                last_pending_region = pending_region
            if pending_created_step is not None and idx is not None:
                pending_ages.append(idx - pending_created_step)
        else:
            pending_created_step = None
            last_pending_region = ""
        if prev is not None:
            prev_current = _region(prev)
            prev_pending = _region(prev, "pending_strategy")
            if (
                _is_target_anchor(prev_current)
                and prev_pending
                and _specificity(prev_pending) <= _specificity(prev_current)
                and not _is_target_anchor(current_region)
                and ("frontier_reached" in (step.get("planner_reasons") or []) or step.get("pending_promoted"))
            ):
                harmful.append(
                    {
                        "step_idx": idx,
                        "from_region": prev_current,
                        "pending_region_previous_step": prev_pending,
                        "to_region": current_region,
                        "planner_reasons": step.get("planner_reasons") or [],
                    }
                )
        if _is_target_anchor(current_region) and not pending_region and prev is not None:
            prev_pending = _region(prev, "pending_strategy")
            if prev_pending and _specificity(prev_pending) <= _specificity(_region(prev)):
                stale_discard_steps.append(idx)
    status = "fail" if harmful else "pass"
    return _status(
        status,
        {
            "harmful_pending_promotion_count": len(harmful),
            "harmful_pending_promotions": harmful[:8],
            "stale_pending_discard_steps": stale_discard_steps[:20],
            "pending_age_max": max(pending_ages) if pending_ages else 0,
            "pending_age_median": median(pending_ages) if pending_ages else 0,
        },
    )


def _audit_attempt(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = []
    stall_replan_steps = []
    for step in steps:
        attempt = step.get("target_anchor_attempt")
        if isinstance(attempt, dict) and attempt.get("active"):
            summaries.append({"step_idx": _safe_int(step.get("step_idx")), **attempt})
        if step.get("target_anchor_stall_replan_triggered"):
            stall_replan_steps.append(_safe_int(step.get("step_idx")))
    max_stall = max([_safe_int(item.get("stall_updates")) or 0 for item in summaries], default=0)
    status = "warn" if stall_replan_steps else "pass"
    return _status(
        status,
        {
            "attempt_observation_count": len(summaries),
            "max_stall_updates": max_stall,
            "stall_replan_steps": stall_replan_steps,
            "last_attempt_summary": summaries[-1] if summaries else None,
        },
    )


def _audit_monitor(steps: Sequence[Dict[str, Any]], monitor_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    unhandled_target_steps = []
    trigger_counts = Counter()
    for step in steps:
        events = _event_types(step)
        trigger_counts.update(events)
        if "target_candidate_detected" in events and not step.get("planner_called") and not step.get("monitor_called"):
            unhandled_target_steps.append(_safe_int(step.get("step_idx")))
    skip_reasons = Counter()
    for step in steps:
        reason = step.get("monitor_trigger_reason") or step.get("monitor_reason")
        if reason and not step.get("monitor_called"):
            skip_reasons[str(reason)] += 1
    status = "fail" if unhandled_target_steps else "pass"
    return _status(
        status,
        {
            "monitor_call_count": len(monitor_calls),
            "event_type_counts": dict(trigger_counts),
            "unhandled_target_candidate_steps": unhandled_target_steps[:50],
            "unhandled_target_candidate_count": len(unhandled_target_steps),
            "monitor_skip_reasons": dict(skip_reasons),
        },
    )


def _audit_executor(summary: Dict[str, Any], episode_results: Any,
                    steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    adoption_changes = sum(1 for step in steps if step.get("executor_adopted_goal_changed"))
    override_keys = [
        "temp_goal_override_ratio",
        "stuck_goal_override_ratio",
        "global_goal_override_ratio",
        "visible_target_override_ratio",
        "executor_override_ratio",
    ]
    overrides = {key: _safe_float(summary.get(key)) for key in override_keys if summary.get(key) is not None}
    distances = []
    for step in steps:
        gd = step.get("graph_delta") or {}
        val = _safe_float(gd.get("dist_to_goal"))
        if val is not None:
            distances.append(((_safe_int(step.get("step_idx")) or 0), val))
    slope = None
    if len(distances) >= 2:
        first = distances[0]
        last = distances[-1]
        denom = max(1, last[0] - first[0])
        slope = (last[1] - first[1]) / denom
    pause_ratio = _safe_float(summary.get("avg_pause_ratio"))
    if pause_ratio is None and isinstance(episode_results, list) and episode_results:
        pause_ratio = _safe_float(episode_results[-1].get("pause_duration_ratio"))
    status = "pass"
    if pause_ratio is not None and pause_ratio > 0.45:
        status = "warn"
    if overrides.get("executor_override_ratio", 0.0) and overrides.get("executor_override_ratio", 0.0) > 0.1:
        status = "warn"
    return _status(
        status,
        {
            "executor_adoption_changes": adoption_changes,
            "executor_override_ratios": overrides,
            "dist_to_goal_slope": slope,
            "pause_ratio": pause_ratio,
            "direction_reversals": summary.get("avg_direction_reversals"),
            "strategy_switch_count": summary.get("strategy_switch_count"),
        },
    )


def _audit_terminal(summary: Dict[str, Any], episode_results: Any,
                    steps: Sequence[Dict[str, Any]], terminal_outcome: str) -> Dict[str, Any]:
    final_decision = {}
    for step in reversed(steps):
        decision = step.get("terminal_decision") or {}
        if decision and decision.get("outcome") not in RUNNING_OUTCOMES:
            final_decision = decision
            break
    sr = _safe_float(summary.get("SR"))
    spl = _safe_float(summary.get("SPL"))
    total_steps = None
    if isinstance(episode_results, list) and episode_results:
        total_steps = episode_results[-1].get("total_steps")
    status = "pass"
    if terminal_outcome not in SUCCESS_OUTCOMES and sr and sr > 0:
        status = "fail"
    return _status(
        status,
        {
            "terminal_outcome": terminal_outcome,
            "terminal_decision": final_decision,
            "SR": sr,
            "SPL": spl,
            "total_steps": total_steps,
            "summary_terminal_outcome_counts": summary.get("terminal_outcome_counts") or summary.get("terminal_outcomes"),
        },
    )


def _audit_replay(run_dir: Path) -> Dict[str, Any]:
    artifacts = {
        "target_branch_capture": run_dir / "target_branch_capture.json",
        "pending_fix_counterfactual": run_dir / "pending_fix_counterfactual.json",
        "target_anchor_attempt_analysis": run_dir / "target_anchor_attempt_analysis.json",
        "target_anchor_attempt_replay": run_dir / "target_anchor_attempt_replay.json",
    }
    present = {name: path.is_file() for name, path in artifacts.items()}
    loaded = {}
    for name, path in artifacts.items():
        if path.is_file():
            loaded[name] = _read_json(path, {})
    status = "pass" if all(present.values()) else "warn"
    return _status(
        status,
        {
            "present_artifacts": present,
            "pending_fix_verdict": (loaded.get("pending_fix_counterfactual") or {}).get("verdict"),
            "attempt_replay_verdict": (loaded.get("target_anchor_attempt_replay") or {}).get("verdict"),
        },
    )


def _audit_grounding_snapshots(run_dir: Path, steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    files = sorted((run_dir / "grounding_snapshots").glob("episode_*.jsonl"))
    records: List[Dict[str, Any]] = []
    for path in files:
        records.extend(_read_jsonl(path))
    target_branch_seen = any(_is_target_anchor(_region(step)) for step in steps)
    if not records:
        return _status(
            "warn" if target_branch_seen else "pass",
            {
                "snapshot_file_count": 0,
                "record_count": 0,
                "target_record_count": 0,
                "target_branch_seen": bool(target_branch_seen),
            },
            ["target branch had no replay snapshots"] if target_branch_seen else [],
        )

    try:
        from smoothnav.frontier_scoring import replay_frontier_value_snapshot
    except Exception as exc:
        return _status(
            "warn",
            {
                "snapshot_file_count": len(files),
                "record_count": len(records),
                "replay_import_error": str(exc),
            },
            ["could not import replay helper"],
        )

    replay_results = []
    for record in records:
        snapshot = record.get("snapshot") if isinstance(record.get("snapshot"), dict) else record
        try:
            replay = replay_frontier_value_snapshot(snapshot)
        except Exception as exc:
            replay = {"status": "error", "reason": str(exc), "checks": {}}
        replay_results.append(
            {
                "step_idx": record.get("step_idx"),
                "trigger": record.get("trigger"),
                "active_target_region": snapshot.get("active_target_region"),
                "status": replay.get("status"),
                "checks": replay.get("checks") or {},
                "semantic_term": replay.get("semantic_term"),
                "semantic_term_share": replay.get("semantic_term_share"),
                "target_progress_score": (
                    replay.get("selected_score_breakdown") or {}
                ).get("target_progress_score"),
            }
        )
    target_records = [
        item for item in replay_results
        if _is_target_anchor(str(item.get("active_target_region") or ""))
    ]
    failing = [item for item in replay_results if item.get("status") != "pass"]
    target_progress_zero = [
        item for item in target_records
        if (_safe_float(item.get("target_progress_score")) or 0.0) <= 0.0
    ]
    semantic_zero = [
        item for item in target_records
        if (_safe_float(item.get("semantic_term")) or 0.0) <= 0.0
    ]
    status = "pass"
    observations: List[str] = []
    if failing or target_progress_zero or semantic_zero:
        status = "fail"
        observations.append("grounding snapshot replay failed staged checks")
    elif target_branch_seen and not target_records:
        status = "warn"
        observations.append("target branch seen but no target replay snapshots")
    return _status(
        status,
        {
            "snapshot_file_count": len(files),
            "record_count": len(records),
            "target_record_count": len(target_records),
            "fail_count": len(failing),
            "target_progress_zero_count": len(target_progress_zero),
            "target_semantic_term_zero_count": len(semantic_zero),
            "sample_results": replay_results[:8],
        },
        observations,
    )


def _choose_dominant_layer(layers: Dict[str, Dict[str, Any]], terminal_outcome: str) -> Tuple[Optional[str], str]:
    if terminal_outcome in SUCCESS_OUTCOMES:
        return None, "terminal success"
    if layers["suite_artifact_config"]["status"] == "fail":
        return "suite_artifact_config", "missing required artifacts"
    if layers["dataset_scene_episode"]["status"] == "fail":
        return "dataset_scene_episode", "dataset or episode metadata invalid"
    branch = layers["determinism_branch_reproducibility"]["metrics"]
    perception = layers["perception_graph_target_evidence"]["metrics"]
    if not branch.get("target_branch_covered") and perception.get("target_event_count", 0) == 0:
        return "determinism_branch_reproducibility", "target branch was not covered in this run"
    if layers["planner_menu_choice"]["status"] == "fail":
        return "planner_menu_choice", "target evidence was not consumed by planner"
    if layers["strategy_contract"]["status"] == "fail":
        return "strategy_contract", "semantic strategy contract violation"
    if layers["controller_pending_prefetch"]["status"] == "fail":
        return "controller_pending_prefetch", "harmful stale pending promotion"
    if layers["grounding_frontier_value"]["status"] == "fail":
        return "grounding_frontier_value", "target frontier value lost semantic signal"
    if layers["grounding_stage_snapshot_replay"]["status"] == "fail":
        return "grounding_stage_snapshot_replay", "target grounding replay hook failed"
    if layers["local_map_lifecycle"]["status"] == "fail":
        return "local_map_lifecycle", "local-map recenter left stale target goal"
    if layers["monitor_trigger_coverage"]["status"] == "fail":
        return "monitor_trigger_coverage", "target trigger was unhandled"
    if layers["target_anchor_attempt"]["status"] == "warn":
        return "target_anchor_attempt", "target-anchor stall/decommit fired"
    if layers["executor_low_level_control"]["status"] == "warn":
        return "executor_low_level_control", "execution degraded after high-level goal"
    return "terminal_metric_arbitration", "failed without a more specific first-failing contract"


def audit_run_dir(run_dir: Path | str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    summary = _read_json(run_dir / "summary.json", {}) or {}
    config = _read_json(run_dir / "effective_config.json", {}) or {}
    episode_results = _read_json(run_dir / "episode_results.json", []) or []
    steps = _load_trace(run_dir)
    planner_calls = _load_jsonl_many(_planner_call_files(run_dir))
    monitor_calls = _load_jsonl_many(_monitor_call_files(run_dir))
    terminal_outcome = _terminal_outcome(summary, episode_results, steps)
    if config.get("goal_type"):
        goal_type = str(config.get("goal_type"))
    elif steps:
        goal_type = str((steps[0].get("task_spec") or {}).get("task_type") or "")
    else:
        goal_type = ""

    layers = {
        "suite_artifact_config": _audit_artifacts(run_dir, steps, planner_calls, monitor_calls),
        "dataset_scene_episode": _audit_dataset(config, steps),
        "determinism_branch_reproducibility": _audit_branch_repro(config, steps, planner_calls),
        "perception_graph_target_evidence": _audit_perception(steps, terminal_outcome, goal_type),
        "planner_menu_choice": _audit_planner(steps, planner_calls),
        "strategy_contract": _audit_strategy(steps),
        "grounding_frontier_value": _audit_grounding(steps, terminal_outcome),
        "grounding_stage_snapshot_replay": _audit_grounding_snapshots(run_dir, steps),
        "local_map_lifecycle": _audit_local_map(steps),
        "controller_pending_prefetch": _audit_pending(steps),
        "target_anchor_attempt": _audit_attempt(steps),
        "monitor_trigger_coverage": _audit_monitor(steps, monitor_calls),
        "executor_low_level_control": _audit_executor(summary, episode_results, steps),
        "terminal_metric_arbitration": _audit_terminal(summary, episode_results, steps, terminal_outcome),
        "replay_counterfactual_coverage": _audit_replay(run_dir),
    }
    dominant_layer, reason = _choose_dominant_layer(layers, terminal_outcome)
    return {
        "run_dir": str(run_dir),
        "run_id": config.get("run_id") or summary.get("run_id") or run_dir.name,
        "profile": config.get("controller_profile") or summary.get("controller_profile"),
        "episode_id": config.get("episode_id"),
        "terminal_outcome": terminal_outcome,
        "dominant_failure_layer": dominant_layer,
        "dominant_failure_reason": reason,
        "layers": layers,
    }


def _discover_run_dirs(paths: Sequence[str]) -> List[Path]:
    result: Dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == "summary.json":
            result[str(path.parent)] = path.parent
        elif path.is_dir() and (path / "summary.json").is_file():
            result[str(path)] = path
        elif path.is_dir():
            for summary in path.glob("**/summary.json"):
                # Exclude analysis helper summaries outside actual run dirs by requiring a step trace dir.
                run_dir = summary.parent
                if (run_dir / "step_traces").is_dir() or (run_dir / "effective_config.json").is_file():
                    result[str(run_dir)] = run_dir
    return [result[key] for key in sorted(result)]


def _compare_branch_reproducibility(run_reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for report in run_reports:
        groups[(report.get("episode_id"), report.get("profile"))].append(report)
    comparisons = []
    for (episode_id, profile), reports in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        if len(reports) < 2:
            continue
        target_steps = [
            report["layers"]["determinism_branch_reproducibility"]["metrics"].get("first_target_step")
            for report in reports
        ]
        target_counts = [
            report["layers"]["determinism_branch_reproducibility"]["metrics"].get("target_event_count")
            for report in reports
        ]
        branch_covered = [
            bool(report["layers"]["determinism_branch_reproducibility"]["metrics"].get("target_branch_covered"))
            for report in reports
        ]
        first_sequences = [
            report["layers"]["determinism_branch_reproducibility"]["metrics"].get("first_strategy_sequence") or []
            for report in reports
        ]
        first_divergence_step = None
        if first_sequences:
            min_len = min(len(seq) for seq in first_sequences)
            for idx in range(min_len):
                regions = {json.dumps(seq[idx], sort_keys=True) for seq in first_sequences}
                if len(regions) > 1:
                    first_divergence_step = min((seq[idx].get("step_idx") for seq in first_sequences if idx < len(seq) and seq[idx].get("step_idx") is not None), default=None)
                    break
        comparisons.append(
            {
                "episode_id": episode_id,
                "profile": profile,
                "num_runs": len(reports),
                "first_target_steps": target_steps,
                "target_event_counts": target_counts,
                "target_branch_covered_flags": branch_covered,
                "target_branch_coverage_rate": _rate(sum(1 for item in branch_covered if item), len(branch_covered)),
                "first_strategy_divergence_step": first_divergence_step,
                "run_ids": [report.get("run_id") for report in reports],
            }
        )
    return comparisons


def audit_paths(paths: Sequence[str]) -> Dict[str, Any]:
    run_dirs = _discover_run_dirs(paths)
    reports = [audit_run_dir(path) for path in run_dirs]
    layer_counts: Dict[str, Counter] = {layer: Counter() for layer in LAYER_ORDER}
    dominant_counts = Counter()
    for report in reports:
        dominant_counts[str(report.get("dominant_failure_layer"))] += 1
        for layer, layer_report in report["layers"].items():
            layer_counts[layer][layer_report.get("status", "unknown")] += 1
    return {
        "schema_version": "smoothnav_trace_contract_audit_v1",
        "input_paths": list(paths),
        "num_runs": len(reports),
        "layer_status_counts": {layer: dict(counter) for layer, counter in layer_counts.items()},
        "dominant_failure_layer_counts": dict(dominant_counts),
        "branch_reproducibility_comparisons": _compare_branch_reproducibility(reports),
        "runs": reports,
    }


def _print_table(report: Dict[str, Any]) -> None:
    print("run_id profile episode terminal dominant_layer reason")
    for item in report.get("runs", []):
        print(
            f"{item.get('run_id')} {item.get('profile')} {item.get('episode_id')} "
            f"{item.get('terminal_outcome')} {item.get('dominant_failure_layer')} "
            f"{item.get('dominant_failure_reason')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run dirs or results roots to audit")
    parser.add_argument("--output", help="Write JSON report to this path")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args()

    report = audit_paths(args.paths)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_table(report)
        if args.output:
            print(f"\nWrote audit report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
