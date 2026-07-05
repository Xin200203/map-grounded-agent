#!/usr/bin/env python3
"""Count gate-binding events to test whether E3 gate-off ablations are vacuous.

For each run directory, tallies from planner_calls and step_traces:
  - object_choices / low_rel_object_choices: planner object picks whose best
    resolved candidate has target_relevance < 0.75 (relevance gate surface)
  - direct_exec_used: grounding events where graph_debug.direct_object_goal
    was executed; direct_exec_blocked: gate refusals (low relevance / nonlocal)
  - anchor_active_steps / target_progress_hits: steps with an active target
    anchor, and grounding events whose selected breakdown had a nonzero
    target-progress term (PONI-lite gate surface)
  - stall_replans / max_stall: anchor stall decommit firings and the largest
    stall counter observed (decommit gate surface)
"""

import argparse
import glob
import json
import os


def tally_profile(profile_root: str):
    t = {
        "runs": 0,
        "object_choices": 0,
        "low_rel_object_choices": 0,
        "direct_exec_used": 0,
        "direct_exec_blocked": 0,
        "anchor_active_steps": 0,
        "target_progress_hits": 0,
        "stall_replans": 0,
        "max_stall": 0,
    }
    for run_dir in sorted(glob.glob(os.path.join(profile_root, "*", "*_text_*"))):
        if not os.path.isdir(os.path.join(run_dir, "step_traces")):
            continue
        t["runs"] += 1
        for pc in glob.glob(os.path.join(run_dir, "planner_calls", "*.jsonl")):
            for line in open(pc):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("choice_type") != "object":
                    continue
                t["object_choices"] += 1
                cands = rec.get("resolved_object_candidates") or []
                if cands:
                    rel = cands[0].get("target_relevance")
                    if rel is not None and float(rel) < 0.75:
                        t["low_rel_object_choices"] += 1
        for st in glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl")):
            for line in open(st):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                anchor = rec.get("target_anchor_attempt") or {}
                if anchor.get("active"):
                    t["anchor_active_steps"] += 1
                    t["max_stall"] = max(t["max_stall"], int(anchor.get("stall_updates") or 0))
                if rec.get("target_anchor_stall_replan_triggered"):
                    t["stall_replans"] += 1
                for ev in rec.get("grounding_events") or []:
                    dbg = ev.get("graph_debug") or {}
                    if dbg.get("direct_object_goal") is not None:
                        t["direct_exec_used"] += 1
                    if dbg.get("direct_object_goal_blocked_reason"):
                        t["direct_exec_blocked"] += 1
                    breakdown = ev.get("selected_frontier_score_breakdown") or {}
                    if float(breakdown.get("target_progress_score") or 0.0) > 0.0:
                        t["target_progress_hits"] += 1
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled-roots", nargs="+", required=True,
                        help="label=path pairs, e.g. full=results/e1_main_cross12_20260706/smoothnav-full")
    args = parser.parse_args()
    for pair in args.labelled_roots:
        label, _, root = pair.partition("=")
        t = tally_profile(root)
        print(
            f"{label:>18} (n={t['runs']}): obj_choices={t['object_choices']} "
            f"low_rel={t['low_rel_object_choices']} direct_used={t['direct_exec_used']} "
            f"direct_blocked={t['direct_exec_blocked']} anchor_steps={t['anchor_active_steps']} "
            f"tp_hits={t['target_progress_hits']} stall_replans={t['stall_replans']} "
            f"max_stall={t['max_stall']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
