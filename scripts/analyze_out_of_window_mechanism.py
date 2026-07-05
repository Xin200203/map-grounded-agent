#!/usr/bin/env python3
"""Attribute out-of-local-window behavior to repair vs avoidance mechanisms.

For each run under <suite_root>/<profile>/, walks step_traces and tallies:
  - oow_events: grounding attempts that failed with out_of_local_window
  - repaired_same_step: patch moved the local map and the retry projected
  - deferred: patch could not resolve (goal parked as deferred)
  - repair_reasons: planner_reasons markers for the retry path
  - post_patch_next_grounding_ok: next grounding event after a deferral succeeded

Prints one line per profile so numbers can go straight into the paper table.
"""

import argparse
import glob
import json
import os


def analyze_run(run_dir: str):
    stats = {
        "steps": 0,
        "grounding_attempts": 0,
        "oow_noops": 0,
        "patch_handled": 0,
        "patch_resolved_same_step": 0,
        "patch_deferred": 0,
        "oow_after_deferral_recovered": 0,
        "local_projection_valid_steps": 0,
    }
    pending_deferral = False
    for trace_path in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        for line in open(trace_path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stats["steps"] += 1
            events = rec.get("grounding_events") or []
            stats["grounding_attempts"] += len(events)
            oow_here = sum(
                1 for e in events if (e.get("noop_reason") or e.get("reason")) == "out_of_local_window"
            )
            stats["oow_noops"] += oow_here
            if rec.get("grounding_patch_reason"):
                stats["patch_handled"] += 1
                if rec.get("grounding_deferred"):
                    stats["patch_deferred"] += 1
                    pending_deferral = True
                elif rec.get("grounding_patch_moved_local_map"):
                    stats["patch_resolved_same_step"] += 1
            if pending_deferral and events and not oow_here:
                changed = any(e.get("changed") for e in events)
                if changed:
                    stats["oow_after_deferral_recovered"] += 1
                    pending_deferral = False
            if rec.get("local_projection_valid"):
                stats["local_projection_valid_steps"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    args = parser.parse_args()

    for profile in args.profiles:
        total = None
        run_count = 0
        for run_dir in sorted(glob.glob(os.path.join(args.suite_root, profile, "*", "*_text_*"))):
            if not os.path.isdir(os.path.join(run_dir, "step_traces")):
                continue
            stats = analyze_run(run_dir)
            run_count += 1
            if total is None:
                total = dict(stats)
            else:
                for key, value in stats.items():
                    total[key] += value
        if total is None:
            print(f"{profile}: no runs")
            continue
        print(
            f"{profile} (n={run_count} runs): "
            f"oow_noops={total['oow_noops']} "
            f"patch_handled={total['patch_handled']} "
            f"resolved_same_step={total['patch_resolved_same_step']} "
            f"deferred={total['patch_deferred']} "
            f"recovered_after_deferral={total['oow_after_deferral_recovered']} "
            f"grounding_attempts={total['grounding_attempts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
