#!/usr/bin/env python3

import argparse
import glob
import json
import os
import statistics
from collections import Counter


NUMERIC_KEYS = [
    "SR",
    "SPL",
    "avg_high_level_calls",
    "avg_low_level_calls",
    "strategy_switch_count",
    "goal_update_delay_steps",
    "executor_override_ratio",
    "pending_created_count",
    "pending_promoted_count",
    "missing_epoch_trace_steps",
]

DICT_KEYS = [
    "grounding_noop_reason_counts",
    "terminal_outcome_counts",
]


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _find_run_records(results_root, profiles, episodes):
    dedup = {}
    for profile in profiles:
        summary_paths = sorted(
            glob.glob(os.path.join(results_root, profile, "*", "*", "summary.json"))
        )
        for path in summary_paths:
            run_dir = os.path.dirname(path)
            effective_config_path = os.path.join(run_dir, "effective_config.json")
            if not os.path.isfile(effective_config_path):
                continue
            effective_config = _load_json(effective_config_path)
            episode_id = effective_config.get("episode_id")
            if episode_id not in episodes:
                continue
            summary = _load_json(path)
            record = {
                "profile": profile,
                "episode_id": episode_id,
                "run_dir": run_dir,
                "summary": summary,
            }
            dedup_key = (profile, episode_id)
            previous = dedup.get(dedup_key)
            if previous is None or record["run_dir"] > previous["run_dir"]:
                dedup[dedup_key] = record
    return sorted(dedup.values(), key=lambda record: (record["profile"], record["episode_id"]))


def _mean(values):
    valid = [value for value in values if isinstance(value, (int, float))]
    if not valid:
        return None
    return statistics.mean(valid)


def _aggregate_profile(records):
    aggregate = {
        "num_runs": len(records),
        "episodes": sorted(record["episode_id"] for record in records),
    }
    for key in NUMERIC_KEYS:
        aggregate[key] = _mean([record["summary"].get(key) for record in records])
    for key in DICT_KEYS:
        merged = Counter()
        for record in records:
            merged.update(record["summary"].get(key, {}) or {})
        aggregate[key] = dict(merged)
    aggregate["run_dirs"] = [record["run_dir"] for record in records]
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--episodes", nargs="+", required=True, type=int)
    parser.add_argument("--profiles", nargs="+", required=True)
    args = parser.parse_args()

    records = _find_run_records(args.results_root, args.profiles, set(args.episodes))
    grouped = {}
    for record in records:
        grouped.setdefault(record["profile"], []).append(record)

    report = {
        "results_root": args.results_root,
        "episodes_requested": args.episodes,
        "profiles_requested": args.profiles,
        "profiles": {},
    }

    for profile in args.profiles:
        profile_records = grouped.get(profile, [])
        report["profiles"][profile] = _aggregate_profile(profile_records)

    output_path = os.path.join(args.results_root, "suite_summary.json")
    os.makedirs(args.results_root, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote aggregate summary to {output_path}")


if __name__ == "__main__":
    main()
