#!/usr/bin/env python3
"""Matched-episode aggregation for E1-style profile suites.

Walks <suite_root>/<profile>/<date>/<*_text_*>/ runs, keys each run by the
--episode_id parsed from manifest.json, and prints:
  - per-episode matched table (success/SPL/outcome per profile)
  - per-profile aggregates over the COMMON completed episode set
  - pairwise win/loss/tie on SPL vs a reference profile

Dependency-free; safe to run while the suite is still filling in.
"""

import argparse
import glob
import json
import os
import re
import sys


def collect_profile_runs(suite_root: str, profile: str):
    """Return {episode_id: run_record}, keeping the LATEST run per episode."""
    runs = {}
    pattern = os.path.join(suite_root, profile, "*", "*_text_*")
    for run_dir in sorted(glob.glob(pattern)):
        manifest_path = os.path.join(run_dir, "manifest.json")
        summary_path = os.path.join(run_dir, "summary.json")
        results_path = os.path.join(run_dir, "episode_results.json")
        if not (os.path.exists(manifest_path) and os.path.exists(summary_path)):
            continue
        try:
            manifest = json.load(open(manifest_path))
            command = str(manifest.get("command", ""))
            match = re.search(r"--episode_id\s+(\d+)", command)
            episode_id = int(match.group(1)) if match else -1
            episode_results = (
                json.load(open(results_path)) if os.path.exists(results_path) else []
            )
            if not episode_results:
                continue
            entry = episode_results[0]
            summary = json.load(open(summary_path))
            runs[episode_id] = {
                "run_dir": run_dir,
                "success": float(entry.get("success") or 0.0),
                "spl": float(entry.get("spl") or 0.0),
                "outcome": str(entry.get("terminal_outcome", "")),
                "high_calls": entry.get("high_level_calls"),
                "low_calls": entry.get("low_level_calls"),
                "smoothness": entry.get("smoothness_score"),
                "pause_count": entry.get("pause_count"),
                "summary_sr": summary.get("SR"),
            }
        except Exception as exc:  # noqa: BLE001 - suite may be mid-write
            print(f"  [skip unreadable {run_dir}: {exc}]", file=sys.stderr)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline-explore", "baseline-periodic", "smoothnav-full"],
    )
    parser.add_argument("--reference", default="baseline-periodic")
    parser.add_argument("--episodes", nargs="*", type=int, default=None,
                        help="Optional explicit episode order for display")
    args = parser.parse_args()

    data = {p: collect_profile_runs(args.suite_root, p) for p in args.profiles}
    for profile, runs in data.items():
        print(f"# {profile}: {len(runs)} completed episodes")

    all_eps = set()
    for runs in data.values():
        all_eps |= set(runs)
    common = set.intersection(*[set(runs) for runs in data.values()]) if data else set()
    order = [e for e in (args.episodes or sorted(all_eps)) if e in all_eps]

    print("\n== matched table (episode: profile -> S/SPL/outcome) ==")
    for ep in order:
        cells = []
        for profile in args.profiles:
            rec = data[profile].get(ep)
            if rec is None:
                cells.append(f"{profile}: -")
            else:
                cells.append(
                    f"{profile}: {int(rec['success'])}/{rec['spl']:.3f}/{rec['outcome'][:24]}"
                )
        marker = "*" if ep in common else " "
        print(f"ep{ep:>4}{marker} | " + " | ".join(cells))

    if not common:
        print("\n(no common completed episodes yet)")
        return 0

    print(f"\n== aggregates over {len(common)} COMMON episodes {sorted(common)} ==")
    for profile in args.profiles:
        rows = [data[profile][ep] for ep in sorted(common)]
        sr = sum(r["success"] for r in rows) / len(rows)
        spl = sum(r["spl"] for r in rows) / len(rows)
        outcomes = {}
        for r in rows:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        print(f"{profile:>22}: SR={sr:.4f} SPL={spl:.4f} outcomes={outcomes}")

    ref = args.reference
    if ref in data:
        print(f"\n== pairwise SPL win/loss/tie vs {ref} (common episodes) ==")
        for profile in args.profiles:
            if profile == ref:
                continue
            win = loss = tie = 0
            for ep in common:
                a = data[profile][ep]["spl"]
                b = data[ref][ep]["spl"]
                if abs(a - b) < 1e-9:
                    tie += 1
                elif a > b:
                    win += 1
                else:
                    loss += 1
            print(f"{profile:>22}: win={win} loss={loss} tie={tie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
