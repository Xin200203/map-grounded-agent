#!/usr/bin/env python3
"""For anchored-failed episodes: did the agent ever reach its anchor coordinate?

Splits anchored_failed into:
  reached_anchor_vicinity  - min agent->anchor distance < near_cells:
                             anchor was reached; failure is anchor-precision
                             (node center off-target) or stop/instance layer
  never_reached_anchor     - agent never got near its own anchor:
                             reachability / approach failure

Uses step-trace fields only (pose_before.map_x/y vs
target_anchor_attempt.full_map_coord).
"""

import argparse
import glob
import json
import math
import os


def analyze_run(run_dir, near_cells):
    try:
        entry = json.load(open(os.path.join(run_dir, "episode_results.json")))[0]
    except Exception:
        return None
    if float(entry.get("success") or 0.0) > 0:
        return ("success", None)
    min_dist = None
    anchored = False
    for tp in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        for line in open(tp):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            anchor = rec.get("target_anchor_attempt") or {}
            coord = anchor.get("full_map_coord")
            if not (anchor.get("active") and coord):
                continue
            anchored = True
            pose = rec.get("pose_before") or {}
            try:
                dist = math.dist(
                    [float(pose.get("map_x")), float(pose.get("map_y"))],
                    [float(coord[0]), float(coord[1])],
                )
            except Exception:
                continue
            if min_dist is None or dist < min_dist:
                min_dist = dist
    if not anchored:
        return ("no_anchor", None)
    if min_dist is not None and min_dist < near_cells:
        return ("reached_anchor_vicinity", min_dist)
    return ("never_reached_anchor", min_dist)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled-roots", nargs="+", required=True)
    parser.add_argument("--near-cells", type=float, default=20.0,
                        help="vicinity threshold in map cells (20 cells = 1 m at 5 cm)")
    args = parser.parse_args()
    for pair in args.labelled_roots:
        label, _, root = pair.partition("=")
        counts = {}
        dists = []
        for run_dir in sorted(glob.glob(os.path.join(root, "*", "*_text_*"))):
            res = analyze_run(run_dir, args.near_cells)
            if res is None:
                continue
            bucket, dist = res
            counts[bucket] = counts.get(bucket, 0) + 1
            if bucket == "never_reached_anchor" and dist is not None:
                dists.append(dist)
        med = sorted(dists)[len(dists) // 2] if dists else None
        print(f"{label}: {counts} median_unreached_dist_cells={med and round(med,1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
