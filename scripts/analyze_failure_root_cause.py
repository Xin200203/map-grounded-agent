#!/usr/bin/env python3
"""Root-cause split for never-detected failures.

For every failed episode, gather all node captions that ever entered the graph
(step_traces new_node_captions) and score them against the goal text with the
production matcher. Buckets:

  category_never_captioned  - no caption of the goal category all episode:
                              detector/graph recall or coverage failure
  captioned_below_threshold - goal-category caption existed but production
                              score < 0.75: matcher/threshold failure
  candidate_surfaced        - target candidate did fire (failure is downstream:
                              anchoring/reachability; matches pyramid buckets)

Also prints per-goal-category counts and graph richness stats.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from smoothnav.target_matching import score_caption_against_goal  # noqa: E402

EPISODE_CATEGORY = {
    228: "tv_monitor", 527: "chair", 661: "chair", 64: "chair", 159: "sofa",
    358: "tv_monitor", 486: "chair", 574: "plant", 717: "plant", 778: "plant",
    859: "plant", 955: "bed",
    65: "bed", 79: "chair", 93: "chair", 160: "chair", 170: "plant", 180: "chair",
    229: "chair", 248: "chair", 267: "plant", 359: "bed", 372: "chair", 385: "sofa",
    487: "chair", 500: "chair", 513: "toilet", 528: "chair", 543: "chair",
    558: "sofa", 575: "plant", 585: "chair", 595: "plant", 662: "tv_monitor",
    674: "chair", 686: "chair", 718: "bed", 734: "tv_monitor", 750: "plant",
    779: "chair", 788: "plant", 797: "plant", 860: "chair", 886: "chair",
    912: "plant", 956: "tv_monitor", 970: "chair", 984: "chair",
}


def goal_text_of(run_dir):
    for pc in glob.glob(os.path.join(run_dir, "planner_calls", "*.jsonl")):
        for line in open(pc):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("goal_description")
            if text:
                return str(text)
    return ""


def episode_id_of(run_dir):
    import re
    try:
        manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
        m = re.search(r"--episode_id\s+(\d+)", str(manifest.get("command", "")))
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def analyze_run(run_dir):
    try:
        entry = json.load(open(os.path.join(run_dir, "episode_results.json")))[0]
    except Exception:
        return None
    success = float(entry.get("success") or 0.0) > 0
    captions = set()
    candidate = False
    for tp in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        for line in open(tp):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for cap in rec.get("new_node_captions") or []:
                captions.add(str(cap))
            delta = rec.get("graph_delta") or {}
            if delta.get("target_candidate_captions"):
                candidate = True
    goal_text = goal_text_of(run_dir)
    best = 0.0
    best_cap = ""
    for cap in captions:
        score = score_caption_against_goal(cap, goal_text)["score"]
        if score > best:
            best, best_cap = score, cap
    ep = episode_id_of(run_dir)
    if success:
        bucket = "success"
    elif candidate:
        bucket = "candidate_surfaced"
    elif best >= 0.75:
        bucket = "captioned_but_candidate_missed"  # should be rare: instrumentation gap
    elif best > 0.0:
        bucket = "captioned_below_threshold"
    else:
        bucket = "category_never_captioned"
    return {
        "episode": ep,
        "category": EPISODE_CATEGORY.get(ep, "?"),
        "bucket": bucket,
        "best_score": round(best, 2),
        "best_caption": best_cap[:40],
        "n_captions": len(captions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled-roots", nargs="+", required=True)
    parser.add_argument("--per-episode", action="store_true")
    args = parser.parse_args()
    for pair in args.labelled_roots:
        label, _, root = pair.partition("=")
        buckets = {}
        by_cat = {}
        rich = []
        rows = []
        for run_dir in sorted(glob.glob(os.path.join(root, "*", "*_text_*"))):
            res = analyze_run(run_dir)
            if res is None:
                continue
            rows.append(res)
            buckets[res["bucket"]] = buckets.get(res["bucket"], 0) + 1
            if res["bucket"] == "category_never_captioned":
                by_cat[res["category"]] = by_cat.get(res["category"], 0) + 1
            rich.append(res["n_captions"])
        avg_rich = sum(rich) / len(rich) if rich else 0
        print(f"{label} (n={len(rows)}, avg_unique_captions={avg_rich:.1f}): {buckets}")
        if by_cat:
            print(f"   never_captioned by goal category: {by_cat}")
        if args.per_episode:
            for res in rows:
                if res["bucket"] != "success":
                    print(f"   ep{res['episode']:>4} {res['category']:<10} {res['bucket']:<30} best={res['best_score']} '{res['best_caption']}' caps={res['n_captions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
