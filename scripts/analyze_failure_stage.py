#!/usr/bin/env python3
"""GT-grounded per-episode failure-stage classifier (measured, not inferred).

For every episode under the given suite roots, walk the step trace and, with
the dataset GT goal position, measure exactly how far down the pipeline the
episode got before failing:

  ladder (each stage is a measured fact from trace + GT):
    committed        current_strategy.target_region ever named the goal category
    candidate        a target_candidate_detected event ever fired
    caption_reco     any accumulated node caption scored >= 0.75 vs goal text
    approach_2m/1m   agent pose (pose_before, GT-calibrated) ever within 2m/1m

Then classify each FAILURE into the deepest stage it reached. This replaces
the graph-internal buckets (which trust the graph's own candidate logic) with
a GT-anchored measurement of where the pipeline actually drops each target.

Known limit (closed by scripts/dump graph nodes, step 2): with node map
positions absent from these traces, "no candidate ever" cannot yet be split
into never-seen vs seen-but-not-recognized. That split needs the per-episode
graph-node dump; this script measures everything the existing traces support.

  python scripts/analyze_failure_stage.py --roots results/c45_probe12_20260707 \
      results/c45_exp36_20260707 --label c45 --transform x=+fwd,y=-right
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base_UniGoal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_goal_distance import (  # noqa: E402
    CANDIDATE_TRANSFORMS, MAP_CENTER_M, distances, episodic_goal_offsets,
)
from audit_hm3d_scene_loadability import load_habitat_modules  # noqa: E402
from smoothnav.target_matching import (  # noqa: E402
    normalize_text, primary_goal_categories, score_caption_against_goal,
    CATEGORY_ALIASES,
)


def goal_category_aliases(goal_text):
    """Alias strings for the goal's primary categories (to match target_region)."""
    cats = primary_goal_categories(goal_text)
    aliases = set()
    for c in cats:
        aliases |= CATEGORY_ALIASES.get(c, {c})
        aliases.add(c)
    return {normalize_text(a) for a in aliases if a}


def walk_trace(run_dir, goal_text):
    """One pass over a run's step traces -> measured stage facts."""
    aliases = goal_category_aliases(goal_text)
    committed = candidate = False
    best_caption = 0.0
    poses = []
    node_count_max = 0
    captions_seen = set()
    for tf in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        with open(tf) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                strat = rec.get("current_strategy") or {}
                region = normalize_text(strat.get("target_region", ""))
                if region and any(a in region for a in aliases):
                    committed = True
                gd = rec.get("graph_delta") or {}
                if "target_candidate_detected" in (gd.get("event_types") or []):
                    candidate = True
                for cap in rec.get("new_node_captions") or []:
                    if cap not in captions_seen:
                        captions_seen.add(cap)
                        s = score_caption_against_goal(cap, goal_text)["score"]
                        best_caption = max(best_caption, s)
                node_count_max = max(node_count_max, int(rec.get("graph_node_count", 0) or 0))
                pose = rec.get("pose_before")
                if isinstance(pose, dict) and "x" in pose and "y" in pose:
                    poses.append((float(pose["x"]), float(pose["y"])))
    return {
        "committed": committed, "candidate": candidate,
        "best_caption": round(best_caption, 3), "poses": poses,
        "node_count_max": node_count_max, "unique_captions": len(captions_seen),
    }


def classify(rec):
    """Deepest measured stage reached (mutually exclusive)."""
    if rec["min_dist"] is not None and rec["min_dist"] <= 1.0:
        return "approached_1m"      # within success radius yet scored failure
    if rec["min_dist"] is not None and rec["min_dist"] <= 2.0:
        return "approached_2m"      # got close, last-meter/pose miss
    if rec["committed"]:
        return "committed_far"      # believed it found target, never got near
    if rec["candidate"] or rec["best_caption"] >= 0.75:
        return "candidate_no_commit"  # target recognized, never committed
    return "no_candidate"           # target never recognized at all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--transform", default="x=+fwd,y=-right",
                    choices=sorted(CANDIDATE_TRANSFORMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    get_config, make_dataset, OmegaConf = load_habitat_modules()
    cfg = get_config(config_path=os.path.abspath("base_UniGoal/configs/tasks/instance_imagenav.yaml"))
    OmegaConf.set_readonly(cfg, False)
    cfg.habitat.dataset.split = "val"
    OmegaConf.set_readonly(cfg, True)
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    episodes = {idx: ep for idx, ep in enumerate(dataset.episodes)}
    tf = CANDIDATE_TRANSFORMS[args.transform]

    rows = []
    for root in args.roots:
        for f in glob.glob(os.path.join(root, "**", "episode_results.json"), recursive=True):
            run = os.path.dirname(f)
            try:
                e = json.load(open(f))[0]
            except Exception:
                continue
            ep = int(e.get("habitat_episode_no", -1))
            if ep not in episodes:
                continue
            # prefer the mission text logged in the trace (full description);
            # fall back to the dataset object_category
            goal_text = _mission_text(run) or str(
                getattr(episodes[ep], "object_category", "") or ""
            )
            w = walk_trace(run, goal_text)
            offsets = episodic_goal_offsets(episodes[ep])
            goals = [(MAP_CENTER_M + tf(r, fwd)[0], MAP_CENTER_M + tf(r, fwd)[1]) for r, fwd in offsets]
            dmin, dfin = distances(w["poses"], goals) if w["poses"] else (None, None)
            row = {
                "episode": ep, "success": bool(e.get("success")),
                "committed": w["committed"], "candidate": w["candidate"],
                "best_caption": w["best_caption"], "nodes": w["node_count_max"],
                "min_dist": round(dmin, 2) if dmin is not None else None,
                "run": run,
            }
            row["stage"] = "success" if row["success"] else classify(row)
            rows.append(row)

    rows.sort(key=lambda r: (r["episode"], r["run"]))
    print(f"{'ep':>5} {'succ':>4} {'commit':>6} {'cand':>4} {'capRel':>6} {'nodes':>5} {'minD':>6}  stage")
    for r in rows:
        print(f"{r['episode']:>5} {str(r['success'])[0]:>4} {str(r['committed'])[0]:>6} "
              f"{str(r['candidate'])[0]:>4} {r['best_caption']:>6.2f} {r['nodes']:>5} "
              f"{('%.2f'%r['min_dist']) if r['min_dist'] is not None else '  -  ':>6}  {r['stage']}")

    fails = [r for r in rows if not r["success"]]
    from collections import Counter
    order = ["approached_1m", "approached_2m", "committed_far", "candidate_no_commit", "no_candidate"]
    hist = Counter(r["stage"] for r in fails)
    print(f"\n== {args.label}: {len(rows)} episodes, {sum(r['success'] for r in rows)} success, "
          f"{len(fails)} failures ==")
    print("where failures die (deepest measured stage):")
    for s in order:
        print(f"  {s:>20}: {hist.get(s,0)}")
    print(f"committed-ever among failures: {sum(r['committed'] for r in fails)}/{len(fails)}")
    print(f"candidate-ever among failures: {sum(r['candidate'] for r in fails)}/{len(fails)}")
    print(f"caption>=0.75 among failures:  {sum(1 for r in fails if r['best_caption']>=0.75)}/{len(fails)}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    return 0


def _mission_text(run_dir):
    for tf in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        with open(tf) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ts = rec.get("task_spec") or {}
                if ts.get("primary_goal"):
                    return str(ts["primary_goal"])
                ms = rec.get("mission_state_summary") or {}
                if ms.get("mission_text"):
                    return str(ms["mission_text"])
        break
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
