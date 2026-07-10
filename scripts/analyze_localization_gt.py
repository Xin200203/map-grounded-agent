#!/usr/bin/env python3
"""GT-grounded target-localization diagnosis from per-episode graph-node dumps.

Splits the dominant failure bucket (recognized-and-committed but agent never
reached GT) into its two mechanisms, measured against GT:

  node_near_GT   a graph node whose caption matches the target was mapped
                 within R meters of the true goal  -> target localized right,
                 the agent just never navigated there (navigation / reach)
  no_node_near   no target-matching node within R of GT, though the episode
                 committed -> the committed target was localized elsewhere
                 (wrong instance / coordinate error) or never mapped at GT

Node centers are stored in flipped map-cell coordinates (graph.py set_center:
[int(x_m*20), map_size-1-int(y_m*20)]). We recover metric coordinates and
CALIBRATE the episodic->map axis on success episodes (a success almost always
has a target node mapped at the goal), trying flip/no-flip x the 8 axis
permutations and picking the combo that puts a target node nearest GT on the
most successes. No axis convention is assumed.

  python scripts/analyze_localization_gt.py --roots results/c45redump_probe12_20260710 \
      results/c45redump_exp36_20260710 --label c45 --radius 1.5
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base_UniGoal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_goal_distance import CANDIDATE_TRANSFORMS, MAP_CENTER_M, episodic_goal_offsets  # noqa: E402
from audit_hm3d_scene_loadability import load_habitat_modules  # noqa: E402
from smoothnav.target_matching import score_caption_against_goal  # noqa: E402

MAP_SIZE = 720
CELLS_PER_M = 20.0


def node_metric(center, flip_y):
    """Recover (x_m, y_m) from a stored node center [x_cell, y_cell]."""
    x_cell, y_cell = float(center[0]), float(center[1])
    x_m = x_cell / CELLS_PER_M
    y_m = ((MAP_SIZE - 1 - y_cell) if flip_y else y_cell) / CELLS_PER_M
    return x_m, y_m


def load_dump(run_dir):
    for f in glob.glob(os.path.join(run_dir, "graph_nodes", "*.json")):
        try:
            return json.load(open(f))
        except Exception:
            return None
    return None


def mission_text(run_dir):
    for tf in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        for line in open(tf):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("task_spec") or {}
            if ts.get("primary_goal"):
                return str(ts["primary_goal"])
        break
    return ""


def gather(roots):
    recs = []
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "episode_results.json"), recursive=True):
            run = os.path.dirname(f)
            dump = load_dump(run)
            if dump is None:
                continue
            try:
                e = json.load(open(f))[0]
            except Exception:
                continue
            recs.append({
                "episode": int(e.get("habitat_episode_no", -1)),
                "success": bool(e.get("success")),
                "nodes": dump.get("nodes", []),
                "goal_text": mission_text(run),
                "run": run,
            })
    return recs


def best_target_node_dist(nodes, goal_text, gx, gy, flip_y, min_rel=0.75):
    """Distance to the nearest target-matching node (caption rel >= min_rel);
    falls back to nearest node of any caption if none match."""
    best_match = best_any = None
    for nd in nodes:
        c = nd.get("center")
        if not c or len(c) < 2:
            continue
        x_m, y_m = node_metric(c, flip_y)
        d = ((x_m - gx) ** 2 + (y_m - gy) ** 2) ** 0.5
        best_any = d if best_any is None else min(best_any, d)
        rel = score_caption_against_goal(nd.get("caption", ""), goal_text)["score"]
        if rel >= min_rel:
            best_match = d if best_match is None else min(best_match, d)
    return best_match, best_any


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    get_config, make_dataset, OmegaConf = load_habitat_modules()
    cfg = get_config(config_path=os.path.abspath("base_UniGoal/configs/tasks/instance_imagenav.yaml"))
    OmegaConf.set_readonly(cfg, False)
    cfg.habitat.dataset.split = "val"
    OmegaConf.set_readonly(cfg, True)
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    episodes = {idx: ep for idx, ep in enumerate(dataset.episodes)}

    recs = gather(args.roots)
    for r in recs:
        if r["episode"] in episodes:
            r["offsets"] = episodic_goal_offsets(episodes[r["episode"]])
        if not r["goal_text"]:
            r["goal_text"] = str(getattr(episodes.get(r["episode"]), "object_category", "") or "")

    # calibrate (flip_y, transform) on successes: maximize successes with a
    # target-matching node within radius of GT.
    successes = [r for r in recs if r["success"] and r.get("offsets")]
    best = None
    for flip_y in (True, False):
        for tname, tf in CANDIDATE_TRANSFORMS.items():
            hit = 0
            for r in successes:
                goals = [(MAP_CENTER_M + tf(rr, ff)[0], MAP_CENTER_M + tf(rr, ff)[1])
                         for rr, ff in r["offsets"]]
                ok = False
                for gx, gy in goals:
                    m, a = best_target_node_dist(r["nodes"], r["goal_text"], gx, gy, flip_y)
                    d = m if m is not None else a
                    if d is not None and d <= args.radius:
                        ok = True
                        break
                hit += 1 if ok else 0
            if best is None or hit > best[0]:
                best = (hit, flip_y, tname, tf)
    hit, flip_y, tname, tf = best
    print(f"calibration: flip_y={flip_y} transform={tname} "
          f"(successes with target node within {args.radius}m of GT: {hit}/{len(successes)})")

    rows = []
    for r in recs:
        if not r.get("offsets"):
            continue
        goals = [(MAP_CENTER_M + tf(rr, ff)[0], MAP_CENTER_M + tf(rr, ff)[1]) for rr, ff in r["offsets"]]
        bm = ba = None
        for gx, gy in goals:
            m, a = best_target_node_dist(r["nodes"], r["goal_text"], gx, gy, flip_y)
            if m is not None:
                bm = m if bm is None else min(bm, m)
            if a is not None:
                ba = a if ba is None else min(ba, a)
        rows.append({
            "episode": r["episode"], "success": r["success"], "n_nodes": len(r["nodes"]),
            "target_node_gt_dist": round(bm, 2) if bm is not None else None,
            "any_node_gt_dist": round(ba, 2) if ba is not None else None,
        })

    rows.sort(key=lambda x: (x["success"], x["episode"]))
    print(f"\n{'ep':>5} {'succ':>4} {'nodes':>5} {'tgtNodeD':>8} {'anyNodeD':>8}")
    for r in rows:
        t = ('%.2f' % r['target_node_gt_dist']) if r['target_node_gt_dist'] is not None else 'none'
        a = ('%.2f' % r['any_node_gt_dist']) if r['any_node_gt_dist'] is not None else 'none'
        print(f"{r['episode']:>5} {str(r['success'])[0]:>4} {r['n_nodes']:>5} {t:>8} {a:>8}")

    fails = [r for r in rows if not r["success"]]
    R = args.radius
    localized = [r for r in fails if r["target_node_gt_dist"] is not None and r["target_node_gt_dist"] <= R]
    mapped_not_reco = [r for r in fails if r["target_node_gt_dist"] is None
                       and r["any_node_gt_dist"] is not None and r["any_node_gt_dist"] <= R]
    no_node = [r for r in fails if (r["any_node_gt_dist"] is None or r["any_node_gt_dist"] > R)]
    print(f"\n== {args.label}: {len(fails)} failures, GT localization split (R={R}m) ==")
    print(f"  target node mapped near GT (recognized, agent didn't reach -> NAV): {len(localized)}")
    print(f"  some node near GT but not target-recognized (reco gap):             {len(mapped_not_reco)}")
    print(f"  no node near GT (localized elsewhere / never mapped at GT):         {len(no_node)}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
