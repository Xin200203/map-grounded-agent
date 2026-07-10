#!/usr/bin/env python3
"""Per-episode closest-approach distance to the GT goal, from step traces.

For each run under the given results roots, reads the agent pose trajectory
(pose_before, metric meters; map cell = metric * 20) from step traces and
the episode's GT goal position(s) from the textnav dataset, and reports the
minimum and final Euclidean ground-plane distance to the nearest goal.

The episodic->map axis convention is CALIBRATED, not assumed: over episodes
with success=1 the final pose must sit within ~1 m of a goal; the axis
permutation/sign combo that satisfies the most successes wins and is then
applied everywhere. Run on the server:

  python scripts/analyze_goal_distance.py --roots results/c45_probe12_20260707 \
      results/c45_exp36_20260707 --label c45
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base_UniGoal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_hm3d_scene_loadability import load_habitat_modules  # noqa: E402

MAP_CENTER_M = 18.0  # 720 cells * 5 cm / 2: episode starts at map center


def episodic_goal_offsets(episode):
    """Goal offsets in the episodic frame as (right, forward) meter pairs."""
    import numpy as np
    import quaternion  # noqa: F401  (numpy-quaternion, dependency of habitat)
    from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector

    start = np.asarray(episode.start_position, dtype=float)
    rot = quaternion_from_coeff(episode.start_rotation)
    offsets = []
    for goal in episode.goals:
        delta = np.asarray(goal.position, dtype=float) - start
        local = quaternion_rotate_vector(rot.inverse(), delta)
        # habitat episodic frame: x right, y up, z backward
        offsets.append((float(local[0]), float(-local[2])))
    return offsets


CANDIDATE_TRANSFORMS = {
    # name: (right, forward) -> (dx, dy) added to map center, in meters
    "x=+fwd,y=+right": lambda r, f: (f, r),
    "x=+fwd,y=-right": lambda r, f: (f, -r),
    "x=-fwd,y=+right": lambda r, f: (-f, r),
    "x=-fwd,y=-right": lambda r, f: (-f, -r),
    "x=+right,y=+fwd": lambda r, f: (r, f),
    "x=+right,y=-fwd": lambda r, f: (r, -f),
    "x=-right,y=+fwd": lambda r, f: (-r, f),
    "x=-right,y=-fwd": lambda r, f: (-r, -f),
}


def load_trajectory(run_dir):
    poses = []
    for tf in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        with open(tf) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                pose = rec.get("pose_before")
                if isinstance(pose, dict) and "x" in pose and "y" in pose:
                    poses.append((float(pose["x"]), float(pose["y"])))
    return poses


def run_record(run_dir):
    f = os.path.join(run_dir, "episode_results.json")
    try:
        e = json.load(open(f))[0]
    except Exception:
        return None
    return {
        "run": run_dir,
        "episode": int(e.get("habitat_episode_no", -1)),
        "success": bool(e.get("success")),
        "poses": load_trajectory(run_dir),
    }


def distances(poses, goal_xy_list):
    best = final = None
    for px, py in poses:
        d = min(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 for gx, gy in goal_xy_list)
        best = d if best is None or d < best else best
    if poses:
        px, py = poses[-1]
        final = min(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 for gx, gy in goal_xy_list)
    return best, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=None, help="write per-episode JSON here")
    args = parser.parse_args()

    get_config, make_dataset, OmegaConf = load_habitat_modules()
    cfg = get_config(config_path=os.path.abspath("base_UniGoal/configs/tasks/instance_imagenav.yaml"))
    OmegaConf.set_readonly(cfg, False)
    cfg.habitat.dataset.split = "val"
    OmegaConf.set_readonly(cfg, True)
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    episodes = {idx: ep for idx, ep in enumerate(dataset.episodes)}

    runs = []
    for root in args.roots:
        for f in glob.glob(os.path.join(root, "**", "episode_results.json"), recursive=True):
            rec = run_record(os.path.dirname(f))
            if rec and rec["episode"] in episodes and rec["poses"]:
                rec["goal_offsets"] = episodic_goal_offsets(episodes[rec["episode"]])
                runs.append(rec)

    # calibrate axis convention on successes
    scores = {}
    for name, tf in CANDIDATE_TRANSFORMS.items():
        hit = tot = 0
        for rec in runs:
            if not rec["success"]:
                continue
            goals = [(MAP_CENTER_M + tf(r, fwd)[0], MAP_CENTER_M + tf(r, fwd)[1])
                     for r, fwd in rec["goal_offsets"]]
            _, final = distances(rec["poses"], goals)
            tot += 1
            hit += 1 if final is not None and final <= 1.5 else 0
        scores[name] = (hit, tot)
    best_name = max(scores, key=lambda k: scores[k][0])
    tf = CANDIDATE_TRANSFORMS[best_name]
    print(f"calibration: {best_name} " +
          " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(scores.items())))

    rows = []
    for rec in runs:
        goals = [(MAP_CENTER_M + tf(r, fwd)[0], MAP_CENTER_M + tf(r, fwd)[1])
                 for r, fwd in rec["goal_offsets"]]
        dmin, dfin = distances(rec["poses"], goals)
        rows.append({
            "episode": rec["episode"], "success": rec["success"],
            "min_dist": round(dmin, 3), "final_dist": round(dfin, 3),
            "run": rec["run"],
        })

    rows.sort(key=lambda x: (x["episode"], x["run"]))
    print(f"\n{'episode':>8} {'success':>7} {'min_d':>7} {'final_d':>8}")
    for r in rows:
        print(f"{r['episode']:>8} {str(r['success']):>7} {r['min_dist']:>7.2f} {r['final_dist']:>8.2f}")

    n = len(rows)
    if n:
        for thr in (1.0, 2.0, 3.0):
            k = sum(1 for r in rows if r["min_dist"] <= thr)
            print(f"min_dist<= {thr:.0f}m: {k}/{n}")
        med = sorted(r["min_dist"] for r in rows)[n // 2]
        print(f"median min_dist: {med:.2f} m   label={args.label}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
