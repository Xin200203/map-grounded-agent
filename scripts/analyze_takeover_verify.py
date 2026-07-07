#!/usr/bin/env python3
"""Tabulate C7' takeover-verification activity per episode.

For each episode run under the given results roots, reads step traces and
reports: verify verdict sequence, takeover adoptions, verified suppressions,
and outcome. Run on the server:

  python scripts/analyze_takeover_verify.py results/c7v_probe12_20260708 \
      results/c7v_exp36_20260708
"""

import glob
import json
import os
import sys
from collections import Counter


def scan_run(run_dir):
    verdicts = []
    sources = Counter()
    trace_files = sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl")))
    for tf in trace_files:
        prev = None
        with open(tf) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                v = rec.get("takeover_verify_verdict")
                src = rec.get("executor_adopted_goal_source")
                if src:
                    sources[src] += 1
                # verdict field repeats while cached; count transitions/new steps
                if v and (prev is None or v != prev.get("v") or
                          int(rec.get("step_idx", rec.get("timestep", 0) or 0))
                          - prev["step"] >= 10):
                    verdicts.append(
                        (int(rec.get("step_idx", rec.get("timestep", 0) or 0)), v)
                    )
                    prev = {"v": v, "step": int(rec.get("step_idx", rec.get("timestep", 0) or 0))}
    episode_file = os.path.join(run_dir, "episode_results.json")
    success = None
    episode_id = None
    if os.path.exists(episode_file):
        try:
            eps = json.load(open(episode_file))
            if eps:
                success = bool(eps[0].get("success"))
                episode_id = eps[0].get(
                    "habitat_episode_no", eps[0].get("episode_id")
                )
        except Exception:
            pass
    return {
        "run": run_dir,
        "episode_id": episode_id,
        "success": success,
        "verdicts": verdicts,
        "takeover_steps": sources.get("visible_target_temp_goal", 0),
        "suppressed_steps": sources.get("exp_goal_text_visible_suppressed", 0),
        "lock_steps": sources.get("visible_target_global_goal", 0),
    }


def main():
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        return 1
    runs = []
    for root in roots:
        for summary in glob.glob(os.path.join(root, "**", "episode_results.json"),
                                 recursive=True):
            runs.append(scan_run(os.path.dirname(summary)))

    total = Counter()
    print(f"{'episode':>8} {'success':>7} {'verify(y/n/u)':>14} {'takeover':>8} "
          f"{'suppressed':>10} {'lock':>5}  first_verdicts")
    for r in sorted(runs, key=lambda x: (x["episode_id"] is None, x["episode_id"])):
        vc = Counter(v for _, v in r["verdicts"])
        total.update(vc)
        total["takeover"] += r["takeover_steps"]
        total["suppressed"] += r["suppressed_steps"]
        total["episodes"] += 1
        total["success"] += 1 if r["success"] else 0
        head = ",".join(f"{s}:{v}" for s, v in r["verdicts"][:4])
        print(f"{str(r['episode_id']):>8} {str(r['success']):>7} "
              f"{vc.get('yes', 0)}/{vc.get('no', 0)}/{vc.get('unsure', 0):>2}"
              f"{'':>6} {r['takeover_steps']:>8} {r['suppressed_steps']:>10} "
              f"{r['lock_steps']:>5}  {head}")
    print(f"\n== aggregate over {total['episodes']} episodes ==")
    print(f"success: {total['success']}  verdicts yes/no/unsure: "
          f"{total.get('yes', 0)}/{total.get('no', 0)}/{total.get('unsure', 0)}")
    print(f"takeover steps: {total['takeover']}  suppressed steps: {total['suppressed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
