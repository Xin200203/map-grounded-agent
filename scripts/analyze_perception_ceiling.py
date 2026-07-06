#!/usr/bin/env python3
"""Failure-attribution pyramid: was the SR ceiling perception-bound?

For every episode run, classify the terminal outcome into:
  success            - episode succeeded
  never_detected     - no target-like candidate ever surfaced by the
                       detector+matcher (graph_delta.target_candidate_captions
                       empty at every step)
  detected_no_anchor - candidates surfaced but the controller never committed
                       a target anchor (target_anchor_attempt.active never true)
                       and the executor never locked a visible target
  anchored_failed    - a target anchor was active at some point, episode failed
  visible_failed     - executor visible-target lock fired, episode still failed
                       (approach/stop/verification failure; strongest evidence
                       against a pure perception explanation)

Buckets are mutually exclusive with priority:
  success > visible_failed > anchored_failed > detected_no_anchor > never_detected
"""

import argparse
import glob
import json
import os


def classify_run(run_dir: str):
    results_path = os.path.join(run_dir, "episode_results.json")
    try:
        entry = json.load(open(results_path))[0]
    except Exception:
        return None
    success = float(entry.get("success") or 0.0) > 0
    detected = False
    anchored = False
    visible = False
    for trace_path in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        for line in open(trace_path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            delta = rec.get("graph_delta") or {}
            if delta.get("target_candidate_captions"):
                detected = True
            anchor = rec.get("target_anchor_attempt") or {}
            if anchor.get("active"):
                anchored = True
            if rec.get("visible_target_override"):
                visible = True
    if success:
        return "success"
    if visible:
        return "visible_failed"
    if anchored:
        return "anchored_failed"
    if detected:
        return "detected_no_anchor"
    return "never_detected"


BUCKETS = [
    "success",
    "visible_failed",
    "anchored_failed",
    "detected_no_anchor",
    "never_detected",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled-roots", nargs="+", required=True,
                        help="label=profile_dir pairs")
    args = parser.parse_args()
    for pair in args.labelled_roots:
        label, _, root = pair.partition("=")
        counts = {bucket: 0 for bucket in BUCKETS}
        runs = 0
        for run_dir in sorted(glob.glob(os.path.join(root, "*", "*_text_*"))):
            bucket = classify_run(run_dir)
            if bucket is None:
                continue
            runs += 1
            counts[bucket] += 1
        failures = runs - counts["success"]
        never_share = counts["never_detected"] / failures if failures else 0.0
        print(
            f"{label:>22} (n={runs}): "
            + " ".join(f"{bucket}={counts[bucket]}" for bucket in BUCKETS)
            + f" | never_detected/failures={never_share:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
