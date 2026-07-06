#!/usr/bin/env python3
"""Paired significance treatment for matched profile suites.

For each comparison profile vs the reference profile over the COMMON episode
set (merged across suite roots):
  - SR: exact McNemar test on discordant pairs (binomial two-sided)
  - SPL: paired bootstrap CI (10k resamples) on the mean per-episode difference

Dependency-free (uses math/random only). Prints one line per comparison,
paper-ready.
"""

import argparse
import math
import random

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_e1_matched import collect_profile_runs  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i)
    p = 2.0 * total * (0.5 ** n)
    return min(1.0, p)


def bootstrap_ci(diffs, iters=10000, seed=7):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return sum(diffs) / n, lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, nargs="+")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    args = parser.parse_args()

    def merged(profile):
        runs = {}
        for root in args.suite_root:
            for ep, rec in collect_profile_runs(root, profile).items():
                runs.setdefault(ep, rec)
        return runs

    ref = merged(args.reference)
    for profile in args.profiles:
        if profile == args.reference:
            continue
        cur = merged(profile)
        common = sorted(set(ref) & set(cur))
        if not common:
            print(f"{profile}: no common episodes with {args.reference}")
            continue
        b = sum(1 for ep in common if cur[ep]["success"] > ref[ep]["success"])
        c = sum(1 for ep in common if cur[ep]["success"] < ref[ep]["success"])
        p_sr = mcnemar_exact(b, c)
        diffs = [cur[ep]["spl"] - ref[ep]["spl"] for ep in common]
        mean_diff, lo, hi = bootstrap_ci(diffs)
        sig_sr = "sig" if p_sr < 0.05 else "n.s."
        sig_spl = "sig" if (lo > 0 or hi < 0) else "n.s."
        sr_cur = sum(cur[ep]["success"] for ep in common) / len(common)
        sr_ref = sum(ref[ep]["success"] for ep in common) / len(common)
        print(
            f"{profile} vs {args.reference} (n={len(common)}): "
            f"SR {sr_cur:.3f} vs {sr_ref:.3f}, discordant +{b}/-{c}, McNemar p={p_sr:.3f} [{sig_sr}] | "
            f"dSPL={mean_diff:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] [{sig_spl}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
