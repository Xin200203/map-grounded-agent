#!/usr/bin/env python3
"""Per-episode conversion-chain walkthrough (S1..S8).

For one run dir, prints a stage timeline showing where the
detected-target -> committed-anchor -> reached-goal chain breaks:
  S1 first graph caption matching the goal category
  S2 target-candidate events
  S3 planner triggers (reasons) after candidates
  S4 whether the planner menu offered the target caption
  S5 what the planner chose instead
  S6 strategies adopted (target_region transitions) + anchor grounding
  S7 anchor lifecycle (activation, stall, decommit)
  S8 executor visible-target locks
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from smoothnav.target_matching import score_caption_against_goal  # noqa: E402


def walk(run_dir: str, max_lines: int = 60):
    goal_text = ""
    planner_events = []
    for pc in sorted(glob.glob(os.path.join(run_dir, "planner_calls", "*.jsonl"))):
        for line in open(pc):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            goal_text = goal_text or str(rec.get("goal_description") or "")
            menu = str(rec.get("raw_prompt") or "")
            objects_menu = ""
            if "OBJECTS" in menu:
                objects_menu = menu.split("OBJECTS", 1)[1].split("ROOMS", 1)[0]
            planner_events.append({
                "step": rec.get("step_idx"),
                "reason": str(rec.get("escalate_reason") or "")[:44],
                "choice": f"{rec.get('choice_type')}:{rec.get('choice_id')}",
                "menu_objects": [
                    seg.strip().lstrip("[object]").strip()
                    for seg in objects_menu.splitlines()
                    if "[object]" in seg
                ],
                "fallback": rec.get("fallback_triggered"),
            })
    entry = json.load(open(os.path.join(run_dir, "episode_results.json")))[0]
    print(f"== {run_dir.split('/')[-1]}  outcome={entry.get('terminal_outcome','')[:28]} spl={entry.get('spl')}")
    print(f"   goal: {goal_text[:110]}")

    lines = []
    first_cat_step = None
    prev_strategy = None
    prev_anchor_active = False
    for tp in sorted(glob.glob(os.path.join(run_dir, "step_traces", "*.jsonl"))):
        for line in open(tp):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            step = rec.get("step_idx")
            for cap in rec.get("new_node_captions") or []:
                score = score_caption_against_goal(cap, goal_text)["score"]
                if score >= 0.75 and first_cat_step is None:
                    first_cat_step = step
                    lines.append(f"S1 step{step:>4}: target-category caption enters graph: '{cap}' (score {score})")
            delta = rec.get("graph_delta") or {}
            cands = delta.get("target_candidate_captions") or []
            if cands:
                lines.append(f"S2 step{step:>4}: target_candidate fired: {cands[:3]}")
            reasons = rec.get("planner_reasons") or []
            if reasons:
                lines.append(f"S3 step{step:>4}: planner triggered: {reasons}")
            strategy = (rec.get("current_strategy") or {}).get("target_region")
            if strategy != prev_strategy and strategy is not None:
                lines.append(f"S6 step{step:>4}: strategy -> {str(strategy)[:44]}")
                prev_strategy = strategy
            anchor = rec.get("target_anchor_attempt") or {}
            active = bool(anchor.get("active"))
            if active != prev_anchor_active:
                if active:
                    lines.append(f"S7 step{step:>4}: ANCHOR ACTIVE: {anchor.get('target_region','')[:40]}")
                else:
                    lines.append(f"S7 step{step:>4}: anchor released: {anchor.get('decommit_reason','')[:40]}")
                prev_anchor_active = active
            if rec.get("visible_target_override"):
                lines.append(f"S8 step{step:>4}: executor visible-target lock")
    for ev in planner_events:
        target_in_menu = any(
            score_caption_against_goal(m, goal_text)["score"] >= 0.75
            for m in ev["menu_objects"]
        )
        lines.append(
            f"S4/S5 step{str(ev['step']):>4}: reason='{ev['reason']}' menu_has_target={target_in_menu} "
            f"menu={ev['menu_objects'][:4]} choice={ev['choice']} fallback={ev['fallback']}"
        )
    lines.sort(key=lambda item: int("".join(ch for ch in item.split(":", 1)[0] if ch.isdigit()) or 0))
    step_of = lambda s: int("".join(ch for ch in s.split(":", 1)[0] if ch.isdigit()) or 0)
    lines.sort(key=step_of)
    for line_out in lines[:max_lines]:
        print("   " + line_out)
    if len(lines) > max_lines:
        print(f"   ... ({len(lines)-max_lines} more)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--max-lines", type=int, default=60)
    args = parser.parse_args()
    import re
    for run_dir in sorted(glob.glob(os.path.join(args.profile_root, "*", "*_text_*"))):
        try:
            manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
            match = re.search(r"--episode_id\s+(\d+)", str(manifest.get("command", "")))
            ep = int(match.group(1)) if match else -1
        except Exception:
            continue
        if ep in args.episodes:
            walk(run_dir, args.max_lines)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
