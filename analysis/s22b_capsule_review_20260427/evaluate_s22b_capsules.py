#!/usr/bin/env python3
"""Offline review helper for s22b BEV MLLM capsules."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "raw" / "run"
TARGET_STEPS = [31, 43, 98, 145]
OUT_IMAGES = ROOT / "bev_images"
OUT_IMAGES.mkdir(exist_ok=True)
OUT_INPUTS = ROOT / "planner_inputs"
OUT_INPUTS.mkdir(exist_ok=True)
OUT_ENRICHED_IMAGES = ROOT / "bev_images_enriched"


AGENT_VISUAL_ASSESSMENT = {
    31: (
        "BEV上B2是靠近机器人/主房间内部的小局部frontier；B1是更大的东北侧延伸，"
        "B3是西侧出口。若目标不可见且策略是搜索living-room/TV，B2看起来更像same-room局部补洞，"
        "不是高价值离房间分支。"
    ),
    43: (
        "B1指向东北侧小房间/延伸区，B2指向西侧出口；在文字策略仍为unexplored north时，"
        "B1几何上可解释，但它更像继续检查相邻小空间，而不是明显进入living-room的主通道。"
    ),
    98: (
        "B1是已访问的东北/北侧分支，B2是主房间内部东侧局部，B3是新出现的西侧出口/楔形区域；"
        "MLLM把B1改成B3在BEV上更合理，属于避免回访并尝试新区域。"
    ),
    145: (
        "BEV已经展开到西南/南侧新区域，B3/B4/B5/B6都是同一片新区域附近的south分支；"
        "B4位于较中心的可行动分支，选择本身可解释。但文字策略仍显示unexplored north，"
        "说明策略层没有随拓扑变化重写语义搜索意图。"
    ),
}


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def step_dir(step: int) -> Path:
    suffix = f"episode_000000_step_{step:06d}_"
    matches = sorted((RUN / "task_frame_capsules").glob(suffix + "*"))
    if not matches:
        raise FileNotFoundError(f"missing capsule for step {step}")
    return matches[0]


def compact_branch(branch):
    terms = branch.get("score_terms_aggregate", {}) or {}
    return {
        "id": branch.get("id"),
        "direction": branch.get("direction_from_agent"),
        "centroid": branch.get("centroid"),
        "representative_coord": branch.get("representative_coord"),
        "candidate_points": branch.get("candidate_point_count"),
        "pixel_count": branch.get("pixel_count"),
        "final_score_max": terms.get("final_score_max"),
        "actionability_score_max": terms.get("actionability_score_max"),
        "novelty_score_max": terms.get("novelty_score_max"),
        "distance_mean": terms.get("distance_mean"),
        "repeat_penalty_max": terms.get("repeat_penalty_max"),
    }


def first_lines(text: str, limit: int = 16):
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    return "\n".join(lines[:limit])


def raw_selected_branch(raw_response: str) -> str:
    """Best-effort extraction for malformed MLLM JSON.

    Step 31 contains a selected_branch_id in the raw model text but the JSON is
    invalid, so the contract verdict correctly rejected it.  For manual review
    we still expose the raw model intent separately from the applied verdict.
    """

    match = re.search(r'"selected_branch_id"\s*:\s*"([^"]+)"', raw_response or "")
    return match.group(1) if match else ""


mllm_by_step = {r.get("step_idx"): r for r in read_jsonl(RUN / "mllm_frontier_calls" / "episode_000000.jsonl")}
planner_by_step = {}
for row in read_jsonl(RUN / "planner_calls" / "episode_000000.jsonl"):
    planner_by_step.setdefault(row.get("step_idx"), []).append(row)
steps = read_jsonl(RUN / "step_traces" / "episode_000000.jsonl")
steps_by_idx = {r.get("step_idx"): r for r in steps}
summary = read_json(RUN / "summary.json")
suite = read_json(ROOT / "raw" / "suite_summary.json")

review_rows = []
for idx, step in enumerate(TARGET_STEPS):
    cap = step_dir(step)
    image_src = cap / "bev_annotated_branches.png"
    image_dst = OUT_IMAGES / f"step_{step:06d}_bev_annotated_branches.png"
    if image_src.exists():
        shutil.copy2(image_src, image_dst)
    enriched_image = OUT_ENRICHED_IMAGES / f"step_{step:06d}_bev_enriched.png"

    branches_payload = read_json(cap / "frontier_branches.json")
    branches = branches_payload.get("branches", []) or []
    mllm = mllm_by_step.get(step, {})
    verdict = mllm.get("verdict") or {}
    dry_debug = mllm.get("dry_debug") or {}
    final_debug = mllm.get("final_debug") or {}
    grounding = read_json(cap / "grounding_result.json")
    scoring = read_json(cap / "scoring_input.json")
    world = read_json(cap / "world_state.json")
    contract = read_json(cap / "planner_contract_verdict.json") if (cap / "planner_contract_verdict.json").exists() else {}

    next_step = TARGET_STEPS[idx + 1] if idx + 1 < len(TARGET_STEPS) else (steps[-1].get("step_idx", step) + 1)
    window = [r for r in steps if step < int(r.get("step_idx", -1)) <= next_step]
    window_20 = [r for r in steps if step < int(r.get("step_idx", -1)) <= step + 20]
    new_captions = [c for r in window for c in (r.get("new_node_captions") or [])]
    new_captions_20 = [c for r in window_20 for c in (r.get("new_node_captions") or [])]
    new_rooms = []
    for r in window:
        gd = r.get("graph_delta") or {}
        new_rooms.extend(gd.get("new_rooms") or [])
    event_counts = Counter(e for r in window for e in ((r.get("graph_delta") or {}).get("event_types") or []))
    no_progress_count = sum(1 for r in window if (r.get("graph_delta") or {}).get("no_progress"))
    frontier_reached_count = sum(1 for r in window if (r.get("graph_delta") or {}).get("frontier_reached"))
    stuck_count = sum(1 for r in window if (r.get("graph_delta") or {}).get("stuck"))

    planner_rows = planner_by_step.get(step, [])
    planner_brief = []
    for p in planner_rows:
        planner_brief.append(
            {
                "choice_type": p.get("choice_type"),
                "choice_id": p.get("choice_id"),
                "fallback": p.get("fallback_triggered"),
                "strategy": (p.get("strategy") or {}).get("target_region"),
                "reasoning": (p.get("parsed_result") or {}).get("reasoning"),
                "prompt_excerpt": first_lines(p.get("raw_prompt", ""), 22),
            }
        )

    prompt_text = mllm.get("prompt") or ""
    raw_response = mllm.get("raw_response") or ""
    prompt_path = OUT_INPUTS / f"step_{step:06d}_mllm_prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    call_summary_path = OUT_INPUTS / f"step_{step:06d}_mllm_call_summary.json"
    call_summary_path.write_text(
        json.dumps(
            {
                "step_idx": step,
                "status": mllm.get("status"),
                "applied": mllm.get("applied"),
                "allowed_branch_ids": mllm.get("allowed_branch_ids"),
                "strategy": mllm.get("strategy") or {},
                "dry_debug": dry_debug,
                "final_debug": final_debug,
                "verdict": verdict,
                "raw_selected_branch_id": raw_selected_branch(raw_response),
                "raw_response": raw_response,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # A conservative automatic classification; BEV visual reasonableness remains human-reviewed.
    semantic_gain = bool(new_captions or new_rooms)
    if semantic_gain:
        auto_class = "new_semantic_evidence"
    elif frontier_reached_count > 0 and no_progress_count < max(3, len(window) // 2):
        auto_class = "new_area_no_semantic_gain"
    elif stuck_count or no_progress_count >= max(3, len(window) // 2):
        auto_class = "same_room_or_dead_end_revisit"
    else:
        auto_class = "unclear_no_semantic_gain"

    review_rows.append(
        {
            "step": step,
            "capsule_dir": str(cap.relative_to(ROOT)),
            "bev_image": str(image_dst.relative_to(ROOT)),
            "bev_image_enriched": (
                str(enriched_image.relative_to(ROOT)) if enriched_image.exists() else ""
            ),
            "trigger": (read_json(cap / "manifest.json")).get("trigger"),
            "strategy": mllm.get("strategy") or {},
            "dry_branch": dry_debug.get("selected_frontier_branch_id"),
            "dry_frontier": dry_debug.get("selected_frontier"),
            "mllm_status": mllm.get("status"),
            "mllm_applied": mllm.get("applied"),
            "mllm_branch": verdict.get("selected_branch_id"),
            "mllm_raw_selected_branch": raw_selected_branch(raw_response),
            "mllm_ranked": verdict.get("ranked_branch_ids"),
            "mllm_raw_response": raw_response,
            "final_branch": final_debug.get("selected_frontier_branch_id"),
            "final_frontier": final_debug.get("selected_frontier") or grounding.get("selected_frontier"),
            "planner_prior_weight": final_debug.get("planner_prior_weight"),
            "allowed_branch_ids": mllm.get("allowed_branch_ids"),
            "branch_changed_by_mllm": dry_debug.get("selected_frontier_branch_id") != final_debug.get("selected_frontier_branch_id"),
            "branches": [compact_branch(b) for b in branches],
            "topk_frontiers": final_debug.get("topk_frontiers") or grounding.get("topk_frontier_scores"),
            "world_state": {
                "frontier_count": world.get("frontier_count"),
                "room_summary": world.get("room_summary"),
                "object_summary": world.get("object_summary"),
                "graph_delta": world.get("graph_delta"),
            },
            "planner_calls_at_step": planner_brief,
            "mllm_prompt_excerpt": first_lines(mllm.get("prompt", ""), 44),
            "planner_input_files": {
                "mllm_prompt": str(prompt_path.relative_to(ROOT)),
                "mllm_call_summary": str(call_summary_path.relative_to(ROOT)),
            },
            "subsequent_window": {
                "from_exclusive": step,
                "to_inclusive": next_step,
                "step_count": len(window),
                "new_node_captions": new_captions,
                "new_node_captions_first_20_steps": new_captions_20,
                "new_rooms": new_rooms,
                "event_counts": dict(event_counts),
                "frontier_reached_count": frontier_reached_count,
                "no_progress_count": no_progress_count,
                "stuck_count": stuck_count,
                "auto_classification": auto_class,
            },
            "agent_bev_reasonableness": AGENT_VISUAL_ASSESSMENT.get(step, ""),
        }
    )

report = {
    "schema_version": "smoothnav.s22b_capsule_branch_review.v1",
    "source_run": "results/phase2_revalidation/s22b_bev_mllm_union_ep228_20260427/smoothnav-full/20260427/smoothnav_text_194639_3b40f51d",
    "suite_summary": suite,
    "run_summary_subset": {
        k: summary.get(k)
        for k in [
            "SR",
            "SPL",
            "terminal_outcome_counts",
            "avg_high_level_calls",
            "grounding_attempt_count",
            "executor_override_ratio",
        ]
    },
    "target_steps": TARGET_STEPS,
    "rows": review_rows,
}
(ROOT / "capsule_branch_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

lines = []
lines.append("# s22b BEV Frontier Capsule Review\n")
lines.append("\n## Local artifacts\n")
lines.append("- raw run copy: `analysis/s22b_capsule_review_20260427/raw/run/`\n")
lines.append("- BEV images: `analysis/s22b_capsule_review_20260427/bev_images/`\n")
lines.append("- enriched/cropped BEV images: `analysis/s22b_capsule_review_20260427/bev_images_enriched/`\n")
lines.append("- per-step MLLM prompt/call summaries: `analysis/s22b_capsule_review_20260427/planner_inputs/`\n")
lines.append("- machine-readable review: `analysis/s22b_capsule_review_20260427/capsule_branch_review.json`\n")
lines.append("\n## Summary table\n")
lines.append("| step | dry | MLLM parsed | MLLM raw | final | changed | subsequent evidence | auto class | BEV image | enriched BEV |\n")
lines.append("| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
for r in review_rows:
    ev = r["subsequent_window"]
    evidence = []
    if ev["new_node_captions"]:
        evidence.append("new objects=" + ", ".join(ev["new_node_captions"]))
    if ev["new_rooms"]:
        evidence.append("new rooms=" + ", ".join(ev["new_rooms"]))
    if not evidence:
        evidence.append("no new semantic evidence")
    lines.append(
        f"| {r['step']} | {r['dry_branch']} | {r['mllm_branch']} | {r['mllm_raw_selected_branch']} | {r['final_branch']} | "
        f"{r['branch_changed_by_mllm']} | {'; '.join(evidence)} | "
        f"{ev['auto_classification']} | `{r['bev_image']}` | `{r['bev_image_enriched']}` |\n"
    )
lines.append("\n## BEV image information gap\n")
lines.append("- Observation: the original saved BEV images mostly contain gray unknown space plus a small free-space patch, robot marker, and branch dots. They do **not** show semantic object positions, room labels, trajectory/history, branch masks, or explicit dead-end/revisit state.\n")
lines.append("- Observation: this means the current online MLLM image alone is under-informative. The MLLM is effectively relying on the text branch table and stale text strategy more than on map perception.\n")
lines.append("- Mitigation added in code: `smoothnav/frontier_branching.py` now crops BEV images to the known map/branch content and overlays candidate samples, branch direction, candidate count, score, actionability, legend, and compass. The enriched s22b images were regenerated on the remote runtime and copied into `bev_images_enriched/`.\n")
lines.append("- Bug found: the original `agent_coord` in branch replay snapshots used the FMM/traversible vertically-flipped row frame, while frontier coordinates and BEV images used the unflipped full-map frame. That is why `A robot` could be drawn outside the room. `base_UniGoal/src/graph/graph.py` now converts the agent row back to the full-map frame before storing branch snapshots; the enriched offline images were overwritten with corrected robot markers from the map current-location channel.\n")
lines.append("- Remaining gap: even the enriched image is still geometry-only; it does not yet draw object/room/visited/dead-end layers. That should be added before trusting MLLM branch choices online.\n")
lines.append("\n## Agent BEV visual screening\n")
lines.append("> This section is an agent-side visual check of the saved BEV images only. Treat it as a screening note for manual review, not as a final claim.\n")
for r in review_rows:
    lines.append(f"- step {r['step']}: {r['agent_bev_reasonableness']}\n")
lines.append("\n## Per-frame details\n")
for r in review_rows:
    ev = r["subsequent_window"]
    lines.append(f"\n### Step {r['step']}\n")
    lines.append(f"- capsule: `{r['capsule_dir']}`\n")
    lines.append(f"- BEV image: `{r['bev_image']}`\n")
    lines.append(f"- enriched BEV image: `{r['bev_image_enriched']}`\n")
    lines.append(f"- dry branch/frontier: `{r['dry_branch']}` / `{r['dry_frontier']}`\n")
    lines.append(
        f"- MLLM status/applied/parsed branch/raw branch: `{r['mllm_status']}` / `{r['mllm_applied']}` / "
        f"`{r['mllm_branch']}` / `{r['mllm_raw_selected_branch']}`\n"
    )
    lines.append(f"- final branch/frontier: `{r['final_branch']}` / `{r['final_frontier']}`\n")
    lines.append(f"- allowed branches: `{r['allowed_branch_ids']}`\n")
    lines.append(f"- planner input files: `{r['planner_input_files']['mllm_prompt']}`, `{r['planner_input_files']['mllm_call_summary']}`\n")
    lines.append(f"- BEV visual screening: {r['agent_bev_reasonableness']}\n")
    lines.append(f"- subsequent window: steps ({ev['from_exclusive']}, {ev['to_inclusive']}], count={ev['step_count']}\n")
    lines.append(f"- subsequent new captions: `{ev['new_node_captions']}`\n")
    lines.append(f"- subsequent new rooms: `{ev['new_rooms']}`\n")
    lines.append(f"- event counts: `{ev['event_counts']}`\n")
    lines.append(f"- auto classification: `{ev['auto_classification']}`\n")
    lines.append("- branch summaries:\n")
    for b in r["branches"]:
        lines.append(
            f"  - `{b['id']}` dir={b['direction']} rep={b['representative_coord']} "
            f"cand={b['candidate_points']} final_max={b['final_score_max']} "
            f"novelty={b['novelty_score_max']} action={b['actionability_score_max']}\n"
        )
    if r["planner_calls_at_step"]:
        lines.append("- text planner calls at this step:\n")
        for p in r["planner_calls_at_step"]:
            lines.append(f"  - {p['choice_type']} `{p['choice_id']}` -> `{p['strategy']}`; reason={p['reasoning']}\n")
    else:
        lines.append("- text planner calls at this exact step: none; MLLM used current carried strategy.\n")
    lines.append("- MLLM raw response excerpt:\n")
    raw = str(r.get("mllm_raw_response") or "").replace("\n", " ")
    lines.append(f"  `{raw[:500]}`\n")

(ROOT / "capsule_branch_review.md").write_text("".join(lines))
print(ROOT / "capsule_branch_review.md")
print(ROOT / "capsule_branch_review.json")
for r in review_rows:
    print(r["step"], r["bev_image"], r["dry_branch"], r["mllm_branch"], r["final_branch"], r["subsequent_window"]["auto_classification"])
