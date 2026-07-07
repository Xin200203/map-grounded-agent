#!/usr/bin/env python3
"""Haiku vs Sonnet verification-VLM probe on ground-truth goal viewpoints.

For each requested episode:
  - reset the Habitat env, grab the START frame (likely-negative view)
  - teleport-render the episode's first goal view_point (guaranteed-positive)
  - ask each candidate VLM, per frame: "is the described target visible?"
Scores each model against ground truth (goal frame = positive, start frame
treated as negative unless the goal is visible from spawn, which we accept as
label noise). Saves frames + verdicts under analysis/vlm_verification_probe/.

Run inside the experiment conda env on the server:
  python scripts/vlm_verification_probe.py --episodes 228 527 661 358 574 955 \
      --models claude-haiku-4-5 claude-sonnet-4-5
"""

import argparse
import json
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base_UniGoal"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from audit_hm3d_scene_loadability import load_habitat_modules  # noqa: E402


PROMPT = (
    "You are verifying a navigation target. Target description: \"{goal}\".\n"
    "Look at the image. Is the described target object itself clearly visible?\n"
    "Answer JSON only: {{\"visible\": true/false, \"confidence\": 0.0-1.0, "
    "\"reason\": \"<max 15 words>\"}}"
)


def goal_text_for(episode, attribute_data):
    key = getattr(episode, "goal_key", None) or ""
    for item in attribute_data or []:
        if item.get("goal_key") == key:
            attrs = item.get("attributes") or {}
            return (
                str(attrs.get("intrinsic_attributes", "")) + " "
                + str(attrs.get("extrinsic_attributes", ""))
            ).strip()
    return str(getattr(episode, "object_category", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--config", default="base_UniGoal/configs/tasks/instance_imagenav.yaml")
    parser.add_argument("--out", default="analysis/vlm_verification_probe")
    args = parser.parse_args()

    get_config, make_dataset, OmegaConf = load_habitat_modules()
    import habitat
    import gzip
    import numpy as np
    from PIL import Image

    from src.utils.llm import VLM

    attribute_data = []
    try:
        with gzip.open("data/datasets/textnav/val/val_text.json.gz") as fh:
            attribute_data = json.load(fh).get("attribute_data", [])
    except Exception:
        pass

    cfg = get_config(config_path=os.path.abspath(args.config))
    OmegaConf.set_readonly(cfg, False)
    cfg.habitat.dataset.split = "val"
    OmegaConf.set_readonly(cfg, True)
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    wanted = {int(e) for e in args.episodes}
    picked = {idx: ep for idx, ep in enumerate(dataset.episodes) if idx in wanted}
    os.makedirs(args.out, exist_ok=True)

    vlms = {
        model: VLM(
            os.environ["VAPEUR_BASE_URL"], os.environ["VAPEUR_API_KEY"], model,
            api_provider="anthropic", api_protocol="anthropic-messages",
        )
        for model in args.models
    }

    def ask(model, image, goal):
        raw = vlms[model](PROMPT.format(goal=goal[:400]), image)
        try:
            payload = json.loads(re.search(r"\{.*\}", str(raw), re.S).group())
            return bool(payload.get("visible")), float(payload.get("confidence", 0)), str(payload.get("reason", ""))[:60]
        except Exception:
            return None, 0.0, str(raw)[:60]

    rows = []
    for idx in sorted(picked):
        episode = picked[idx]
        subset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        subset.episodes = [episode]
        env = habitat.Env(config=cfg, dataset=subset)
        obs = env.reset()
        sim = env.sim
        goal_text = goal_text_for(episode, attribute_data)
        frames = {"start_neg": obs["rgb"]}
        try:
            vp = episode.goals[0].view_points[0].agent_state
            goal_obs = sim.get_observations_at(vp.position, vp.rotation, keep_agent_at_new_pose=False)
            frames["goal_pos"] = goal_obs["rgb"]
        except Exception as exc:
            print(f"ep{idx}: no goal viewpoint ({exc})")
        for tag, rgb in frames.items():
            img = Image.fromarray(np.asarray(rgb)[:, :, :3].astype("uint8"))
            img.save(os.path.join(args.out, f"ep{idx}_{tag}.png"))
            truth = tag.endswith("_pos")
            for model in args.models:
                visible, conf, reason = ask(model, img, goal_text)
                rows.append({
                    "episode": idx, "frame": tag, "truth": truth, "model": model,
                    "visible": visible, "confidence": conf, "reason": reason,
                })
                print(f"ep{idx} {tag:>9} truth={truth} {model}: visible={visible} conf={conf:.2f} {reason}")
        env.close()

    with open(os.path.join(args.out, "verdicts.json"), "w") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)
    print("\n== accuracy ==")
    for model in args.models:
        scored = [r for r in rows if r["model"] == model and r["visible"] is not None]
        correct = sum(1 for r in scored if r["visible"] == r["truth"])
        pos = [r for r in scored if r["truth"]]
        pos_hit = sum(1 for r in pos if r["visible"])
        print(f"{model}: acc={correct}/{len(scored)}  goal-frame recall={pos_hit}/{len(pos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
