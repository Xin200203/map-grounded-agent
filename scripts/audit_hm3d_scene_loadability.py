#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


PROBE_CODE = r"""
import json
import sys
from habitat.config.default import get_config
from habitat import make_dataset, Env
from omegaconf import OmegaConf

episode_idx = int(sys.argv[1])
scene_id = sys.argv[2]
config_path = sys.argv[3]
split = sys.argv[4]

cfg = get_config(config_path=config_path)
OmegaConf.set_readonly(cfg, False)
cfg.habitat.dataset.split = split
cfg.habitat.simulator.scene = scene_id
subset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
subset.episodes = [subset.episodes[episode_idx]]
env = Env(cfg, subset)
env.reset()
env.close()
print(json.dumps({"status": "ok"}))
"""


def load_habitat_modules():
    try:
        from habitat.config.default import get_config
        from habitat import make_dataset
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Habitat dependencies. Run this script inside the experiment "
            "environment, for example `conda activate unigoal` on the server."
        ) from exc
    return get_config, make_dataset, OmegaConf


def load_scene_entries(config_path: str, split: str):
    get_config, make_dataset, OmegaConf = load_habitat_modules()
    cfg = get_config(config_path=config_path)
    OmegaConf.set_readonly(cfg, False)
    cfg.habitat.dataset.split = split
    OmegaConf.set_readonly(cfg, True)
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    by_scene = defaultdict(list)
    for idx, episode in enumerate(dataset.episodes):
        by_scene[episode.scene_id].append(
            {
                "episode_idx": idx,
                "episode_id": getattr(episode, "episode_id", None),
                "object_category": getattr(episode, "object_category", None),
                "goal_key": getattr(episode, "goal_key", None),
            }
        )
    return by_scene


def probe_scene(python_bin: str, config_path: str, split: str, scene_id: str, episode_idx: int, timeout_s: int):
    try:
        proc = subprocess.run(
            [python_bin, "-c", PROBE_CODE, str(episode_idx), scene_id, config_path, split],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "timeout"}

    output = (proc.stdout or "").strip()
    if proc.returncode == 0 and output:
        try:
            payload = json.loads(output.splitlines()[-1])
        except json.JSONDecodeError:
            payload = {"status": "ok", "message": output.splitlines()[-1][:240]}
        return payload

    combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
    first_line = combined.splitlines()[0][:240] if combined else ""
    return {"status": "fail", "message": first_line}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config",
        default="configs/tasks/instance_imagenav.yaml",
        help="Habitat task config path relative to base_UniGoal.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--max-scenes", type=int, default=36)
    parser.add_argument("--known-ok-scene", action="append", default=[])
    parser.add_argument(
        "--output",
        default="results/phase2_revalidation/hm3d_scene_loadability_audit.json",
    )
    args = parser.parse_args()

    by_scene = load_scene_entries(args.task_config, args.split)
    scene_entries = sorted(by_scene.items(), key=lambda item: len(item[1]), reverse=True)
    report = {
        "task_config": args.task_config,
        "split": args.split,
        "max_scenes": args.max_scenes,
        "known_ok_scene": args.known_ok_scene,
        "total_scenes": len(scene_entries),
        "results": [],
    }

    python_bin = sys.executable
    checked = 0
    for scene_id, entries in scene_entries:
        if any(token in scene_id for token in args.known_ok_scene):
            report["results"].append(
                {
                    "scene_id": scene_id,
                    "status": "known_ok_skip",
                    "episode_idx": entries[0]["episode_idx"],
                    "episode_id": entries[0]["episode_id"],
                    "object_category": entries[0]["object_category"],
                    "goal_key": entries[0]["goal_key"],
                }
            )
            continue
        probe_entry = entries[0]
        result = probe_scene(
            python_bin=python_bin,
            config_path=args.task_config,
            split=args.split,
            scene_id=scene_id,
            episode_idx=probe_entry["episode_idx"],
            timeout_s=args.timeout_s,
        )
        report["results"].append(
            {
                "scene_id": scene_id,
                "episode_idx": probe_entry["episode_idx"],
                "episode_id": probe_entry["episode_id"],
                "object_category": probe_entry["object_category"],
                "goal_key": probe_entry["goal_key"],
                **result,
            }
        )
        checked += 1
        if checked >= args.max_scenes:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote HM3D loadability audit to {out_path}")


if __name__ == "__main__":
    main()
