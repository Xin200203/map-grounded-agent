"""Run-level IO helpers for isolated result directories and manifests."""

import json
import os
import shlex
import socket
import subprocess
import uuid
from datetime import datetime
from pathlib import Path


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _redact_secrets(data):
    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            lowered = str(key).lower()
            if lowered in {"api_key", "auth_token", "token", "secret"}:
                redacted[key] = "<redacted>" if value else ""
            else:
                redacted[key] = _redact_secrets(value)
        return redacted
    if isinstance(data, list):
        return [_redact_secrets(v) for v in data]
    return _json_safe(data)


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_api_config(args):
    api_key_env = getattr(args, "api_key_env", "SMOOTHNAV_API_KEY")
    base_url_env = getattr(args, "base_url_env", "SMOOTHNAV_BASE_URL")

    api_key = os.environ.get(api_key_env, "").strip()
    base_url = os.environ.get(base_url_env, "").strip()

    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {api_key_env} before running SmoothNav."
        )
    if not base_url:
        raise RuntimeError(
            f"Missing base URL. Set environment variable {base_url_env} before running SmoothNav."
        )

    args.api_key = api_key
    args.base_url = base_url
    args.api_key_env = api_key_env
    args.base_url_env = base_url_env
    return args


def get_git_hash(repo_root):
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def setup_run_environment(args, argv, prompt_versions):
    repo_root = get_repo_root()
    results_root = getattr(args, "results_root", "") or "results"
    if not os.path.isabs(results_root):
        results_root = os.path.join(repo_root, results_root)

    now = datetime.now()
    date_dir = now.strftime("%Y%m%d")
    time_slug = now.strftime("%H%M%S")
    run_id = f"{args.mode}_{args.goal_type}_{time_slug}_{uuid.uuid4().hex[:8]}"
    run_dir = os.path.join(results_root, date_dir, run_id)
    log_dir = os.path.join(run_dir, "log")
    visualization_dir = os.path.join(run_dir, "visualization")
    video_dir = os.path.join(visualization_dir, "videos")
    step_trace_dir = os.path.join(run_dir, "step_traces")
    planner_call_dir = os.path.join(run_dir, "planner_calls")
    monitor_call_dir = os.path.join(run_dir, "monitor_calls")

    for path in [
        results_root,
        os.path.join(results_root, date_dir),
        run_dir,
        log_dir,
        visualization_dir,
        video_dir,
        step_trace_dir,
        planner_call_dir,
        monitor_call_dir,
    ]:
        os.makedirs(path, exist_ok=True)

    args.repo_root = repo_root
    args.results_root = results_root
    args.run_id = run_id
    args.run_dir = run_dir
    args.log_dir = log_dir
    args.visualization_dir = visualization_dir
    args.step_trace_dir = step_trace_dir
    args.planner_call_dir = planner_call_dir
    args.monitor_call_dir = monitor_call_dir
    args.eval_log_path = os.path.join(log_dir, "eval.log")
    args.manifest_path = os.path.join(run_dir, "manifest.json")
    args.effective_config_path = os.path.join(run_dir, "effective_config.json")
    args.summary_path = os.path.join(run_dir, "summary.json")
    args.episode_results_path = os.path.join(run_dir, "episode_results.json")
    args.action_analysis_path = os.path.join(run_dir, "action_analysis.json")
    args.experiment_id = run_id

    manifest = {
        "run_id": run_id,
        "created_at_local": now.isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "repo_root": repo_root,
        "results_root": results_root,
        "run_dir": run_dir,
        "command": " ".join(shlex.quote(arg) for arg in argv),
        "config_file": getattr(args, "config_file", ""),
        "mode": getattr(args, "mode", ""),
        "goal_type": getattr(args, "goal_type", ""),
        "num_eval": getattr(args, "num_eval", 0),
        "num_eval_episodes": getattr(args, "num_eval_episodes", 0),
        "models": {
            "planner": getattr(args, "llm_model", ""),
            "monitor": getattr(args, "llm_model_fast", ""),
            "vision": getattr(args, "vlm_model", ""),
        },
        "prompt_schema_versions": _json_safe(prompt_versions),
        "git_hash": get_git_hash(repo_root),
        "api_env": {
            "api_key_env": getattr(args, "api_key_env", ""),
            "base_url_env": getattr(args, "base_url_env", ""),
        },
    }
    _write_json(args.manifest_path, manifest)

    effective_config = _redact_secrets(vars(args))
    _write_json(args.effective_config_path, effective_config)
    return args
