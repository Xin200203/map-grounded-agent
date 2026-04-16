"""Controller profile resolution for Phase 2 ablations."""

from types import SimpleNamespace


CONTROLLER_PROFILE_DEFAULTS = {
    "baseline-explore": {
        "mode": "baseline",
        "controller_enable_monitor": False,
        "controller_monitor_policy": "off",
        "controller_enable_prefetch": False,
        "controller_replan_policy": "baseline_explore",
        "controller_enable_stuck_replan": False,
        "controller_stuck_suppression_steps": 0,
        "controller_direction_reuse_limit": 0,
        "controller_grounding_noop_replan_threshold": 0,
        "controller_same_frontier_reuse_threshold": 0,
    },
    "baseline-periodic": {
        "mode": "smoothnav",
        "controller_enable_monitor": False,
        "controller_monitor_policy": "off",
        "controller_enable_prefetch": False,
        "controller_replan_policy": "fixed_interval",
        "controller_enable_stuck_replan": False,
        "controller_stuck_suppression_steps": 0,
        "controller_direction_reuse_limit": 0,
        "controller_grounding_noop_replan_threshold": 2,
        "controller_same_frontier_reuse_threshold": 2,
    },
    "smoothnav-full": {
        "mode": "smoothnav",
        "controller_enable_monitor": True,
        "controller_monitor_policy": "llm_escalation",
        "controller_enable_prefetch": True,
        "controller_replan_policy": "event",
        "controller_enable_stuck_replan": True,
        "controller_stuck_suppression_steps": 8,
        "controller_direction_reuse_limit": 1,
        "controller_grounding_noop_replan_threshold": 2,
        "controller_same_frontier_reuse_threshold": 2,
    },
    "smoothnav-no-monitor": {
        "mode": "smoothnav",
        "controller_enable_monitor": False,
        "controller_monitor_policy": "off",
        "controller_enable_prefetch": True,
        "controller_replan_policy": "event",
        "controller_enable_stuck_replan": True,
        "controller_stuck_suppression_steps": 8,
        "controller_direction_reuse_limit": 1,
        "controller_grounding_noop_replan_threshold": 2,
        "controller_same_frontier_reuse_threshold": 2,
    },
    "smoothnav-rules-only": {
        "mode": "smoothnav",
        "controller_enable_monitor": True,
        "controller_monitor_policy": "rules",
        "controller_enable_prefetch": True,
        "controller_replan_policy": "event",
        "controller_enable_stuck_replan": True,
        "controller_stuck_suppression_steps": 8,
        "controller_direction_reuse_limit": 1,
        "controller_grounding_noop_replan_threshold": 2,
        "controller_same_frontier_reuse_threshold": 2,
    },
    "smoothnav-no-prefetch": {
        "mode": "smoothnav",
        "controller_enable_monitor": True,
        "controller_monitor_policy": "llm_escalation",
        "controller_enable_prefetch": False,
        "controller_replan_policy": "event",
        "controller_enable_stuck_replan": True,
        "controller_stuck_suppression_steps": 8,
        "controller_direction_reuse_limit": 1,
        "controller_grounding_noop_replan_threshold": 2,
        "controller_same_frontier_reuse_threshold": 2,
    },
    "smoothnav-fixed-interval": {
        "mode": "smoothnav",
        "controller_enable_monitor": False,
        "controller_monitor_policy": "off",
        "controller_enable_prefetch": False,
        "controller_replan_policy": "fixed_interval",
        "controller_enable_stuck_replan": True,
        "controller_stuck_suppression_steps": 8,
        "controller_direction_reuse_limit": 1,
        "controller_grounding_noop_replan_threshold": 2,
        "controller_same_frontier_reuse_threshold": 2,
    },
}


def available_controller_profiles():
    return sorted(CONTROLLER_PROFILE_DEFAULTS)


def infer_controller_profile(mode: str) -> str:
    return "baseline-explore" if mode == "baseline" else "smoothnav-full"


def controller_config_dict(args) -> dict:
    return {
        "profile": getattr(args, "controller_profile", ""),
        "enable_monitor": bool(getattr(args, "controller_enable_monitor", False)),
        "monitor_policy": getattr(args, "controller_monitor_policy", "off"),
        "enable_prefetch": bool(getattr(args, "controller_enable_prefetch", False)),
        "replan_policy": getattr(args, "controller_replan_policy", ""),
        "enable_stuck_replan": bool(
            getattr(args, "controller_enable_stuck_replan", False)
        ),
        "stuck_suppression_steps": int(
            getattr(args, "controller_stuck_suppression_steps", 0) or 0
        ),
        "direction_reuse_limit": int(
            getattr(args, "controller_direction_reuse_limit", 0) or 0
        ),
        "grounding_noop_replan_threshold": int(
            getattr(args, "controller_grounding_noop_replan_threshold", 0) or 0
        ),
        "same_frontier_reuse_threshold": int(
            getattr(args, "controller_same_frontier_reuse_threshold", 0) or 0
        ),
        "fixed_plan_interval_steps": int(
            getattr(args, "controller_fixed_plan_interval_steps", 0) or 0
        ),
        "prefetch_near_threshold": float(
            getattr(args, "controller_prefetch_near_threshold", 0.0) or 0.0
        ),
    }


def resolve_controller_config(args):
    controller_cli_overrides = set(getattr(args, "_controller_cli_overrides", []))
    explicit_profile = getattr(args, "controller_profile", None)
    profile = explicit_profile or infer_controller_profile(getattr(args, "mode", "smoothnav"))
    if profile not in CONTROLLER_PROFILE_DEFAULTS:
        raise ValueError(f"Unknown controller profile: {profile}")

    explicit_values = {}
    for key in [
        "controller_enable_monitor",
        "controller_monitor_policy",
        "controller_enable_prefetch",
        "controller_replan_policy",
        "controller_enable_stuck_replan",
        "controller_stuck_suppression_steps",
        "controller_direction_reuse_limit",
        "controller_grounding_noop_replan_threshold",
        "controller_same_frontier_reuse_threshold",
        "controller_fixed_plan_interval_steps",
        "controller_prefetch_near_threshold",
    ]:
        if "controller_profile" in controller_cli_overrides and key not in controller_cli_overrides:
            explicit_values[key] = None
        else:
            explicit_values[key] = getattr(args, key, None)

    defaults = CONTROLLER_PROFILE_DEFAULTS[profile]
    args.mode = defaults["mode"]
    args.controller_profile = profile
    for key, value in defaults.items():
        if key == "mode":
            continue
        setattr(args, key, value)

    if getattr(args, "controller_fixed_plan_interval_steps", None) is None:
        args.controller_fixed_plan_interval_steps = getattr(args, "num_local_steps", 40)
    if getattr(args, "controller_prefetch_near_threshold", None) is None:
        args.controller_prefetch_near_threshold = 10.0

    for key, value in explicit_values.items():
        if value is not None:
            setattr(args, key, value)

    if args.controller_monitor_policy == "off":
        args.controller_enable_monitor = False
    if not args.controller_enable_monitor:
        args.controller_monitor_policy = "off"

    if args.mode == "baseline":
        args.controller_enable_monitor = False
        args.controller_monitor_policy = "off"
        args.controller_enable_prefetch = False
        args.controller_replan_policy = "baseline_explore"
        args.controller_enable_stuck_replan = False
        args.controller_stuck_suppression_steps = 0
        args.controller_direction_reuse_limit = 0
        args.controller_grounding_noop_replan_threshold = 0
        args.controller_same_frontier_reuse_threshold = 0

    if args.controller_replan_policy not in {"event", "fixed_interval", "baseline_explore"}:
        raise ValueError(
            f"Unsupported replan policy: {args.controller_replan_policy}"
        )
    if args.controller_monitor_policy not in {"llm", "llm_escalation", "rules", "off"}:
        raise ValueError(
            f"Unsupported monitor policy: {args.controller_monitor_policy}"
        )

    return args


def controller_namespace(args):
    """Small helper for tests and debugging."""
    return SimpleNamespace(**controller_config_dict(args))
