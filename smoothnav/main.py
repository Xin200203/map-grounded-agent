"""SmoothNav Main Loop with Phase 0/1 observability and controller structure."""

import argparse
import json
import logging
import os
import sys
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'base_UniGoal'))
from src.agent.unigoal.agent import UniGoal_Agent
from src.envs import construct_envs
from src.graph.graph import Graph
from src.map.bev_mapping import BEV_Map
from src.utils.llm import LLM

from smoothnav.control_metrics import compute_run_control_metrics
from smoothnav.controller_config import (
    available_controller_profiles,
    controller_config_dict,
    resolve_controller_config,
)
from smoothnav.controller_logic import (
    build_graph_delta,
    handle_frontier_reached,
    handle_grounding_failure,
    handle_stuck_replan,
    is_room_target,
    maybe_call_monitor,
    maybe_promote_pending,
    plan_strategy,
    update_grounding_failure_state,
)
from smoothnav.controller_state import ControllerState
from smoothnav.controller_runtime import compact_layered_payload, layered_trace_payload
from smoothnav.budget_context import BudgetGovernor
from smoothnav.evidence_ledger import EvidenceLedger
from smoothnav.executor_adapter import ExecutorAdapter, null_geometric_goal
from smoothnav.experiment_io import resolve_api_config, setup_run_environment
from smoothnav.geometric_grounder import GeometricGrounder
from smoothnav.low_level_agent import (
    DisabledMonitor,
    ESCALATION_ONLY_MONITOR_SCHEMA_VERSION,
    LOW_LEVEL_PROMPT_SCHEMA_VERSION,
    LowLevelAction,
    LowLevelAgent,
    EscalationOnlyMonitor,
    RULE_MONITOR_SCHEMA_VERSION,
    RuleBasedMonitor,
)
from smoothnav.mission_state import MissionProgressManager
from smoothnav.metrics import SmoothnessMetrics
from smoothnav.pending_proposals import PendingProposalManager
from smoothnav.planner import PLANNER_PROMPT_SCHEMA_VERSION, HighLevelPlanner
from smoothnav.tactical_arbiter import TacticalArbiter
from smoothnav.task_belief import TaskBeliefUpdater
from smoothnav.task_spec import parse_task_spec
from smoothnav.terminal_arbitration import TerminalArbiter, TerminalDecision
from smoothnav.tracing import RunTracer, strategy_to_dict
from smoothnav.types import TacticalMode, TerminalOutcome, stage_goal_from_strategy
from smoothnav.world_state import WorldStateBuilder


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="base_UniGoal/configs/config_habitat.yaml",
                        metavar="FILE", type=str)
    parser.add_argument("--goal_type", default="text", type=str)
    parser.add_argument("--episode_id", default=-1, type=int)
    parser.add_argument("--goal", default="", type=str)
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--mode", default="smoothnav", choices=["smoothnav", "baseline"],
                        help="smoothnav: semantic controller family; baseline: frontier-explore UniGoal baseline")
    parser.add_argument("--stuck_threshold", default=15, type=int)
    parser.add_argument("--num_eval", default=0, type=int,
                        help="override num_eval_episodes (0=use config)")
    parser.add_argument("--results-root", default="", type=str,
                        help="optional override for results root directory")
    parser.add_argument(
        "--enable-controller-trace",
        dest="enable_controller_trace",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--disable-controller-trace",
        dest="enable_controller_trace",
        action="store_const",
        const=False,
        default=None,
    )
    parser.add_argument(
        "--api-provider",
        dest="api_provider",
        default=None,
        choices=["anthropic", "openai"],
    )
    parser.add_argument(
        "--api-protocol",
        dest="api_protocol",
        default=None,
        choices=["anthropic-messages", "openai-responses", "openai-chat-completions"],
    )
    parser.add_argument(
        "--controller-profile",
        dest="controller_profile",
        default=None,
        choices=available_controller_profiles(),
    )
    parser.add_argument(
        "--monitor-policy",
        dest="controller_monitor_policy",
        default=None,
        choices=["llm", "llm_escalation", "rules", "off"],
    )
    parser.add_argument(
        "--replan-policy",
        dest="controller_replan_policy",
        default=None,
        choices=["event", "fixed_interval", "baseline_explore"],
    )
    parser.add_argument(
        "--fixed-plan-interval-steps",
        dest="controller_fixed_plan_interval_steps",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--prefetch-near-threshold",
        dest="controller_prefetch_near_threshold",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--enable-monitor",
        dest="controller_enable_monitor",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--disable-monitor",
        dest="controller_enable_monitor",
        action="store_const",
        const=False,
        default=None,
    )
    parser.add_argument(
        "--enable-prefetch",
        dest="controller_enable_prefetch",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--disable-prefetch",
        dest="controller_enable_prefetch",
        action="store_const",
        const=False,
        default=None,
    )
    parser.add_argument(
        "--enable-stuck-replan",
        dest="controller_enable_stuck_replan",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--disable-stuck-replan",
        dest="controller_enable_stuck_replan",
        action="store_const",
        const=False,
        default=None,
    )
    parser.add_argument(
        "--stuck-suppression-steps",
        dest="controller_stuck_suppression_steps",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--direction-reuse-limit",
        dest="controller_direction_reuse_limit",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--grounding-noop-replan-threshold",
        dest="controller_grounding_noop_replan_threshold",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--same-frontier-reuse-threshold",
        dest="controller_same_frontier_reuse_threshold",
        default=None,
        type=int,
    )
    parsed_args = parser.parse_args()
    controller_cli_overrides = {
        key
        for key, value in vars(parsed_args).items()
        if key.startswith("controller_") and value is not None
    }

    with open(parsed_args.config_file, "r") as file:
        config = yaml.safe_load(file)
    args_dict = dict(config)
    nullable_passthrough_keys = {"enable_controller_trace"}
    for key, value in vars(parsed_args).items():
        if key == "results_root":
            if value:
                args_dict[key] = value
            elif key not in args_dict:
                args_dict[key] = value
        elif key in nullable_passthrough_keys and value is None:
            continue
        elif value is not None and key.startswith("controller_"):
            args_dict[key] = value
        elif value is None and key.startswith("controller_"):
            continue
        else:
            args_dict[key] = value
    args = SimpleNamespace(**args_dict)
    args._controller_cli_overrides = sorted(controller_cli_overrides)

    args.is_debugging = sys.gettrace() is not None
    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    args.num_scenes = args.num_processes
    if args.num_eval > 0:
        args.num_eval_episodes = args.num_eval
    args.num_episodes = int(args.num_eval_episodes)

    args = resolve_controller_config(args)
    args = resolve_api_config(args)
    args = setup_run_environment(
        args,
        argv=sys.argv,
        prompt_versions={
            "planner": PLANNER_PROMPT_SCHEMA_VERSION,
            "monitor": (
                RULE_MONITOR_SCHEMA_VERSION
                if args.controller_monitor_policy == "rules"
                else ESCALATION_ONLY_MONITOR_SCHEMA_VERSION
                if args.controller_monitor_policy == "llm_escalation"
                else LOW_LEVEL_PROMPT_SCHEMA_VERSION
            ),
        },
    )
    return args


def _goal_description_from_infos(args, infos):
    if args.goal_type == "text":
        text_goal = infos.get("text_goal", infos.get("goal_name", ""))
        if isinstance(text_goal, dict):
            return (
                text_goal.get("intrinsic_attributes", "") + " "
                + text_goal.get("extrinsic_attributes", "")
            ).strip()
        return str(text_goal)
    if args.goal_type == "ins-image":
        return infos.get("goal_name", "")
    return ""


def _reset_graph_goal(args, graph, infos):
    graph.reset()
    graph.set_obj_goal(infos["goal_name"])
    if args.goal_type == "text":
        graph.set_text_goal(infos.get("text_goal", infos.get("goal_name", "")))
    elif args.goal_type == "ins-image":
        graph.set_image_goal(infos["instance_imagegoal"])


def _configure_logging(args):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    file_handler = logging.FileHandler(args.eval_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    )
    root_logger.addHandler(file_handler)


def main():
    args = get_config()
    tracer = RunTracer(
        args.run_dir,
        enable_controller_trace=getattr(args, "enable_controller_trace", True),
    )
    _configure_logging(args)
    logging.info(
        "SmoothNav starting: mode=%s profile=%s run_id=%s",
        args.mode,
        args.controller_profile,
        args.run_id,
    )
    logging.info(
        "Run context: goal_type=%s num_eval=%s results=%s controller=%s",
        args.goal_type,
        args.num_eval_episodes,
        args.run_dir,
        controller_config_dict(args),
    )

    bev_map = BEV_Map(args)
    graph = Graph(args)
    envs = construct_envs(args)
    agent = UniGoal_Agent(args, envs)
    executor_adapter = ExecutorAdapter(agent)
    grounder = GeometricGrounder()
    mission_manager = MissionProgressManager()
    world_builder = WorldStateBuilder()
    pending_manager = PendingProposalManager(max_pending=2)
    budget_governor = BudgetGovernor(
        planner_call_budget=8,
        adjudicator_call_budget=8,
        window_steps=100,
        forced_cooldown_after_replan=0,
        belief_summarization_interval=50,
    )
    terminal_arbiter = TerminalArbiter(
        max_no_progress_steps=max(args.stuck_threshold * 8, 40),
        max_grounding_noops=max(args.controller_grounding_noop_replan_threshold * 6, 12),
        max_stuck_steps=max(args.stuck_threshold * 4, 30),
    )
    tactical_arbiter = TacticalArbiter(
        direction_reuse_limit=args.controller_direction_reuse_limit
    )

    llm_sonnet = LLM(
        args.base_url,
        args.api_key,
        args.llm_model,
        api_provider=args.api_provider,
        api_protocol=args.api_protocol,
    )
    high_planner = HighLevelPlanner(llm_fn=llm_sonnet)
    if args.controller_monitor_policy == "llm":
        llm_haiku = LLM(
            args.base_url,
            args.api_key,
            args.llm_model_fast,
            api_provider=args.api_provider,
            api_protocol=args.api_protocol,
        )
        low_agent = LowLevelAgent(llm_fn=llm_haiku)
    elif args.controller_monitor_policy == "llm_escalation":
        llm_haiku = LLM(
            args.base_url,
            args.api_key,
            args.llm_model_fast,
            api_provider=args.api_provider,
            api_protocol=args.api_protocol,
        )
        low_agent = EscalationOnlyMonitor(
            llm_fn=llm_haiku,
            prefetch_near_threshold=args.controller_prefetch_near_threshold,
        )
    elif args.controller_monitor_policy == "rules":
        low_agent = RuleBasedMonitor(
            prefetch_near_threshold=args.controller_prefetch_near_threshold
        )
    else:
        low_agent = DisabledMonitor()
    smoothness = SmoothnessMetrics()

    episode_results = []
    episode_success = deque(maxlen=args.num_episodes)
    episode_spl = deque(maxlen=args.num_episodes)
    episode_smoothness = deque(maxlen=args.num_episodes)
    ep_action_log = []
    all_ep_action_logs = []

    controller_state = ControllerState()
    finished = False
    wait_env = False

    bev_map.init_map_and_pose()
    obs, rgbd, infos = agent.reset()
    bev_map.mapping(rgbd, infos)

    global_goals = [args.local_width // 2, args.local_height // 2]
    goal_maps = np.zeros((args.local_width, args.local_height))
    goal_maps[global_goals[0], global_goals[1]] = 1

    agent_input = {
        "map_pred": bev_map.local_map[0, 0, :, :].cpu().numpy(),
        "exp_pred": bev_map.local_map[0, 1, :, :].cpu().numpy(),
        "pose_pred": bev_map.planner_pose_inputs[0],
        "goal": goal_maps,
        "exp_goal": goal_maps.copy(),
        "new_goal": 1,
        "found_goal": 0,
        "wait": wait_env or finished,
        "sem_map": bev_map.local_map[0, 4:11, :, :].cpu().numpy(),
    }

    if args.visualize:
        bev_map.local_map[0, 10, :, :] = 1e-5
        agent_input["sem_map_pred"] = (
            bev_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()
        )

    obs, rgbd, done, infos = agent.step(agent_input)
    _reset_graph_goal(args, graph, infos)
    goal_description = _goal_description_from_infos(args, infos)
    mission_manager.reset(goal_description, args.goal_type)
    task_spec = parse_task_spec(
        goal_description,
        args.goal_type,
        task_id="episode_000000",
    )
    evidence_ledger = EvidenceLedger()
    belief_updater = TaskBeliefUpdater(task_spec, evidence_ledger)
    task_belief = belief_updater.belief
    smoothness.reset()
    active_episode_id = 0
    step = 0
    grounding_events = []
    world_state = None
    terminal_decision = TerminalDecision(
        outcome=TerminalOutcome.RUNNING,
        termination_confidence=0.0,
        reason="not_started",
    )

    def apply_strategy_with_trace(strategy, trigger):
        result, geometric_goal = grounder.ground_strategy(
            strategy,
            graph,
            bev_map,
            args,
            global_goals,
            task_belief=task_belief,
            world_state=world_state,
            goal_epoch=controller_state.goal_epoch,
        )
        apply_strategy_with_trace.last_result = result
        apply_strategy_with_trace.last_geometric_goal = geometric_goal
        mission_manager.note_stage_goal(strategy, replan_reason=trigger)
        belief_updater.note_active_stage(
            stage_goal_from_strategy(
                strategy,
                task_epoch=task_belief.task_epoch,
                belief_epoch=task_belief.belief_epoch,
                world_epoch=getattr(world_state, "world_epoch", 0) if world_state else 0,
                stage_epoch=controller_state.strategy_epoch,
            )
        )
        update_grounding_failure_state(controller_state, result)
        event = result.to_dict()
        event["trigger"] = trigger
        event["strategy"] = strategy_to_dict(strategy)
        event["geometric_goal"] = geometric_goal.to_dict()
        event["consecutive_grounding_noops"] = (
            controller_state.consecutive_grounding_noops
        )
        event["same_frontier_reuse_count"] = (
            controller_state.same_frontier_reuse_count
        )
        grounding_events.append(event)
        logging.info(
            "Grounding[%s]: success=%s changed=%s noop=%s frontier=%s projected=%s",
            trigger,
            result.success,
            result.changed,
            result.noop_type or "",
            result.selected_frontier,
            result.projected_goal,
        )
        return result

    apply_strategy_with_trace.last_result = None
    apply_strategy_with_trace.last_geometric_goal = None
    planner_budget_denial_events = []

    def plan_strategy_budgeted(*, fallback_strategy=None, allow_none=False, **kwargs):
        if not budget_governor.can_call_planner(step):
            event = {
                "step_idx": step,
                "reason": kwargs.get("escalate_reason", "planner_call"),
                "fallback": "none" if allow_none else "current_strategy",
            }
            planner_budget_denial_events.append(event)
            logging.warning("Planner budget denied: %s", event)
            return None if allow_none else fallback_strategy
        return plan_strategy(**kwargs)

    print(f"SmoothNav [{args.mode}] starting, {args.num_episodes} episodes")

    try:
        while not finished:
            global_step = (step // args.num_local_steps) % args.num_global_steps
            local_step = step % args.num_local_steps

            if done:
                completed_episode_id = active_episode_id
                spl = infos["spl"]
                success = infos["success"] if infos["success"] is not None else 0.0
                episode_success.append(success)
                episode_spl.append(spl)

                sm_result = smoothness.compute()
                episode_smoothness.append(sm_result)
                all_ep_action_logs.append(ep_action_log)
                ep_action_log = []

                episode_results.append(
                    {
                        "episode": completed_episode_id,
                        "habitat_episode_no": int(
                            infos.get("episode_no", completed_episode_id)
                        ),
                        "success": success,
                        "spl": spl,
                        "terminal_outcome": (
                            terminal_decision.outcome.value
                            if terminal_decision is not None
                            and terminal_decision.outcome != TerminalOutcome.RUNNING
                            else (
                                TerminalOutcome.SUCCESS.value
                                if success
                                else TerminalOutcome.FAILURE_NO_PROGRESS_TIMEOUT.value
                            )
                        ),
                        "terminal_decision": (
                            terminal_decision.to_dict()
                            if terminal_decision is not None
                            else None
                        ),
                        "high_level_calls": high_planner.call_count,
                        "low_level_calls": low_agent.call_count,
                        **sm_result.to_dict(),
                    }
                )

                if len(episode_success) == args.num_episodes:
                    finished = True
                    if args.visualize:
                        video_path = os.path.join(
                            args.visualization_dir,
                            "videos",
                            f"eps_{completed_episode_id:06d}.mp4",
                        )
                        agent.save_visualization(video_path)
                    break

                if args.visualize:
                    video_path = os.path.join(
                        args.visualization_dir,
                        "videos",
                        f"eps_{completed_episode_id:06d}.mp4",
                    )
                    agent.save_visualization(video_path)

                wait_env = True
                bev_map.update_intrinsic_rew()
                bev_map.init_map_and_pose_for_env()
                smoothness.reset()
                high_planner.reset()
                low_agent.reset()
                controller_state = ControllerState()
                world_builder.reset()
                pending_manager.reset()
                budget_governor.reset()
                tactical_arbiter.reset()
                _reset_graph_goal(args, graph, infos)
                goal_description = _goal_description_from_infos(args, infos)
                mission_manager.reset(goal_description, args.goal_type)
                task_spec = parse_task_spec(
                    goal_description,
                    args.goal_type,
                    task_id=f"episode_{completed_episode_id + 1:06d}",
                )
                evidence_ledger = EvidenceLedger()
                belief_updater = TaskBeliefUpdater(task_spec, evidence_ledger)
                task_belief = belief_updater.belief
                terminal_decision = TerminalDecision(
                    outcome=TerminalOutcome.RUNNING,
                    termination_confidence=0.0,
                    reason="reset",
                )
                active_episode_id = completed_episode_id + 1

            bev_map.mapping(rgbd, infos)

            navigate_steps = global_step * args.num_local_steps + local_step
            graph.set_navigate_steps(navigate_steps)
            if not wait_env and navigate_steps % 2 == 0:
                graph.set_observations(obs)
                graph.update_scenegraph()

            pose_pred = bev_map.planner_pose_inputs[0]
            start_x, start_y, start_o = pose_pred[0], pose_pred[1], pose_pred[2]
            agent_map_x = start_x * 100.0 / args.map_resolution
            agent_map_y = start_y * 100.0 / args.map_resolution
            pose_before = {
                "x": float(start_x),
                "y": float(start_y),
                "heading": float(start_o),
            }
            goal_before = list(global_goals)
            planner_calls_before = high_planner.call_count
            monitor_calls_before = low_agent.call_count
            planner_called = False
            monitor_called = False
            planner_reasons = []
            monitor_decision = None
            monitor_reason = ""
            monitor_trigger_reason = ""
            monitor_trigger_event_types = []
            low_level_action = None
            graph_delta = None
            world_state = None
            tactical_decision = None
            executor_command = None
            executor_feedback = None
            budget_state = budget_governor.update(step)
            step_episode_id = active_episode_id
            current_strategy_before = (
                controller_state.current_strategy.target_region
                if controller_state.current_strategy is not None
                else None
            )
            pending_strategy_before = (
                controller_state.pending_strategy.target_region
                if controller_state.pending_strategy is not None
                else None
            )
            pending_created = False
            pending_promoted = False
            pending_created_and_promoted_same_step = False
            pending_create_reason = ""
            pending_strategy_type = ""
            pending_promotion_reason = ""
            pending_proposal_created = None
            pending_proposal_adopted = None
            forced_replan_due_to_direction_reuse = False
            forced_replan_due_to_grounding_failure = False
            controller_stuck_replan_triggered = False
            grounding_failure_reason = ""
            grounding_events = []
            apply_strategy_with_trace.last_result = None

            near_goal = np.linalg.norm(
                np.array([bev_map.local_row, bev_map.local_col]) - np.array(global_goals)
            ) < 10
            frontier_reached = False

            if args.mode == "smoothnav":
                is_planning = False

                if local_step == args.num_local_steps - 1 or near_goal:
                    if wait_env:
                        wait_env = False
                    else:
                        bev_map.update_intrinsic_rew()
                    bev_map.move_local_map()
                    graph.set_full_map(bev_map.full_map)
                    graph.set_full_pose(bev_map.full_pose)
                    frontier_reached = near_goal

                dist_to_goal = np.linalg.norm(
                    np.array([bev_map.local_row, bev_map.local_col]) - np.array(global_goals)
                )

                if controller_state.needs_initial_plan:
                    graph.set_full_map(bev_map.full_map)
                    graph.set_full_pose(bev_map.full_pose)
                    controller_state.current_strategy = plan_strategy_budgeted(
                        fallback_strategy=controller_state.current_strategy,
                        high_planner=high_planner,
                        graph=graph,
                        controller_state=controller_state,
                        goal_description=goal_description,
                        escalate_reason="Episode start",
                        agent_pos=(int(agent_map_x), int(agent_map_y)),
                        map_size=args.map_size,
                        episode_id=step_episode_id,
                        step_idx=step,
                        trace_writer=tracer,
                    )
                    apply_strategy_with_trace(
                        controller_state.current_strategy,
                        "episode_start",
                    )
                    planner_reasons.append("episode_start")
                    controller_state.needs_initial_plan = False
                    frontier_reached = False
                    controller_state.prev_node_count = len(graph.nodes)
                    is_planning = True
                    logging.info(
                        "Step %s: Initial plan -> %s",
                        step,
                        controller_state.current_strategy.target_region,
                    )

                pos = np.array([agent_map_x, agent_map_y])
                if controller_state.last_position is not None:
                    if np.linalg.norm(pos - np.array(controller_state.last_position)) < 0.05:
                        controller_state.no_progress_steps += 1
                    else:
                        controller_state.no_progress_steps = 0
                controller_state.last_position = pos.tolist()

                graph_delta = build_graph_delta(
                    graph=graph,
                    controller_state=controller_state,
                    frontier_near=near_goal,
                    frontier_reached=frontier_reached,
                    no_progress=controller_state.no_progress_steps > 0,
                    stuck=controller_state.no_progress_steps >= args.stuck_threshold,
                    dist_to_goal=dist_to_goal,
                )
                controller_state.prev_room_object_counts = graph_delta.room_object_counts
                controller_state.prev_node_captions = graph_delta.node_captions_snapshot
                world_state = world_builder.build(
                    step_idx=step,
                    pose={
                        **pose_before,
                        "map_x": float(agent_map_x),
                        "map_y": float(agent_map_y),
                    },
                    local_pose=pose_pred,
                    graph=graph,
                    bev_map=bev_map,
                    controller_state=controller_state,
                    graph_delta=graph_delta,
                    agent=agent,
                )
                task_belief = belief_updater.update(world_state)
                budget_state = budget_governor.update(step)
                tactical_decision = tactical_arbiter.decide(
                    world_state=world_state,
                    mission_state=mission_manager.state,
                    current_strategy=controller_state.current_strategy,
                    pending_strategy=controller_state.pending_strategy,
                    needs_initial_plan=controller_state.needs_initial_plan,
                    no_frontiers=(
                        apply_strategy_with_trace.last_result is not None
                        and apply_strategy_with_trace.last_result.graph_no_goal_reason
                        == "no_frontiers"
                    ),
                )

                if (
                    tactical_decision is not None
                    and tactical_decision.mode == TacticalMode.REPLAN_REQUIRED
                    and tactical_decision.reason == "new_room_discovered"
                    and args.controller_replan_policy == "event"
                ):
                    if (
                        controller_state.current_strategy
                        and not is_room_target(controller_state.current_strategy.target_region)
                        and not controller_state.needs_initial_plan
                    ):
                        controller_state.current_strategy = plan_strategy_budgeted(
                            fallback_strategy=controller_state.current_strategy,
                            high_planner=high_planner,
                            graph=graph,
                            controller_state=controller_state,
                            goal_description=goal_description,
                            escalate_reason="New room discovered, can make specific choice",
                            agent_pos=(int(agent_map_x), int(agent_map_y)),
                            map_size=args.map_size,
                            episode_id=step_episode_id,
                            step_idx=step,
                            trace_writer=tracer,
                            mission_state=mission_manager.state,
                            world_state=world_state,
                        )
                        apply_strategy_with_trace(
                            controller_state.current_strategy,
                            "new_room_discovered",
                        )
                        planner_reasons.append("new_room_discovered")
                        is_planning = True
                        logging.info(
                            "Step %s: New room -> %s",
                            step,
                            controller_state.current_strategy.target_region,
                        )

                if (
                    args.controller_enable_monitor
                    and tactical_decision is not None
                    and tactical_decision.should_call_monitor
                ):
                    monitor_called, low_result, monitor_trigger_event_types = maybe_call_monitor(
                        low_agent=low_agent,
                        controller_state=controller_state,
                        graph_delta=graph_delta,
                        graph=graph,
                        episode_id=step_episode_id,
                        step_idx=step,
                        trace_writer=tracer,
                    )
                    if not monitor_called and graph_delta.has_new_nodes:
                        controller_state.prev_node_count = len(graph.nodes)
                else:
                    monitor_called, low_result, monitor_trigger_event_types = False, None, []
                    if graph_delta.has_new_nodes:
                        controller_state.prev_node_count = len(graph.nodes)
                if monitor_called:
                    monitor_trigger_reason = ", ".join(monitor_trigger_event_types)
                    controller_state.prev_node_count = len(graph.nodes)
                    low_level_action = low_result.action
                    monitor_decision = low_result.action.name
                    monitor_reason = low_result.reason
                    if low_result.action == LowLevelAction.ADJUST:
                        if low_result.adjust_bias is not None:
                            controller_state.current_strategy.bias_position = low_result.adjust_bias
                            apply_strategy_with_trace(
                                controller_state.current_strategy,
                                "monitor_adjust",
                            )
                            logging.info(
                                "Step %s: ADJUST bias -> %s reason=%s",
                                step,
                                low_result.adjust_bias,
                                low_result.reason,
                            )
                    elif low_result.action == LowLevelAction.PREFETCH:
                        if (
                            args.controller_enable_prefetch
                            and controller_state.pending_strategy is None
                        ):
                            controller_state.pending_strategy = plan_strategy_budgeted(
                                fallback_strategy=controller_state.pending_strategy,
                                allow_none=True,
                                high_planner=high_planner,
                                graph=graph,
                                controller_state=controller_state,
                                goal_description=goal_description,
                                escalate_reason=f"PREFETCH: {low_result.reason}",
                                agent_pos=(int(agent_map_x), int(agent_map_y)),
                                map_size=args.map_size,
                                episode_id=step_episode_id,
                                step_idx=step,
                                trace_writer=tracer,
                                mission_state=mission_manager.state,
                                world_state=world_state,
                            )
                            if controller_state.pending_strategy is not None:
                                planner_reasons.append("monitor_prefetch")
                                is_planning = True
                                pending_created = True
                                pending_create_reason = "monitor_prefetch"
                                pending_strategy_type = (
                                    "object"
                                    if controller_state.pending_strategy.target_region.startswith("object:")
                                    else "room"
                                    if is_room_target(
                                        controller_state.pending_strategy.target_region
                                    )
                                    else "direction"
                                )
                                logging.info(
                                    "Step %s: PREFETCH -> %s",
                                    step,
                                    controller_state.pending_strategy.target_region,
                                )
                    elif low_result.action == LowLevelAction.ESCALATE:
                        if is_room_target(controller_state.current_strategy.target_region):
                            controller_state.explored_regions.append(
                                f"{controller_state.current_strategy.target_region} (searched, not found)"
                            )
                        controller_state.current_strategy = plan_strategy_budgeted(
                            fallback_strategy=controller_state.current_strategy,
                            high_planner=high_planner,
                            graph=graph,
                            controller_state=controller_state,
                            goal_description=goal_description,
                            escalate_reason=f"ESCALATE: {low_result.reason}",
                            agent_pos=(int(agent_map_x), int(agent_map_y)),
                            map_size=args.map_size,
                            episode_id=step_episode_id,
                            step_idx=step,
                            trace_writer=tracer,
                            mission_state=mission_manager.state,
                            world_state=world_state,
                        )
                        controller_state.pending_strategy = None
                        apply_strategy_with_trace(
                            controller_state.current_strategy,
                            "monitor_escalate",
                        )
                        planner_reasons.append("monitor_escalate")
                        is_planning = True
                        logging.info(
                            "Step %s: ESCALATE -> %s",
                            step,
                            controller_state.current_strategy.target_region,
                        )

                pending_promotion = maybe_promote_pending(
                    controller_state=controller_state,
                    graph=graph,
                    bev_map=bev_map,
                    args=args,
                    global_goals=global_goals,
                    apply_strategy_fn=lambda strategy, graph, bev_map, args, global_goals: apply_strategy_with_trace(
                        strategy,
                        "pending_promotion",
                    ),
                )
                if pending_promotion["promoted"]:
                    pending_promoted = True
                    pending_promotion_reason = pending_promotion["reason"]

                frontier_result = handle_frontier_reached(
                    controller_state=controller_state,
                    graph_delta=graph_delta,
                    graph=graph,
                    bev_map=bev_map,
                    args=args,
                    global_goals=global_goals,
                    high_planner=high_planner,
                    goal_description=goal_description,
                    agent_pos=(int(agent_map_x), int(agent_map_y)),
                    apply_strategy_fn=lambda strategy, graph, bev_map, args, global_goals: apply_strategy_with_trace(
                        strategy,
                        "frontier_reached",
                    ),
                    episode_id=step_episode_id,
                    step_idx=step,
                    trace_writer=tracer,
                    mission_state=mission_manager.state,
                    world_state=world_state,
                )
                if frontier_result["handled"]:
                    planner_reasons.append("frontier_reached")
                    is_planning = True
                    pending_promoted = pending_promoted or frontier_result["pending_promoted"]
                    if frontier_result["pending_promotion_reason"]:
                        pending_promotion_reason = frontier_result["pending_promotion_reason"]
                    forced_replan_due_to_direction_reuse = frontier_result[
                        "forced_replan_due_to_direction_reuse"
                    ]

                if (
                    args.controller_enable_prefetch
                    and args.controller_replan_policy == "event"
                    and controller_state.pending_strategy is None
                    and low_level_action != LowLevelAction.PREFETCH
                    and not frontier_reached
                    and not controller_state.needs_initial_plan
                    and controller_state.current_strategy
                    and (
                        getattr(graph_delta, "frontier_near", False)
                        or dist_to_goal < args.controller_prefetch_near_threshold
                        or (
                            not is_room_target(
                                controller_state.current_strategy.target_region
                            )
                            and (
                                getattr(graph_delta, "has_new_rooms", False)
                                or getattr(graph_delta, "has_room_object_increase", False)
                            )
                        )
                    )
                ):
                    controller_state.pending_strategy = plan_strategy_budgeted(
                        fallback_strategy=controller_state.pending_strategy,
                        allow_none=True,
                        high_planner=high_planner,
                        graph=graph,
                        controller_state=controller_state,
                        goal_description=goal_description,
                        escalate_reason="Auto-PREFETCH: approaching frontier",
                        agent_pos=(int(agent_map_x), int(agent_map_y)),
                        map_size=args.map_size,
                        episode_id=step_episode_id,
                        step_idx=step,
                        trace_writer=tracer,
                        mission_state=mission_manager.state,
                        world_state=world_state,
                    )
                    if controller_state.pending_strategy is not None:
                        planner_reasons.append("auto_prefetch")
                        is_planning = True
                        pending_created = True
                        pending_create_reason = "auto_prefetch"
                        pending_strategy_type = (
                            "object"
                            if controller_state.pending_strategy.target_region.startswith("object:")
                            else "room"
                            if is_room_target(controller_state.pending_strategy.target_region)
                            else "direction"
                        )
                        logging.info(
                            "Step %s: Auto-PREFETCH -> %s",
                            step,
                            controller_state.pending_strategy.target_region,
                        )

                if (
                    args.controller_replan_policy == "fixed_interval"
                    and not controller_state.needs_initial_plan
                    and args.controller_fixed_plan_interval_steps > 0
                    and step > 0
                    and step % args.controller_fixed_plan_interval_steps == 0
                ):
                    controller_state.current_strategy = plan_strategy_budgeted(
                        fallback_strategy=controller_state.current_strategy,
                        high_planner=high_planner,
                        graph=graph,
                        controller_state=controller_state,
                        goal_description=goal_description,
                        escalate_reason="Fixed interval refresh",
                        agent_pos=(int(agent_map_x), int(agent_map_y)),
                        map_size=args.map_size,
                        episode_id=step_episode_id,
                        step_idx=step,
                        trace_writer=tracer,
                        mission_state=mission_manager.state,
                        world_state=world_state,
                    )
                    controller_state.pending_strategy = None
                    apply_strategy_with_trace(
                        controller_state.current_strategy,
                        "fixed_interval_refresh",
                    )
                    planner_reasons.append("fixed_interval_refresh")
                    is_planning = True

                if (
                    args.controller_enable_stuck_replan
                    and tactical_decision is not None
                    and tactical_decision.mode == TacticalMode.RECOVERY
                    and handle_stuck_replan(
                        controller_state=controller_state,
                        graph_delta=graph_delta,
                        graph=graph,
                        bev_map=bev_map,
                        args=args,
                        global_goals=global_goals,
                        high_planner=high_planner,
                        goal_description=goal_description,
                        agent_pos=(int(agent_map_x), int(agent_map_y)),
                        apply_strategy_fn=lambda strategy, graph, bev_map, args, global_goals: apply_strategy_with_trace(
                            strategy,
                            "stuck_replan",
                        ),
                        episode_id=step_episode_id,
                        step_idx=step,
                        trace_writer=tracer,
                        mission_state=mission_manager.state,
                        world_state=world_state,
                    )
                ):
                    planner_reasons.append("stuck_replan")
                    is_planning = True
                    controller_stuck_replan_triggered = True
                    controller_state.executor_stuck_suppression_steps = int(
                        getattr(args, "controller_stuck_suppression_steps", 0) or 0
                    )

                grounding_failure = handle_grounding_failure(
                    controller_state=controller_state,
                    last_grounding_result=(
                        apply_strategy_with_trace.last_result
                        if hasattr(apply_strategy_with_trace, "last_result")
                        else None
                    ),
                    graph=graph,
                    bev_map=bev_map,
                    args=args,
                    global_goals=global_goals,
                    high_planner=high_planner,
                    goal_description=goal_description,
                    agent_pos=(int(agent_map_x), int(agent_map_y)),
                    apply_strategy_fn=lambda strategy, graph, bev_map, args, global_goals: apply_strategy_with_trace(
                        strategy,
                        "grounding_failure_replan",
                    ),
                    episode_id=step_episode_id,
                    step_idx=step,
                    trace_writer=tracer,
                    mission_state=mission_manager.state,
                    world_state=world_state,
                )
                if grounding_failure["replanned"]:
                    planner_reasons.append("grounding_failure_replan")
                    is_planning = True
                    forced_replan_due_to_grounding_failure = True
                    grounding_failure_reason = grounding_failure["grounding_failure_reason"]

                smoothness.record_from_habitat(
                    x=start_x,
                    y=start_y,
                    heading=np.deg2rad(start_o),
                    step=step,
                    action=getattr(agent, "last_action", 1) or 1,
                    is_planning=is_planning,
                )

            else:
                dist_to_goal = np.linalg.norm(
                    np.array([bev_map.local_row, bev_map.local_col]) - np.array(global_goals)
                )
                pos = np.array([agent_map_x, agent_map_y])
                if controller_state.last_position is not None:
                    if np.linalg.norm(pos - np.array(controller_state.last_position)) < 0.05:
                        controller_state.no_progress_steps += 1
                    else:
                        controller_state.no_progress_steps = 0
                controller_state.last_position = pos.tolist()
                is_planning = False
                if local_step == args.num_local_steps - 1 or near_goal:
                    if wait_env:
                        wait_env = False
                    else:
                        bev_map.update_intrinsic_rew()
                    bev_map.move_local_map()

                    is_planning = True
                    graph.set_full_map(bev_map.full_map)
                    graph.set_full_pose(bev_map.full_pose)
                    goal = graph.explore()
                    if hasattr(graph, "frontier_locations_16"):
                        graph.frontier_locations_16[:, 0] -= bev_map.local_map_boundary[0, 0]
                        graph.frontier_locations_16[:, 1] -= bev_map.local_map_boundary[0, 2]
                    if isinstance(goal, (list, np.ndarray)):
                        goal = list(goal)
                        goal[0] -= bev_map.local_map_boundary[0, 0]
                        goal[1] -= bev_map.local_map_boundary[0, 2]
                        if 0 <= goal[0] < args.local_width and 0 <= goal[1] < args.local_height:
                            global_goals = goal

                graph_delta = build_graph_delta(
                    graph=graph,
                    controller_state=controller_state,
                    frontier_near=near_goal,
                    frontier_reached=False,
                    no_progress=controller_state.no_progress_steps > 0,
                    stuck=controller_state.no_progress_steps >= args.stuck_threshold,
                    dist_to_goal=dist_to_goal,
                )
                controller_state.prev_room_object_counts = graph_delta.room_object_counts
                controller_state.prev_node_captions = graph_delta.node_captions_snapshot
                controller_state.prev_node_count = len(graph.nodes)
                world_state = world_builder.build(
                    step_idx=step,
                    pose={
                        **pose_before,
                        "map_x": float(agent_map_x),
                        "map_y": float(agent_map_y),
                    },
                    local_pose=pose_pred,
                    graph=graph,
                    bev_map=bev_map,
                    controller_state=controller_state,
                    graph_delta=graph_delta,
                    agent=agent,
                )
                task_belief = belief_updater.update(world_state)
                budget_state = budget_governor.update(step)
                tactical_decision = tactical_arbiter.decide(
                    world_state=world_state,
                    mission_state=mission_manager.state,
                    current_strategy=controller_state.current_strategy,
                    pending_strategy=controller_state.pending_strategy,
                    needs_initial_plan=controller_state.needs_initial_plan,
                    no_frontiers=False,
                )

                smoothness.record_from_habitat(
                    x=start_x,
                    y=start_y,
                    heading=np.deg2rad(start_o),
                    step=step,
                    action=getattr(agent, "last_action", 1) or 1,
                    is_planning=is_planning,
                )

            controller_state.planner_call_count = high_planner.call_count
            controller_state.monitor_call_count = low_agent.call_count
            planner_called = high_planner.call_count > planner_calls_before
            monitor_called = monitor_called or (low_agent.call_count > monitor_calls_before)
            if planner_called:
                for _ in range(high_planner.call_count - planner_calls_before):
                    budget_state = budget_governor.record_planner_call(step)
            if monitor_called:
                for _ in range(low_agent.call_count - monitor_calls_before):
                    budget_state = budget_governor.record_adjudicator_call(step)
            goal_after = list(global_goals)
            controller_state.last_goal = list(goal_after)
            current_strategy_after = (
                controller_state.current_strategy.target_region
                if controller_state.current_strategy is not None
                else None
            )
            pending_strategy_after = (
                controller_state.pending_strategy.target_region
                if controller_state.pending_strategy is not None
                else None
            )
            strategy_switched = (
                current_strategy_before != current_strategy_after
                and current_strategy_after is not None
            )
            if strategy_switched:
                controller_state.direction_reuse_count = 0
            pending_created = pending_created or (
                pending_strategy_before is None and pending_strategy_after is not None
            )
            if (
                pending_strategy_before is not None
                and pending_strategy_after is None
                and current_strategy_after == pending_strategy_before
            ):
                pending_promoted = True
            pending_created_and_promoted_same_step = pending_created and pending_promoted
            if pending_created and controller_state.pending_strategy is not None:
                pending_stage_goal = stage_goal_from_strategy(
                    controller_state.pending_strategy,
                    task_epoch=task_belief.task_epoch,
                    belief_epoch=task_belief.belief_epoch,
                    world_epoch=getattr(world_state, "world_epoch", 0) if world_state else 0,
                    stage_epoch=controller_state.strategy_epoch + 1,
                )
                if pending_stage_goal is not None:
                    pending_proposal_created = pending_manager.create(
                        pending_stage_goal,
                        task_belief=task_belief,
                        world_state=world_state,
                        created_reason=pending_create_reason or "legacy_pending_created",
                    )
            if pending_promoted:
                pending_proposal_adopted = pending_manager.adopt_best(
                    task_belief=task_belief,
                    world_state=world_state,
                    ledger=evidence_ledger,
                    reason=pending_promotion_reason or "legacy_pending_promoted",
                )
                if pending_proposal_adopted is None and controller_state.current_strategy is not None:
                    adopted_stage_goal = stage_goal_from_strategy(
                        controller_state.current_strategy,
                        task_epoch=task_belief.task_epoch,
                        belief_epoch=task_belief.belief_epoch,
                        world_epoch=getattr(world_state, "world_epoch", 0) if world_state else 0,
                        stage_epoch=controller_state.strategy_epoch + 1,
                    )
                    if adopted_stage_goal is not None:
                        pending_proposal_created = pending_manager.create(
                            adopted_stage_goal,
                            task_belief=task_belief,
                            world_state=world_state,
                            created_reason="promoted_without_recorded_pending",
                        )
                        pending_proposal_adopted = pending_manager.adopt_best(
                            task_belief=task_belief,
                            world_state=world_state,
                            ledger=evidence_ledger,
                            reason=pending_promotion_reason or "legacy_pending_promoted",
                        )
            goal_updated = goal_after != goal_before
            goal_epoch_advanced = False
            if goal_updated:
                controller_state.goal_epoch += 1
                goal_epoch_advanced = True
            grounding_attempt_count = len(grounding_events)
            grounding_noop_count = sum(
                1 for event in grounding_events if not event.get("changed", False)
            )
            grounding_changed_count = sum(
                1 for event in grounding_events if event.get("changed", False)
            )
            control_epoch_advanced = False
            if current_strategy_after is not None and (
                strategy_switched
                or grounding_changed_count > 0
                or controller_state.strategy_epoch == 0
            ):
                controller_state.strategy_epoch += 1
                control_epoch_advanced = True

            latest_grounding = grounding_events[-1] if grounding_events else {}

            goal_maps = np.zeros((args.local_width, args.local_height))
            gx = int(np.clip(global_goals[0], 0, args.local_width - 1))
            gy = int(np.clip(global_goals[1], 0, args.local_height - 1))
            goal_maps[gx, gy] = 1
            latest_geometric_goal = (
                apply_strategy_with_trace.last_geometric_goal
                if apply_strategy_with_trace.last_geometric_goal is not None
                else null_geometric_goal()
            )
            executor_command = executor_adapter.build_command(
                geometric_goal=latest_geometric_goal,
                strategy_epoch=controller_state.strategy_epoch,
                goal_epoch=controller_state.goal_epoch,
                allow_target_lock=True,
                allow_recovery=not (
                    controller_state.executor_stuck_suppression_steps > 0
                ),
                clear_temp_goal=False,
            )

            agent_input = {
                "map_pred": bev_map.local_map[0, 0, :, :].cpu().numpy(),
                "exp_pred": bev_map.local_map[0, 1, :, :].cpu().numpy(),
                "pose_pred": bev_map.planner_pose_inputs[0],
                "goal": goal_maps,
                "exp_goal": goal_maps.copy(),
                "new_goal": 0,
                "found_goal": 0,
                "strategy_epoch": controller_state.strategy_epoch,
                "goal_epoch": controller_state.goal_epoch,
                "step_idx": step,
                "suppress_stuck_override": (
                    controller_state.executor_stuck_suppression_steps > 0
                ),
                "wait": wait_env or finished,
                "sem_map": bev_map.local_map[0, 4:11, :, :].cpu().numpy(),
            }

            if args.visualize:
                bev_map.local_map[0, 10, :, :] = 1e-5
                agent_input["sem_map_pred"] = (
                    bev_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()
                )

            executor_result = executor_adapter.step(agent_input, executor_command)
            obs = executor_result.obs
            rgbd = executor_result.rgbd
            done = executor_result.done
            infos = executor_result.infos
            executor_feedback = executor_result.executor_feedback
            if world_state is not None:
                task_belief = belief_updater.update(
                    world_state,
                    executor_feedback=executor_feedback,
                )
            terminal_decision = terminal_arbiter.decide(
                done=done,
                infos=infos,
                task_belief=task_belief,
                budget_state=budget_state,
                controller_state=controller_state,
                latest_grounding=apply_strategy_with_trace.last_result,
                executor_feedback=executor_feedback,
            )
            if controller_state.executor_stuck_suppression_steps > 0:
                controller_state.executor_stuck_suppression_steps -= 1

            pose_after = getattr(agent, "last_pose_after_action", None)
            if pose_after is None:
                pose_after = {
                    "x": float(start_x),
                    "y": float(start_y),
                    "heading": float(start_o),
                }

            ep_action_log.append(
                {
                    "action": agent.last_action,
                    "x_before": float(start_x),
                    "y_before": float(start_y),
                    "x_after": float(pose_after["x"]),
                    "y_after": float(pose_after["y"]),
                    "heading": float(start_o),
                }
            )
            layered_payload = compact_layered_payload(
                layered_trace_payload(
                    world_state=world_state,
                    mission_state=mission_manager.state,
                    task_spec=task_spec,
                    task_belief=task_belief,
                    evidence_ledger=evidence_ledger,
                    pending_proposals=pending_manager,
                    current_strategy=controller_state.current_strategy,
                    pending_strategy=controller_state.pending_strategy,
                    tactical_decision=tactical_decision,
                    geometric_goal=latest_geometric_goal,
                    executor_command=executor_command,
                    executor_feedback=executor_feedback,
                    budget_state=budget_state,
                    terminal_decision=terminal_decision,
                )
            )

            tracer.record_step(
                step_episode_id,
                {
                    "episode_id": step_episode_id,
                    "step_idx": step,
                    "mode": args.mode,
                    **layered_payload,
                    "pose_before": pose_before,
                    "pose_after": pose_after,
                    "graph_node_count": getattr(graph_delta, "graph_node_count", len(graph.nodes)),
                    "new_node_count": len(getattr(graph_delta, "new_nodes", [])),
                    "new_node_captions": getattr(graph_delta, "new_node_captions", []),
                    "graph_delta": {
                        "event_types": getattr(graph_delta, "event_types", []),
                        "current_strategy_type": getattr(
                            graph_delta, "current_strategy_type", "none"
                        ),
                        "new_rooms": getattr(graph_delta, "new_rooms", []),
                        "room_object_count_changes": getattr(graph_delta, "room_object_count_changes", {}),
                        "room_object_count_increase_rooms": getattr(
                            graph_delta, "room_object_count_increase_rooms", []
                        ),
                        "node_caption_changed": getattr(graph_delta, "node_caption_changed", False),
                        "frontier_near": getattr(graph_delta, "frontier_near", False),
                        "frontier_reached": getattr(graph_delta, "frontier_reached", False),
                        "no_progress": getattr(graph_delta, "no_progress", False),
                        "stuck": getattr(graph_delta, "stuck", False),
                        "dist_to_goal": getattr(graph_delta, "dist_to_goal", None),
                    },
                    "current_strategy": strategy_to_dict(controller_state.current_strategy),
                    "pending_strategy": strategy_to_dict(controller_state.pending_strategy),
                    "planner_called": planner_called,
                    "planner_calls_this_step": high_planner.call_count - planner_calls_before,
                    "planner_reasons": planner_reasons,
                    "planner_budget_denials": [
                        event
                        for event in planner_budget_denial_events
                        if event.get("step_idx") == step
                    ],
                    "monitor_called": monitor_called,
                    "monitor_calls_this_step": low_agent.call_count - monitor_calls_before,
                    "monitor_decision": monitor_decision,
                    "monitor_reason": monitor_reason,
                    "monitor_trigger_reason": monitor_trigger_reason,
                    "monitor_trigger_event_types": monitor_trigger_event_types,
                    "goal_before": goal_before,
                    "goal_after": goal_after,
                    "goal_updated": goal_updated,
                    "goal_epoch": controller_state.goal_epoch,
                    "goal_epoch_advanced": goal_epoch_advanced,
                    "task_epoch": task_belief.task_epoch,
                    "belief_epoch": task_belief.belief_epoch,
                    "stage_epoch": controller_state.strategy_epoch,
                    "mode_epoch": (
                        tactical_decision.mode_epoch
                        if tactical_decision is not None
                        else None
                    ),
                    "world_epoch": (
                        world_state.world_epoch if world_state is not None else None
                    ),
                    "evidence_ids_used": list(
                        task_belief.evidence_ids_used_recently
                    ),
                    "pending_proposal_epoch": pending_manager.proposal_epoch,
                    "pending_proposal_created": (
                        pending_proposal_created.to_dict()
                        if pending_proposal_created is not None
                        else None
                    ),
                    "pending_proposal_adopted": (
                        pending_proposal_adopted.to_dict()
                        if pending_proposal_adopted is not None
                        else None
                    ),
                    "strategy_switched": strategy_switched,
                    "pending_created": pending_created,
                    "pending_created_and_promoted_same_step": (
                        pending_created_and_promoted_same_step
                    ),
                    "pending_create_reason": pending_create_reason,
                    "pending_strategy_type": pending_strategy_type,
                    "pending_promoted": pending_promoted,
                    "pending_promotion_reason": pending_promotion_reason,
                    "grounding_events": grounding_events,
                    "grounding_attempt_count": grounding_attempt_count,
                    "grounding_noop_count": grounding_noop_count,
                    "grounding_changed_count": grounding_changed_count,
                    "bias_input": latest_grounding.get("bias_input"),
                    "selected_frontier": latest_grounding.get("selected_frontier"),
                    "selected_frontier_same_as_prev": latest_grounding.get(
                        "selected_frontier_same_as_prev",
                        latest_grounding.get("selected_frontier_same_as_previous"),
                    ),
                    "grounding_success": latest_grounding.get("success"),
                    "grounding_changed": latest_grounding.get("changed"),
                    "grounding_noop_reason": latest_grounding.get("noop_reason"),
                    "grounding_no_goal_reason": latest_grounding.get(
                        "graph_no_goal_reason"
                    ),
                    "local_projection_valid": latest_grounding.get("local_projection_valid"),
                    "topk_frontier_scores": latest_grounding.get("topk_frontier_scores", []),
                    "selected_frontier_score_breakdown": latest_grounding.get(
                        "selected_frontier_score_breakdown", {}
                    ),
                    "top1_top2_gap": latest_grounding.get("top1_top2_gap"),
                    "candidate_frontier_count_after_bias_filter": latest_grounding.get(
                        "candidate_frontier_count_after_bias_filter"
                    ),
                    "raw_frontier_count": latest_grounding.get("raw_frontier_count"),
                    "filtered_frontier_count": latest_grounding.get(
                        "filtered_frontier_count"
                    ),
                    "frontier_filter_fallback_mode": latest_grounding.get(
                        "frontier_filter_fallback_mode"
                    ),
                    "candidate_distance_fallback_mode": latest_grounding.get(
                        "candidate_distance_fallback_mode"
                    ),
                    "used_raw_frontier_fallback": latest_grounding.get(
                        "used_raw_frontier_fallback"
                    ),
                    "used_relaxed_distance_fallback": latest_grounding.get(
                        "used_relaxed_distance_fallback"
                    ),
                    "selected_from_bias_filtered_subset": latest_grounding.get(
                        "selected_from_bias_filtered_subset"
                    ),
                    "consecutive_grounding_noops": controller_state.consecutive_grounding_noops,
                    "same_frontier_reuse_count": controller_state.same_frontier_reuse_count,
                    "forced_replan_due_to_grounding_failure": (
                        forced_replan_due_to_grounding_failure
                    ),
                    "grounding_failure_reason": grounding_failure_reason,
                    "strategy_epoch": controller_state.strategy_epoch,
                    "control_epoch_advanced": control_epoch_advanced,
                    "controller_profile": args.controller_profile,
                    "controller_monitor_policy": args.controller_monitor_policy,
                    "controller_replan_policy": args.controller_replan_policy,
                    "terminal_outcome": terminal_decision.outcome.value,
                    "terminal_decision": terminal_decision.to_dict(),
                    "budget_state_trace": budget_state.to_dict(),
                    "visible_target_override": bool(agent.last_override_info.get("visible_target_override")),
                    "temp_goal_override": bool(agent.last_override_info.get("temp_goal_override")),
                    "stuck_goal_override": bool(agent.last_override_info.get("stuck_goal_override")),
                    "global_goal_override": bool(agent.last_override_info.get("global_goal_override")),
                    "executor_adopted_goal_source": agent.last_override_info.get("adopted_goal_source"),
                    "executor_feedback_trace": (
                        executor_feedback.to_dict()
                        if executor_feedback is not None
                        else None
                    ),
                    "adopted_goal_source": agent.last_override_info.get("adopted_goal_source"),
                    "adopted_goal_before": agent.last_override_info.get("adopted_goal_before"),
                    "adopted_goal_after": agent.last_override_info.get("adopted_goal_after"),
                    "adopted_goal_epoch": agent.last_override_info.get("goal_epoch"),
                    "executor_adopted_goal_summary": agent.last_override_info.get("adopted_goal_summary"),
                    "executor_adopted_goal_changed": bool(
                        agent.last_override_info.get("adopted_goal_changed")
                    ),
                    "executor_adoption_changed": bool(
                        agent.last_override_info.get("adopted_goal_changed")
                    ),
                    "executor_strategy_epoch": agent.last_override_info.get("strategy_epoch"),
                    "executor_temp_goal_epoch": agent.last_override_info.get("temp_goal_epoch"),
                    "stale_temp_goal_cleared": bool(
                        agent.last_override_info.get("stale_temp_goal_cleared")
                    ),
                    "temp_goal_cleared_on_strategy_switch": bool(
                        agent.last_override_info.get("temp_goal_cleared_on_strategy_switch")
                    ),
                    "temp_goal_suppressed_by_epoch": bool(
                        agent.last_override_info.get("temp_goal_suppressed_by_epoch")
                    ),
                    "stuck_override_suppressed": bool(
                        agent.last_override_info.get("stuck_override_suppressed")
                    ),
                    "executor_stuck_override_suppressed": bool(
                        agent.last_override_info.get("stuck_override_suppressed")
                    ),
                    "controller_stuck_replan_triggered": controller_stuck_replan_triggered,
                    "stuck_suppression_steps_remaining": (
                        controller_state.executor_stuck_suppression_steps
                    ),
                    "direction_reuse_count": controller_state.direction_reuse_count,
                    "forced_replan_due_to_direction_reuse": (
                        forced_replan_due_to_direction_reuse
                    ),
                    "action": agent.last_action,
                    "sensor_pose_delta": infos.get("sensor_pose"),
                    "done": done,
                },
            )

            if step % args.log_interval == 0:
                total_success = list(episode_success)
                total_spl = list(episode_spl)
                log = f"step {step}, ep {step_episode_id}"
                if total_spl:
                    log += f" | SR={np.mean(total_success):.3f} SPL={np.mean(total_spl):.3f}"
                if episode_smoothness:
                    avg_smooth = np.mean([s.smoothness_score for s in episode_smoothness])
                    avg_pauses = np.mean([s.pause_count for s in episode_smoothness])
                    log += f" | Smooth={avg_smooth:.3f} Pauses={avg_pauses:.1f}"
                if args.mode == "smoothnav":
                    log += f" | H={high_planner.call_count} L={low_agent.call_count}"
                print(log)
                logging.info(log)

            step += 1

        total_success = list(episode_success)
        total_spl = list(episode_spl)
        summary = {
            "run_id": args.run_id,
            "mode": args.mode,
            "controller_profile": args.controller_profile,
            "controller": controller_config_dict(args),
            "goal_type": args.goal_type,
            "num_episodes": len(total_success),
            "SR": float(np.mean(total_success)) if total_success else 0,
            "SPL": float(np.mean(total_spl)) if total_spl else 0,
        }

        summary["avg_high_level_calls"] = float(np.mean(
            [r["high_level_calls"] for r in episode_results]
        )) if episode_results else 0
        summary["avg_low_level_calls"] = float(np.mean(
            [r["low_level_calls"] for r in episode_results]
        )) if episode_results else 0
        terminal_outcome_counts = {}
        for result in episode_results:
            outcome = result.get("terminal_outcome", TerminalOutcome.RUNNING.value)
            terminal_outcome_counts[outcome] = terminal_outcome_counts.get(outcome, 0) + 1
        summary["terminal_outcomes"] = terminal_outcome_counts

        if episode_smoothness:
            summary.update(
                {
                    "avg_smoothness": float(np.mean([s.smoothness_score for s in episode_smoothness])),
                    "avg_sigma_v": float(np.mean([s.sigma_v for s in episode_smoothness])),
                    "avg_sigma_omega": float(np.mean([s.sigma_omega for s in episode_smoothness])),
                    "avg_jerk": float(np.mean([s.jerk for s in episode_smoothness])),
                    "avg_pause_count": float(np.mean([s.pause_count for s in episode_smoothness])),
                    "avg_pause_ratio": float(np.mean([s.pause_duration_ratio for s in episode_smoothness])),
                    "avg_direction_reversals": float(np.mean([s.direction_reversals for s in episode_smoothness])),
                }
            )

        summary.update(
            compute_run_control_metrics(
                args.run_dir,
                episode_ids=[result["episode"] for result in episode_results],
            )
        )

        log = f"\n{'=' * 60}\nFinal Results ({args.mode}):\n"
        for key, value in summary.items():
            log += f"  {key}: {value}\n"
        log += f"{'=' * 60}"
        print(log)
        logging.info(log)

        with open(args.episode_results_path, "w") as f:
            json.dump(episode_results, f, indent=2)
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        _analyze_actions(all_ep_action_logs, args)
    finally:
        tracer.close()


def _analyze_actions(all_ep_logs, args):
    grand_totals = {0: 0, 1: 0, 2: 0, 3: 0, None: 0}
    for ep_log in all_ep_logs:
        for step_entry in ep_log:
            action = step_entry["action"]
            grand_totals[action] = grand_totals.get(action, 0) + 1

    total_steps = sum(grand_totals.values())
    if total_steps == 0:
        return

    with open(args.action_analysis_path, "w") as f:
        json.dump(
            {
                "grand_totals": {str(k): v for k, v in grand_totals.items()},
                "total_steps": total_steps,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
