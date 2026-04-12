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

from smoothnav.controller_logic import (
    build_graph_delta,
    handle_frontier_reached,
    handle_stuck_replan,
    is_room_target,
    maybe_call_monitor,
    maybe_promote_pending,
    plan_strategy,
)
from smoothnav.controller_state import ControllerState
from smoothnav.experiment_io import resolve_api_config, setup_run_environment
from smoothnav.low_level_agent import (
    LOW_LEVEL_PROMPT_SCHEMA_VERSION,
    LowLevelAction,
    LowLevelAgent,
)
from smoothnav.metrics import SmoothnessMetrics
from smoothnav.planner import PLANNER_PROMPT_SCHEMA_VERSION, HighLevelPlanner
from smoothnav.tracing import RunTracer, strategy_to_dict


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="base_UniGoal/configs/config_habitat.yaml",
                        metavar="FILE", type=str)
    parser.add_argument("--goal_type", default="text", type=str)
    parser.add_argument("--episode_id", default=-1, type=int)
    parser.add_argument("--goal", default="", type=str)
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--mode", default="smoothnav", choices=["smoothnav", "baseline"],
                        help="smoothnav: three-tier hierarchical; baseline: per-interval (UniGoal)")
    parser.add_argument("--stuck_threshold", default=15, type=int)
    parser.add_argument("--num_eval", default=0, type=int,
                        help="override num_eval_episodes (0=use config)")
    parser.add_argument("--results-root", default="", type=str,
                        help="optional override for results root directory")
    parsed_args = parser.parse_args()

    with open(parsed_args.config_file, "r") as file:
        config = yaml.safe_load(file)
    args_dict = dict(config)
    for key, value in vars(parsed_args).items():
        if key == "results_root":
            if value:
                args_dict[key] = value
            elif key not in args_dict:
                args_dict[key] = value
        else:
            args_dict[key] = value
    args = SimpleNamespace(**args_dict)

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

    args = resolve_api_config(args)
    args = setup_run_environment(
        args,
        argv=sys.argv,
        prompt_versions={
            "planner": PLANNER_PROMPT_SCHEMA_VERSION,
            "monitor": LOW_LEVEL_PROMPT_SCHEMA_VERSION,
        },
    )
    return args


def _apply_strategy(strategy, graph, bev_map, args, global_goals):
    graph.set_full_map(bev_map.full_map)
    graph.set_full_pose(bev_map.full_pose)

    bias = strategy.bias_position if strategy else None
    goal = graph.get_goal(goal=bias)

    if goal is not None:
        goal = list(goal)
        goal[0] -= bev_map.local_map_boundary[0, 0]
        goal[1] -= bev_map.local_map_boundary[0, 2]
        if 0 <= goal[0] < args.local_width and 0 <= goal[1] < args.local_height:
            global_goals[0] = goal[0]
            global_goals[1] = goal[1]


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
    tracer = RunTracer(args.run_dir)
    _configure_logging(args)
    logging.info("SmoothNav starting: mode=%s run_id=%s", args.mode, args.run_id)
    logging.info(
        "Run context: goal_type=%s num_eval=%s results=%s",
        args.goal_type,
        args.num_eval_episodes,
        args.run_dir,
    )

    bev_map = BEV_Map(args)
    graph = Graph(args)
    envs = construct_envs(args)
    agent = UniGoal_Agent(args, envs)

    llm_sonnet = LLM(args.base_url, args.api_key, args.llm_model)
    llm_haiku = LLM(args.base_url, args.api_key, args.llm_model_fast)
    high_planner = HighLevelPlanner(llm_fn=llm_sonnet)
    low_agent = LowLevelAgent(llm_fn=llm_haiku)
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
    smoothness.reset()
    active_episode_id = int(infos.get("episode_no", 0))
    step = 0

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
                        "success": success,
                        "spl": spl,
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

                wait_env = True
                bev_map.update_intrinsic_rew()
                bev_map.init_map_and_pose_for_env()
                smoothness.reset()
                high_planner.reset()
                low_agent.reset()
                controller_state = ControllerState()
                _reset_graph_goal(args, graph, infos)
                goal_description = _goal_description_from_infos(args, infos)
                active_episode_id = int(infos.get("episode_no", completed_episode_id + 1))

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
            monitor_decision = None
            monitor_reason = ""
            low_level_action = None
            graph_delta = None
            step_episode_id = active_episode_id

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
                    controller_state.current_strategy = plan_strategy(
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
                    _apply_strategy(controller_state.current_strategy, graph, bev_map, args, global_goals)
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

                if graph_delta.has_new_rooms:
                    if (
                        controller_state.current_strategy
                        and not is_room_target(controller_state.current_strategy.target_region)
                        and not controller_state.needs_initial_plan
                    ):
                        controller_state.current_strategy = plan_strategy(
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
                        )
                        _apply_strategy(controller_state.current_strategy, graph, bev_map, args, global_goals)
                        is_planning = True
                        logging.info(
                            "Step %s: New room -> %s",
                            step,
                            controller_state.current_strategy.target_region,
                        )

                monitor_called, low_result = maybe_call_monitor(
                    low_agent=low_agent,
                    controller_state=controller_state,
                    graph_delta=graph_delta,
                    graph=graph,
                    episode_id=step_episode_id,
                    step_idx=step,
                    trace_writer=tracer,
                )
                if monitor_called:
                    controller_state.prev_node_count = len(graph.nodes)
                    low_level_action = low_result.action
                    monitor_decision = low_result.action.name
                    monitor_reason = low_result.reason
                    if low_result.action == LowLevelAction.ADJUST:
                        if low_result.adjust_bias is not None:
                            controller_state.current_strategy.bias_position = low_result.adjust_bias
                            _apply_strategy(controller_state.current_strategy, graph, bev_map, args, global_goals)
                            logging.info(
                                "Step %s: ADJUST bias -> %s reason=%s",
                                step,
                                low_result.adjust_bias,
                                low_result.reason,
                            )
                    elif low_result.action == LowLevelAction.PREFETCH:
                        if controller_state.pending_strategy is None:
                            controller_state.pending_strategy = plan_strategy(
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
                            )
                            is_planning = True
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
                        controller_state.current_strategy = plan_strategy(
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
                        )
                        controller_state.pending_strategy = None
                        _apply_strategy(controller_state.current_strategy, graph, bev_map, args, global_goals)
                        is_planning = True
                        logging.info(
                            "Step %s: ESCALATE -> %s",
                            step,
                            controller_state.current_strategy.target_region,
                        )

                if maybe_promote_pending(
                    controller_state=controller_state,
                    graph=graph,
                    bev_map=bev_map,
                    args=args,
                    global_goals=global_goals,
                    apply_strategy_fn=_apply_strategy,
                ):
                    pass

                if handle_frontier_reached(
                    controller_state=controller_state,
                    graph_delta=graph_delta,
                    graph=graph,
                    bev_map=bev_map,
                    args=args,
                    global_goals=global_goals,
                    high_planner=high_planner,
                    goal_description=goal_description,
                    agent_pos=(int(agent_map_x), int(agent_map_y)),
                    apply_strategy_fn=_apply_strategy,
                    episode_id=step_episode_id,
                    step_idx=step,
                    trace_writer=tracer,
                ):
                    is_planning = True

                if (
                    dist_to_goal < 10
                    and controller_state.pending_strategy is None
                    and low_level_action != LowLevelAction.PREFETCH
                    and not frontier_reached
                    and not controller_state.needs_initial_plan
                    and controller_state.current_strategy
                    and is_room_target(controller_state.current_strategy.target_region)
                ):
                    controller_state.pending_strategy = plan_strategy(
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
                    )
                    is_planning = True
                    logging.info(
                        "Step %s: Auto-PREFETCH -> %s",
                        step,
                        controller_state.pending_strategy.target_region,
                    )

                if handle_stuck_replan(
                    controller_state=controller_state,
                    graph_delta=graph_delta,
                    graph=graph,
                    bev_map=bev_map,
                    args=args,
                    global_goals=global_goals,
                    high_planner=high_planner,
                    goal_description=goal_description,
                    agent_pos=(int(agent_map_x), int(agent_map_y)),
                    apply_strategy_fn=_apply_strategy,
                    episode_id=step_episode_id,
                    step_idx=step,
                    trace_writer=tracer,
                ):
                    is_planning = True

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
                controller_state.prev_node_count = len(graph.nodes)

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
            goal_after = list(global_goals)
            controller_state.last_goal = list(goal_after)

            goal_maps = np.zeros((args.local_width, args.local_height))
            gx = int(np.clip(global_goals[0], 0, args.local_width - 1))
            gy = int(np.clip(global_goals[1], 0, args.local_height - 1))
            goal_maps[gx, gy] = 1

            agent_input = {
                "map_pred": bev_map.local_map[0, 0, :, :].cpu().numpy(),
                "exp_pred": bev_map.local_map[0, 1, :, :].cpu().numpy(),
                "pose_pred": bev_map.planner_pose_inputs[0],
                "goal": goal_maps,
                "exp_goal": goal_maps.copy(),
                "new_goal": 0,
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

            tracer.record_step(
                step_episode_id,
                {
                    "episode_id": step_episode_id,
                    "step_idx": step,
                    "mode": args.mode,
                    "pose_before": pose_before,
                    "pose_after": pose_after,
                    "graph_node_count": getattr(graph_delta, "graph_node_count", len(graph.nodes)),
                    "new_node_count": len(getattr(graph_delta, "new_nodes", [])),
                    "new_node_captions": getattr(graph_delta, "new_node_captions", []),
                    "graph_delta": {
                        "new_rooms": getattr(graph_delta, "new_rooms", []),
                        "room_object_count_changes": getattr(graph_delta, "room_object_count_changes", {}),
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
                    "monitor_called": monitor_called,
                    "monitor_calls_this_step": low_agent.call_count - monitor_calls_before,
                    "monitor_decision": monitor_decision,
                    "monitor_reason": monitor_reason,
                    "goal_before": goal_before,
                    "goal_after": goal_after,
                    "visible_target_override": bool(agent.last_override_info.get("visible_target_override")),
                    "temp_goal_override": bool(agent.last_override_info.get("temp_goal_override")),
                    "stuck_goal_override": bool(agent.last_override_info.get("stuck_goal_override")),
                    "global_goal_override": bool(agent.last_override_info.get("global_goal_override")),
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
            "goal_type": args.goal_type,
            "num_episodes": len(total_success),
            "SR": float(np.mean(total_success)) if total_success else 0,
            "SPL": float(np.mean(total_spl)) if total_spl else 0,
        }

        if args.mode == "smoothnav":
            summary["avg_high_level_calls"] = float(np.mean(
                [r["high_level_calls"] for r in episode_results]
            )) if episode_results else 0
            summary["avg_low_level_calls"] = float(np.mean(
                [r["low_level_calls"] for r in episode_results]
            )) if episode_results else 0

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
