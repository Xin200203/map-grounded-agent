"""Controller helpers extracted from the SmoothNav main loop."""

import logging
from typing import Tuple

from smoothnav.graph_delta import (
    build_graph_delta,
    is_room_target,
    strategy_specificity,
)
from smoothnav.low_level_agent import LowLevelAction
from smoothnav.planner import serialize_for_planner

logger = logging.getLogger(__name__)


def plan_strategy(high_planner, graph, controller_state, goal_description: str,
                  escalate_reason: str, agent_pos: Tuple[int, int], map_size: int,
                  episode_id: int, step_idx: int, trace_writer=None,
                  mission_state=None, world_state=None):
    if mission_state is not None and world_state is not None and hasattr(
        high_planner, "plan_stage_goal"
    ):
        stage_goal = high_planner.plan_stage_goal(
            mission_state=mission_state,
            world_state=world_state,
            reason=escalate_reason,
            episode_id=episode_id,
            step_idx=step_idx,
            trace_writer=trace_writer,
        )
        controller_state.planner_call_count = high_planner.call_count
        return stage_goal

    scene_text = serialize_for_planner(graph, controller_state.explored_regions)
    strategy = high_planner.plan(
        scene_text=scene_text,
        goal_description=goal_description,
        explored_regions=controller_state.explored_regions,
        escalate_reason=escalate_reason,
        graph=graph,
        agent_pos=agent_pos,
        map_size=map_size,
        episode_id=episode_id,
        step_idx=step_idx,
        trace_writer=trace_writer,
    )
    controller_state.planner_call_count = high_planner.call_count
    return strategy


def maybe_call_monitor(low_agent, controller_state, graph_delta, graph,
                       episode_id: int, step_idx: int, trace_writer=None):
    if controller_state.current_strategy is None or controller_state.needs_initial_plan:
        return False, None, []
    trigger_event_types = []
    if graph_delta.has_new_nodes:
        trigger_event_types.append("new_nodes")
    if graph_delta.has_new_rooms:
        trigger_event_types.append("new_rooms")
    if graph_delta.has_caption_changes:
        trigger_event_types.append("node_caption_changed")
    if graph_delta.has_room_object_increase:
        trigger_event_types.append("room_object_count_increase")
    if graph_delta.frontier_near:
        trigger_event_types.append("frontier_near")
    if graph_delta.no_progress:
        trigger_event_types.append("no_progress")
    if graph_delta.stuck:
        trigger_event_types.append("stuck")
    if not trigger_event_types:
        return False, None, []
    if hasattr(low_agent, "should_evaluate") and not low_agent.should_evaluate(
        strategy=controller_state.current_strategy,
        graph_delta=graph_delta,
        no_progress_steps=controller_state.no_progress_steps,
        dist_to_goal=graph_delta.dist_to_goal,
    ):
        return False, None, trigger_event_types

    result = low_agent.evaluate(
        strategy=controller_state.current_strategy,
        new_nodes=graph_delta.new_nodes,
        dist_to_goal=graph_delta.dist_to_goal,
        total_nodes=graph_delta.graph_node_count,
        graph=graph,
        graph_delta=graph_delta,
        no_progress_steps=controller_state.no_progress_steps,
        episode_id=episode_id,
        step_idx=step_idx,
        trace_writer=trace_writer,
    )
    controller_state.monitor_call_count = low_agent.call_count
    return True, result, trigger_event_types


def maybe_promote_pending(controller_state, graph, bev_map, args, global_goals,
                          apply_strategy_fn):
    if controller_state.pending_strategy is None:
        return {"promoted": False, "reason": ""}
    if controller_state.current_strategy is None:
        return {"promoted": False, "reason": ""}
    current_specificity = strategy_specificity(
        controller_state.current_strategy.target_region
    )
    pending_specificity = strategy_specificity(
        controller_state.pending_strategy.target_region
    )
    if pending_specificity <= current_specificity:
        return {"promoted": False, "reason": ""}

    controller_state.current_strategy = controller_state.pending_strategy
    controller_state.pending_strategy = None
    controller_state.direction_reuse_count = 0
    apply_strategy_fn(controller_state.current_strategy, graph, bev_map, args, global_goals)
    logger.info(
        "Early promote pending -> %s",
        controller_state.current_strategy.target_region,
    )
    return {"promoted": True, "reason": "pending_more_specific"}


def handle_frontier_reached(controller_state, graph_delta, graph, bev_map, args,
                            global_goals, high_planner, goal_description,
                            agent_pos: Tuple[int, int], apply_strategy_fn,
                            episode_id: int, step_idx: int, trace_writer=None,
                            mission_state=None, world_state=None):
    if controller_state.needs_initial_plan or not graph_delta.frontier_reached:
        return {
            "handled": False,
            "pending_promoted": False,
            "pending_promotion_reason": "",
            "forced_replan_due_to_direction_reuse": False,
        }

    if controller_state.pending_strategy is not None:
        if controller_state.current_strategy and is_room_target(
            controller_state.current_strategy.target_region
        ):
            controller_state.explored_regions.append(
                controller_state.current_strategy.target_region
            )
        controller_state.current_strategy = controller_state.pending_strategy
        controller_state.pending_strategy = None
        controller_state.direction_reuse_count = 0
        logger.info(
            "Step %s: Apply pending -> %s",
            step_idx,
            controller_state.current_strategy.target_region,
        )
        outcome = {
            "handled": True,
            "pending_promoted": True,
            "pending_promotion_reason": "frontier_reached_pending",
            "forced_replan_due_to_direction_reuse": False,
        }
    elif controller_state.current_strategy and not is_room_target(
        controller_state.current_strategy.target_region
    ):
        controller_state.direction_reuse_count += 1
        reuse_limit = int(getattr(args, "controller_direction_reuse_limit", 1) or 1)
        if controller_state.direction_reuse_count <= reuse_limit:
            logger.info(
                "Step %s: Direction reuse (%s/%s) -> %s",
                step_idx,
                controller_state.direction_reuse_count,
                reuse_limit,
                controller_state.current_strategy.target_region,
            )
            outcome = {
                "handled": True,
                "pending_promoted": False,
                "pending_promotion_reason": "",
                "forced_replan_due_to_direction_reuse": False,
            }
        else:
            controller_state.direction_reuse_count = 0
            controller_state.current_strategy = plan_strategy(
                high_planner=high_planner,
                graph=graph,
                controller_state=controller_state,
                goal_description=goal_description,
                escalate_reason="Direction reuse limit reached, need new target",
                agent_pos=agent_pos,
                map_size=args.map_size,
                episode_id=episode_id,
                step_idx=step_idx,
                trace_writer=trace_writer,
                mission_state=mission_state,
                world_state=world_state,
            )
            logger.info(
                "Step %s: Direction reuse limit reached -> %s",
                step_idx,
                controller_state.current_strategy.target_region,
            )
            outcome = {
                "handled": True,
                "pending_promoted": False,
                "pending_promotion_reason": "",
                "forced_replan_due_to_direction_reuse": True,
            }
    else:
        if controller_state.current_strategy and is_room_target(
            controller_state.current_strategy.target_region
        ):
            controller_state.explored_regions.append(
                controller_state.current_strategy.target_region
            )
        controller_state.current_strategy = plan_strategy(
            high_planner=high_planner,
            graph=graph,
            controller_state=controller_state,
            goal_description=goal_description,
            escalate_reason="Frontier reached, need new target",
            agent_pos=agent_pos,
            map_size=args.map_size,
            episode_id=episode_id,
            step_idx=step_idx,
            trace_writer=trace_writer,
            mission_state=mission_state,
            world_state=world_state,
        )
        controller_state.direction_reuse_count = 0
        logger.info(
            "Step %s: Frontier reached -> %s",
            step_idx,
            controller_state.current_strategy.target_region,
        )
        outcome = {
            "handled": True,
            "pending_promoted": False,
            "pending_promotion_reason": "",
            "forced_replan_due_to_direction_reuse": False,
        }

    apply_strategy_fn(controller_state.current_strategy, graph, bev_map, args, global_goals)
    return outcome


def handle_stuck_replan(controller_state, graph_delta, graph, bev_map, args,
                        global_goals, high_planner, goal_description,
                        agent_pos: Tuple[int, int], apply_strategy_fn,
                        episode_id: int, step_idx: int, trace_writer=None,
                        mission_state=None, world_state=None):
    if controller_state.needs_initial_plan or not graph_delta.stuck:
        return False

    controller_state.no_progress_steps = 0
    if controller_state.current_strategy and is_room_target(
        controller_state.current_strategy.target_region
    ):
        controller_state.explored_regions.append(
            f"{controller_state.current_strategy.target_region} (stuck)"
        )
    controller_state.current_strategy = plan_strategy(
        high_planner=high_planner,
        graph=graph,
        controller_state=controller_state,
        goal_description=goal_description,
        escalate_reason="Agent stuck, need alternative route",
        agent_pos=agent_pos,
        map_size=args.map_size,
        episode_id=episode_id,
        step_idx=step_idx,
        trace_writer=trace_writer,
        mission_state=mission_state,
        world_state=world_state,
    )
    controller_state.pending_strategy = None
    controller_state.direction_reuse_count = 0
    apply_strategy_fn(controller_state.current_strategy, graph, bev_map, args, global_goals)
    logger.info(
        "Step %s: STUCK -> %s",
        step_idx,
        controller_state.current_strategy.target_region,
    )
    return True


def update_grounding_failure_state(controller_state, grounding_result):
    selected_frontier = grounding_result.selected_frontier
    selected_frontier_list = (
        list(selected_frontier) if selected_frontier is not None else None
    )

    if not grounding_result.changed:
        controller_state.consecutive_grounding_noops += 1
    else:
        controller_state.consecutive_grounding_noops = 0

    if selected_frontier_list is None:
        controller_state.same_frontier_reuse_count = 0
    elif controller_state.last_grounding_selected_frontier == selected_frontier_list:
        controller_state.same_frontier_reuse_count += 1
    else:
        controller_state.same_frontier_reuse_count = 1

    controller_state.last_grounding_selected_frontier = selected_frontier_list


def handle_grounding_failure(controller_state, last_grounding_result, graph, bev_map,
                             args, global_goals, high_planner,
                             goal_description, agent_pos: Tuple[int, int],
                             apply_strategy_fn, episode_id: int, step_idx: int,
                             trace_writer=None, mission_state=None, world_state=None):
    if controller_state.needs_initial_plan or controller_state.current_strategy is None:
        return {
            "replanned": False,
            "forced_replan_due_to_grounding_failure": False,
            "grounding_failure_reason": "",
        }
    if last_grounding_result is None or last_grounding_result.changed:
        return {
            "replanned": False,
            "forced_replan_due_to_grounding_failure": False,
            "grounding_failure_reason": "",
        }

    noop_threshold = int(
        getattr(args, "controller_grounding_noop_replan_threshold", 2) or 2
    )
    same_frontier_threshold = int(
        getattr(args, "controller_same_frontier_reuse_threshold", 2) or 2
    )
    should_replan = (
        controller_state.consecutive_grounding_noops >= noop_threshold
        or controller_state.same_frontier_reuse_count >= same_frontier_threshold
    )
    if not should_replan:
        return {
            "replanned": False,
            "forced_replan_due_to_grounding_failure": False,
            "grounding_failure_reason": "",
        }

    controller_state.current_strategy = plan_strategy(
        high_planner=high_planner,
        graph=graph,
        controller_state=controller_state,
        goal_description=goal_description,
        escalate_reason=(
            "Grounding failure replan: "
            f"noop={last_grounding_result.noop_reason or last_grounding_result.reason}, "
            f"same_frontier_reuse={controller_state.same_frontier_reuse_count}, "
            f"consecutive_noops={controller_state.consecutive_grounding_noops}"
        ),
        agent_pos=agent_pos,
        map_size=args.map_size,
        episode_id=episode_id,
        step_idx=step_idx,
        trace_writer=trace_writer,
        mission_state=mission_state,
        world_state=world_state,
    )
    controller_state.pending_strategy = None
    controller_state.direction_reuse_count = 0
    apply_strategy_fn(controller_state.current_strategy, graph, bev_map, args, global_goals)
    logger.info(
        "Step %s: grounding failure replan -> %s",
        step_idx,
        controller_state.current_strategy.target_region,
    )
    return {
        "replanned": True,
        "forced_replan_due_to_grounding_failure": True,
        "grounding_failure_reason": (
            last_grounding_result.noop_reason or last_grounding_result.reason or ""
        ),
    }
