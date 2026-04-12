"""Controller helpers extracted from the SmoothNav main loop."""

import logging
from typing import Dict, List, Tuple

from smoothnav.controller_events import GraphDelta
from smoothnav.low_level_agent import LowLevelAction
from smoothnav.planner import serialize_for_planner

logger = logging.getLogger(__name__)


def is_room_target(target_region: str) -> bool:
    if not target_region:
        return False
    return (not target_region.startswith("unexplored")
            and not target_region.startswith("object:"))


def build_room_object_counts(graph) -> Dict[str, int]:
    counts = {}
    for room_node in getattr(graph, "room_nodes", []):
        count = len(getattr(room_node, "nodes", []))
        if count > 0:
            counts[room_node.caption] = count
    return counts


def build_graph_delta(graph, controller_state, frontier_near: bool,
                      frontier_reached: bool, no_progress: bool,
                      stuck: bool, dist_to_goal: float) -> GraphDelta:
    new_nodes = graph.nodes[controller_state.prev_node_count:]
    new_node_captions = [
        node.caption for node in new_nodes
        if hasattr(node, "caption") and node.caption
    ]

    room_counts = build_room_object_counts(graph)
    room_count_changes = {}
    new_rooms = []
    all_rooms = set(room_counts) | set(controller_state.prev_room_object_counts)
    for room_name in sorted(all_rooms):
        before = controller_state.prev_room_object_counts.get(room_name, 0)
        after = room_counts.get(room_name, 0)
        if before != after:
            room_count_changes[room_name] = {"before": before, "after": after}
        if before == 0 and after > 0:
            new_rooms.append(room_name)

    return GraphDelta(
        new_nodes=new_nodes,
        new_node_captions=new_node_captions,
        new_rooms=new_rooms,
        room_object_count_changes=room_count_changes,
        frontier_near=frontier_near,
        frontier_reached=frontier_reached,
        no_progress=no_progress,
        stuck=stuck,
        dist_to_goal=float(dist_to_goal),
        graph_node_count=len(graph.nodes),
        room_object_counts=room_counts,
    )


def plan_strategy(high_planner, graph, controller_state, goal_description: str,
                  escalate_reason: str, agent_pos: Tuple[int, int], map_size: int,
                  episode_id: int, step_idx: int, trace_writer=None):
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
        return False, None
    if not graph_delta.has_new_nodes:
        return False, None

    result = low_agent.evaluate(
        strategy=controller_state.current_strategy,
        new_nodes=graph_delta.new_nodes,
        dist_to_goal=graph_delta.dist_to_goal,
        total_nodes=graph_delta.graph_node_count,
        graph=graph,
        episode_id=episode_id,
        step_idx=step_idx,
        trace_writer=trace_writer,
    )
    controller_state.monitor_call_count = low_agent.call_count
    return True, result


def maybe_promote_pending(controller_state, graph, bev_map, args, global_goals,
                          apply_strategy_fn):
    if controller_state.pending_strategy is None:
        return False
    if controller_state.current_strategy is None:
        return False
    if is_room_target(controller_state.current_strategy.target_region):
        return False
    if not is_room_target(controller_state.pending_strategy.target_region):
        return False

    controller_state.current_strategy = controller_state.pending_strategy
    controller_state.pending_strategy = None
    apply_strategy_fn(controller_state.current_strategy, graph, bev_map, args, global_goals)
    logger.info(
        "Early promote pending -> %s",
        controller_state.current_strategy.target_region,
    )
    return True


def handle_frontier_reached(controller_state, graph_delta, graph, bev_map, args,
                            global_goals, high_planner, goal_description,
                            agent_pos: Tuple[int, int], apply_strategy_fn,
                            episode_id: int, step_idx: int, trace_writer=None):
    if controller_state.needs_initial_plan or not graph_delta.frontier_reached:
        return False

    if controller_state.pending_strategy is not None:
        if controller_state.current_strategy and is_room_target(
            controller_state.current_strategy.target_region
        ):
            controller_state.explored_regions.append(
                controller_state.current_strategy.target_region
            )
        controller_state.current_strategy = controller_state.pending_strategy
        controller_state.pending_strategy = None
        logger.info(
            "Step %s: Apply pending -> %s",
            step_idx,
            controller_state.current_strategy.target_region,
        )
    elif controller_state.current_strategy and not is_room_target(
        controller_state.current_strategy.target_region
    ):
        logger.info(
            "Step %s: Direction reuse -> %s",
            step_idx,
            controller_state.current_strategy.target_region,
        )
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
        )
        logger.info(
            "Step %s: Frontier reached -> %s",
            step_idx,
            controller_state.current_strategy.target_region,
        )

    apply_strategy_fn(controller_state.current_strategy, graph, bev_map, args, global_goals)
    return True


def handle_stuck_replan(controller_state, graph_delta, graph, bev_map, args,
                        global_goals, high_planner, goal_description,
                        agent_pos: Tuple[int, int], apply_strategy_fn,
                        episode_id: int, step_idx: int, trace_writer=None):
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
    )
    controller_state.pending_strategy = None
    apply_strategy_fn(controller_state.current_strategy, graph, bev_map, args, global_goals)
    logger.info(
        "Step %s: STUCK -> %s",
        step_idx,
        controller_state.current_strategy.target_region,
    )
    return True
