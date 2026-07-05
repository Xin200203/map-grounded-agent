"""Controller helpers extracted from the SmoothNav main loop."""

import logging
import math
from typing import Tuple

from smoothnav.graph_delta import (
    build_graph_delta,
    is_object_target,
    is_target_search_anchor,
    is_room_target,
    strategy_specificity,
)
from smoothnav.low_level_agent import LowLevelAction
from smoothnav.planner import serialize_for_planner

logger = logging.getLogger(__name__)
TARGET_ANCHOR_UPDATE_TRIGGERS = {
    "target_candidate_detected",
    "target_anchor_local_map_refresh",
    "frontier_reached",
}


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
            explored_regions_override=list(controller_state.explored_regions),
            current_target_region=(
                getattr(controller_state.current_strategy, "target_region", "")
                if controller_state.current_strategy is not None
                else ""
            ),
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
        current_target_region=(
            getattr(controller_state.current_strategy, "target_region", "")
            if controller_state.current_strategy is not None
            else ""
        ),
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
    if getattr(graph_delta, "has_target_candidates", False):
        trigger_event_types.append("target_candidate_detected")
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


def empty_response_plan_is_more_specific(candidate_strategy, fallback_strategy) -> bool:
    """Return whether an empty-response fallback result should replace current plan.

    Empty LLM responses often produce generic deterministic direction fallbacks,
    which should not churn an already active plan. A target-aware/object fallback,
    however, is a deliberate semantic upgrade and must be allowed through.
    """

    candidate_region = getattr(candidate_strategy, "target_region", "") or ""
    fallback_region = getattr(fallback_strategy, "target_region", "") or ""
    return strategy_specificity(candidate_region) > strategy_specificity(fallback_region)


def reset_target_anchor_attempt(controller_state):
    if (
        controller_state.target_anchor_target_region
        and not controller_state.target_anchor_decommit_reason
    ):
        controller_state.target_anchor_decommit_reason = "reset"
    controller_state.target_anchor_target_region = ""
    controller_state.target_anchor_best_distance = None
    controller_state.target_anchor_last_distance = None
    controller_state.target_anchor_stall_updates = 0
    controller_state.target_anchor_update_count = 0
    controller_state.target_anchor_last_update_step = None
    controller_state.target_anchor_last_improvement_step = None
    controller_state.target_anchor_last_trigger = ""
    controller_state.target_anchor_label = ""
    controller_state.target_anchor_full_map_coord = None
    controller_state.target_anchor_last_projected_local_coord = None
    controller_state.target_anchor_last_value_source = ""
    controller_state.target_anchor_last_seen_step = None


def target_anchor_attempt_summary(controller_state):
    return {
        "active": bool(controller_state.target_anchor_target_region),
        "attempt_id": int(controller_state.target_anchor_attempt_id),
        "target_region": controller_state.target_anchor_target_region,
        "best_distance": controller_state.target_anchor_best_distance,
        "last_distance": controller_state.target_anchor_last_distance,
        "stall_updates": int(controller_state.target_anchor_stall_updates),
        "update_count": int(controller_state.target_anchor_update_count),
        "last_update_step": controller_state.target_anchor_last_update_step,
        "last_improvement_step": controller_state.target_anchor_last_improvement_step,
        "last_trigger": controller_state.target_anchor_last_trigger,
        "target_label": controller_state.target_anchor_label,
        "full_map_coord": controller_state.target_anchor_full_map_coord,
        "last_projected_local_coord": (
            controller_state.target_anchor_last_projected_local_coord
        ),
        "last_value_source": controller_state.target_anchor_last_value_source,
        "last_seen_step": controller_state.target_anchor_last_seen_step,
        "decommit_reason": controller_state.target_anchor_decommit_reason,
    }


def _target_anchor_distance(grounding_result):
    bias = getattr(grounding_result, "bias_input", None)
    frontier = getattr(grounding_result, "selected_frontier", None)
    if bias is None or frontier is None:
        return None
    try:
        return float(
            math.dist(
                [float(bias[0]), float(bias[1])],
                [float(frontier[0]), float(frontier[1])],
            )
        )
    except Exception:
        return None


def update_target_anchor_attempt(controller_state, strategy, grounding_result,
                                 trigger: str, step_idx: int, args):
    target_region = getattr(strategy, "target_region", "") if strategy is not None else ""
    if not is_target_search_anchor(target_region):
        if controller_state.target_anchor_target_region:
            controller_state.target_anchor_decommit_reason = (
                f"switched_to:{target_region}" if target_region else "switched_away"
            )
            reset_target_anchor_attempt(controller_state)
        return target_anchor_attempt_summary(controller_state)

    label = str(getattr(strategy, "anchor_object", "") or "")
    if not label and ":" in str(target_region):
        label = str(target_region).split(":", 1)[1].strip()
    bias = getattr(grounding_result, "bias_input", None)
    projected = getattr(grounding_result, "projected_goal", None)
    graph_debug = getattr(grounding_result, "graph_debug", {}) or {}

    if str(trigger or "") not in TARGET_ANCHOR_UPDATE_TRIGGERS:
        if controller_state.target_anchor_target_region != str(target_region):
            controller_state.target_anchor_attempt_id += 1
            controller_state.target_anchor_target_region = str(target_region)
            controller_state.target_anchor_last_trigger = str(trigger or "")
            controller_state.target_anchor_label = label
            controller_state.target_anchor_decommit_reason = ""
        return target_anchor_attempt_summary(controller_state)

    if controller_state.target_anchor_target_region != str(target_region):
        controller_state.target_anchor_attempt_id += 1
        reset_target_anchor_attempt(controller_state)
        controller_state.target_anchor_target_region = str(target_region)
        controller_state.target_anchor_label = label
        controller_state.target_anchor_decommit_reason = ""

    controller_state.target_anchor_label = label
    controller_state.target_anchor_last_seen_step = int(step_idx)
    if bias is not None:
        try:
            controller_state.target_anchor_full_map_coord = [
                int(bias[0]),
                int(bias[1]),
            ]
        except Exception:
            pass
    if projected is not None:
        try:
            controller_state.target_anchor_last_projected_local_coord = [
                int(projected[0]),
                int(projected[1]),
            ]
        except Exception:
            pass
    if graph_debug:
        if graph_debug.get("target_progress_active"):
            controller_state.target_anchor_last_value_source = "target_progress"
        elif graph_debug.get("object_anchor_goal") is not None:
            controller_state.target_anchor_last_value_source = "object_anchor"
        elif graph_debug.get("direct_object_goal") is not None:
            controller_state.target_anchor_last_value_source = "direct_object"

    distance = _target_anchor_distance(grounding_result)
    controller_state.target_anchor_last_update_step = int(step_idx)
    controller_state.target_anchor_last_trigger = str(trigger or "")
    if distance is None:
        return target_anchor_attempt_summary(controller_state)

    controller_state.target_anchor_update_count += 1
    controller_state.target_anchor_last_distance = float(distance)
    min_improvement = float(
        getattr(args, "controller_target_anchor_min_improvement", 1.0) or 1.0
    )

    if controller_state.target_anchor_best_distance is None:
        controller_state.target_anchor_best_distance = float(distance)
        controller_state.target_anchor_last_improvement_step = int(step_idx)
        controller_state.target_anchor_stall_updates = 0
        return target_anchor_attempt_summary(controller_state)

    improvement = (
        float(controller_state.target_anchor_best_distance) - float(distance)
    )
    if improvement >= min_improvement:
        controller_state.target_anchor_best_distance = float(distance)
        controller_state.target_anchor_last_improvement_step = int(step_idx)
        controller_state.target_anchor_stall_updates = 0
    elif str(trigger or "") == "target_candidate_detected":
        controller_state.target_anchor_stall_updates = 0
    else:
        controller_state.target_anchor_stall_updates += 1
    return target_anchor_attempt_summary(controller_state)


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

    pending_handled = False
    if controller_state.pending_strategy is not None:
        current_target_region = (
            controller_state.current_strategy.target_region
            if controller_state.current_strategy is not None
            else ""
        )
        pending_target_region = (
            controller_state.pending_strategy.target_region
            if controller_state.pending_strategy is not None
            else ""
        )
        current_specificity = strategy_specificity(current_target_region)
        pending_specificity = strategy_specificity(pending_target_region)
        if is_target_search_anchor(current_target_region) and pending_specificity <= current_specificity:
            logger.info(
                "Step %s: Discard stale pending behind target anchor -> %s",
                step_idx,
                pending_target_region,
            )
            controller_state.pending_strategy = None
        else:
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
            pending_handled = True
    if pending_handled:
        pass
    elif (
        controller_state.pending_strategy is None
        and controller_state.current_strategy
        and is_target_search_anchor(controller_state.current_strategy.target_region)
    ):
        hold_limit = int(
            getattr(args, "controller_target_anchor_stall_patience_updates", 4) or 4
        )
        if controller_state.target_anchor_stall_updates >= hold_limit:
            controller_state.explored_regions.append(
                f"{controller_state.current_strategy.target_region} (stalled)"
            )
            reset_target_anchor_attempt(controller_state)
            controller_state.current_strategy = plan_strategy(
                high_planner=high_planner,
                graph=graph,
                controller_state=controller_state,
                goal_description=goal_description,
                escalate_reason="Target anchor stalled after repeated non-improving updates, need alternative target",
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
                "Step %s: Target-anchor stall replan -> %s",
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
            controller_state.direction_reuse_count = 0
            logger.info(
                "Step %s: Target-anchor hold on frontier (%s/%s) -> %s",
                step_idx,
                controller_state.target_anchor_stall_updates,
                hold_limit,
                controller_state.current_strategy.target_region,
            )
            outcome = {
                "handled": True,
                "pending_promoted": False,
                "pending_promotion_reason": "",
                "forced_replan_due_to_direction_reuse": False,
            }
    elif controller_state.pending_strategy is None and controller_state.current_strategy and is_object_target(
        controller_state.current_strategy.target_region
    ):
        hold_limit = int(
            getattr(args, "controller_object_same_goal_replan_threshold", 3)
            or getattr(args, "controller_same_frontier_reuse_threshold", 2)
            or 3
        )
        if controller_state.same_goal_hold_count >= hold_limit:
            controller_state.explored_regions.append(
                f"{controller_state.current_strategy.target_region} (stagnant)"
            )
            controller_state.current_strategy = plan_strategy(
                high_planner=high_planner,
                graph=graph,
                controller_state=controller_state,
                goal_description=goal_description,
                escalate_reason="Object target stagnated at same local goal, need alternative target",
                agent_pos=agent_pos,
                map_size=args.map_size,
                episode_id=episode_id,
                step_idx=step_idx,
                trace_writer=trace_writer,
                mission_state=mission_state,
                world_state=world_state,
            )
            controller_state.same_goal_hold_count = 0
            logger.info(
                "Step %s: Object stagnation replan -> %s",
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
            controller_state.direction_reuse_count = 0
            logger.info(
                "Step %s: Object target hold on frontier (%s/%s) -> %s",
                step_idx,
                controller_state.same_goal_hold_count,
                hold_limit,
                controller_state.current_strategy.target_region,
            )
            outcome = {
                "handled": True,
                "pending_promoted": False,
                "pending_promotion_reason": "",
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
    current_target = getattr(controller_state.current_strategy, "target_region", "")
    if str(current_target).startswith("object:"):
        logger.info(
            "Step %s: object target stuck; decommit and request alternative -> %s",
            step_idx,
            current_target,
        )
        controller_state.explored_regions.append(f"{current_target} (stuck)")
        controller_state.pending_strategy = None

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
    noop_reason = getattr(grounding_result, "noop_reason", None) or getattr(
        grounding_result, "reason", ""
    )
    if noop_reason == "same_goal_as_prev":
        controller_state.same_goal_hold_count += 1
    else:
        controller_state.same_goal_hold_count = 0

    if not grounding_result.changed and noop_reason != "same_goal_as_prev":
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


def handle_out_of_local_window(controller_state, last_grounding_result, graph, bev_map,
                               args, global_goals, apply_strategy_fn,
                               tactical_arbiter=None):
    if controller_state.needs_initial_plan or controller_state.current_strategy is None:
        return {
            "handled": False,
            "resolved": False,
            "deferred": False,
            "moved_local_map": False,
            "retried": False,
            "stale_local_goal_invalidated": False,
            "target_anchor_reprojected": False,
            "target_anchor_reprojection_failed": False,
            "grounding_patch_reason": "",
            "grounding_patch_mode": "",
        }

    failure_code = getattr(last_grounding_result, "failure_code", None) or getattr(
        last_grounding_result, "graph_no_goal_reason", ""
    ) or getattr(last_grounding_result, "noop_type", "")
    if failure_code != "out_of_local_window":
        return {
            "handled": False,
            "resolved": False,
            "deferred": False,
            "moved_local_map": False,
            "retried": False,
            "stale_local_goal_invalidated": False,
            "target_anchor_reprojected": False,
            "target_anchor_reprojection_failed": False,
            "grounding_patch_reason": "",
            "grounding_patch_mode": "",
        }

    patch_decision = (
        tactical_arbiter.post_grounding_patch(last_grounding_result)
        if tactical_arbiter is not None
        else None
    )
    patch_reason = getattr(patch_decision, "reason", "") if patch_decision else ""
    patch_mode = getattr(getattr(patch_decision, "mode", None), "value", "") if patch_decision else ""

    primary_goal = getattr(last_grounding_result, "primary_goal", None) or {}
    full_goal_coord = None
    if isinstance(primary_goal, dict):
        full_goal_coord = primary_goal.get("full_map_coord")
    if full_goal_coord is None and getattr(last_grounding_result, "selected_frontier", None) is not None:
        full_goal_coord = list(last_grounding_result.selected_frontier)

    moved_local_map = False
    retry_result = None
    target_anchor_active = is_target_search_anchor(
        getattr(controller_state.current_strategy, "target_region", "")
    )
    stale_local_goal_invalidated = False
    if hasattr(bev_map, "move_local_map"):
        bev_map.move_local_map()
        moved_local_map = True
        stale_local_goal_invalidated = bool(target_anchor_active)
        if stale_local_goal_invalidated:
            controller_state.last_goal = None
        retry_result = apply_strategy_fn(
            controller_state.current_strategy,
            graph,
            bev_map,
            args,
            global_goals,
        )

    if retry_result is not None and getattr(retry_result, "local_projection_valid", False):
        controller_state.grounding_deferred = False
        controller_state.grounding_deferred_reason = ""
        controller_state.grounding_deferred_full_goal = None
        logger.info(
            "Grounding patch resolved out_of_local_window after local-map move -> %s",
            controller_state.current_strategy.target_region,
        )
        return {
            "handled": True,
            "resolved": True,
            "deferred": False,
            "moved_local_map": moved_local_map,
            "retried": retry_result is not None,
            "stale_local_goal_invalidated": stale_local_goal_invalidated,
            "target_anchor_reprojected": bool(target_anchor_active),
            "target_anchor_reprojection_failed": False,
            "grounding_patch_reason": patch_reason,
            "grounding_patch_mode": patch_mode,
        }

    controller_state.grounding_deferred = True
    controller_state.grounding_deferred_reason = failure_code
    controller_state.grounding_deferred_full_goal = (
        list(full_goal_coord) if full_goal_coord is not None else None
    )
    logger.info(
        "Grounding deferred: failure=%s full_goal=%s strategy=%s",
        failure_code,
        controller_state.grounding_deferred_full_goal,
        controller_state.current_strategy.target_region,
    )
    return {
        "handled": True,
        "resolved": False,
        "deferred": True,
        "moved_local_map": moved_local_map,
        "retried": retry_result is not None,
        "stale_local_goal_invalidated": stale_local_goal_invalidated,
        "target_anchor_reprojected": False,
        "target_anchor_reprojection_failed": bool(target_anchor_active and moved_local_map),
        "grounding_patch_reason": patch_reason,
        "grounding_patch_mode": patch_mode,
    }


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
    failure_code = getattr(last_grounding_result, "failure_code", None) or getattr(
        last_grounding_result, "graph_no_goal_reason", ""
    ) or getattr(last_grounding_result, "noop_type", "")
    noop_reason = getattr(last_grounding_result, "noop_reason", None) or getattr(
        last_grounding_result, "reason", ""
    )
    if failure_code == "out_of_local_window" or controller_state.grounding_deferred:
        return {
            "replanned": False,
            "forced_replan_due_to_grounding_failure": False,
            "grounding_failure_reason": failure_code or controller_state.grounding_deferred_reason,
        }
    if noop_reason == "same_goal_as_prev":
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
