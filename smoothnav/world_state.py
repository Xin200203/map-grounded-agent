"""World-state assembly for SmoothNav's Layer 0 world model."""

from typing import Any, Dict, List, Optional

from smoothnav.types import WorldState


def _safe_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def summarize_frontiers(graph, limit: int = 8) -> List[Dict[str, Any]]:
    frontiers = getattr(graph, "frontier_locations_16", None)
    if frontiers is None:
        frontiers = getattr(graph, "frontier_locations", None)
    result = []
    for idx, coord in enumerate(_safe_list(frontiers)[:limit]):
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            result.append({"rank": idx + 1, "coord": [int(coord[0]), int(coord[1])]})
    return result


def summarize_rooms(graph, limit: int = 20) -> List[Dict[str, Any]]:
    rooms = []
    for room_node in getattr(graph, "room_nodes", [])[:limit]:
        nodes = list(getattr(room_node, "nodes", []) or [])
        rooms.append(
            {
                "caption": getattr(room_node, "caption", ""),
                "object_count": len(nodes),
                "objects": [
                    getattr(node, "caption", "")
                    for node in nodes[:10]
                    if getattr(node, "caption", "")
                ],
            }
        )
    return rooms


def summarize_objects(graph, limit: int = 30) -> List[Dict[str, Any]]:
    objects = []
    for node in getattr(graph, "nodes", [])[:limit]:
        center = getattr(node, "center", None)
        objects.append(
            {
                "caption": getattr(node, "caption", ""),
                "center": (
                    [int(center[0]), int(center[1])]
                    if center is not None and len(center) >= 2
                    else None
                ),
            }
        )
    return objects


def summarize_visible_targets(agent=None) -> List[Dict[str, Any]]:
    if agent is None:
        return []
    override = getattr(agent, "last_override_info", {}) or {}
    if not override.get("visible_target_override"):
        return []
    goal = override.get("adopted_goal_summary")
    return [
        {
            "source": override.get("adopted_goal_source", "visible_target"),
            "goal": goal,
        }
    ]


def build_world_state(
    *,
    step_idx: int,
    world_epoch: int = 0,
    pose,
    local_pose,
    graph,
    bev_map,
    controller_state,
    graph_delta,
    agent=None,
) -> WorldState:
    frontier_summary = summarize_frontiers(graph)
    frontier_locations = getattr(graph, "frontier_locations_16", None)
    if frontier_locations is None:
        frontier_locations = getattr(graph, "frontier_locations", [])
    return WorldState(
        step_idx=int(step_idx),
        world_epoch=int(world_epoch),
        pose=pose,
        local_pose=local_pose,
        graph=graph,
        bev_map=bev_map,
        explored_regions=list(getattr(controller_state, "explored_regions", []) or []),
        frontier_count=len(_safe_list(frontier_locations)),
        frontier_summary=frontier_summary,
        room_summary=summarize_rooms(graph),
        object_summary=summarize_objects(graph),
        visible_targets=summarize_visible_targets(agent),
        visible_target_summary=summarize_visible_targets(agent),
        stuck_signal=bool(getattr(graph_delta, "stuck", False)),
        no_progress_steps=int(getattr(controller_state, "no_progress_steps", 0) or 0),
        graph_delta=graph_delta,
    )


class WorldStateBuilder:
    """Single writer for monotonic WorldState epochs."""

    def __init__(self):
        self.world_epoch = 0
        self._last_signature = None

    def reset(self) -> None:
        self.world_epoch = 0
        self._last_signature = None

    def build(self, **kwargs) -> WorldState:
        graph = kwargs.get("graph")
        graph_delta = kwargs.get("graph_delta")
        frontier_locations = getattr(graph, "frontier_locations_16", None)
        if frontier_locations is None:
            frontier_locations = getattr(graph, "frontier_locations", [])
        signature = (
            len(getattr(graph, "nodes", []) or []),
            len(_safe_list(frontier_locations)),
            tuple(getattr(graph_delta, "event_types", []) or []),
            bool(getattr(graph_delta, "stuck", False)),
            bool(getattr(graph_delta, "frontier_reached", False)),
        )
        if self._last_signature is None or signature != self._last_signature:
            self.world_epoch += 1
            self._last_signature = signature
        kwargs["world_epoch"] = self.world_epoch
        return build_world_state(**kwargs)
