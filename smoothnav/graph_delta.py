"""Graph-delta construction for the SmoothNav world model layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from smoothnav.target_matching import is_target_candidate, target_candidate_details


@dataclass
class GraphDelta:
    new_nodes: List[Any] = field(default_factory=list)
    new_node_captions: List[str] = field(default_factory=list)
    new_rooms: List[str] = field(default_factory=list)
    room_object_count_changes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    room_object_count_increase_rooms: List[str] = field(default_factory=list)
    target_candidate_captions: List[str] = field(default_factory=list)
    target_candidate_details: List[Dict[str, Any]] = field(default_factory=list)
    node_caption_changed: bool = False
    node_captions_snapshot: Dict[int, str] = field(default_factory=dict)
    frontier_near: bool = False
    frontier_reached: bool = False
    no_progress: bool = False
    stuck: bool = False
    dist_to_goal: float = 0.0
    graph_node_count: int = 0
    room_object_counts: Dict[str, int] = field(default_factory=dict)
    event_types: List[str] = field(default_factory=list)
    current_strategy_type: str = "none"

    @property
    def has_new_nodes(self):
        return len(self.new_nodes) > 0

    @property
    def has_new_rooms(self):
        return len(self.new_rooms) > 0

    @property
    def has_room_object_increase(self):
        return len(self.room_object_count_increase_rooms) > 0

    @property
    def has_target_candidates(self):
        return len(self.target_candidate_captions) > 0

    @property
    def has_caption_changes(self):
        return bool(self.node_caption_changed)

    def to_dict(self):
        return {
            "event_types": list(self.event_types),
            "current_strategy_type": self.current_strategy_type,
            "new_node_count": len(self.new_nodes),
            "new_node_captions": list(self.new_node_captions),
            "new_rooms": list(self.new_rooms),
            "room_object_count_changes": dict(self.room_object_count_changes),
            "room_object_count_increase_rooms": list(
                self.room_object_count_increase_rooms
            ),
            "target_candidate_captions": list(self.target_candidate_captions),
            "target_candidate_details": [
                dict(item) for item in self.target_candidate_details
            ],
            "node_caption_changed": bool(self.node_caption_changed),
            "frontier_near": bool(self.frontier_near),
            "frontier_reached": bool(self.frontier_reached),
            "no_progress": bool(self.no_progress),
            "stuck": bool(self.stuck),
            "dist_to_goal": float(self.dist_to_goal),
            "graph_node_count": int(self.graph_node_count),
            "room_object_counts": dict(self.room_object_counts),
        }


def is_room_target(target_region: str) -> bool:
    if not target_region:
        return False
    return (
        not str(target_region).startswith("unexplored")
        and not str(target_region).startswith("object:")
    )


def is_object_target(target_region: str) -> bool:
    return bool(target_region and str(target_region).startswith("object:"))


def is_target_search_anchor(target_region: str) -> bool:
    return bool(target_region and str(target_region).startswith("unexplored target:"))


def strategy_type(target_region: str) -> str:
    if is_object_target(target_region):
        return "object"
    if is_room_target(target_region):
        return "room"
    if target_region:
        return "direction"
    return "none"


def strategy_specificity(target_region: str) -> int:
    if is_object_target(target_region):
        return 3
    if is_target_search_anchor(target_region):
        return 2
    if is_room_target(target_region):
        return 1
    return 0


def build_room_object_counts(graph) -> Dict[str, int]:
    counts = {}
    for room_node in getattr(graph, "room_nodes", []):
        count = len(getattr(room_node, "nodes", []))
        if count > 0:
            counts[room_node.caption] = count
    return counts


def _node_key(node, fallback_index: int) -> int:
    for attr in ("node_id", "id", "idx"):
        value = getattr(node, attr, None)
        if isinstance(value, int):
            return value
    return int(fallback_index)


def build_graph_delta(
    graph,
    controller_state,
    frontier_near: bool,
    frontier_reached: bool,
    no_progress: bool,
    stuck: bool,
    dist_to_goal: float,
) -> GraphDelta:
    new_nodes = graph.nodes[controller_state.prev_node_count:]
    new_node_captions = [
        node.caption for node in new_nodes
        if hasattr(node, "caption") and node.caption
    ]
    goal_text = (
        getattr(graph, "text_goal", None)
        or getattr(graph, "obj_goal", None)
        or ""
    )
    target_threshold = float(
        getattr(
            getattr(graph, "args", None),
            "graph_text_goal_direct_relevance_threshold",
            0.75,
        )
    )
    room_counts = build_room_object_counts(graph)
    room_count_changes = {}
    room_object_count_increase_rooms = []
    new_rooms = []
    all_rooms = set(room_counts) | set(controller_state.prev_room_object_counts)
    for room_name in sorted(all_rooms):
        before = controller_state.prev_room_object_counts.get(room_name, 0)
        after = room_counts.get(room_name, 0)
        if before != after:
            room_count_changes[room_name] = {"before": before, "after": after}
        if after > before:
            room_object_count_increase_rooms.append(room_name)
        if before == 0 and after > 0:
            new_rooms.append(room_name)

    node_captions_snapshot = {}
    node_caption_changed = False
    caption_changed_target_nodes = []
    prev_node_captions = getattr(controller_state, "prev_node_captions", {})
    for idx, node in enumerate(getattr(graph, "nodes", [])):
        caption = getattr(node, "caption", "")
        if not caption:
            continue
        key = _node_key(node, idx)
        node_captions_snapshot[key] = caption
        if key in prev_node_captions and prev_node_captions[key] != caption:
            node_caption_changed = True
            if is_target_candidate(caption, goal_text, threshold=target_threshold):
                caption_changed_target_nodes.append(node)

    target_nodes = list(new_nodes)
    seen_target_ids = {id(node) for node in target_nodes}
    for node in caption_changed_target_nodes:
        if id(node) not in seen_target_ids:
            target_nodes.append(node)
            seen_target_ids.add(id(node))
    target_details = target_candidate_details(
        target_nodes, goal_text, threshold=target_threshold
    )
    target_captions = [
        str(item.get("caption", ""))
        for item in target_details
        if item.get("caption")
    ]

    event_types = []
    if new_nodes:
        event_types.append("new_nodes")
    if target_captions:
        event_types.append("target_candidate_detected")
    if new_rooms:
        event_types.append("new_rooms")
    if room_object_count_increase_rooms:
        event_types.append("room_object_count_increase")
    if node_caption_changed:
        event_types.append("node_caption_changed")
    if frontier_near:
        event_types.append("frontier_near")
    if frontier_reached:
        event_types.append("frontier_reached")
    if no_progress:
        event_types.append("no_progress")
    if stuck:
        event_types.append("stuck")

    current_strategy = getattr(controller_state, "current_strategy", None)
    return GraphDelta(
        new_nodes=new_nodes,
        new_node_captions=new_node_captions,
        new_rooms=new_rooms,
        room_object_count_changes=room_count_changes,
        room_object_count_increase_rooms=room_object_count_increase_rooms,
        target_candidate_captions=target_captions,
        target_candidate_details=target_details,
        node_captions_snapshot=node_captions_snapshot,
        node_caption_changed=node_caption_changed,
        frontier_near=frontier_near,
        frontier_reached=frontier_reached,
        no_progress=no_progress,
        stuck=stuck,
        dist_to_goal=float(dist_to_goal),
        graph_node_count=len(getattr(graph, "nodes", [])),
        room_object_counts=room_counts,
        event_types=event_types,
        current_strategy_type=(
            strategy_type(current_strategy.target_region)
            if current_strategy is not None
            else "none"
        ),
    )
