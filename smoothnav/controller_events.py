"""Minimal control-event abstraction for SmoothNav."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GraphDelta:
    new_nodes: List[Any] = field(default_factory=list)
    new_node_captions: List[str] = field(default_factory=list)
    new_rooms: List[str] = field(default_factory=list)
    room_object_count_changes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    frontier_near: bool = False
    frontier_reached: bool = False
    no_progress: bool = False
    stuck: bool = False
    dist_to_goal: float = 0.0
    graph_node_count: int = 0
    room_object_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def has_new_nodes(self):
        return len(self.new_nodes) > 0

    @property
    def has_new_rooms(self):
        return len(self.new_rooms) > 0
