"""Explicit controller state for generation-2 SmoothNav orchestration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ControllerState:
    current_strategy: Optional[Any] = None
    pending_strategy: Optional[Any] = None
    explored_regions: List[str] = field(default_factory=list)
    prev_node_count: int = 0
    prev_room_object_counts: Dict[str, int] = field(default_factory=dict)
    no_progress_steps: int = 0
    last_position: Optional[List[float]] = None
    last_goal: Optional[List[int]] = None
    planner_call_count: int = 0
    monitor_call_count: int = 0
    needs_initial_plan: bool = True
