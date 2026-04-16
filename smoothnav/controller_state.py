"""Explicit controller state for generation-2 SmoothNav orchestration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ControllerState:
    current_strategy: Optional[Any] = None
    pending_strategy: Optional[Any] = None
    explored_regions: List[str] = field(default_factory=list)
    strategy_epoch: int = 0
    goal_epoch: int = 0
    prev_node_count: int = 0
    prev_room_object_counts: Dict[str, int] = field(default_factory=dict)
    prev_node_captions: Dict[int, str] = field(default_factory=dict)
    no_progress_steps: int = 0
    direction_reuse_count: int = 0
    consecutive_grounding_noops: int = 0
    same_frontier_reuse_count: int = 0
    last_grounding_selected_frontier: Optional[List[int]] = None
    executor_stuck_suppression_steps: int = 0
    last_position: Optional[List[float]] = None
    last_goal: Optional[List[int]] = None
    planner_call_count: int = 0
    monitor_call_count: int = 0
    needs_initial_plan: bool = True
