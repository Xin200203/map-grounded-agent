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
    same_goal_hold_count: int = 0
    consecutive_grounding_noops: int = 0
    same_frontier_reuse_count: int = 0
    last_grounding_selected_frontier: Optional[List[int]] = None
    grounding_deferred: bool = False
    grounding_deferred_reason: str = ""
    grounding_deferred_full_goal: Optional[List[int]] = None
    executor_stuck_suppression_steps: int = 0
    last_position: Optional[List[float]] = None
    last_goal: Optional[List[int]] = None
    planner_call_count: int = 0
    monitor_call_count: int = 0
    planner_prefetch_cooldown_until_step: int = 0
    planner_empty_response_streak: int = 0
    target_anchor_attempt_id: int = 0
    target_anchor_target_region: str = ""
    target_anchor_best_distance: Optional[float] = None
    target_anchor_last_distance: Optional[float] = None
    target_anchor_stall_updates: int = 0
    target_anchor_update_count: int = 0
    target_anchor_last_update_step: Optional[int] = None
    target_anchor_last_improvement_step: Optional[int] = None
    target_anchor_last_trigger: str = ""
    target_anchor_label: str = ""
    target_anchor_full_map_coord: Optional[List[int]] = None
    target_anchor_last_projected_local_coord: Optional[List[int]] = None
    target_anchor_last_value_source: str = ""
    target_anchor_last_seen_step: Optional[int] = None
    target_anchor_decommit_reason: str = ""
    target_commit_label: str = ""
    target_commit_approach_retries: int = 0
    target_commit_stuck_strikes: int = 0
    needs_initial_plan: bool = True
