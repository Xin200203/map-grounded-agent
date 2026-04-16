"""Mission/progress state for SmoothNav's Layer 1 manager."""

from typing import Any, Optional

from smoothnav.types import MissionState, stage_goal_from_strategy


class MissionProgressManager:
    """Small progress manager that makes stage state explicit.

    The first version intentionally avoids a full instruction parser. It records
    the active semantic stage, why it changed, and which stages were completed
    or blocked so the planner is no longer operating from only raw graph text.
    """

    def __init__(self, mission_text: str = "", mission_type: str = ""):
        self.state = MissionState(mission_text=mission_text, mission_type=mission_type)

    def reset(self, mission_text: str, mission_type: str = "") -> MissionState:
        self.state = MissionState(
            mission_text=mission_text or "",
            mission_type=mission_type or "",
            stage_status="not_started",
        )
        return self.state

    def note_stage_goal(
        self,
        strategy_or_stage_goal: Any,
        *,
        replan_reason: Optional[str] = None,
        status: str = "active",
    ) -> MissionState:
        stage_goal = stage_goal_from_strategy(strategy_or_stage_goal)
        if stage_goal is None:
            return self.state

        desc = stage_goal.semantic_intent or stage_goal.target_region or ""
        if desc != self.state.current_stage_desc:
            self.state.current_stage_id += 1
        self.state.current_stage_desc = desc
        self.state.stage_status = status
        self.state.stop_condition = stage_goal.stop_condition
        self.state.replan_reason = replan_reason
        required = stage_goal.target_object or stage_goal.target_region
        self.state.required_evidence = [required] if required else []
        return self.state

    def mark_completed(self, stage_desc: Optional[str] = None) -> MissionState:
        desc = stage_desc or self.state.current_stage_desc
        if desc and desc not in self.state.completed_stages:
            self.state.completed_stages.append(desc)
        self.state.stage_status = "completed"
        return self.state

    def mark_blocked(self, stage_desc: Optional[str] = None, reason: str = "") -> MissionState:
        desc = stage_desc or self.state.current_stage_desc
        blocked = f"{desc}: {reason}" if reason else desc
        if blocked and blocked not in self.state.blocked_stages:
            self.state.blocked_stages.append(blocked)
        self.state.stage_status = "blocked"
        self.state.replan_reason = reason or self.state.replan_reason
        return self.state

    def observe_evidence(self, evidence: str) -> MissionState:
        if evidence and evidence not in self.state.obtained_evidence:
            self.state.obtained_evidence.append(evidence)
        return self.state
