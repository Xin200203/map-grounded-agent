"""
SmoothNav Low-Level Agent: haiku-based event-driven monitor and bias adjuster.

Triggered when scene graph has new nodes. Evaluates whether current strategy
is still valid and outputs one of four actions:
  CONTINUE / ADJUST / PREFETCH / ESCALATE
"""

import re
import json
import logging
from enum import IntEnum
from typing import List, Optional, Tuple
from dataclasses import dataclass

from smoothnav.tracing import hash_text

logger = logging.getLogger(__name__)
LOW_LEVEL_PROMPT_SCHEMA_VERSION = "smoothnav_monitor_v1"
RULE_MONITOR_SCHEMA_VERSION = "smoothnav_monitor_rules_v1"
ESCALATION_ONLY_MONITOR_SCHEMA_VERSION = "smoothnav_monitor_escalation_v1"


class LowLevelAction(IntEnum):
    CONTINUE = 0   # maintain current target
    ADJUST = 1     # micro-adjust bias within current strategy
    PREFETCH = 2   # pre-trigger high-level planner (non-blocking)
    ESCALATE = 3   # strategy is stale, need new high-level plan


@dataclass
class LowLevelResult:
    action: LowLevelAction
    reason: str = ""
    adjust_anchor: str = ""                        # object name for ADJUST
    adjust_bias: Optional[Tuple[int, int]] = None  # resolved coords for ADJUST


LOW_LEVEL_PROMPT = """You are monitoring a robot's navigation. The robot is searching for a target object.

CURRENT STRATEGY: Search in "{target_region}"
CURRENT STRATEGY TYPE: {strategy_type}
STRATEGY REASONING: {reasoning}

NEW OBJECTS DETECTED: {new_objects}
NEW OBJECTS IN TARGET ROOM: {new_objects_in_target_room}
NEW ROOMS: {new_rooms}
EVENT TYPES: {event_types}
DISTANCE TO CURRENT FRONTIER GOAL: {dist_to_goal} map units (closer = arriving soon)
FRONTIER NEAR: {frontier_near}
NO PROGRESS STEPS: {no_progress_steps}
TOTAL OBJECTS OBSERVED: {total_nodes}

Decide one action:
- CONTINUE: new objects are consistent with current strategy, keep going
- ADJUST: new objects suggest the target is near a specific observed object. Provide its name in adjust_anchor.
- PREFETCH: agent is approaching its frontier goal (distance < 15) and will need a new strategy soon
- ESCALATE: new evidence shows the current strategy is wrong (e.g., found a bathroom but strategy says kitchen)

Output JSON only:
{{"action": "CONTINUE", "reason": "<brief>", "adjust_anchor": ""}}"""


ESCALATION_ONLY_PROMPT = """You are monitoring whether a robot's current navigation strategy has become stale.

CURRENT STRATEGY: Search in "{target_region}"
CURRENT STRATEGY TYPE: {strategy_type}
STRATEGY REASONING: {reasoning}

NEW OBJECTS DETECTED: {new_objects}
NEW OBJECTS IN TARGET ROOM: {new_objects_in_target_room}
NEW ROOMS: {new_rooms}
EVENT TYPES: {event_types}
DISTANCE TO CURRENT FRONTIER GOAL: {dist_to_goal} map units
FRONTIER NEAR: {frontier_near}
NO PROGRESS STEPS: {no_progress_steps}
TOTAL OBJECTS OBSERVED: {total_nodes}

Only decide whether the current strategy should be replaced.

Choose exactly one:
- CONTINUE: current strategy still makes sense
- ESCALATE: current strategy is stale or contradicted and needs a new high-level plan

Output JSON only:
{{"action": "CONTINUE", "reason": "<brief>"}}"""


def _is_room_target(target_region: str) -> bool:
    if not target_region:
        return False
    return (not target_region.startswith("unexplored")
            and not target_region.startswith("object:"))


def _strategy_type(target_region: str) -> str:
    if target_region.startswith("object:"):
        return "object"
    if _is_room_target(target_region):
        return "room"
    return "direction"


class LowLevelAgent:
    """Haiku-based low-level agent. Event-driven: called when scene graph changes."""

    def __init__(self, llm_fn, max_retries: int = 1):
        """
        Args:
            llm_fn: callable(prompt=str) -> str. LLM instance with haiku model.
            max_retries: number of retries on parse failure.
        """
        self.llm = llm_fn
        self.max_retries = max_retries
        self._call_count: int = 0

    def reset(self):
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def should_evaluate(
        self,
        strategy,
        graph_delta=None,
        no_progress_steps: int = 0,
        dist_to_goal: float = 0.0,
    ) -> bool:
        event_types = list(getattr(graph_delta, "event_types", []))
        return bool(event_types or getattr(graph_delta, "new_nodes", []))

    def evaluate(self, strategy, new_nodes: list, dist_to_goal: float,
                 total_nodes: int, graph=None, graph_delta=None,
                 no_progress_steps: int = 0,
                 episode_id: Optional[int] = None,
                 step_idx: Optional[int] = None,
                 trace_writer=None) -> LowLevelResult:
        """
        Evaluate current situation and decide action.

        Args:
            strategy: current Strategy from high-level planner
            new_nodes: list of new ObjectNode instances since last evaluation
            dist_to_goal: distance to current frontier goal (map units)
            total_nodes: total scene graph node count
            graph: Graph instance for resolving ADJUST coordinates

        Returns:
            LowLevelResult with action and optional bias adjustment.
        """
        new_captions = []
        for n in new_nodes:
            if hasattr(n, 'caption') and n.caption:
                new_captions.append(n.caption)
        event_types = list(getattr(graph_delta, "event_types", []))
        strategy_type = _strategy_type(strategy.target_region)
        new_rooms = list(getattr(graph_delta, "new_rooms", []))
        room_increase_rooms = list(
            getattr(graph_delta, "room_object_count_increase_rooms", [])
        )
        new_objects_in_target_room = []
        if strategy_type == "room" and strategy.target_region in (
            set(new_rooms) | set(room_increase_rooms)
        ):
            new_objects_in_target_room = list(new_captions)
        monitor_prompt_event_summary = {
            "strategy_type": strategy_type,
            "event_types": event_types,
            "new_rooms": new_rooms,
            "new_objects_in_target_room": new_objects_in_target_room,
            "frontier_near": bool(getattr(graph_delta, "frontier_near", False)),
            "no_progress_steps": int(no_progress_steps),
        }

        if not new_captions and not event_types:
            result = LowLevelResult(action=LowLevelAction.CONTINUE,
                                    reason="no_new_objects")
            if trace_writer is not None and episode_id is not None:
                trace_writer.record_monitor_call(
                    episode_id,
                    {
                        "step_idx": step_idx,
                        "schema_version": LOW_LEVEL_PROMPT_SCHEMA_VERSION,
                        "prompt_hash": "",
                        "raw_prompt": "",
                        "raw_response": "",
                        "parsed_result": {
                            "action": result.action.name,
                            "reason": result.reason,
                            "adjust_anchor": result.adjust_anchor,
                            "adjust_bias": result.adjust_bias,
                        },
                        "fallback_triggered": False,
                        "monitor_fallback_used": False,
                        "llm_called": False,
                        "strategy_target_region": strategy.target_region,
                        "strategy_type": strategy_type,
                        "new_objects": new_captions,
                        "new_objects_in_target_room": new_objects_in_target_room,
                        "new_rooms": new_rooms,
                        "event_types": event_types,
                        "dist_to_goal": int(dist_to_goal),
                        "frontier_near": bool(getattr(graph_delta, "frontier_near", False)),
                        "no_progress_steps": int(no_progress_steps),
                        "total_nodes": total_nodes,
                        "monitor_prompt_event_summary": monitor_prompt_event_summary,
                        "monitor_parsed_action": result.action.name,
                    },
                )
            return result

        prompt = LOW_LEVEL_PROMPT.format(
            target_region=strategy.target_region,
            strategy_type=strategy_type,
            reasoning=strategy.reasoning,
            new_objects=', '.join(new_captions) or "None",
            new_objects_in_target_room=', '.join(new_objects_in_target_room) or "None",
            new_rooms=', '.join(new_rooms) or "None",
            event_types=', '.join(event_types) or "none",
            dist_to_goal=int(dist_to_goal),
            frontier_near=str(bool(getattr(graph_delta, "frontier_near", False))).lower(),
            no_progress_steps=int(no_progress_steps),
            total_nodes=total_nodes,
        )

        result = None
        raw_response = ""
        error_message = ""
        used_fallback = False
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm(prompt=prompt)
                raw_response = response
                if not str(response).strip():
                    error_message = "empty_response"
                    logger.warning("LowLevelAgent received empty response; using fallback.")
                    break
                result = self._parse(response)
                if result:
                    break
            except Exception as e:
                error_message = str(e)
                logger.warning(f"LowLevelAgent attempt {attempt} failed: {e}")

        self._call_count += 1

        if result is None:
            used_fallback = True
            result = LowLevelResult(action=LowLevelAction.CONTINUE,
                                    reason="parse_failure")

        # Resolve ADJUST anchor to map coordinates
        if result.action == LowLevelAction.ADJUST and result.adjust_anchor and graph:
            anchor_lower = result.adjust_anchor.lower()
            for node in graph.nodes:
                if (hasattr(node, 'caption') and anchor_lower in node.caption.lower()
                        and node.center is not None):
                    result.adjust_bias = (int(node.center[0]), int(node.center[1]))
                    break

        logger.info(f"LowLevelAgent: {result.action.name} reason={result.reason} "
                     f"anchor={result.adjust_anchor} bias={result.adjust_bias}")

        if trace_writer is not None and episode_id is not None:
            trace_writer.record_monitor_call(
                episode_id,
                {
                    "step_idx": step_idx,
                    "schema_version": LOW_LEVEL_PROMPT_SCHEMA_VERSION,
                    "prompt_hash": hash_text(prompt),
                    "raw_prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_result": {
                        "action": result.action.name,
                        "reason": result.reason,
                        "adjust_anchor": result.adjust_anchor,
                        "adjust_bias": result.adjust_bias,
                    },
                    "fallback_triggered": used_fallback,
                    "monitor_fallback_used": used_fallback,
                    "llm_called": True,
                    "error_message": error_message,
                    "strategy_target_region": strategy.target_region,
                    "strategy_type": strategy_type,
                    "new_objects": new_captions,
                    "new_objects_in_target_room": new_objects_in_target_room,
                    "new_rooms": new_rooms,
                    "event_types": event_types,
                    "dist_to_goal": int(dist_to_goal),
                    "frontier_near": bool(getattr(graph_delta, "frontier_near", False)),
                    "no_progress_steps": int(no_progress_steps),
                    "total_nodes": total_nodes,
                    "monitor_prompt_event_summary": monitor_prompt_event_summary,
                    "monitor_parsed_action": result.action.name,
                },
            )
        return result

    def _parse(self, response: str) -> Optional[LowLevelResult]:
        """Parse JSON from haiku response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            action_str = str(data.get('action', '')).upper()
            action_map = {
                'CONTINUE': LowLevelAction.CONTINUE,
                'ADJUST': LowLevelAction.ADJUST,
                'PREFETCH': LowLevelAction.PREFETCH,
                'ESCALATE': LowLevelAction.ESCALATE,
            }
            if action_str not in action_map:
                return None
            return LowLevelResult(
                action=action_map[action_str],
                reason=data.get('reason', ''),
                adjust_anchor=data.get('adjust_anchor', ''),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


class RuleBasedMonitor:
    """Deterministic low-level monitor for Phase 2 ablations."""

    def __init__(self, prefetch_near_threshold: float = 10.0):
        self.prefetch_near_threshold = float(prefetch_near_threshold)
        self._call_count: int = 0

    def reset(self):
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def should_evaluate(
        self,
        strategy,
        graph_delta=None,
        no_progress_steps: int = 0,
        dist_to_goal: float = 0.0,
    ) -> bool:
        event_types = list(getattr(graph_delta, "event_types", []))
        return bool(event_types or getattr(graph_delta, "new_nodes", []))

    def evaluate(self, strategy, new_nodes: list, dist_to_goal: float,
                 total_nodes: int, graph=None, graph_delta=None,
                 no_progress_steps: int = 0,
                 episode_id: Optional[int] = None,
                 step_idx: Optional[int] = None,
                 trace_writer=None) -> LowLevelResult:
        self._call_count += 1

        new_captions = []
        for node in new_nodes:
            if hasattr(node, "caption") and node.caption:
                new_captions.append(node.caption)

        strategy_type = _strategy_type(strategy.target_region)
        event_types = list(getattr(graph_delta, "event_types", []))
        new_rooms = list(getattr(graph_delta, "new_rooms", []))
        room_increase_rooms = list(
            getattr(graph_delta, "room_object_count_increase_rooms", [])
        )
        frontier_near = bool(getattr(graph_delta, "frontier_near", False))
        no_progress = bool(getattr(graph_delta, "no_progress", False))
        stuck = bool(getattr(graph_delta, "stuck", False))
        new_objects_in_target_room = []
        if strategy_type == "room" and strategy.target_region in (
            set(new_rooms) | set(room_increase_rooms)
        ):
            new_objects_in_target_room = list(new_captions)

        action = LowLevelAction.CONTINUE
        reason = "rules_continue"
        adjust_anchor = ""
        if stuck:
            action = LowLevelAction.ESCALATE
            reason = "rules_stuck_escalate"
        elif no_progress and no_progress_steps >= 2:
            action = LowLevelAction.ESCALATE
            reason = "rules_no_progress_escalate"
        elif strategy_type == "direction" and new_rooms:
            action = LowLevelAction.ESCALATE
            reason = "rules_new_room_escalate"
        elif frontier_near or dist_to_goal < self.prefetch_near_threshold:
            action = LowLevelAction.PREFETCH
            reason = "rules_prefetch_near_frontier"
        elif strategy_type == "room" and new_captions:
            action = LowLevelAction.ADJUST
            adjust_anchor = new_captions[0]
            reason = "rules_room_adjust_to_new_object"

        result = LowLevelResult(action=action, reason=reason, adjust_anchor=adjust_anchor)
        if result.action == LowLevelAction.ADJUST and result.adjust_anchor and graph:
            anchor_lower = result.adjust_anchor.lower()
            for node in graph.nodes:
                if (hasattr(node, "caption") and anchor_lower in node.caption.lower()
                        and node.center is not None):
                    result.adjust_bias = (int(node.center[0]), int(node.center[1]))
                    break
        if trace_writer is not None and episode_id is not None:
            trace_writer.record_monitor_call(
                episode_id,
                {
                    "step_idx": step_idx,
                    "schema_version": RULE_MONITOR_SCHEMA_VERSION,
                    "prompt_hash": "",
                    "raw_prompt": "",
                    "raw_response": "",
                    "parsed_result": {
                        "action": result.action.name,
                        "reason": result.reason,
                        "adjust_anchor": result.adjust_anchor,
                        "adjust_bias": result.adjust_bias,
                    },
                    "fallback_triggered": False,
                    "monitor_fallback_used": False,
                    "llm_called": False,
                    "rule_based": True,
                    "strategy_target_region": strategy.target_region,
                    "strategy_type": strategy_type,
                    "new_objects": new_captions,
                    "new_objects_in_target_room": new_objects_in_target_room,
                    "new_rooms": new_rooms,
                    "event_types": event_types,
                    "dist_to_goal": int(dist_to_goal),
                    "frontier_near": frontier_near,
                    "no_progress_steps": int(no_progress_steps),
                    "total_nodes": total_nodes,
                    "monitor_prompt_event_summary": {
                        "strategy_type": strategy_type,
                        "event_types": event_types,
                        "new_rooms": new_rooms,
                        "new_objects_in_target_room": new_objects_in_target_room,
                        "frontier_near": frontier_near,
                        "no_progress_steps": int(no_progress_steps),
                    },
                    "monitor_parsed_action": result.action.name,
                },
            )
        return result


class EscalationOnlyMonitor:
    """Heuristic-first monitor that asks the LLM only for escalation decisions."""

    def __init__(self, llm_fn, prefetch_near_threshold: float = 10.0, max_retries: int = 1):
        self.llm = llm_fn
        self.prefetch_near_threshold = float(prefetch_near_threshold)
        self.max_retries = max_retries
        self._call_count: int = 0

    def reset(self):
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def should_evaluate(
        self,
        strategy,
        graph_delta=None,
        no_progress_steps: int = 0,
        dist_to_goal: float = 0.0,
    ) -> bool:
        if strategy is None or graph_delta is None:
            return False
        if getattr(graph_delta, "frontier_near", False):
            return False
        if dist_to_goal < self.prefetch_near_threshold:
            return False
        if getattr(graph_delta, "stuck", False):
            return False
        if getattr(graph_delta, "no_progress", False) and no_progress_steps >= 2:
            return False
        event_types = set(getattr(graph_delta, "event_types", []))
        semantic_events = {
            "new_rooms",
            "room_object_count_increase",
            "node_caption_changed",
        }
        if not (event_types & semantic_events):
            return False
        strategy_type = _strategy_type(strategy.target_region)
        if strategy_type == "room" and strategy.target_region in set(
            getattr(graph_delta, "new_rooms", [])
        ):
            return False
        return True

    def evaluate(self, strategy, new_nodes: list, dist_to_goal: float,
                 total_nodes: int, graph=None, graph_delta=None,
                 no_progress_steps: int = 0,
                 episode_id: Optional[int] = None,
                 step_idx: Optional[int] = None,
                 trace_writer=None) -> LowLevelResult:
        new_captions = []
        for n in new_nodes:
            if hasattr(n, "caption") and n.caption:
                new_captions.append(n.caption)
        event_types = list(getattr(graph_delta, "event_types", []))
        strategy_type = _strategy_type(strategy.target_region)
        new_rooms = list(getattr(graph_delta, "new_rooms", []))
        room_increase_rooms = list(
            getattr(graph_delta, "room_object_count_increase_rooms", [])
        )
        frontier_near = bool(getattr(graph_delta, "frontier_near", False))
        new_objects_in_target_room = []
        if strategy_type == "room" and strategy.target_region in (
            set(new_rooms) | set(room_increase_rooms)
        ):
            new_objects_in_target_room = list(new_captions)
        prompt_event_summary = {
            "strategy_type": strategy_type,
            "event_types": event_types,
            "new_rooms": new_rooms,
            "new_objects_in_target_room": new_objects_in_target_room,
            "frontier_near": frontier_near,
            "no_progress_steps": int(no_progress_steps),
        }

        heuristic_reason = ""
        if not self.should_evaluate(
            strategy=strategy,
            graph_delta=graph_delta,
            no_progress_steps=no_progress_steps,
            dist_to_goal=dist_to_goal,
        ):
            heuristic_reason = "heuristic_continue"
        elif strategy_type == "room" and new_objects_in_target_room:
            heuristic_reason = "heuristic_supportive_room_evidence"

        if heuristic_reason:
            result = LowLevelResult(
                action=LowLevelAction.CONTINUE,
                reason=heuristic_reason,
            )
            if trace_writer is not None and episode_id is not None:
                trace_writer.record_monitor_call(
                    episode_id,
                    {
                        "step_idx": step_idx,
                        "schema_version": ESCALATION_ONLY_MONITOR_SCHEMA_VERSION,
                        "prompt_hash": "",
                        "raw_prompt": "",
                        "raw_response": "",
                        "parsed_result": {
                            "action": result.action.name,
                            "reason": result.reason,
                            "adjust_anchor": result.adjust_anchor,
                            "adjust_bias": result.adjust_bias,
                        },
                        "fallback_triggered": False,
                        "monitor_fallback_used": False,
                        "llm_called": False,
                        "rule_based": True,
                        "escalation_only": True,
                        "strategy_target_region": strategy.target_region,
                        "strategy_type": strategy_type,
                        "new_objects": new_captions,
                        "new_objects_in_target_room": new_objects_in_target_room,
                        "new_rooms": new_rooms,
                        "event_types": event_types,
                        "dist_to_goal": int(dist_to_goal),
                        "frontier_near": frontier_near,
                        "no_progress_steps": int(no_progress_steps),
                        "total_nodes": total_nodes,
                        "monitor_prompt_event_summary": prompt_event_summary,
                        "monitor_parsed_action": result.action.name,
                    },
                )
            return result

        prompt = ESCALATION_ONLY_PROMPT.format(
            target_region=strategy.target_region,
            strategy_type=strategy_type,
            reasoning=strategy.reasoning,
            new_objects=", ".join(new_captions) or "None",
            new_objects_in_target_room=", ".join(new_objects_in_target_room) or "None",
            new_rooms=", ".join(new_rooms) or "None",
            event_types=", ".join(event_types) or "none",
            dist_to_goal=int(dist_to_goal),
            frontier_near=str(frontier_near).lower(),
            no_progress_steps=int(no_progress_steps),
            total_nodes=total_nodes,
        )

        result = None
        raw_response = ""
        error_message = ""
        used_fallback = False
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm(prompt=prompt)
                raw_response = response
                if not str(response).strip():
                    error_message = "empty_response"
                    logger.warning(
                        "EscalationOnlyMonitor received empty response; using fallback."
                    )
                    break
                result = self._parse(response)
                if result:
                    break
            except Exception as e:
                error_message = str(e)
                logger.warning(f"EscalationOnlyMonitor attempt {attempt} failed: {e}")

        self._call_count += 1

        if result is None:
            used_fallback = True
            result = LowLevelResult(
                action=LowLevelAction.CONTINUE,
                reason="parse_failure",
            )

        if trace_writer is not None and episode_id is not None:
            trace_writer.record_monitor_call(
                episode_id,
                {
                    "step_idx": step_idx,
                    "schema_version": ESCALATION_ONLY_MONITOR_SCHEMA_VERSION,
                    "prompt_hash": hash_text(prompt),
                    "raw_prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_result": {
                        "action": result.action.name,
                        "reason": result.reason,
                        "adjust_anchor": result.adjust_anchor,
                        "adjust_bias": result.adjust_bias,
                    },
                    "fallback_triggered": used_fallback,
                    "monitor_fallback_used": used_fallback,
                    "llm_called": True,
                    "error_message": error_message,
                    "escalation_only": True,
                    "strategy_target_region": strategy.target_region,
                    "strategy_type": strategy_type,
                    "new_objects": new_captions,
                    "new_objects_in_target_room": new_objects_in_target_room,
                    "new_rooms": new_rooms,
                    "event_types": event_types,
                    "dist_to_goal": int(dist_to_goal),
                    "frontier_near": frontier_near,
                    "no_progress_steps": int(no_progress_steps),
                    "total_nodes": total_nodes,
                    "monitor_prompt_event_summary": prompt_event_summary,
                    "monitor_parsed_action": result.action.name,
                },
            )
        return result

    def _parse(self, response: str) -> Optional[LowLevelResult]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            action_str = str(data.get("action", "")).upper()
            if action_str not in {"CONTINUE", "ESCALATE"}:
                return None
            return LowLevelResult(
                action=(
                    LowLevelAction.ESCALATE
                    if action_str == "ESCALATE"
                    else LowLevelAction.CONTINUE
                ),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


class DisabledMonitor:
    """Monitor placeholder used when the ablation disables monitoring entirely."""

    def reset(self):
        return None

    @property
    def call_count(self) -> int:
        return 0

    def should_evaluate(
        self,
        strategy,
        graph_delta=None,
        no_progress_steps: int = 0,
        dist_to_goal: float = 0.0,
    ) -> bool:
        return False
