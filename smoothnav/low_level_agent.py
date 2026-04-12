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
STRATEGY REASONING: {reasoning}

NEW OBJECTS DETECTED: {new_objects}
DISTANCE TO CURRENT FRONTIER GOAL: {dist_to_goal} map units (closer = arriving soon)
TOTAL OBJECTS OBSERVED: {total_nodes}

Decide one action:
- CONTINUE: new objects are consistent with current strategy, keep going
- ADJUST: new objects suggest the target is near a specific observed object. Provide its name in adjust_anchor.
- PREFETCH: agent is approaching its frontier goal (distance < 15) and will need a new strategy soon
- ESCALATE: new evidence shows the current strategy is wrong (e.g., found a bathroom but strategy says kitchen)

Output JSON only:
{{"action": "CONTINUE", "reason": "<brief>", "adjust_anchor": ""}}"""


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

    def evaluate(self, strategy, new_nodes: list, dist_to_goal: float,
                 total_nodes: int, graph=None,
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

        if not new_captions:
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
                        "llm_called": False,
                        "strategy_target_region": strategy.target_region,
                        "new_objects": new_captions,
                        "dist_to_goal": int(dist_to_goal),
                        "total_nodes": total_nodes,
                    },
                )
            return result

        prompt = LOW_LEVEL_PROMPT.format(
            target_region=strategy.target_region,
            reasoning=strategy.reasoning,
            new_objects=', '.join(new_captions),
            dist_to_goal=int(dist_to_goal),
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
                    "llm_called": True,
                    "error_message": error_message,
                    "strategy_target_region": strategy.target_region,
                    "new_objects": new_captions,
                    "dist_to_goal": int(dist_to_goal),
                    "total_nodes": total_nodes,
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

    def evaluate(self, strategy, new_nodes: list, dist_to_goal: float,
                 total_nodes: int, graph=None,
                 episode_id: Optional[int] = None,
                 step_idx: Optional[int] = None,
                 trace_writer=None) -> LowLevelResult:
        self._call_count += 1

        new_captions = []
        for node in new_nodes:
            if hasattr(node, "caption") and node.caption:
                new_captions.append(node.caption)

        action = LowLevelAction.CONTINUE
        reason = "rules_continue"
        if dist_to_goal < self.prefetch_near_threshold:
            action = LowLevelAction.PREFETCH
            reason = "rules_prefetch_near_frontier"

        result = LowLevelResult(action=action, reason=reason)
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
                    "llm_called": False,
                    "rule_based": True,
                    "strategy_target_region": strategy.target_region,
                    "new_objects": new_captions,
                    "dist_to_goal": int(dist_to_goal),
                    "total_nodes": total_nodes,
                },
            )
        return result


class DisabledMonitor:
    """Monitor placeholder used when the ablation disables monitoring entirely."""

    def reset(self):
        return None

    @property
    def call_count(self) -> int:
        return 0
