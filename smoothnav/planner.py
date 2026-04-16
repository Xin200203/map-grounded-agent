"""
SmoothNav High-Level Planner: sonnet-based scene graph-conditioned strategy planning.

Outputs a semantic Strategy (target region + bias position for FMM),
NOT frontier selection. Called 3-8 times per episode.
"""

import re
import json
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from smoothnav.tracing import hash_text
from smoothnav.types import StageGoal, stage_goal_from_strategy

logger = logging.getLogger(__name__)
PLANNER_PROMPT_SCHEMA_VERSION = "smoothnav_planner_v1"


@dataclass
class Strategy:
    """High-level search strategy output."""
    target_region: str                          # "kitchen" / "unexplored east"
    bias_position: Optional[Tuple[int, int]]    # full_map coords for get_goal()
    reasoning: str                              # for low-level agent to judge staleness
    explored_regions: List[str] = field(default_factory=list)
    anchor_object: str = ""                     # specific object name used for resolve


HIGH_LEVEL_PROMPT = """You are a robot searching for a target object inside an unknown indoor environment.

TARGET: {goal_description}

{scene_text}

ALREADY SEARCHED: {explored_text}

REASON FOR REPLANNING: {escalate_reason}

Choose the best navigation target from the options below.

{choices_text}

Rules:
1. If you see the target object in the object list, pick it directly (use "object" choice_type).
2. Use common sense: kitchens have appliances/food, bedrooms have beds, bathrooms have toilets, etc.
3. Do NOT revisit already-searched regions.
4. If no room seems promising, pick a direction to explore unknown areas.
5. Prefer observed room/object evidence over generic direction exploration whenever it is plausible.

Output JSON only:
{{"choice_type": "object" or "room" or "direction", "choice_id": "<object name, room id, or direction>", "reasoning": "<1-2 sentence explanation>"}}"""


def serialize_for_planner(graph, explored_regions: List[str]) -> str:
    """Serialize scene graph context: rooms + objects + spatial relations.

    This is the observation context (what the robot has seen), NOT the choices menu.
    """
    lines = []

    # Room structure with objects
    has_rooms = False
    for rn in graph.room_nodes:
        if len(rn.nodes) > 0:
            obj_list = [n.caption for n in rn.nodes]
            lines.append(f"  {rn.caption}: {', '.join(obj_list)}")
            has_rooms = True

    if has_rooms:
        lines.insert(0, "ROOMS AND OBJECTS:")
    else:
        lines.append("No objects observed yet.")

    # Spatial relationships
    if hasattr(graph, 'get_edges'):
        edges = graph.get_edges()
        if edges:
            rel_lines = []
            for edge in edges[:20]:
                if edge.relation:
                    rel_lines.append(
                        f"  {edge.node1.caption} {edge.relation} {edge.node2.caption}")
            if rel_lines:
                lines.append("SPATIAL RELATIONSHIPS:")
                lines.extend(rel_lines)

    return '\n'.join(lines)


def build_choices_text(graph, explored_regions: List[str]) -> str:
    """Build structured choices menu. Every listed choice is resolvable to coordinates.

    Design invariant: if the LLM picks any valid choice from this list,
    resolve_bias_position() will return non-None coordinates.
    - Objects: only those with known center → resolvable
    - Rooms: only those with objects (computable centroid) → resolvable
    - Directions: always resolvable (offset from agent position)
    """
    lines = []
    explored_lower = set()
    for r in explored_regions:
        # "kitchen (searched, not found)" → "kitchen"
        clean = r.lower().split(' (')[0].strip()
        explored_lower.add(clean)

    # 1. Objects with known positions
    obj_names = []
    for node in graph.nodes:
        if hasattr(node, 'caption') and node.caption and node.center is not None:
            obj_names.append(node.caption)
    if obj_names:
        lines.append("OBJECTS (pick if you recognize the target):")
        for name in obj_names[:30]:
            lines.append(f"  [object] {name}")

    # 2. Rooms with objects (resolvable centroid), excluding already-searched
    room_lines = []
    for rn in graph.room_nodes:
        if rn.caption.lower() in explored_lower:
            continue
        if len(rn.nodes) > 0:
            obj_list = [n.caption for n in rn.nodes]
            room_lines.append(f"  [room] {rn.caption}: {', '.join(obj_list)}")
    if room_lines:
        lines.append("\nROOMS (pick based on common sense about where the target would be):")
        lines.extend(room_lines)

    # 3. Directions (always resolvable)
    lines.append("\nDIRECTIONS (pick to explore unknown areas):")
    for d in ['north', 'south', 'east', 'west']:
        lines.append(f"  [direction] {d}")

    return '\n'.join(lines)


def resolve_bias_position(parsed: dict, graph, agent_pos: Tuple[int, int],
                          map_size: int, return_debug: bool = False):
    """Convert structured choice to map coordinates for get_goal().

    Uses choice_type + choice_id from constrained LLM output.
    Every valid choice listed in build_choices_text() is guaranteed resolvable.
    """
    choice_type = parsed.get('choice_type', '').lower().strip()
    choice_id = parsed.get('choice_id', '').strip()
    debug = {
        "choice_type": choice_type,
        "resolved_object_candidates": [],
        "direction_bias_mode": "",
    }
    if not choice_id:
        return (None, debug) if return_debug else None
    choice_lower = choice_id.lower()

    if choice_type == 'object':
        candidates = []
        for node in graph.nodes:
            if (hasattr(node, 'caption') and choice_lower in node.caption.lower()
                    and node.center is not None):
                distance = (
                    abs(int(node.center[0]) - int(agent_pos[0]))
                    + abs(int(node.center[1]) - int(agent_pos[1]))
                )
                candidates.append(
                    (
                        distance,
                        int(node.center[0]),
                        int(node.center[1]),
                        node.caption,
                    )
                )
        if candidates:
            candidates.sort()
            debug["resolved_object_candidates"] = [
                {
                    "caption": caption,
                    "agent_distance": int(distance),
                    "center": [int(cx), int(cy)],
                }
                for distance, cx, cy, caption in candidates[:5]
            ]
            _, cx, cy, _ = candidates[0]
            bias = (cx, cy)
            return (bias, debug) if return_debug else bias

    elif choice_type == 'room':
        for rn in graph.room_nodes:
            if choice_lower in rn.caption.lower() and len(rn.nodes) > 0:
                centers = [n.center for n in rn.nodes if n.center is not None]
                if centers:
                    cx = int(sum(c[0] for c in centers) / len(centers))
                    cy = int(sum(c[1] for c in centers) / len(centers))
                    bias = (cx, cy)
                    return (bias, debug) if return_debug else bias

    elif choice_type == 'direction':
        direction_offsets = {
            'east': (0, 1), 'west': (0, -1),
            'north': (-1, 0), 'south': (1, 0),
            'northeast': (-1, 1), 'northwest': (-1, -1),
            'southeast': (1, 1), 'southwest': (1, -1),
        }
        for dir_name, (dr, dc) in direction_offsets.items():
            if dir_name == choice_lower:
                frontier_locations = getattr(graph, 'frontier_locations_16', None)
                if frontier_locations is not None and len(frontier_locations) > 0:
                    scored = []
                    for coord in frontier_locations:
                        fx, fy = int(coord[0]), int(coord[1])
                        dx = fx - int(agent_pos[0])
                        dy = fy - int(agent_pos[1])
                        projection = dx * dr + dy * dc
                        if projection <= 0:
                            continue
                        lateral = abs(dx * dc - dy * dr)
                        scored.append((projection, -lateral, fx, fy))
                    if scored:
                        scored.sort(reverse=True)
                        top = scored[: min(15, len(scored))]
                        bx = int(round(sum(item[2] for item in top) / len(top)))
                        by = int(round(sum(item[3] for item in top) / len(top)))
                        debug["direction_bias_mode"] = "frontier_cluster"
                        bias = (bx, by)
                        return (bias, debug) if return_debug else bias
                offset = map_size // 3
                bx = int(min(max(agent_pos[0] + dr * offset, 0), map_size - 1))
                by = int(min(max(agent_pos[1] + dc * offset, 0), map_size - 1))
                debug["direction_bias_mode"] = "fixed_offset"
                bias = (bx, by)
                return (bias, debug) if return_debug else bias

    return (None, debug) if return_debug else None


class HighLevelPlanner:
    """Sonnet-based high-level planner. Outputs semantic Strategy."""

    def __init__(self, llm_fn, max_retries: int = 2):
        """
        Args:
            llm_fn: callable(prompt=str) -> str. LLM instance with sonnet model.
            max_retries: number of retries on parse failure.
        """
        self.llm = llm_fn
        self.max_retries = max_retries
        self._call_count: int = 0
        self._choice_counts = {"object": 0, "room": 0, "direction": 0}

    def reset(self):
        self._call_count = 0
        self._choice_counts = {"object": 0, "room": 0, "direction": 0}

    @property
    def call_count(self) -> int:
        return self._call_count

    def plan(self, scene_text: str, goal_description: str,
             explored_regions: List[str],
             escalate_reason: str = "Initial planning",
             graph=None, agent_pos: Tuple[int, int] = (360, 360),
             map_size: int = 720,
             episode_id: Optional[int] = None,
             step_idx: Optional[int] = None,
             trace_writer=None) -> Strategy:
        """Single sonnet LLM call. Returns Strategy with resolved bias_position."""
        explored_text = ', '.join(explored_regions) if explored_regions else "None yet"

        # Build structured choices from current graph state
        choices_text = ""
        if graph is not None:
            choices_text = build_choices_text(graph, explored_regions)
        if not choices_text:
            choices_text = ("DIRECTIONS (pick to explore unknown areas):\n"
                           "  [direction] north\n  [direction] south\n"
                           "  [direction] east\n  [direction] west")

        prompt = HIGH_LEVEL_PROMPT.format(
            goal_description=goal_description,
            scene_text=scene_text,
            explored_text=explored_text,
            escalate_reason=escalate_reason,
            choices_text=choices_text,
        )

        parsed = None
        raw_response = ""
        error_message = ""
        used_fallback = False
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm(prompt=prompt)
                raw_response = response
                if not str(response).strip():
                    error_message = "empty_response"
                    logger.warning("HighLevelPlanner received empty response; using fallback.")
                    break
                parsed = self._parse(response)
                if parsed:
                    break
            except Exception as e:
                error_message = str(e)
                logger.warning(f"HighLevelPlanner attempt {attempt} failed: {e}")

        self._call_count += 1

        if parsed is None:
            used_fallback = True
            parsed = {'choice_type': 'direction', 'choice_id': 'north',
                      'reasoning': 'parse_failure_fallback'}

        # Resolve structured choice to map coordinates
        bias = None
        resolution_debug = {
            "resolved_object_candidates": [],
            "direction_bias_mode": "",
        }
        if graph is not None:
            bias, resolution_debug = resolve_bias_position(
                parsed,
                graph,
                agent_pos,
                map_size,
                return_debug=True,
            )

        # Map choice_type/choice_id to Strategy fields
        choice_type = parsed.get('choice_type', 'direction')
        choice_id = parsed.get('choice_id', '')
        self._choice_counts[choice_type] = self._choice_counts.get(choice_type, 0) + 1
        if choice_type == 'object':
            target_region = f"object: {choice_id}"
            anchor_object = choice_id
        elif choice_type == 'room':
            target_region = choice_id
            anchor_object = ''
        else:
            target_region = f"unexplored {choice_id}"
            anchor_object = ''

        strategy = Strategy(
            target_region=target_region,
            bias_position=bias,
            reasoning=parsed.get('reasoning', ''),
            explored_regions=list(explored_regions),
            anchor_object=anchor_object,
        )

        if trace_writer is not None and episode_id is not None:
            trace_writer.record_planner_call(
                episode_id,
                {
                    "step_idx": step_idx,
                    "trace_kind": "planner_call",
                    "schema_version": PLANNER_PROMPT_SCHEMA_VERSION,
                    "prompt_hash": hash_text(prompt),
                    "raw_prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_result": parsed,
                    "fallback_triggered": used_fallback,
                    "error_message": error_message,
                    "choice_type": choice_type,
                    "choice_id": choice_id,
                    "resolved_bias": bias,
                    "resolved_bias_position": bias,
                    "resolved_object_candidates": resolution_debug.get(
                        "resolved_object_candidates", []
                    ),
                    "direction_bias_mode": resolution_debug.get(
                        "direction_bias_mode", ""
                    ),
                    "planner_choice_distribution": dict(self._choice_counts),
                    "strategy": {
                        "target_region": strategy.target_region,
                        "bias_position": strategy.bias_position,
                        "reasoning": strategy.reasoning,
                        "explored_regions": strategy.explored_regions,
                        "anchor_object": strategy.anchor_object,
                    },
                    "goal_description": goal_description,
                    "escalate_reason": escalate_reason,
                },
            )

        logger.info(f"HighLevelPlanner: type={choice_type} id={choice_id} "
                     f"bias={strategy.bias_position} reason={strategy.reasoning}")
        return strategy

    def plan_stage_goal(self, mission_state, world_state, reason: str,
                        episode_id: Optional[int] = None,
                        step_idx: Optional[int] = None,
                        trace_writer=None) -> StageGoal:
        """Layer-2 planner interface that returns a StageGoal contract."""

        graph = world_state.graph if world_state is not None else None
        explored_regions = (
            list(getattr(world_state, "explored_regions", []) or [])
            if world_state is not None
            else []
        )
        scene_text = serialize_for_planner(graph, explored_regions) if graph else ""
        pose = getattr(world_state, "pose", {}) if world_state is not None else {}
        agent_pos = (
            int(pose.get("map_x", 360)) if isinstance(pose, dict) else 360,
            int(pose.get("map_y", 360)) if isinstance(pose, dict) else 360,
        )
        map_size = (
            int(getattr(world_state.bev_map, "args", None).map_size)
            if world_state is not None
            and getattr(world_state, "bev_map", None) is not None
            and getattr(getattr(world_state, "bev_map", None), "args", None) is not None
            and hasattr(getattr(world_state.bev_map, "args"), "map_size")
            else 720
        )
        strategy = self.plan(
            scene_text=scene_text,
            goal_description=getattr(mission_state, "mission_text", ""),
            explored_regions=explored_regions,
            escalate_reason=reason or getattr(mission_state, "replan_reason", None) or "",
            graph=graph,
            agent_pos=agent_pos,
            map_size=map_size,
            episode_id=episode_id,
            step_idx=step_idx,
            trace_writer=trace_writer,
        )
        return stage_goal_from_strategy(strategy)

    def _parse(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            choice_type = str(data.get('choice_type', '')).strip().lower()
            choice_id = str(data.get('choice_id', '')).strip()
            if choice_type not in {"object", "room", "direction"}:
                return None
            if not choice_id:
                return None
            data["choice_type"] = choice_type
            data["choice_id"] = choice_id
            return data
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
