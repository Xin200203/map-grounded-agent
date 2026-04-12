"""Legacy generation-1 tests kept only for reference, not default collection."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from smoothnav.planner import Planner, Subgoal, PlanResult
from smoothnav.smooth_executor import SmoothExecutor, HabitatAction, ReplanTrigger


# --- Mock LLM ---
class MockLLM:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def __call__(self, prompt="", **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = '[]'
        self.call_count += 1
        return resp


# ===================== Planner Tests =====================

def test_decompose_basic():
    llm = MockLLM([
        '[{"landmark": "stairs", "action": "walk_to", "spatial_hint": "ahead"},'
        '{"landmark": "door", "action": "go_through", "spatial_hint": "on left"}]'
    ])
    p = Planner(llm_fn=llm)
    sgs = p.decompose_instruction("Walk to the stairs and go through the door on the left")
    assert len(sgs) == 2
    assert sgs[0].landmark == "stairs"
    assert sgs[1].action == "go_through"
    assert p.plan_count == 1
    print("PASS: decompose_basic")


def test_decompose_fallback():
    """When LLM returns garbage, should fallback to default subgoal."""
    llm = MockLLM(["this is not json"])
    p = Planner(llm_fn=llm)
    sgs = p.decompose_instruction("Go somewhere")
    assert len(sgs) == 1
    assert sgs[0].landmark == "destination"
    print("PASS: decompose_fallback")


def test_subgoal_progression():
    llm = MockLLM([
        '[{"landmark": "A", "action": "walk_to", "spatial_hint": ""},'
        '{"landmark": "B", "action": "stop_near", "spatial_hint": ""}]'
    ])
    p = Planner(llm_fn=llm)
    p.decompose_instruction("Go to A then B")

    assert p.get_current_subgoal().landmark == "A"
    assert not p.is_done()

    has_more = p.advance_subgoal()
    assert has_more
    assert p.get_current_subgoal().landmark == "B"

    has_more = p.advance_subgoal()
    assert not has_more
    assert p.is_done()
    print("PASS: subgoal_progression")


def test_plan_response():
    llm = MockLLM([
        '[{"landmark": "chair", "action": "walk_to", "spatial_hint": ""}]',
        '{"target_type": "frontier", "target_id": 2, "reason": "unexplored area"}'
    ])
    p = Planner(llm_fn=llm)
    p.decompose_instruction("Find the chair")
    result = p.plan("SCENE: empty", (100, 100), "F0: (50,50)\nF1: (150,50)\nF2: (100,200)")
    assert result is not None
    assert result.target_type == "frontier"
    assert result.target_id == 2
    assert p.plan_count == 2  # 1 decompose + 1 plan
    print("PASS: plan_response")


def test_remaining_landmarks():
    llm = MockLLM([
        '[{"landmark": "A", "action": "walk_to", "spatial_hint": ""},'
        '{"landmark": "B", "action": "turn_left", "spatial_hint": ""},'
        '{"landmark": "C", "action": "stop_near", "spatial_hint": ""}]'
    ])
    p = Planner(llm_fn=llm)
    p.decompose_instruction("A B C")
    assert p.get_remaining_landmarks() == ["A", "B", "C"]
    p.advance_subgoal()
    assert p.get_remaining_landmarks() == ["B", "C"]
    print("PASS: remaining_landmarks")


# ===================== Executor Tests =====================

def test_goal_management():
    """Test set/get/clear/needs_replan cycle."""
    ex = SmoothExecutor()
    assert ex.needs_replan()
    assert ex.get_goal() is None

    ex.set_goal((50, 80), source="planner")
    assert not ex.needs_replan()
    assert ex.get_goal() == (50, 80)

    ex.clear_goal()
    assert ex.needs_replan()
    assert ex.get_goal() is None
    print("PASS: goal_management")


def test_interrupt_stuck():
    """Stuck detection after N steps with no progress."""
    ex = SmoothExecutor(stuck_threshold=5)
    ex.set_goal((50, 80))
    pos = np.array([1.0, 1.0])

    for i in range(5):
        trigger = ex.check_interrupts(pos)
        assert trigger == ReplanTrigger.NONE, f"Step {i}: expected NONE, got {trigger}"

    # 6th check (5 no-progress after first baseline) → stuck
    trigger = ex.check_interrupts(pos)
    assert trigger == ReplanTrigger.STUCK
    print("PASS: interrupt_stuck")


def test_interrupt_found_goal():
    ex = SmoothExecutor()
    ex.set_goal((50, 80))
    trigger = ex.check_interrupts(np.array([1.0, 1.0]), found_goal=True)
    assert trigger == ReplanTrigger.GOAL_VISIBLE
    print("PASS: interrupt_found_goal")


def test_interrupt_subgoal_reached():
    ex = SmoothExecutor()
    ex.set_goal((50, 80))
    trigger = ex.check_interrupts(np.array([1.0, 1.0]), subgoal_reached=True)
    assert trigger == ReplanTrigger.SUBGOAL_REACHED
    print("PASS: interrupt_subgoal_reached")


def test_smooth_override_oscillation():
    """L-R-L pattern should be overridden with FORWARD."""
    ex = SmoothExecutor()
    L, R, F = HabitatAction.TURN_LEFT, HabitatAction.TURN_RIGHT, HabitatAction.MOVE_FORWARD

    ex.record_action(L)
    ex.record_action(R)
    ex.record_action(L)
    override = ex.get_smooth_override()
    assert override == int(F), f"Expected FORWARD override, got {override}"

    # Same for R-L-R
    ex.reset()
    ex.record_action(R)
    ex.record_action(L)
    ex.record_action(R)
    override = ex.get_smooth_override()
    assert override == int(F)
    print("PASS: smooth_override_oscillation")


def test_smooth_override_spinning():
    """Many consecutive turns should be broken with FORWARD."""
    ex = SmoothExecutor(max_consecutive_turns=4)
    L = HabitatAction.TURN_LEFT

    for _ in range(4):
        ex.record_action(L)
    override = ex.get_smooth_override()
    assert override == int(HabitatAction.MOVE_FORWARD)
    print("PASS: smooth_override_spinning")


def test_smooth_no_override():
    """Normal action patterns should not trigger override."""
    ex = SmoothExecutor()
    F = HabitatAction.MOVE_FORWARD

    ex.record_action(F)
    ex.record_action(F)
    ex.record_action(F)
    override = ex.get_smooth_override()
    assert override is None
    print("PASS: smooth_no_override")


def test_executor_reset():
    ex = SmoothExecutor()
    ex.set_goal((10, 20))
    ex.record_action(1)
    ex.record_action(2)
    ex.reset()
    assert ex.needs_replan()
    assert ex.get_goal() is None
    assert len(ex.action_history) == 0
    assert ex.total_overrides == 0
    print("PASS: executor_reset")


if __name__ == "__main__":
    test_decompose_basic()
    test_decompose_fallback()
    test_subgoal_progression()
    test_plan_response()
    test_remaining_landmarks()
    test_goal_management()
    test_interrupt_stuck()
    test_interrupt_found_goal()
    test_interrupt_subgoal_reached()
    test_smooth_override_oscillation()
    test_smooth_override_spinning()
    test_smooth_no_override()
    test_executor_reset()
    print("\nAll tests passed!")
