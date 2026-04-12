"""Unit tests for SmoothNav smoothness metrics."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from smoothnav.metrics import (
    SmoothnessMetrics, SmoothnessResult, AgentState,
    angle_diff, compute_smoothness_from_positions,
)


def test_angle_diff():
    """Test shortest angular difference."""
    a = np.array([0.1, np.pi - 0.1, -np.pi + 0.1])
    b = np.array([0.0, np.pi + 0.1, np.pi - 0.1])
    diff = angle_diff(a, b)
    np.testing.assert_allclose(diff[0], 0.1, atol=1e-10)
    # Wrapping around pi
    np.testing.assert_allclose(diff[1], -0.2, atol=1e-10)
    np.testing.assert_allclose(diff[2], 0.2, atol=1e-10)
    print("PASS: angle_diff")


def test_straight_line():
    """Perfectly smooth: straight line, constant speed, no turns."""
    m = SmoothnessMetrics()
    for i in range(20):
        m.record(AgentState(x=i * 0.25, y=0.0, heading=0.0,
                            step=i, action=1, is_planning=False))
    r = m.compute()
    assert r.sigma_v < 1e-10, f"sigma_v should be ~0, got {r.sigma_v}"
    assert r.sigma_omega < 1e-10, f"sigma_omega should be ~0, got {r.sigma_omega}"
    assert r.pause_count == 0
    assert r.direction_reversals == 0
    assert r.smoothness_score > 0.99, f"score should be ~1.0, got {r.smoothness_score}"
    print(f"PASS: straight_line (score={r.smoothness_score:.4f})")


def test_zigzag():
    """Jerky: alternating left-right turns."""
    m = SmoothnessMetrics()
    heading = 0.0
    x, y = 0.0, 0.0
    for i in range(30):
        turn = 0.5 if i % 2 == 0 else -0.5  # alternating ±0.5 rad
        heading += turn
        x += 0.25 * np.cos(heading)
        y += 0.25 * np.sin(heading)
        m.record(AgentState(x=x, y=y, heading=heading,
                            step=i, action=2 if turn > 0 else 3))
    r = m.compute()
    assert r.direction_reversals > 10, f"expected many reversals, got {r.direction_reversals}"
    assert r.sigma_omega > 0.1, f"sigma_omega should be high, got {r.sigma_omega}"
    print(f"PASS: zigzag (reversals={r.direction_reversals}, sigma_omega={r.sigma_omega:.4f}, score={r.smoothness_score:.4f})")


def test_stop_and_go():
    """Jerky: alternating between moving and stopping."""
    m = SmoothnessMetrics()
    x = 0.0
    for i in range(30):
        if i % 3 == 0:
            # stopped
            m.record(AgentState(x=x, y=0.0, heading=0.0, step=i, action=0))
        else:
            x += 0.25
            m.record(AgentState(x=x, y=0.0, heading=0.0, step=i, action=1))
    r = m.compute()
    assert r.pause_count >= 5, f"expected pauses, got {r.pause_count}"
    assert r.pause_duration_ratio > 0.2, f"expected high pause ratio, got {r.pause_duration_ratio}"
    assert r.sigma_v > 0.05, f"sigma_v should be high, got {r.sigma_v}"
    print(f"PASS: stop_and_go (pauses={r.pause_count}, pause_ratio={r.pause_duration_ratio:.3f}, score={r.smoothness_score:.4f})")


def test_smooth_curve():
    """Smooth navigation: gentle curve."""
    m = SmoothnessMetrics()
    for i in range(40):
        angle = i * 0.05  # gentle constant turn rate
        x = 2.0 * np.sin(angle)
        y = 2.0 * (1 - np.cos(angle))
        m.record(AgentState(x=x, y=y, heading=angle,
                            step=i, action=1))
    r = m.compute()
    assert r.direction_reversals == 0, f"no reversals expected, got {r.direction_reversals}"
    assert r.pause_count == 0
    assert r.smoothness_score > 0.5, f"should be reasonably smooth, got {r.smoothness_score}"
    print(f"PASS: smooth_curve (score={r.smoothness_score:.4f}, sigma_omega={r.sigma_omega:.6f})")


def test_llm_tracking():
    """Verify LLM call counting."""
    m = SmoothnessMetrics()
    for i in range(20):
        m.record(AgentState(x=i * 0.25, y=0.0, heading=0.0,
                            step=i, action=1, is_planning=(i % 10 == 0)))
    r = m.compute()
    assert r.llm_call_count == 2, f"expected 2 LLM calls, got {r.llm_call_count}"
    assert abs(r.llm_call_ratio - 0.1) < 1e-10
    print(f"PASS: llm_tracking (calls={r.llm_call_count}, ratio={r.llm_call_ratio:.3f})")


def test_short_trajectory():
    """Edge case: very short trajectory."""
    m = SmoothnessMetrics()
    m.record(AgentState(x=0, y=0, heading=0, step=0, action=1))
    r = m.compute()
    assert r.total_steps == 1
    assert r.smoothness_score == 0.0
    print("PASS: short_trajectory")


def test_convenience_function():
    """Test compute_smoothness_from_positions."""
    positions = np.array([[i * 0.25, 0.0] for i in range(15)])
    headings = np.zeros(15)
    r = compute_smoothness_from_positions(positions, headings, planning_steps=[0, 7])
    assert r.llm_call_count == 2
    assert r.sigma_v < 1e-10
    print(f"PASS: convenience_function (score={r.smoothness_score:.4f})")


def test_comparison_smooth_vs_jerky():
    """The core test: smooth trajectory should score higher than jerky one."""
    # Smooth
    m_smooth = SmoothnessMetrics()
    for i in range(40):
        angle = i * 0.05
        x = 2.0 * np.sin(angle)
        y = 2.0 * (1 - np.cos(angle))
        m_smooth.record(AgentState(x=x, y=y, heading=angle, step=i, action=1))
    r_smooth = m_smooth.compute()

    # Jerky (zigzag + pauses)
    m_jerky = SmoothnessMetrics()
    heading = 0.0
    x, y = 0.0, 0.0
    for i in range(40):
        if i % 5 == 0:
            m_jerky.record(AgentState(x=x, y=y, heading=heading, step=i, action=0))
        else:
            turn = 0.4 if i % 2 == 0 else -0.4
            heading += turn
            x += 0.25 * np.cos(heading)
            y += 0.25 * np.sin(heading)
            m_jerky.record(AgentState(x=x, y=y, heading=heading, step=i, action=1))
    r_jerky = m_jerky.compute()

    assert r_smooth.smoothness_score > r_jerky.smoothness_score, \
        f"smooth ({r_smooth.smoothness_score:.4f}) should beat jerky ({r_jerky.smoothness_score:.4f})"
    assert r_smooth.direction_reversals < r_jerky.direction_reversals
    assert r_smooth.pause_count < r_jerky.pause_count
    print(f"PASS: comparison (smooth={r_smooth.smoothness_score:.4f} > jerky={r_jerky.smoothness_score:.4f})")


if __name__ == "__main__":
    test_angle_diff()
    test_straight_line()
    test_zigzag()
    test_stop_and_go()
    test_smooth_curve()
    test_llm_tracking()
    test_short_trajectory()
    test_convenience_function()
    test_comparison_smooth_vs_jerky()
    print("\nAll tests passed!")
