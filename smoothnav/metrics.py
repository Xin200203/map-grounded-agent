"""Smoothness metrics for evaluating navigation trajectory quality."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class AgentState:
    """Single timestep agent state."""
    x: float          # position x (meters)
    y: float          # position y (meters)
    heading: float    # heading in radians
    step: int         # timestep index
    action: int       # action taken (0=stop, 1=forward, 2=left, 3=right)
    is_planning: bool = False  # whether LLM was called at this step


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute shortest angular difference between two angle arrays (radians)."""
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi


@dataclass
class SmoothnessResult:
    """Container for all smoothness metrics."""
    # Core metrics
    sigma_v: float = 0.0           # speed variance
    sigma_omega: float = 0.0       # angular velocity variance
    jerk: float = 0.0              # mean absolute jerk
    pause_count: int = 0           # number of pauses (moving→stopped transitions)
    pause_duration_ratio: float = 0.0  # fraction of steps where agent is stationary
    direction_reversals: int = 0   # number of angular velocity sign changes

    # Composite score (higher = smoother)
    smoothness_score: float = 0.0

    # Efficiency metrics
    total_steps: int = 0
    llm_call_count: int = 0        # number of steps where LLM was invoked
    llm_call_ratio: float = 0.0    # fraction of steps with LLM calls

    def to_dict(self) -> dict:
        return {
            'sigma_v': self.sigma_v,
            'sigma_omega': self.sigma_omega,
            'jerk': self.jerk,
            'pause_count': self.pause_count,
            'pause_duration_ratio': self.pause_duration_ratio,
            'direction_reversals': self.direction_reversals,
            'smoothness_score': self.smoothness_score,
            'total_steps': self.total_steps,
            'llm_call_count': self.llm_call_count,
            'llm_call_ratio': self.llm_call_ratio,
        }


class SmoothnessMetrics:
    """Compute navigation smoothness metrics from agent trajectory."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.5):
        """
        Args:
            alpha: weight for sigma_v in composite score
            beta: weight for sigma_omega in composite score
            gamma: weight for pause_count/T in composite score
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.trajectory: List[AgentState] = []

    def reset(self):
        self.trajectory = []

    def record(self, state: AgentState):
        """Record a single timestep."""
        self.trajectory.append(state)

    def record_from_habitat(self, x: float, y: float, heading: float,
                            step: int, action: int, is_planning: bool = False):
        """Convenience method to record from Habitat agent state."""
        self.record(AgentState(
            x=x, y=y, heading=heading,
            step=step, action=action,
            is_planning=is_planning
        ))

    def compute(self) -> SmoothnessResult:
        """Compute all smoothness metrics from recorded trajectory."""
        if len(self.trajectory) < 3:
            return SmoothnessResult(total_steps=len(self.trajectory))

        positions = np.array([(s.x, s.y) for s in self.trajectory])
        headings = np.array([s.heading for s in self.trajectory])

        T = len(self.trajectory)

        # --- Speed (step distances) ---
        displacements = np.diff(positions, axis=0)
        speeds = np.linalg.norm(displacements, axis=1)  # (T-1,)

        # --- Angular velocity ---
        angular_velocities = angle_diff(headings[1:], headings[:-1])  # (T-1,)

        # --- Acceleration and Jerk ---
        accelerations = np.diff(speeds)  # (T-2,)
        jerks = np.diff(accelerations)   # (T-3,)

        # --- Pause detection ---
        eps = 1e-4
        paused = speeds < eps  # (T-1,)
        # Count transitions from moving to stopped
        pause_transitions = np.diff(paused.astype(int))
        pause_count = int(np.sum(pause_transitions == 1))

        # --- Direction reversals ---
        # Count sign changes in angular velocity (ignoring near-zero)
        omega_thresh = 1e-3  # radians, ignore tiny rotations
        significant_omega = np.abs(angular_velocities) > omega_thresh
        omega_signs = np.sign(angular_velocities)
        sign_changes = np.diff(omega_signs)
        # Only count reversals where both steps had significant rotation
        reversals = np.sum(
            (np.abs(sign_changes) == 2) &
            significant_omega[:-1] &
            significant_omega[1:]
        )

        # --- LLM call stats ---
        llm_calls = sum(1 for s in self.trajectory if s.is_planning)

        # --- Compute metrics ---
        sigma_v = float(np.std(speeds))
        sigma_omega = float(np.std(angular_velocities))
        mean_jerk = float(np.mean(np.abs(jerks))) if len(jerks) > 0 else 0.0
        pause_ratio = float(np.mean(paused))

        # Composite smoothness score: higher is better
        smoothness_score = 1.0 / (
            1.0
            + self.alpha * sigma_v
            + self.beta * sigma_omega
            + self.gamma * (pause_count / max(T, 1))
        )

        return SmoothnessResult(
            sigma_v=sigma_v,
            sigma_omega=sigma_omega,
            jerk=mean_jerk,
            pause_count=pause_count,
            pause_duration_ratio=pause_ratio,
            direction_reversals=int(reversals),
            smoothness_score=smoothness_score,
            total_steps=T,
            llm_call_count=llm_calls,
            llm_call_ratio=llm_calls / max(T, 1),
        )


def compute_smoothness_from_positions(
    positions: np.ndarray,
    headings: np.ndarray,
    planning_steps: Optional[List[int]] = None,
) -> SmoothnessResult:
    """
    Convenience function: compute smoothness from raw arrays.

    Args:
        positions: (T, 2) array of (x, y) positions
        headings: (T,) array of headings in radians
        planning_steps: list of step indices where LLM was called
    """
    metrics = SmoothnessMetrics()
    planning_set = set(planning_steps or [])
    for i in range(len(positions)):
        metrics.record(AgentState(
            x=positions[i, 0],
            y=positions[i, 1],
            heading=headings[i],
            step=i,
            action=1,  # placeholder
            is_planning=i in planning_set,
        ))
    return metrics.compute()
