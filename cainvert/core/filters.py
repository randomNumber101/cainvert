"""State and trajectory activity filters for rejecting trivial / degenerate dynamics."""

from typing import List, Tuple
import numpy as np


def compute_density(state: np.ndarray) -> float:
    """Compute the active (1s) fraction of a binary state array."""
    return float(np.mean(state == 1))


def is_density_valid(state: np.ndarray, min_density: float = 0.25, max_density: float = 0.75) -> bool:
    """Check if state density is within [min_density, max_density]."""
    rho = compute_density(state)
    return min_density <= rho <= max_density


def compute_step_hamming_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Normalized Hamming distance between two consecutive states."""
    return float(np.mean(s1 != s2))


def is_trajectory_dynamic(
    trajectory: List[np.ndarray],
    min_density: float = 0.25,
    max_density: float = 0.75,
    min_step_activity: float = 0.05,
) -> Tuple[bool, str]:
    """Validate that every state in trajectory is non-trivial and actively mutating.

    Args:
        trajectory: List of consecutive states [X_0, X_1, ..., X_d].
        min_density: Lower bound on active 1-bits (default: 0.25).
        max_density: Upper bound on active 1-bits (default: 0.75).
        min_step_activity: Minimum normalized Hamming change between steps.

    Returns:
        (is_valid, reason_if_invalid)
    """
    for t, s in enumerate(trajectory):
        rho = compute_density(s)
        if rho < min_density:
            return False, f"State at t={t} too sparse: density={rho:.3f} < {min_density}"
        if rho > max_density:
            return False, f"State at t={t} too dense: density={rho:.3f} > {max_density}"

    # Check temporal mutation rate
    for t in range(len(trajectory) - 1):
        activity = compute_step_hamming_distance(trajectory[t], trajectory[t + 1])
        if activity < min_step_activity:
            return False, f"Trajectory frozen / still-life at step {t}->{t+1}: activity={activity:.3f} < {min_step_activity}"

    return True, "valid"
