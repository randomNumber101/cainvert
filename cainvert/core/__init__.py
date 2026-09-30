"""Core types, filters, and mathematical solvers for cainvert."""

from .types import PreimageResult, TreeData
from .filters import compute_density, is_density_valid, is_trajectory_dynamic
from .hitting_set import solve_greedy_hitting_set, verify_disambiguation

__all__ = [
    "PreimageResult",
    "TreeData",
    "compute_density",
    "is_density_valid",
    "is_trajectory_dynamic",
    "solve_greedy_hitting_set",
    "verify_disambiguation",
]
