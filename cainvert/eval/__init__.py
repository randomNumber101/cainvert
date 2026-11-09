"""Evaluation protocol for CAInvert: verifier, solve@k, null control, strata, reference solver."""

from .verifier import eca_forward, residual, is_solution, check_ground_truth
from .metrics import pass_at_k, null_control, layer_passes, strata, summarise, evaluate_rollouts
from .reference import solve_exact, solve_records

__all__ = [
    "eca_forward", "residual", "is_solution", "check_ground_truth",
    "pass_at_k", "null_control", "layer_passes", "strata", "summarise", "evaluate_rollouts",
    "solve_exact", "solve_records",
]
