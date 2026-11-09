"""Preimage solvers and simulation engines: exact 1D ECA (core) and 2D Life via SAT (legacy, needs python-sat)."""

from .base import BaseEngine
from .eca_1d import ECA1DEngine

try:
    from .sat_2d import SAT2DEngine
except ImportError:  # python-sat not installed: 2D engine unavailable
    SAT2DEngine = None  # type: ignore[assignment]

__all__ = ["BaseEngine", "ECA1DEngine", "SAT2DEngine"]
