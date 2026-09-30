"""Preimage solvers and simulation engines for 1D and 2D cellular automata."""

from .base import BaseEngine
from .eca_1d import ECA1DEngine
from .sat_2d import SAT2DEngine

__all__ = ["BaseEngine", "ECA1DEngine", "SAT2DEngine"]
