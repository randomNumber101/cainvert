"""Abstract BaseEngine for 1D and 2D Cellular Automata Preimage Solvers."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from ..core.types import PreimageResult


class BaseEngine(ABC):
    """Abstract interface for CA forward execution and exact backward preimage solving."""

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Return '1d' or '2d'."""
        pass

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Name of the cellular automaton rule."""
        pass

    @abstractmethod
    def forward_step(self, state: np.ndarray) -> np.ndarray:
        """Evaluate one forward deterministic CA step under periodic boundary conditions."""
        pass

    @abstractmethod
    def count_preimages(self, target_state: np.ndarray) -> int:
        """Compute exact number of preimages for target_state."""
        pass

    @abstractmethod
    def get_preimages(
        self,
        target_state: np.ndarray,
        max_count: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> PreimageResult:
        """Enumerate all exact preimages for target_state.

        Args:
            target_state: Target state array.
            max_count: If specified, stops searching once max_count preimages are found.
            timeout_seconds: Maximum time allowed for search.

        Returns:
            PreimageResult with count, array of preimages, and is_capped flag.
        """
        pass
