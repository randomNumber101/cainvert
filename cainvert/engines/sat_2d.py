"""High-performance 2D SAT-based Preimage Solver for Life-like / Outer-Totalistic Cellular Automata.

Uses PySAT with state-of-the-art CDCL solvers (CaDiCaL195, Glucose4) to compute exact
predecessor configurations on 2D lattices with periodic (toroidal) boundary conditions.
"""

import itertools
import time
from typing import List, Optional, Set, Tuple, Union
import numpy as np
from pysat.formula import CNF
from pysat.solvers import Solver

from .base import BaseEngine
from ..core.types import PreimageResult


def parse_life_rule(rule_str: str) -> Tuple[Set[int], Set[int]]:
    """Parse Life-like rule strings like 'B3/S23' or 'b36/s23' into birth and survival sets."""
    rule_str = rule_str.upper().strip()
    parts = rule_str.split("/")
    b_set: Set[int] = set()
    s_set: Set[int] = set()

    for part in parts:
        if part.startswith("B"):
            b_set = {int(ch) for ch in part[1:] if ch.isdigit()}
        elif part.startswith("S"):
            s_set = {int(ch) for ch in part[1:] if ch.isdigit()}
        else:
            raise ValueError(f"Unrecognized rule part '{part}' in '{rule_str}'")

    return b_set, s_set


class SAT2DEngine(BaseEngine):
    """Exact 2D Preimage Solver for Outer-Totalistic Cellular Automata via SAT."""

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (9, 9),
        rule_str: str = "B3/S23",
        solver_name: str = "cadical195",
    ):
        self.H, self.W = grid_shape
        assert self.H >= 3 and self.W >= 3, f"Grid shape must be at least (3, 3), got {grid_shape}"
        self.total_cells = self.H * self.W
        self.rule_str = rule_str
        self.solver_name = solver_name
        self.birth_set, self.survival_set = parse_life_rule(rule_str)

        # Precompute invalid bit patterns for (x_center, n0..n7)
        # when target y=1 vs target y=0
        self.invalid_when_y1: List[Tuple[int, ...]] = []
        self.invalid_when_y0: List[Tuple[int, ...]] = []

        for bits in itertools.product([0, 1], repeat=9):
            x = bits[0]
            k = sum(bits[1:])
            is_active = (x == 0 and k in self.birth_set) or (x == 1 and k in self.survival_set)
            if not is_active:
                self.invalid_when_y1.append(bits)
            else:
                self.invalid_when_y0.append(bits)

    @property
    def dimension(self) -> str:
        return "2d"

    @property
    def rule_name(self) -> str:
        return f"life_{self.rule_str.replace('/', '_').lower()}"

    def cell_var(self, r: int, c: int) -> int:
        """1-based variable index for cell at (r, c) on toroidal grid."""
        return (r % self.H) * self.W + (c % self.W) + 1

    def forward_step(self, state: np.ndarray) -> np.ndarray:
        """Vectorized 2D forward CA step under periodic boundary conditions."""
        state = np.asarray(state, dtype=np.uint8).reshape((self.H, self.W))
        # Compute 8-neighbor sum using np.roll
        nbr_sum = np.zeros((self.H, self.W), dtype=np.int32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nbr_sum += np.roll(np.roll(state, dr, axis=0), dc, axis=1)

        next_state = np.zeros((self.H, self.W), dtype=np.uint8)
        # Birth: state == 0 and nbr_sum in birth_set
        for b in self.birth_set:
            next_state |= ((state == 0) & (nbr_sum == b)).astype(np.uint8)
        # Survival: state == 1 and nbr_sum in survival_set
        for s in self.survival_set:
            next_state |= ((state == 1) & (nbr_sum == s)).astype(np.uint8)

        return next_state

    def _build_cnf(self, target_state: np.ndarray) -> CNF:
        """Construct CNF formula constraining predecessor to produce target_state."""
        target = np.asarray(target_state, dtype=np.uint8).reshape((self.H, self.W))
        cnf = CNF()

        for r in range(self.H):
            for c in range(self.W):
                # 9 literals: center cell followed by 8 neighbors in fixed order
                lits = [self.cell_var(r, c)]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        lits.append(self.cell_var(r + dr, c + dc))

                target_val = target[r, c]
                invalids = self.invalid_when_y1 if target_val == 1 else self.invalid_when_y0

                for pattern in invalids:
                    # Pattern is forbidden -> add clause negating the pattern
                    clause = [-lits[i] if pattern[i] == 1 else lits[i] for i in range(9)]
                    cnf.append(clause)

        return cnf

    def count_preimages(self, target_state: np.ndarray, max_count: int = 128) -> int:
        """Count preimages up to max_count using All-SAT blocking clauses."""
        result = self.get_preimages(target_state, max_count=max_count)
        return result.count

    def get_preimages(
        self,
        target_state: np.ndarray,
        max_count: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> PreimageResult:
        """Enumerate exact preimages using CDCL SAT solving with blocking clauses."""
        target = np.asarray(target_state, dtype=np.uint8).reshape((self.H, self.W))
        cnf = self._build_cnf(target)

        preimages: List[np.ndarray] = []
        is_capped = False
        t_start = time.perf_counter()

        with Solver(name=self.solver_name, bootstrap_with=cnf) as solver:
            while solver.solve():
                model = set(solver.get_model())
                sol = np.zeros((self.H, self.W), dtype=np.uint8)
                blocking_clause = []

                for r in range(self.H):
                    for c in range(self.W):
                        var = self.cell_var(r, c)
                        is_one = var in model
                        sol[r, c] = 1 if is_one else 0
                        blocking_clause.append(-var if is_one else var)

                preimages.append(sol)
                solver.add_clause(blocking_clause)

                if max_count is not None and len(preimages) >= max_count:
                    is_capped = True
                    break

                if timeout_seconds is not None and (time.perf_counter() - t_start) > timeout_seconds:
                    is_capped = True
                    break

        if len(preimages) == 0:
            return PreimageResult(
                count=0,
                preimages=np.empty((0, self.H, self.W), dtype=np.uint8),
                is_capped=False,
            )

        return PreimageResult(
            count=len(preimages),
            preimages=np.array(preimages, dtype=np.uint8),
            is_capped=is_capped,
        )
