"""High-performance Trellis / De Bruijn Transfer-Matrix Preimage Solver for 1D Elementary Cellular Automata.

Computes exact predecessor states of a 1D lattice of width W with periodic boundary conditions
in linear time O(16 * W).
"""

from typing import Optional
import numpy as np

from .base import BaseEngine
from ..core.types import PreimageResult


class ECA1DEngine(BaseEngine):
    """Exact Preimage Solver for 1D Elementary Cellular Automata (Radius r=1)."""

    def __init__(self, rule_number: int = 110):
        self._rule_number = int(rule_number)
        self.rule_table = np.array([(self._rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)

        # 4 states in De Bruijn graph: u = (s_{i-1}, s_i) in {0, 1, 2, 3}
        self.transitions = {0: [[] for _ in range(4)], 1: [[] for _ in range(4)]}
        self.M = {0: np.zeros((4, 4), dtype=np.int64), 1: np.zeros((4, 4), dtype=np.int64)}

        for u in range(4):
            u0 = (u >> 1) & 1
            u1 = u & 1
            for b in (0, 1):
                v = (u1 << 1) | b
                idx = (u0 << 2) | (u1 << 1) | b
                out_bit = int(self.rule_table[idx])
                self.transitions[out_bit][u].append((v, b))
                self.M[out_bit][u, v] += 1

    @property
    def dimension(self) -> str:
        return "1d"

    @property
    def rule_name(self) -> str:
        return f"rule_{self._rule_number}"

    def forward_step(self, state: np.ndarray) -> np.ndarray:
        """Evaluate one forward step under periodic boundary conditions."""
        state = np.asarray(state, dtype=np.uint8).flatten()
        l = np.roll(state, 1)
        c = state
        r = np.roll(state, -1)
        idx = (l << 2) | (c << 1) | r
        return self.rule_table[idx]

    def count_preimages(self, target_state: np.ndarray) -> int:
        """Compute exact number of preimages using transfer matrix trace product in O(16 * W)."""
        target_state = np.asarray(target_state, dtype=np.uint8).flatten()
        T = np.eye(4, dtype=object)
        for bit in target_state:
            T = T @ self.M[int(bit)]
        return int(np.trace(T))

    def get_preimages(
        self,
        target_state: np.ndarray,
        max_count: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> PreimageResult:
        """Enumerate exact preimages using DP reachability and cycle backtracking."""
        target_state = np.asarray(target_state, dtype=np.uint8).flatten()
        W = len(target_state)

        num_pre = self.count_preimages(target_state)
        if num_pre == 0:
            return PreimageResult(count=0, preimages=np.empty((0, W), dtype=np.uint8))

        # Backward DP reachability filter:
        # reach[i, u, u_init] = True if state u at pos i can reach u_init at pos W
        reach = np.zeros((W + 1, 4, 4), dtype=bool)
        for u in range(4):
            reach[W, u, u] = True

        for i in range(W - 1, -1, -1):
            bit = int(target_state[i])
            for u in range(4):
                for v, b in self.transitions[bit][u]:
                    reach[i, u, :] |= reach[i + 1, v, :]

        # Forward cycle backtracking
        preimages = []
        for u_init in range(4):
            if not reach[0, u_init, u_init]:
                continue
            stack = [(0, u_init, [])]
            while stack:
                pos, u, bits = stack.pop()
                if pos == W:
                    if u == u_init:
                        s0 = int(u_init & 1)
                        preimage = np.array([s0] + bits[:-1], dtype=np.uint8)
                        preimages.append(preimage)
                        if max_count is not None and len(preimages) >= max_count:
                            return PreimageResult(
                                count=num_pre,
                                preimages=np.array(preimages, dtype=np.uint8),
                                is_capped=True,
                            )
                    continue

                bit = int(target_state[pos])
                for v, b in self.transitions[bit][u]:
                    if reach[pos + 1, v, u_init]:
                        stack.append((pos + 1, v, bits + [b]))

        return PreimageResult(
            count=len(preimages),
            preimages=np.array(preimages, dtype=np.uint8),
            is_capped=False,
        )
