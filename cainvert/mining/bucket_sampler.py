"""2D Stratified Bucket Sampler ensuring uniform coverage across Depth and Leaf-Count / Branching."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.types import TreeData


@dataclass
class BucketSpec:
    """Specification of 2D difficulty buckets."""
    depths: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    leaf_bounds: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 1),        # Pure bijective recursion
            (2, 8),        # Low branching
            (9, 64),       # Moderate branching
            (65, 512),     # Broad branching
            (513, 4096),   # Explosive combinatorial search
        ]
    )
    samples_per_depth: int = 500
    max_per_leaf_bucket: int = 150  # Cap per leaf-bucket within each depth to ensure diversity

    def get_leaf_bucket_idx(self, leaf_count: int) -> Optional[int]:
        """Return the index of leaf bound range for a given leaf count."""
        for idx, (low, high) in enumerate(self.leaf_bounds):
            if low <= leaf_count <= high:
                return idx
        if leaf_count > self.leaf_bounds[-1][1]:
            return len(self.leaf_bounds) - 1
        if leaf_count < self.leaf_bounds[0][0]:
            return 0
        return None

    def get_depth_idx(self, depth: int) -> Optional[int]:
        """Return index of depth in bucket specification."""
        if depth in self.depths:
            return self.depths.index(depth)
        return None


class StratifiedBucketManager:
    """Tracks and balances mined samples across depths and leaf bounds."""

    def __init__(self, spec: Optional[BucketSpec] = None):
        self.spec = spec or BucketSpec()
        self.num_depths = len(self.spec.depths)
        self.num_leaf_buckets = len(self.spec.leaf_bounds)
        self.counts = np.zeros((self.num_depths, self.num_leaf_buckets), dtype=np.int32)
        self.collected_samples: Dict[Tuple[int, int], List[TreeData]] = {
            (d_i, l_i): []
            for d_i in range(self.num_depths)
            for l_i in range(self.num_leaf_buckets)
        }

    def depth_count(self, d_idx: int) -> int:
        return int(np.sum(self.counts[d_idx, :]))

    def can_accept(self, tree: TreeData, allow_fallback: bool = False) -> bool:
        """Check if this tree belongs to an under-filled depth and leaf bucket."""
        d_idx = self.spec.get_depth_idx(tree.depth)
        l_idx = self.spec.get_leaf_bucket_idx(tree.leaf_count)
        if d_idx is None or l_idx is None:
            return False
        if self.depth_count(d_idx) >= self.spec.samples_per_depth:
            return False
        if self.counts[d_idx, l_idx] < self.spec.max_per_leaf_bucket:
            return True
        return allow_fallback

    def add_tree(self, tree: TreeData, allow_fallback: bool = False) -> bool:
        """Add tree to bucket if capacity remains. Returns True if accepted."""
        if not self.can_accept(tree, allow_fallback=allow_fallback):
            return False

        d_idx = self.spec.get_depth_idx(tree.depth)
        l_idx = self.spec.get_leaf_bucket_idx(tree.leaf_count)
        self.counts[d_idx, l_idx] += 1
        self.collected_samples[(d_idx, l_idx)].append(tree)
        return True

    def is_complete(self) -> bool:
        """True if all depths have reached their target sample count."""
        return all(self.depth_count(d_idx) >= self.spec.samples_per_depth for d_idx in range(self.num_depths))

    def total_collected(self) -> int:
        return int(np.sum(self.counts))

    def get_fill_percentage(self) -> float:
        total_target = self.num_depths * self.spec.samples_per_depth
        return float(self.total_collected() / max(1, total_target) * 100.0)

    def summary_table(self) -> str:
        """Format fill matrix as human-readable table."""
        headers = ["Depth"] + [f"L:[{low}-{high}]" for low, high in self.spec.leaf_bounds]
        lines = [" | ".join(f"{h:<12}" for h in headers)]
        lines.append("-" * len(lines[0]))

        for d_i, d in enumerate(self.spec.depths):
            row = [f"d={d:<10}"]
            for l_i in range(self.num_leaf_buckets):
                cnt = self.counts[d_i, l_i]
                row.append(f"{cnt:<12}")
            row.append(f"Total: {self.depth_count(d_i)}/{self.spec.samples_per_depth}")
            lines.append(" | ".join(row))

        return "\n".join(lines)
