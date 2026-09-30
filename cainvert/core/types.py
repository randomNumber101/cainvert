"""Core data structures and types for the cainvert benchmark engine."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PreimageResult:
    """Represents pre-images found for a given state."""
    count: int
    preimages: np.ndarray  # Shape: (K, *grid_shape)
    is_capped: bool = False


@dataclass
class TreeData:
    """Represents an unrolled pre-image computation tree."""
    tree_id: str
    dimension: str  # "1d" or "2d"
    rule_name: str
    grid_shape: Tuple[int, ...]  # (W,) or (H, W)
    depth: int
    leaf_count: int
    effective_branching: float
    root_target: np.ndarray       # Observed state X_d
    ground_truth_s0: np.ndarray   # True ancestor X_0*
    ground_truth_leaf_idx: int
    leaves_s0: np.ndarray         # Shape (leaf_count, *grid_shape)
    trajectory: List[np.ndarray]  # [X_0, X_1, ..., X_d]
    hint_indices: List[int]
    hints: np.ndarray             # Flattened hint tape (-1 for mask, {0,1} for hint)
    hint_density: float
    mean_activity: float
    x0_hash: str = ""
    tree_uid: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_flattened_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for storage / serialization."""
        return {
            "tree_id": self.tree_id,
            "tree_uid": self.tree_uid or self.tree_id,
            "x0_hash": self.x0_hash,
            "dimension": self.dimension,
            "rule_name": self.rule_name,
            "grid_shape": list(self.grid_shape),
            "depth": self.depth,
            "leaf_count": self.leaf_count,
            "effective_branching": self.effective_branching,
            "target_state": self.root_target.flatten().tolist(),
            "ground_truth_s0": self.ground_truth_s0.flatten().tolist(),
            "hints_s0": self.hints.flatten().tolist(),
            "num_hints": len(self.hint_indices),
            "hint_density": self.hint_density,
            "mean_activity": self.mean_activity,
            "metadata": self.metadata,
        }
