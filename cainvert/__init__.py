"""cainvert: Scalable 1D & 2D Cellular Automata Preimage Tree Mining & Benchmark Engine."""

__version__ = "0.1.0"

from .core.types import PreimageResult, TreeData
from .core.filters import is_density_valid, is_trajectory_dynamic
from .core.hitting_set import solve_greedy_hitting_set, verify_disambiguation
from .engines.base import BaseEngine
from .engines.eca_1d import ECA1DEngine
from .engines.sat_2d import SAT2DEngine
from .mining.tree_miner import TreeMiner
from .mining.bucket_sampler import BucketSpec, StratifiedBucketManager
from .storage.parquet_io import export_trees_to_parquet, read_trees_from_parquet
from .storage.catalog import TreeCatalog
from .storage.torch_dataset import CAInversionDataset

__all__ = [
    "__version__",
    "PreimageResult",
    "TreeData",
    "is_density_valid",
    "is_trajectory_dynamic",
    "solve_greedy_hitting_set",
    "verify_disambiguation",
    "BaseEngine",
    "ECA1DEngine",
    "SAT2DEngine",
    "TreeMiner",
    "BucketSpec",
    "StratifiedBucketManager",
    "export_trees_to_parquet",
    "read_trees_from_parquet",
    "TreeCatalog",
    "CAInversionDataset",
]
