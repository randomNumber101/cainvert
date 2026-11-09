"""cainvert: exact preimage-tree catalogs and an evaluation protocol for cellular-automata inversion.

Core (numpy / pyarrow / duckdb only):
    cainvert.engines.eca_1d        exact 1D preimage enumeration (de Bruijn transfer matrix)
    cainvert.mining.exact_miner    exact, branching-controlled catalogs (schema v0.2.1)
    cainvert.storage.exact_catalog loading splits, tensors, hint control
    cainvert.eval                  verifier, solve@k, null control, strata, reference solver
Optional:
    torch (ExactCatalogDataset, residual_torch), python-sat (2D Life engine, legacy v0.1),
    huggingface_hub (catalog download / upload).
"""

__version__ = "0.2.0"

from .core.types import PreimageResult, TreeData
from .core.filters import is_density_valid, is_trajectory_dynamic
from .core.hitting_set import solve_greedy_hitting_set, verify_disambiguation
from .engines.base import BaseEngine
from .engines.eca_1d import ECA1DEngine
from .mining.tree_miner import TreeMiner
from .mining.bucket_sampler import BucketSpec, StratifiedBucketManager
from .mining.exact_miner import SCHEMA_VERSION, exact_tree, greedy_hint_order, assign_split, state_hash
from .storage.parquet_io import export_trees_to_parquet, read_trees_from_parquet
from .storage.catalog import TreeCatalog
from .storage.exact_catalog import load_records, records_to_arrays, encode_input, hints_with_k_revealed
from . import eval  # noqa: F401  (cainvert.eval.*)

try:  # legacy 2D engine needs python-sat
    from .engines.sat_2d import SAT2DEngine
except ImportError:  # pragma: no cover
    SAT2DEngine = None  # type: ignore[assignment]

try:  # torch datasets are optional
    from .storage.torch_dataset import CAInversionDataset
    from .storage.exact_catalog import ExactCatalogDataset
except ImportError:  # pragma: no cover
    CAInversionDataset = None  # type: ignore[assignment]
    ExactCatalogDataset = None  # type: ignore[assignment]

__all__ = [
    "__version__", "SCHEMA_VERSION",
    "PreimageResult", "TreeData",
    "is_density_valid", "is_trajectory_dynamic",
    "solve_greedy_hitting_set", "verify_disambiguation",
    "BaseEngine", "ECA1DEngine", "SAT2DEngine",
    "TreeMiner", "BucketSpec", "StratifiedBucketManager",
    "exact_tree", "greedy_hint_order", "assign_split", "state_hash",
    "export_trees_to_parquet", "read_trees_from_parquet", "TreeCatalog",
    "load_records", "records_to_arrays", "encode_input", "hints_with_k_revealed",
    "CAInversionDataset", "ExactCatalogDataset",
    "eval",
]
