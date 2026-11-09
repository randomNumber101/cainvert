"""Storage: Parquet I/O, DuckDB catalog, exact-catalog loaders, Hugging Face Hub helpers."""

from .parquet_io import export_trees_to_parquet, read_trees_from_parquet
from .catalog import TreeCatalog
from .exact_catalog import (
    available_splits, load_records, records_to_arrays, encode_input, hints_with_k_revealed, load_splits,
)
from .hub import download_catalog_from_hub, push_catalog_to_hub, get_hub_catalog_manifest

try:
    from .torch_dataset import CAInversionDataset
    from .exact_catalog import ExactCatalogDataset
except ImportError:  # pragma: no cover
    CAInversionDataset = None  # type: ignore[assignment]
    ExactCatalogDataset = None  # type: ignore[assignment]

__all__ = [
    "export_trees_to_parquet", "read_trees_from_parquet", "TreeCatalog",
    "available_splits", "load_records", "records_to_arrays", "encode_input", "hints_with_k_revealed", "load_splits",
    "CAInversionDataset", "ExactCatalogDataset",
    "download_catalog_from_hub", "push_catalog_to_hub", "get_hub_catalog_manifest",
]
