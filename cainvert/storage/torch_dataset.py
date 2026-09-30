"""PyTorch Dataset integration for SQL-queried CA Inversion Catalogs.

Provides dynamic, fast dataset creation from Parquet Tree Catalogs via DuckDB queries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

from cainvert.storage.catalog import TreeCatalog


class CAInversionDataset(Dataset):
    """Dynamic PyTorch Dataset constructed from Parquet Tree Catalogs via SQL queries.

    Allows filtering by depth, leaf count, branching factor, dimension, and rules
    at SQL speeds across millions of samples.

    Constructs standard sequence tensors for recursive model training:
        Input: [S_D (N tokens)] + [SEP=3] + [Hints_S0 (N tokens, where 2 is MASK)]
        Target: Ground Truth S_0 (N tokens)

    Example:
        >>> # Dynamic slice: only depths 4 to 8 with non-trivial ambiguity
        >>> dataset = CAInversionDataset(
        ...     catalog_path="data/catalogs/catalog_1d_w81",
        ...     depth_min=4,
        ...     depth_max=8,
        ...     leaf_min=16,
        ... )
        >>> sample = dataset[0]
        >>> input_ids, target, depth = sample["input_ids"], sample["target"], sample["depth"]
    """

    def __init__(
        self,
        catalog_path: str,
        depth_min: Optional[int] = None,
        depth_max: Optional[int] = None,
        leaf_min: Optional[int] = None,
        leaf_max: Optional[int] = None,
        dimension: Optional[str] = None,
        rule_name: Optional[str] = None,
        sql_filter: Optional[str] = None,
        limit: Optional[int] = None,
        sep_token_id: int = 3,
        mask_token_id: int = 2,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use CAInversionDataset. Please install torch.")

        super().__init__()
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id

        # Normalize catalog glob
        if os.path.isdir(catalog_path):
            catalog_glob = os.path.join(catalog_path, "**", "*.parquet")
        elif catalog_path.endswith(".parquet") and "*" not in catalog_path:
            catalog_glob = catalog_path
        else:
            catalog_glob = catalog_path

        self.catalog = TreeCatalog(catalog_glob)
        
        # Build query
        conditions = []
        if depth_min is not None:
            conditions.append(f"depth >= {depth_min}")
        if depth_max is not None:
            conditions.append(f"depth <= {depth_max}")
        if leaf_min is not None:
            conditions.append(f"leaf_count >= {leaf_min}")
        if leaf_max is not None:
            conditions.append(f"leaf_count <= {leaf_max}")
        if dimension is not None:
            conditions.append(f"dimension = '{dimension}'")
        if rule_name is not None:
            conditions.append(f"rule_name = '{rule_name}'")
        if sql_filter is not None:
            conditions.append(f"({sql_filter})")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        sql = f"""
            SELECT 
                tree_id, dimension, rule_name, grid_shape, depth,
                leaf_count, effective_branching, num_hints, hint_density, mean_activity,
                target_state, ground_truth_s0, hints_s0
            FROM read_parquet('{self.catalog.parquet_glob}', union_by_name=True)
            {where_clause}
            {limit_clause}
        """

        self.records: List[Dict[str, Any]] = self.catalog.conn.execute(sql).to_arrow_table().to_pylist() if self.catalog.has_files() else []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]

        target_state = np.frombuffer(row["target_state"], dtype=np.uint8)
        ground_truth = np.frombuffer(row["ground_truth_s0"], dtype=np.uint8)
        raw_hints = np.frombuffer(row["hints_s0"], dtype=np.int8)

        # Convert hints: -1 (unrevealed) -> mask_token_id (2), 0/1 remain unchanged
        hint_tokens = np.where(raw_hints == -1, self.mask_token_id, raw_hints).astype(np.int64)

        # Construct 2N+1 sequence: [S_D] + [SEP] + [H_0]
        sep = np.array([self.sep_token_id], dtype=np.int64)
        input_seq = np.concatenate([target_state.astype(np.int64), sep, hint_tokens])

        return {
            "input_ids": torch.from_numpy(input_seq),
            "target": torch.from_numpy(ground_truth.astype(np.int64)),
            "hint_mask": torch.from_numpy(raw_hints != -1),
            "depth": torch.tensor(row["depth"], dtype=torch.long),
            "leaf_count": row["leaf_count"],
            "effective_branching": row["effective_branching"],
            "tree_id": row["tree_id"],
        }
