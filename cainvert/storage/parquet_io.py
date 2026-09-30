"""High-efficiency Apache Arrow / Parquet storage for massive tree collections."""

import os
from typing import Dict, List, Optional, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.types import TreeData


import hashlib


def get_parquet_schema() -> pa.Schema:
    """Standardized Arrow schema for tree catalog."""
    return pa.schema([
        ("tree_id", pa.string()),
        ("tree_uid", pa.string()),
        ("x0_hash", pa.string()),
        ("dimension", pa.string()),
        ("rule_name", pa.string()),
        ("grid_shape", pa.list_(pa.int32())),
        ("depth", pa.int32()),
        ("leaf_count", pa.int32()),
        ("effective_branching", pa.float32()),
        ("num_hints", pa.int32()),
        ("hint_density", pa.float32()),
        ("mean_activity", pa.float32()),
        ("target_state", pa.binary()),      # Bit-packed or raw uint8 bytes
        ("ground_truth_s0", pa.binary()),  # Raw uint8 bytes
        ("hints_s0", pa.binary()),         # Raw int8 bytes (-1, 0, 1)
    ])


def export_trees_to_parquet(
    trees: List[TreeData],
    output_path: str,
    compression: str = "snappy",
) -> str:
    """Export a list of TreeData objects to a compact Parquet file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    records = {
        "tree_id": [],
        "tree_uid": [],
        "x0_hash": [],
        "dimension": [],
        "rule_name": [],
        "grid_shape": [],
        "depth": [],
        "leaf_count": [],
        "effective_branching": [],
        "num_hints": [],
        "hint_density": [],
        "mean_activity": [],
        "target_state": [],
        "ground_truth_s0": [],
        "hints_s0": [],
    }

    for tree in trees:
        x0_bytes = tree.ground_truth_s0.astype(np.uint8).tobytes()
        x0_h = tree.x0_hash if tree.x0_hash else hashlib.sha256(x0_bytes).hexdigest()[:16]
        uid = tree.tree_uid if tree.tree_uid else f"{tree.dimension}_{tree.rule_name}_d{tree.depth}_L{tree.leaf_count}_{x0_h}"

        records["tree_id"].append(tree.tree_id)
        records["tree_uid"].append(uid)
        records["x0_hash"].append(x0_h)
        records["dimension"].append(tree.dimension)
        records["rule_name"].append(tree.rule_name)
        records["grid_shape"].append(list(tree.grid_shape))
        records["depth"].append(int(tree.depth))
        records["leaf_count"].append(int(tree.leaf_count))
        records["effective_branching"].append(float(tree.effective_branching))
        records["num_hints"].append(int(len(tree.hint_indices)))
        records["hint_density"].append(float(tree.hint_density))
        records["mean_activity"].append(float(tree.mean_activity))
        records["target_state"].append(tree.root_target.astype(np.uint8).tobytes())
        records["ground_truth_s0"].append(x0_bytes)
        records["hints_s0"].append(tree.hints.astype(np.int8).tobytes())

    table = pa.Table.from_pydict(records, schema=get_parquet_schema())
    pq.write_table(table, output_path, compression=compression)
    return output_path


def read_trees_from_parquet(parquet_path: str) -> pa.Table:
    """Read full Parquet dataset table."""
    return pq.read_table(parquet_path)
