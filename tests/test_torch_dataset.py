import os
import tempfile
import numpy as np
import pytest
import torch

from cainvert.core.types import TreeData
from cainvert.storage.parquet_io import export_trees_to_parquet
from cainvert.storage.torch_dataset import CAInversionDataset


def test_torch_dataset_dynamic_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a few synthetic trees
        trees = []
        for i in range(10):
            depth = 2 if i < 5 else 6
            tree = TreeData(
                tree_id=f"tree_{i}",
                dimension="1d",
                rule_name="Rule110",
                grid_shape=(10,),
                depth=depth,
                leaf_count=4 * depth,
                effective_branching=2.0,
                root_target=np.ones(10, dtype=np.uint8),
                ground_truth_s0=np.zeros(10, dtype=np.uint8),
                ground_truth_leaf_idx=0,
                leaves_s0=np.zeros((4 * depth, 10), dtype=np.uint8),
                trajectory=[np.zeros(10, dtype=np.uint8)] * (depth + 1),
                hint_indices=[0, 2],
                hints=np.array([0, -1, 1, -1, 0, -1, 0, -1, 1, -1], dtype=np.int8),
                hint_density=0.2,
                mean_activity=0.4,
                x0_hash=f"hash_{i}",
                tree_uid=f"uid_{i}",
            )
            trees.append(tree)

        p1 = os.path.join(tmpdir, "shard_1.parquet")
        export_trees_to_parquet(trees, p1)

        # 1. Load full dataset
        ds_full = CAInversionDataset(tmpdir)
        assert len(ds_full) == 10

        # Check item format
        sample = ds_full[0]
        # Input sequence: 10 target + 1 sep + 10 hints = 21 tokens
        assert sample["input_ids"].shape == (21,)
        assert sample["target"].shape == (10,)
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["target"], torch.Tensor)
        assert isinstance(sample["depth"], torch.Tensor)

        # 2. Dynamic query: only depth 6
        ds_d6 = CAInversionDataset(tmpdir, depth_min=6, depth_max=6)
        assert len(ds_d6) == 5
        for s in ds_d6:
            assert s["depth"].item() == 6

        # 3. Dynamic query: leaf count filter
        ds_leaf = CAInversionDataset(tmpdir, leaf_min=20)
        assert len(ds_leaf) == 5
