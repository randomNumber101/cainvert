"""Integration test for TreeMiner, Parquet storage, and DuckDB Catalog."""

import os
import shutil
import tempfile
import numpy as np
import pytest

from cainvert.engines.eca_1d import ECA1DEngine
from cainvert.engines.sat_2d import SAT2DEngine
from cainvert.mining.tree_miner import TreeMiner
from cainvert.storage.parquet_io import export_trees_to_parquet, read_trees_from_parquet
from cainvert.storage.catalog import TreeCatalog


def test_mine_and_store_1d():
    engine = ECA1DEngine(rule_number=110)
    miner = TreeMiner(engine=engine, leaf_cap=256, max_branch_per_node=8)

    rng = np.random.RandomState(42)
    trees = []
    attempts = 0
    while len(trees) < 5 and attempts < 100:
        attempts += 1
        tree = miner.mine_single_tree(grid_shape=(16,), target_depth=4, rng=rng)
        if tree is not None:
            trees.append(tree)

    assert len(trees) >= 3, f"Expected at least 3 valid trees, got {len(trees)}"

    # Test Parquet export and DuckDB query
    tmpdir = tempfile.mkdtemp()
    try:
        pq_path = os.path.join(tmpdir, "test_trees_1d.parquet")
        export_trees_to_parquet(trees, pq_path)
        assert os.path.exists(pq_path)

        table = read_trees_from_parquet(pq_path)
        assert len(table) == len(trees)

        catalog = TreeCatalog(pq_path)
        assert catalog.count() == len(trees)

        results = catalog.query(depth_min=1, depth_max=4)
        assert len(results) == len(trees)
    finally:
        shutil.rmtree(tmpdir)


def test_mine_2d_tree():
    engine = SAT2DEngine(grid_shape=(6, 6), rule_str="B3/S23")
    miner = TreeMiner(engine=engine, leaf_cap=64, max_branch_per_node=8, min_density=0.20, max_density=0.80)

    rng = np.random.RandomState(123)
    trees = []
    attempts = 0
    while len(trees) < 2 and attempts < 50:
        attempts += 1
        tree = miner.mine_single_tree(grid_shape=(6, 6), target_depth=2, rng=rng)
        if tree is not None:
            trees.append(tree)

    assert len(trees) >= 1, f"Expected at least 1 valid 2D tree, got {len(trees)}"
    t = trees[0]
    assert t.dimension == "2d"
    assert t.depth >= 1
    assert t.leaf_count >= 1
    assert len(t.trajectory) == t.depth + 1
