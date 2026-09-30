import os
import shutil
import pytest
from cainvert.mining.parallel_miner import run_parallel_mining
from cainvert.storage.catalog import TreeCatalog


def test_append_only_deduplication(tmp_path):
    catalog_dir = str(tmp_path / "test_dedup_catalog")

    # Run 1: Mine 24 trees (3 per depth across depths [1, 2, 4], 2 workers)
    run_parallel_mining(
        dimension="1d",
        rule=110,
        width=81,
        depths=[1, 2, 4],
        total_samples=24,
        workers=2,
        leaf_cap=512,
        max_branch=16,
        max_expand=128,
        flush_interval=6,
        seed=100,
        out_dir=catalog_dir,
        wandb_mode="disabled",
    )

    catalog1 = TreeCatalog(os.path.join(catalog_dir, "**", "*.parquet"))
    count1 = catalog1.count()
    hashes1 = catalog1.get_existing_x0_hashes()
    assert count1 == 24
    assert len(hashes1) == 24, "All 24 generated trees must have unique x0 hashes"

    # Run 2: Mine 24 MORE trees into the EXACT same catalog directory
    run_parallel_mining(
        dimension="1d",
        rule=110,
        width=81,
        depths=[1, 2, 4],
        total_samples=24,
        workers=2,
        leaf_cap=512,
        max_branch=16,
        max_expand=128,
        flush_interval=6,
        seed=100,  # Same starting base seed, should offset automatically
        out_dir=catalog_dir,
        wandb_mode="disabled",
    )

    catalog2 = TreeCatalog(os.path.join(catalog_dir, "**", "*.parquet"))
    count2 = catalog2.count()
    hashes2 = catalog2.get_existing_x0_hashes()

    assert count2 == 48, f"Expected 48 total trees after append-only run, got {count2}"
    assert len(hashes2) == 48, f"Expected 48 unique x0 hashes across both runs, got {len(hashes2)}"
