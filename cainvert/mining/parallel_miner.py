"""High-throughput Multiprocessing Preimage Tree Miner with Apache Parquet Sharding and W&B Live Monitoring."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# Ensure cainvert root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_cainvert_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _cainvert_root not in sys.path:
    sys.path.insert(0, _cainvert_root)

from cainvert.core.types import TreeData
from cainvert.engines.eca_1d import ECA1DEngine
from cainvert.engines.sat_2d import SAT2DEngine
from cainvert.mining.tree_miner import TreeMiner
from cainvert.mining.bucket_sampler import BucketSpec, StratifiedBucketManager
from cainvert.storage.parquet_io import export_trees_to_parquet
from cainvert.storage.catalog import TreeCatalog


import uuid


def _worker_entrypoint(
    worker_id: int,
    dimension: str,
    rule_arg: Any,
    grid_shape: Tuple[int, ...],
    depths: List[int],
    samples_per_depth_worker: int,
    max_per_leaf_worker: int,
    leaf_cap: int,
    max_branch: int,
    max_expand: int,
    flush_interval: int,
    min_density: float,
    max_density: float,
    node_timeout: float,
    seed: int,
    out_dir: str,
    run_id: str,
    existing_hashes: Optional[Set[str]],
    queue: mp.Queue,
) -> None:
    """Worker process that mines an independent Parquet shard."""
    if dimension == "1d":
        engine = ECA1DEngine(rule_number=int(rule_arg))
    elif dimension == "2d":
        engine = SAT2DEngine(grid_shape=grid_shape, rule_str=str(rule_arg))
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    miner = TreeMiner(
        engine=engine,
        leaf_cap=leaf_cap,
        max_branch_per_node=max_branch,
        max_nodes_per_level=max_expand,
        min_density=min_density,
        max_density=max_density,
    )

    spec = BucketSpec(
        depths=depths,
        samples_per_depth=samples_per_depth_worker,
        max_per_leaf_bucket=max_per_leaf_worker,
    )
    bucket_mgr = StratifiedBucketManager(spec=spec)

    import signal
    stop_requested = [False]

    def _sig_handler(signum, frame):
        stop_requested[0] = True

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass

    worker_hashes: Set[str] = set(existing_hashes) if existing_hashes else set()
    rng = np.random.RandomState(seed + worker_id * 10007)
    attempts = 0
    t_start = time.time()
    last_report_count = 0
    report_interval = max(2, min(50, flush_interval))

    # Periodic chunk flushing buffer
    pending_trees: List[TreeData] = []
    part_idx = 0
    flushed_files: List[str] = []

    while not bucket_mgr.is_complete() and not stop_requested[0]:
        attempts += 1
        active_depths = [
            d for d_i, d in enumerate(depths)
            if bucket_mgr.depth_count(d_i) < samples_per_depth_worker
        ]
        if not active_depths:
            break

        target_d = int(rng.choice(active_depths))
        tree = miner.mine_single_tree(
            grid_shape=grid_shape,
            target_depth=target_d,
            rng=rng,
            tree_idx=attempts,
            node_timeout_sec=node_timeout,
            existing_hashes=worker_hashes,
        )

        if tree is not None:
            allow_fallback = (attempts > (len(depths) * samples_per_depth_worker * 2)) or (dimension == "2d")
            accepted = bucket_mgr.add_tree(tree, allow_fallback=allow_fallback)
            collected = bucket_mgr.total_collected()

            if accepted:
                if tree.x0_hash:
                    worker_hashes.add(tree.x0_hash)
                pending_trees.append(tree)

                # Periodic flush to disk with collision-free naming
                if len(pending_trees) >= flush_interval:
                    chunk_path = os.path.join(out_dir, f"shard_{run_id}_w{worker_id:03d}_p{part_idx:04d}.parquet")
                    export_trees_to_parquet(pending_trees, chunk_path)
                    flushed_files.append(chunk_path)
                    pending_trees = []
                    part_idx += 1

                if collected - last_report_count >= report_interval or bucket_mgr.is_complete():
                    delta = collected - last_report_count
                    last_report_count = collected
                    queue.put({
                        "type": "progress",
                        "worker_id": worker_id,
                        "delta": delta,
                        "total_worker": collected,
                        "attempts": attempts,
                    })

    # Flush any remaining trees in buffer
    if pending_trees:
        chunk_path = os.path.join(out_dir, f"shard_{run_id}_w{worker_id:03d}_p{part_idx:04d}.parquet")
        export_trees_to_parquet(pending_trees, chunk_path)
        flushed_files.append(chunk_path)
        pending_trees = []
        part_idx += 1

    elapsed = time.time() - t_start
    total_samples = bucket_mgr.total_collected()
    queue.put({
        "type": "done",
        "worker_id": worker_id,
        "sample_count": total_samples,
        "attempts": attempts,
        "elapsed": elapsed,
        "depth_counts": {d: bucket_mgr.depth_count(d_i) for d_i, d in enumerate(depths)},
    })


def run_parallel_mining(
    dimension: str = "1d",
    rule: int = 110,
    rule_str: str = "B3/S23",
    width: int = 81,
    height: int = 81,
    depths: Optional[List[int]] = None,
    total_samples: int = 1000000,
    workers: Optional[int] = None,
    leaf_cap: int = 4096,
    max_branch: int = 16,
    max_expand: int = 4,
    flush_interval: int = 2500,
    min_density: float = 0.25,
    max_density: float = 0.75,
    node_timeout: float = 1.0,
    seed: int = 42,
    out_dir: str = "catalogs/catalog_1d_w81_1M",
    wandb_project: str = "cainvert-mining",
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
) -> str:
    """Launch multiprocessing mining run with real-time W&B monitoring."""
    if depths is None:
        depths = [1, 2, 3, 4, 6, 8, 10, 12]

    num_workers = workers or max(1, mp.cpu_count())
    grid_shape = (width,) if dimension == "1d" else (height, width)
    rule_arg = rule if dimension == "1d" else rule_str

    samples_per_depth_total = total_samples // len(depths)
    samples_per_depth_worker = max(1, samples_per_depth_total // num_workers)
    actual_target = samples_per_depth_worker * num_workers * len(depths)
    max_per_leaf_worker = max(10, samples_per_depth_worker // 3)

    os.makedirs(out_dir, exist_ok=True)

    # Preload existing hashes and dynamic offset for append-only mining
    catalog_glob = os.path.join(out_dir, "**", "*.parquet")
    existing_catalog = TreeCatalog(catalog_glob)
    existing_hashes = existing_catalog.get_existing_x0_hashes()
    existing_count = len(existing_hashes)

    print("\n" + "=" * 75)
    print(f" CAINVERT MASSIVE TREE MINER (Target: {actual_target:,} Samples)")
    print("=" * 75)
    print(f"Dimension:            {dimension.upper()}")
    print(f"Rule:                 {rule if dimension == '1d' else rule_str}")
    print(f"Grid Shape:           {grid_shape}")
    print(f"Depths:               {depths}")
    print(f"Workers:              {num_workers} parallel processes")
    print(f"Per-Worker / Depth:   {samples_per_depth_worker:,} trees")
    print(f"Output Directory:     {out_dir}")
    if existing_count > 0:
        print(f"Existing Catalog:     {existing_count:,} unique x0 states (Appending)")
        print(f"Seed Offset:          +{existing_count * 10007}")
    print(f"W&B Project:          {wandb_project} (Mode: {wandb_mode})")
    print("=" * 75 + "\n")

    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # Initialize W&B
    use_wandb = False
    run_name = f"mine_{dimension}_{rule if dimension=='1d' else rule_str}_W{width}_{actual_target // 1000}k"
    try:
        import wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            mode=wandb_mode,
            config={
                "dimension": dimension,
                "rule": rule if dimension == "1d" else rule_str,
                "grid_shape": grid_shape,
                "depths": depths,
                "target_samples": actual_target,
                "num_workers": num_workers,
                "leaf_cap": leaf_cap,
                "max_branch": max_branch,
                "max_expand": max_expand,
                "flush_interval": flush_interval,
                "min_density": min_density,
                "max_density": max_density,
                "seed": seed,
                "existing_samples": existing_count,
            }
        )
        use_wandb = True
        print(f"[W&B] Initialized run: {wandb.run.name} ({wandb.run.url if wandb.run else 'local'})")
    except Exception as e:
        print(f"[W&B] Could not initialize wandb ({e}). Continuing with console logging only.")

    queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for w_id in range(num_workers):
        worker_seed = seed + existing_count * 10007 + w_id * 10003
        p = mp.Process(
            target=_worker_entrypoint,
            kwargs={
                "worker_id": w_id,
                "dimension": dimension,
                "rule_arg": rule_arg,
                "grid_shape": grid_shape,
                "depths": depths,
                "samples_per_depth_worker": samples_per_depth_worker,
                "max_per_leaf_worker": max_per_leaf_worker,
                "leaf_cap": leaf_cap,
                "max_branch": max_branch,
                "max_expand": max_expand,
                "flush_interval": flush_interval,
                "min_density": min_density,
                "max_density": max_density,
                "node_timeout": node_timeout,
                "seed": worker_seed,
                "out_dir": out_dir,
                "run_id": run_id,
                "existing_hashes": existing_hashes,
                "queue": queue,
            }
        )
        p.start()
        processes.append(p)

    total_collected = 0
    total_attempts = 0
    workers_done = 0
    worker_results: List[Dict[str, Any]] = []
    t_start = time.time()
    last_log_time = t_start

    try:
        while workers_done < num_workers:
            msg = queue.get()
            if msg["type"] == "progress":
                total_collected += msg["delta"]
                total_attempts += msg.get("attempts", 0)
            elif msg["type"] == "done":
                workers_done += 1
                worker_results.append(msg)
                print(f"\n[Worker {msg['worker_id']} Done] Saved {msg['sample_count']:,} samples in {msg['elapsed']:.1f}s")

            now = time.time()
            if now - last_log_time >= 3.0 or workers_done == num_workers:
                elapsed = max(1e-5, now - t_start)
                rate = total_collected / elapsed
                remaining = max(0, actual_target - total_collected)
                eta_sec = remaining / max(1e-5, rate)
                pct = (total_collected / actual_target) * 100.0

                print(
                    f"[{total_collected:,}/{actual_target:,} ({pct:5.1f}%)] "
                    f"Rate: {rate:6.1f} s/sec | ETA: {eta_sec / 60:4.1f}m | "
                    f"Active Workers: {num_workers - workers_done}/{num_workers}",
                    end="\r",
                    flush=True,
                )

                if use_wandb:
                    wandb.log({
                        "mining/total_collected": total_collected,
                        "mining/rate_samples_per_sec": rate,
                        "mining/fill_percentage": pct,
                        "mining/eta_minutes": eta_sec / 60.0,
                        "mining/elapsed_seconds": elapsed,
                        "mining/active_workers": num_workers - workers_done,
                    })

                last_log_time = now

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\n[Interrupt] Terminating worker processes...")
        for p in processes:
            p.terminate()
            p.join()
        raise

    total_elapsed = time.time() - t_start
    final_rate = total_collected / max(1e-5, total_elapsed)

    print("\n\n" + "=" * 75)
    print(f" MINING COMPLETE: {total_collected:,} trees in {total_elapsed:.1f}s ({final_rate:.1f} s/sec)")
    print("=" * 75)

    # Compile manifest
    manifest = {
        "catalog_name": os.path.basename(out_dir),
        "dimension": dimension,
        "rule": rule if dimension == "1d" else rule_str,
        "grid_shape": list(grid_shape),
        "depths": depths,
        "total_samples": total_collected,
        "num_shards": num_workers,
        "elapsed_seconds": round(total_elapsed, 2),
        "rate_samples_per_sec": round(final_rate, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path = os.path.join(out_dir, "CATALOG_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to: {manifest_path}")

    # Inspect total catalog with DuckDB
    pq_glob = os.path.join(out_dir, "**", "*.parquet")
    catalog = TreeCatalog(pq_glob)
    verified_count = catalog.count()
    summary_stats = catalog.summary()
    print(f"DuckDB Verified Total Across All Shards: {verified_count:,} trees ({summary_stats.get('unique_x0_states', 0):,} unique x0 states)")

    if use_wandb:
        wandb.log({
            "final/total_verified_trees": verified_count,
            "final/unique_x0_states": summary_stats.get("unique_x0_states", 0),
            "final/total_time_seconds": total_elapsed,
            "final/mean_samples_per_sec": final_rate,
        })
        wandb.finish()

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Massive Multiprocessing CA Inversion Preimage Tree Miner")
    parser.add_argument("--dimension", type=str, default="1d", choices=["1d", "2d"])
    parser.add_argument("--rule", type=int, default=110, help="1D Rule number")
    parser.add_argument("--rule-str", type=str, default="B3/S23", help="2D Rule string")
    parser.add_argument("--width", type=int, default=81, help="Grid width (81 for Sudoku-scale, 400, 900)")
    parser.add_argument("--height", type=int, default=81, help="Grid height (2D)")
    parser.add_argument("--depths", type=str, default="1,2,3,4,6,8,10,12", help="Comma-separated target depths")
    parser.add_argument("--total-samples", type=int, default=1000000, help="Total samples to mine (e.g. 1000000)")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU worker processes")
    parser.add_argument("--leaf-cap", type=int, default=4096, help="Maximum leaves per tree")
    parser.add_argument("--max-branch", type=int, default=16, help="Maximum branching per node")
    parser.add_argument("--max-expand", type=int, default=256, help="Maximum nodes to expand per tree level")
    parser.add_argument("--flush-interval", type=int, default=2500, help="Flush to parquet every N accepted samples")
    parser.add_argument("--min-density", type=float, default=0.25, help="Minimum active 1-fraction")
    parser.add_argument("--max-density", type=float, default=0.75, help="Maximum active 1-fraction")
    parser.add_argument("--node-timeout", type=float, default=1.0, help="Timeout per node in seconds")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-dir", type=str, default="data/catalogs/catalog_1d_w81_1M", help="Output directory")
    parser.add_argument("--wandb-project", type=str, default="cainvert-mining", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/user")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    args = parser.parse_args()
    depth_list = [int(d.strip()) for d in args.depths.split(",")]

    run_parallel_mining(
        dimension=args.dimension,
        rule=args.rule,
        rule_str=args.rule_str,
        width=args.width,
        height=args.height,
        depths=depth_list,
        total_samples=args.total_samples,
        workers=args.workers,
        leaf_cap=args.leaf_cap,
        max_branch=args.max_branch,
        max_expand=args.max_expand,
        flush_interval=args.flush_interval,
        min_density=args.min_density,
        max_density=args.max_density,
        node_timeout=args.node_timeout,
        seed=args.seed,
        out_dir=args.out_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )


if __name__ == "__main__":
    main()
