"""Command-line interface for the cainvert benchmark engine.

    cainvert mine-exact  --width 48 --depth 3 --quota 3000 --out-dir data/catalogs/catalog_1d_w48_v02_exact
    cainvert resplit     --in data/catalogs/catalog_1d_w48_v02_exact --out data/catalogs/catalog_1d_w48_v021_exact
    cainvert verify      --catalog data/catalogs/catalog_1d_w48_v021_exact
    cainvert inspect     --catalog data/catalogs/catalog_1d_w48_v021_exact
    cainvert mine ...    (legacy v0.1 sampled miner, 1D and 2D)
"""

import argparse
import sys
import time
from typing import Optional
import numpy as np

from cainvert.core.filters import is_density_valid
from cainvert.engines.eca_1d import ECA1DEngine
from cainvert.mining.tree_miner import TreeMiner
from cainvert.mining.bucket_sampler import BucketSpec, StratifiedBucketManager
from cainvert.storage.parquet_io import export_trees_to_parquet
from cainvert.storage.catalog import TreeCatalog


def cmd_mine(args: argparse.Namespace) -> None:
    """Execute the legacy (v0.1) sampled mining run with Stratified Bucket Sampling."""
    dim = args.dimension.lower()
    print(f"\n{'='*70}")
    print(f" CAINVERT MINING ENGINE (Dimension: {dim.upper()}) -- legacy sampled miner (v0.1)")
    print(f" For exact, branching-controlled 1D catalogs use: cainvert mine-exact")
    print(f"{'='*70}")

    if dim == "1d":
        grid_shape = (args.width,)
        engine = ECA1DEngine(rule_number=args.rule)
        print(f"Engine: 1D ECA Rule {args.rule} | Width: {args.width}")
    elif dim == "2d":
        from cainvert.engines.sat_2d import SAT2DEngine
        grid_shape = (args.height, args.width)
        engine = SAT2DEngine(grid_shape=grid_shape, rule_str=args.rule_str)
        print(f"Engine: 2D Life-like {args.rule_str} | Grid: {args.height}x{args.width}")
    else:
        raise ValueError(f"Unknown dimension '{dim}', must be '1d' or '2d'")

    miner = TreeMiner(
        engine=engine,
        leaf_cap=args.max_leaves,
        max_branch_per_node=args.max_branch,
        min_density=args.min_density,
        max_density=args.max_density,
    )

    bucket_spec = BucketSpec(
        depths=[int(d) for d in args.depths.split(",")],
        samples_per_depth=args.samples_per_depth,
        max_per_leaf_bucket=args.max_per_leaf,
    )
    bucket_mgr = StratifiedBucketManager(spec=bucket_spec)

    print(f"Target depths: {bucket_spec.depths}")
    print(f"Samples per depth: {bucket_spec.samples_per_depth} (Total Target: {len(bucket_spec.depths) * bucket_spec.samples_per_depth})")
    print(f"Max per leaf-bucket: {bucket_spec.max_per_leaf_bucket}")
    print(f"Output Parquet: {args.out}\n")

    t_start = time.time()
    attempts = 0
    rng = np.random.RandomState(args.seed)

    while not bucket_mgr.is_complete() and attempts < args.max_attempts:
        attempts += 1
        active_depths = [
            d for d_i, d in enumerate(bucket_spec.depths)
            if bucket_mgr.depth_count(d_i) < bucket_spec.samples_per_depth
        ]
        if not active_depths:
            break

        target_d = int(rng.choice(active_depths))
        tree = miner.mine_single_tree(
            grid_shape=grid_shape,
            target_depth=target_d,
            rng=rng,
            tree_idx=attempts,
            node_timeout_sec=args.node_timeout,
        )

        if tree is not None:
            # If we've made many attempts, allow relaxing the max_per_leaf limit
            allow_fallback = (attempts > (len(bucket_spec.depths) * bucket_spec.samples_per_depth * 10))
            accepted = bucket_mgr.add_tree(tree, allow_fallback=allow_fallback)
            if accepted and bucket_mgr.total_collected() % max(1, args.log_interval) == 0:
                elapsed = time.time() - t_start
                rate = bucket_mgr.total_collected() / max(1e-5, elapsed)
                pct = bucket_mgr.get_fill_percentage()
                print(
                    f"[{bucket_mgr.total_collected()} samples ({pct:4.1f}%)] "
                    f"Speed: {rate:5.1f} s/sec | Attempts: {attempts} | "
                    f"Elapsed: {elapsed:5.1f}s",
                    end="\r",
                    flush=True,
                )

    print("\n\n" + bucket_mgr.summary_table() + "\n")

    # Collect all accepted trees
    all_trees = []
    for trees_list in bucket_mgr.collected_samples.values():
        all_trees.extend(trees_list)

    if all_trees:
        export_trees_to_parquet(all_trees, args.out)
        print(f"Successfully saved {len(all_trees)} trees to: {args.out}")
    else:
        print("Warning: No trees were collected.")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect an existing Parquet tree pool using DuckDB."""
    catalog = TreeCatalog(args.catalog if "*" in args.catalog or args.catalog.endswith(".parquet")
                          else args.catalog.rstrip("/\\") + "/**/*.parquet")
    total = catalog.count()
    print(f"\nCatalog: {args.catalog}")
    print(f"Total Trees: {total:,}")

    rows = catalog.query(limit=5)
    print("\nSample Trees:")
    for r in rows:
        extra = ""
        if "search_work" in r:
            extra = f", search_work={r['search_work']}, split={r.get('split')}, {r.get('catalog_version')}"
        print(
            f" - {r['tree_id']}: Depth={r['depth']}, Leaves={r['leaf_count']}, "
            f"b_eff={r['effective_branching']:.2f}, Hints={r['num_hints']} ({r['hint_density']*100:.1f}%){extra}"
        )


def cmd_mine_exact(argv) -> None:
    """Exact, branching-controlled 1D catalogs (schema v0.2.1); see cainvert.mining.exact_miner."""
    from cainvert.mining.exact_miner import main as exact_main
    exact_main(argv)


def cmd_resplit(argv) -> None:
    """Re-key an existing exact catalog to schema v0.2.1 (sd_hash ids, dedup, train/val/test)."""
    from cainvert.mining.resplit import main as resplit_main
    resplit_main(argv)


def cmd_verify(args: argparse.Namespace) -> None:
    """Audit an exact catalog: unique ids, no target state in two splits, ground truth passes the
    verifier, the reference solver recovers exactly the ground truth, search_work matches."""
    import duckdb
    from cainvert.eval import check_ground_truth, solve_records
    from cainvert.storage.exact_catalog import available_splits, catalog_glob, load_records

    g = catalog_glob(args.catalog)
    con = duckdb.connect()
    n, n_ids, n_sd = con.execute(
        f"SELECT count(*), count(DISTINCT tree_id), count(DISTINCT (target_state, depth)) "
        f"FROM read_parquet('{g}', union_by_name=True)").fetchone()
    leak = con.execute(
        f"SELECT count(*) FROM (SELECT target_state, depth FROM read_parquet('{g}', union_by_name=True) "
        f"GROUP BY target_state, depth HAVING count(DISTINCT split) > 1)").fetchone()[0]
    print(f"{args.catalog}: {n:,} rows | unique ids {n_ids:,} | unique (target, depth) {n_sd:,} | "
          f"targets in >1 split: {leak}")
    ok = (n == n_ids) and (n == n_sd) and (leak == 0)
    for split in available_splits(args.catalog):
        recs = load_records(args.catalog, split, per_cell=args.per_cell)
        gt = check_ground_truth(recs)
        ref = solve_records(recs, level_cap=args.level_cap)
        work_ok = ref["search_work"] == ref["catalog_search_work"]
        print(f"  split {split:5s}: checked {len(recs):5d} | gt passes verifier {gt.mean() * 100:6.2f}% | "
              f"reference solver unique & == gt {ref['matches_gt'].mean() * 100:6.2f}% | "
              f"search_work matches {work_ok.mean() * 100:6.2f}%")
        ok = ok and bool(gt.all()) and bool(ref["matches_gt"].all()) and bool(work_ok.all())
    print("OK" if ok else "PROBLEMS FOUND")
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cainvert", description="Cellular Automata Inversion Benchmark Engine")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: mine (legacy v0.1 sampled miner)
    mine_parser = subparsers.add_parser("mine", help="Legacy v0.1 sampled miner (1D/2D, capped branching)")
    mine_parser.add_argument("--dimension", type=str, default="1d", choices=["1d", "2d"])
    mine_parser.add_argument("--rule", type=int, default=110, help="1D Rule number (e.g. 110, 30)")
    mine_parser.add_argument("--rule-str", type=str, default="B3/S23", help="2D Rule string (e.g. B3/S23)")
    mine_parser.add_argument("--width", type=int, default=32, help="Lattice width W")
    mine_parser.add_argument("--height", type=int, default=32, help="Lattice height H (2D only)")
    mine_parser.add_argument("--depths", type=str, default="1,2,3,4,6,8", help="Comma-separated target depths")
    mine_parser.add_argument("--samples-per-depth", type=int, default=100, help="Target count per depth")
    mine_parser.add_argument("--max-per-leaf", type=int, default=30, help="Maximum samples allowed per leaf bound bucket per depth")
    mine_parser.add_argument("--max-leaves", type=int, default=4096, help="Maximum leaves per tree")
    mine_parser.add_argument("--max-branch", type=int, default=16, help="Max branching per node")
    mine_parser.add_argument("--min-density", type=float, default=0.25, help="Minimum active 1-fraction")
    mine_parser.add_argument("--max-density", type=float, default=0.75, help="Maximum active 1-fraction")
    mine_parser.add_argument("--node-timeout", type=float, default=1.0, help="Per-node SAT solver timeout (sec)")
    mine_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    mine_parser.add_argument("--max-attempts", type=int, default=100000, help="Maximum mining attempts")
    mine_parser.add_argument("--log-interval", type=int, default=10, help="Progress log interval")
    mine_parser.add_argument("--out", type=str, default="trees.parquet", help="Output Parquet filepath")

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect an existing tree catalog")
    inspect_parser.add_argument("--catalog", type=str, required=True, help="Directory, path or glob of Parquet file(s)")

    # Subcommands with their own argument parsers (remaining argv is forwarded)
    subparsers.add_parser("mine-exact", help="Mine exact, branching-controlled 1D catalogs (schema v0.2.1)", add_help=False)
    subparsers.add_parser("resplit", help="Re-key an existing exact catalog to schema v0.2.1", add_help=False)

    # Subcommand: verify
    verify_parser = subparsers.add_parser("verify", help="Audit an exact catalog (ids, splits, verifier, reference solver)")
    verify_parser.add_argument("--catalog", type=str, required=True, help="Directory of exact_*.parquet files")
    verify_parser.add_argument("--per-cell", type=int, default=20, help="Rows per (depth, Lambda bin) to solve exactly")
    verify_parser.add_argument("--level-cap", type=int, default=50000)

    if len(sys.argv) > 1 and sys.argv[1] in ("mine-exact", "resplit"):
        (cmd_mine_exact if sys.argv[1] == "mine-exact" else cmd_resplit)(sys.argv[2:])
        return

    args = parser.parse_args()
    if args.subcommand == "mine":
        cmd_mine(args)
    elif args.subcommand == "inspect":
        cmd_inspect(args)
    elif args.subcommand == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
