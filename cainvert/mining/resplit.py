"""Re-key an existing exact catalog (schema v0.2.0) to schema v0.2.1 without re-mining.

v0.2.0 catalogs keyed ids and the train/val split by the hash of x0.  Two different x0 can lead to the
same target S_D, which (a) duplicates a task and (b) can put the same S_D into train and val with
different ground truths.  This tool

* adds ``sd_hash`` (hash of the target state),
* drops duplicate targets within a (W, d) file, keeping the row with the smallest ``x0_hash``
  (deterministic), and
* re-assigns ``tree_id``/``tree_uid`` and a train/val/test split from ``sd_hash``.

All other columns (trees, hints, hint_order, level_widths, ...) are copied unchanged.

    python -m cainvert.mining.resplit --in data/catalogs/catalog_1d_w48_v02_exact \
        --out data/catalogs/catalog_1d_w48_v021_exact
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import pyarrow.parquet as pq

from .exact_miner import SCHEMA_VERSION, rows_to_table, state_hash, write_catalog


def resplit_table(rows: List[Dict], W: int, d: int, rule: int, val_frac: float, test_frac: float):
    """Deduplicate by target state and rebuild the table. Returns (table, n_dropped)."""
    for r in rows:
        r["sd_hash"] = state_hash(r["target_state"])
        r.setdefault("x0_hash", state_hash(r["ground_truth_s0"]))
    best: Dict[str, Dict] = {}
    for r in rows:
        k = r["sd_hash"]
        if k not in best or r["x0_hash"] < best[k]["x0_hash"]:
            best[k] = r
    kept = list(best.values())
    return rows_to_table(kept, W, d, rule, val_frac, test_frac), len(rows) - len(kept)


def resplit_file(src: str, out_dir: str, val_frac: float = 0.15, test_frac: float = 0.15) -> str:
    t = pq.read_table(src)
    rows = t.to_pylist()
    if not rows:
        raise ValueError(f"{src}: empty")
    W = int(rows[0]["grid_shape"][0])
    d = int(rows[0]["depth"])
    rule = int(str(rows[0]["rule_name"]).split("_")[-1])
    assert all(int(r["depth"]) == d for r in rows), "one depth per file expected"
    table, dropped = resplit_table(rows, W, d, rule, val_frac, test_frac)
    stem = os.path.splitext(os.path.basename(src))[0]
    meta: Dict = {}
    jsrc = os.path.splitext(src)[0] + ".json"
    if os.path.exists(jsrc):
        with open(jsrc) as f:
            meta["mining_meta"] = json.load(f)
    meta.update(width=W, depth=d, rule=rule, val_frac=val_frac, test_frac=test_frac,
                resplit_from=os.path.abspath(src), source_rows=len(rows), duplicate_targets_dropped=dropped)
    path, _ = write_catalog(table, out_dir, stem, meta)
    print(f"[resplit] {src}: {len(rows)} rows -> {table.num_rows} ({dropped} duplicate targets dropped) -> {path}", flush=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> List[str]:
    ap = argparse.ArgumentParser(description=f"Re-key exact catalogs to schema {SCHEMA_VERSION}")
    ap.add_argument("--in", dest="inp", required=True, help="parquet file or directory of exact_*.parquet files")
    ap.add_argument("--out", dest="out", required=True, help="output directory")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.inp, "*.parquet"))) if os.path.isdir(args.inp) else [args.inp]
    if not files:
        raise SystemExit(f"no parquet files under {args.inp}")
    return [resplit_file(f, args.out, args.val_frac, args.test_frac) for f in files]


if __name__ == "__main__":
    main()
