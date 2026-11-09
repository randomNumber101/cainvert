"""CAInvert v0.2 exact miner: branching-controlled preimage catalogs for 1D elementary CA.

Differences to the v0.1 ``TreeMiner`` (which caps every node at 16 preimages and subsamples levels):

* the backward tree is enumerated EXACTLY (no per-node cap, no subsampling) -> ``leaf_count`` is the
  true number of d-step preimages Lambda; trees whose levels exceed ``--level-cap`` are rejected
* hints are a greedy hitting set against ALL preimages -> the hinted solution is provably unique
* ``hint_order`` stores the greedy positions followed by all remaining positions in random order, so
  any prefix of length h >= num_hints is also a unique-solution hint set (fixed-hint experiments)
* ``level_widths`` w_1..w_d (true number of k-step preimages) and ``search_work`` = sum_k w_k
* stratified by (depth, Lambda bin) with per-bin quotas (rejection sampling)
* unique ids and a deterministic train/val/test split, both keyed by the hash of the TARGET state
  S_D (schema v0.2.1; v0.2.0 keyed them by x0, which allowed the same S_D in train and val)
* one task per (S_D, depth): a second x0 that leads to an already accepted S_D is dropped

Sampling semantics (state this in a data card): x0 is drawn uniformly from {0,1}^W, the trajectory
x0 -> f(x0) -> ... -> f^d(x0) = S_D must pass ``is_trajectory_dynamic`` (density 0.25..0.75 at every
step, >= 5 % of cells changing per step), and the ground-truth answer is this x0.  Conditioned on S_D
the ground truth is therefore a uniformly drawn leaf (up to the trajectory filter); Lambda bins are
filled by rejection sampling, so the catalog is NOT the natural Lambda distribution.

Usage (one depth per call / array task)::

    python -m cainvert.mining.exact_miner --width 48 --depth 2 --quota 3000 --workers 16 \
        --bins "1-1,2-3,4-15,16-63,64-255,256-1023,1024-4096" --out-dir data/catalogs/catalog_1d_w48_v02_exact

or ``cainvert mine-exact ...`` with the same arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.filters import is_trajectory_dynamic
from ..engines.eca_1d import ECA1DEngine

SCHEMA_VERSION = "v0.2.1-exact"
DEFAULT_BINS = "1-1,2-3,4-15,16-63,64-255,256-1023,1024-4096"
SPLIT_BUCKETS = 1000  # split = f(int(sd_hash, 16) % SPLIT_BUCKETS)

_ENGINE: Optional[ECA1DEngine] = None


def _engine(rule: int) -> ECA1DEngine:
    global _ENGINE
    if _ENGINE is None or _ENGINE._rule_number != rule:
        _ENGINE = ECA1DEngine(rule)
    return _ENGINE


# ----------------------------------------------------------------------------------------------
# schema helpers (shared with resplit.py)
# ----------------------------------------------------------------------------------------------
def state_hash(state: np.ndarray | bytes) -> str:
    """16 hex chars of sha256 over the raw uint8 bytes of a state."""
    raw = state if isinstance(state, (bytes, bytearray)) else np.asarray(state, dtype=np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def assign_split(sd_hash: str, val_frac: float = 0.15, test_frac: float = 0.15) -> str:
    """Deterministic split from the target-state hash: buckets [0, test) -> test,
    [test, test+val) -> val, rest -> train (fractions in units of 1/SPLIT_BUCKETS)."""
    b = int(sd_hash, 16) % SPLIT_BUCKETS
    n_test = int(round(test_frac * SPLIT_BUCKETS))
    n_val = int(round(val_frac * SPLIT_BUCKETS))
    if b < n_test:
        return "test"
    if b < n_test + n_val:
        return "val"
    return "train"


def make_tree_id(rule: int, W: int, d: int, sd_hash: str) -> str:
    return f"1d_rule_{rule}_W{W}_d{d}_{sd_hash}"


def parse_bins(spec: str) -> List[Tuple[int, int]]:
    return [tuple(int(v) for v in b.split("-")) for b in spec.split(",")]  # type: ignore[misc]


def bin_index(L: int, bins: Sequence[Tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= L <= hi:
            return i
    return -1


def rows_to_table(rows: List[Dict], W: int, d: int, rule: int, val_frac: float, test_frac: float) -> pa.Table:
    """Build the catalog table (schema v0.2.1). ``rows`` must contain the keys produced by ``_work``."""
    rows = sorted(rows, key=lambda r: r["sd_hash"])
    n = len(rows)
    ids = [make_tree_id(rule, W, d, r["sd_hash"]) for r in rows]
    return pa.table({
        "tree_id": ids,
        "tree_uid": ids,
        "sd_hash": [r["sd_hash"] for r in rows],
        "x0_hash": [r["x0_hash"] for r in rows],
        "dimension": ["1d"] * n,
        "rule_name": [f"rule_{rule}"] * n,
        "grid_shape": pa.array([[W]] * n, type=pa.list_(pa.int32())),
        "depth": pa.array([d] * n, type=pa.int32()),
        "leaf_count": pa.array([r["leaf_count"] for r in rows], type=pa.int32()),
        "effective_branching": pa.array([r["leaf_count"] ** (1.0 / d) for r in rows], type=pa.float32()),
        "num_hints": pa.array([r["num_hints"] for r in rows], type=pa.int32()),
        "hint_density": pa.array([r["num_hints"] / W for r in rows], type=pa.float32()),
        "mean_activity": pa.array([r["mean_activity"] for r in rows], type=pa.float32()),
        "target_state": pa.array([r["target_state"] for r in rows], type=pa.binary()),
        "ground_truth_s0": pa.array([r["ground_truth_s0"] for r in rows], type=pa.binary()),
        "hints_s0": pa.array([r["hints_s0"] for r in rows], type=pa.binary()),
        # v0.2 extras
        "leaf_bin": pa.array([r["leaf_bin"] for r in rows], type=pa.int32()),
        "level_widths": pa.array([list(r["level_widths"]) for r in rows], type=pa.list_(pa.int32())),
        "search_work": pa.array([r["search_work"] for r in rows], type=pa.int64()),
        "hint_order": pa.array([list(r["hint_order"]) for r in rows], type=pa.list_(pa.int16())),
        "split": [assign_split(r["sd_hash"], val_frac, test_frac) for r in rows],
        "catalog_version": [SCHEMA_VERSION] * n,
    })


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_of_package() -> Optional[str]:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def write_catalog(table: pa.Table, out_dir: str, stem: str, meta: Dict) -> Tuple[str, str]:
    """Write ``<out_dir>/<stem>.parquet`` (zstd) and ``<stem>.json`` (meta + sha256 of the parquet)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.parquet")
    pq.write_table(table, path, compression="zstd")
    meta = dict(meta)
    meta.update(schema_version=SCHEMA_VERSION, n=table.num_rows, parquet_sha256=sha256_file(path),
                split_counts={s: int((np.array(table.column("split").to_pylist()) == s).sum())
                              for s in ("train", "val", "test")},
                written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    jpath = os.path.join(out_dir, f"{stem}.json")
    with open(jpath, "w") as f:
        json.dump(meta, f, indent=1)
    return path, jpath


# ----------------------------------------------------------------------------------------------
# exact enumeration + hints
# ----------------------------------------------------------------------------------------------
def exact_tree(engine: ECA1DEngine, root: np.ndarray, depth: int, level_cap: int):
    """Exact backward BFS. Returns (level_widths, leaves) or (widths, None) if capped/dead."""
    level = root[None, :]
    widths: List[int] = []
    for _ in range(depth):
        nxt = []
        for s in level:
            r = engine.get_preimages(s)
            if r.count:
                nxt.append(r.preimages)
        if not nxt:
            return widths + [0], None
        level = np.unique(np.concatenate(nxt), axis=0)
        widths.append(int(len(level)))
        if len(level) > level_cap:
            return widths, None
    return widths, level


def greedy_hint_order(gt: np.ndarray, distractors: np.ndarray, rng: np.random.RandomState) -> Tuple[List[int], List[int]]:
    """Greedy hitting set (in selection order) + remaining positions shuffled.

    The greedy set is a log-factor approximation of the minimum hitting set, not the minimum itself."""
    W = gt.shape[0]
    chosen: List[int] = []
    if len(distractors):
        D = distractors != gt[None, :]
        assert D.any(1).all(), "distractor identical to ground truth"
        unc = np.ones(len(D), dtype=bool)
        while unc.any():
            sc = D[unc].sum(0)
            sc[chosen] = -1
            j = int(np.argmax(sc))
            chosen.append(j)
            unc &= ~D[:, j]
    rest = [i for i in range(W) if i not in chosen]
    rng.shuffle(rest)
    return chosen, chosen + rest


def _work(args):
    """Worker: draw random x0 until it finds trees for still-open bins (or gives up)."""
    seed, W, depth, rule, bins, open_bins, level_cap, max_tries, max_found = args
    rng = np.random.RandomState(seed)
    eng = _engine(rule)
    found, tries = [], 0
    while tries < max_tries and len(found) < max_found:
        tries += 1
        x0 = rng.randint(0, 2, W).astype(np.uint8)
        traj = [x0]
        for _ in range(depth):
            traj.append(eng.forward_step(traj[-1]))
        if not is_trajectory_dynamic(traj)[0]:
            continue
        widths, leaves = exact_tree(eng, traj[-1], depth, level_cap)
        if leaves is None:
            continue
        L = int(len(leaves))
        b = bin_index(L, bins)
        if b < 0 or b not in open_bins:
            continue
        gt_rows = np.where((leaves == x0[None, :]).all(1))[0]
        if len(gt_rows) != 1:
            continue
        distract = np.delete(leaves, gt_rows[0], axis=0)
        chosen, order = greedy_hint_order(x0, distract, rng)
        hints = np.full(W, -1, dtype=np.int8)
        hints[chosen] = x0[chosen]
        # exact uniqueness check: no other preimage agrees with all hints
        if len(distract) and ((distract[:, chosen] == x0[chosen][None, :]).all(1)).any():
            continue
        act = float(np.mean([np.mean(traj[i] != traj[i + 1]) for i in range(depth)]))
        found.append(dict(
            sd_hash=state_hash(traj[-1]), x0_hash=state_hash(x0), depth=depth, leaf_count=L, leaf_bin=b,
            level_widths=widths, search_work=int(sum(widths)), num_hints=len(chosen), hint_order=order,
            target_state=traj[-1].tobytes(), ground_truth_s0=x0.tobytes(), hints_s0=hints.tobytes(),
            mean_activity=act,
        ))
    return found, tries


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAInvert v0.2 exact branching-controlled miner (1D ECA)")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--rule", type=int, default=110)
    ap.add_argument("--bins", default=DEFAULT_BINS)
    ap.add_argument("--quota", type=int, default=3000, help="trees per (depth, Lambda bin)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--level-cap", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit-min", type=float, default=200.0,
                    help="stop after this many minutes even if quotas are not full (output then depends on speed)")
    ap.add_argument("--max-tries-per-task", type=int, default=400)
    ap.add_argument("--out-dir", required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = build_parser().parse_args(argv)
    bins = parse_bins(args.bins)
    nb = len(bins)
    counts = np.zeros(nb, dtype=int)
    seen_sd: set = set()
    n_dup = 0
    rows: List[Dict] = []
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    tries_total, task_id = 0, 0
    base_seed = 1_000_003 * (args.seed + 1) + 7919 * args.depth + 104_729 * args.width
    print(f"[exact_miner] W={args.width} d={args.depth} bins={bins} quota={args.quota} workers={args.workers}", flush=True)

    def make_task():
        nonlocal task_id
        task_id += 1
        open_bins = [i for i in range(nb) if counts[i] < args.quota]
        return (base_seed + task_id, args.width, args.depth, args.rule, bins, open_bins,
                args.level_cap, args.max_tries_per_task, 50)

    last = time.time()
    with Pool(args.workers) as pool:
        # rounds of 2*workers tasks, so the set of open bins is refreshed between rounds; results are
        # consumed in task order (imap), so the accepted set is a deterministic function of the seed
        # and of the number of completed rounds
        while (counts < args.quota).any() and (time.time() - t0) / 60 <= args.time_limit_min:
            batch = [make_task() for _ in range(2 * args.workers)]
            for found, tries in pool.imap(_work, batch, chunksize=1):
                tries_total += tries
                for r in found:
                    if r["sd_hash"] in seen_sd:
                        n_dup += 1
                        continue
                    if counts[r["leaf_bin"]] >= args.quota:
                        continue
                    seen_sd.add(r["sd_hash"])
                    counts[r["leaf_bin"]] += 1
                    rows.append(r)
            if time.time() - last > 60:
                last = time.time()
                print(f"  {(last - t0) / 60:6.1f} min | tries {tries_total:,} | counts {counts.tolist()}", flush=True)

    W, d = args.width, args.depth
    table = rows_to_table(rows, W, d, args.rule, args.val_frac, args.test_frac)
    meta = dict(width=W, depth=d, rule=args.rule, bins=bins, quota=args.quota, counts=counts.tolist(),
                tries=tries_total, minutes=round((time.time() - t0) / 60, 1), level_cap=args.level_cap,
                val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
                quotas_complete=bool((counts >= args.quota).all()), duplicate_targets_dropped=n_dup,
                args=vars(args), miner_git_commit=git_commit_of_package(), python=sys.version.split()[0])
    path, _ = write_catalog(table, args.out_dir, f"exact_w{W}_d{d}_seed{args.seed}", meta)
    print(f"[exact_miner] done: {json.dumps({k: meta[k] for k in ('counts', 'tries', 'minutes', 'duplicate_targets_dropped')})} -> {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
