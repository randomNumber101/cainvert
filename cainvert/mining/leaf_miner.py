"""Exp 28: CAInvert leaf catalog - ALL d-step preimages of each target state, for training on the
distribution over preimages (no hints).

Per tree (target S_D = f^d(x0) of a random dynamic trajectory):
  * leaves             all d-step preimages (exact backward enumeration, Lambda = leaf_count)
  * p_walk             probability to reach each leaf with a random backward walk that picks uniformly
                       among VIABLE children (children with at least one depth-d descendant)
  * p_walk_naive       uniform among all children (dead ends allowed), renormalised over the leaves
  * leaf_ham_sd        Hamming distance leaf <-> S_D
  * leaf_density       fraction of 1-cells of the leaf
  * leaf_activity      mean fraction of cells changing per step along f^0..f^d(leaf)
  * x0_index           index of the start state that generated S_D (size-biased "natural" preimage)
Stratified by (depth, Lambda bin) with quotas; unique per (S_D, d); split by id hash (15 % val).

  python -m cainvert.mining.leaf_miner --width 32 --depth 2 --quota 4000 --workers 16 \
      --bins "1-1,2-3,4-15,16-63,64-128" --out-dir data/catalogs/catalog_1d_w32_leaves
"""
import argparse, hashlib, json, os, time
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.filters import is_trajectory_dynamic
from ..engines.eca_1d import ECA1DEngine

_ENGINE: Optional[ECA1DEngine] = None


def _engine(rule: int) -> ECA1DEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ECA1DEngine(rule)
    return _ENGINE


def eca_forward_batch(x: np.ndarray, rule: int = 110) -> np.ndarray:
    """One step of an elementary CA on a periodic ring for a batch (N, W); same neighbourhood
    convention as the training verifier (left = roll(+1))."""
    tab = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    return tab[(np.roll(x, 1, axis=1) << 2) | (x << 1) | np.roll(x, -1, axis=1)]


def tree_with_structure(eng, root: np.ndarray, depth: int, level_cap: int):
    levels = [root[None, :].astype(np.uint8)]
    parents: List[np.ndarray] = []
    for _ in range(depth):
        nxt, par = [], []
        for i, s in enumerate(levels[-1]):
            r = eng.get_preimages(s)
            if r.count:
                nxt.append(np.asarray(r.preimages, dtype=np.uint8))
                par.append(np.full(r.count, i, dtype=np.int64))
        if not nxt:
            return None
        L = np.concatenate(nxt); P = np.concatenate(par)
        if len(L) > level_cap:
            return None
        levels.append(L); parents.append(P)
    return levels, parents


def walk_probabilities(levels, parents):
    d = len(parents)
    viable = [None] * (d + 1)
    viable[d] = np.ones(len(levels[d]), dtype=bool)
    for k in range(d - 1, -1, -1):
        viable[k] = np.bincount(parents[k][viable[k + 1]], minlength=len(levels[k])) > 0
    pv = np.ones(1); pn = np.ones(1)
    for k in range(d):
        P = parents[k]
        n_all = np.bincount(P, minlength=len(levels[k])).astype(float)
        n_via = np.bincount(P, weights=viable[k + 1].astype(float), minlength=len(levels[k]))
        pn = pn[P] / n_all[P]
        pv = np.where(viable[k + 1], pv[P] / np.maximum(n_via[P], 1), 0.0)
    pn = pn / pn.sum()
    return pv.astype(np.float32), pn.astype(np.float32)


def bin_index(L: int, bins) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= L <= hi:
            return i
    return -1


def _work(args):
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
        res = tree_with_structure(eng, traj[-1], depth, level_cap)
        if res is None:
            continue
        levels, parents = res
        leaves = levels[-1]
        L = int(len(leaves))
        b = bin_index(L, bins)
        if b < 0 or b not in open_bins:
            continue
        if len(np.unique(leaves, axis=0)) != L:
            continue                                   # sanity: preimage sets must be disjoint
        xi = np.where((leaves == x0[None, :]).all(1))[0]
        if len(xi) != 1:
            continue
        sd = traj[-1].astype(np.uint8)
        # sanity: every leaf maps onto S_D in d steps
        y = leaves.copy()
        for _ in range(depth):
            y = eca_forward_batch(y, rule)
        if not (y == sd[None, :]).all():
            raise RuntimeError('leaf does not map to S_D - neighbourhood convention mismatch')
        pv, pn = walk_probabilities(levels, parents)
        tr = [leaves]
        for _ in range(depth):
            tr.append(eca_forward_batch(tr[-1], rule))
        act = np.mean([np.mean(tr[k] != tr[k + 1], axis=1) for k in range(depth)], axis=0)
        tid = hashlib.sha256(sd.tobytes() + bytes([depth])).hexdigest()[:16]
        found.append(dict(
            tree_id=tid, depth=depth, leaf_count=L, leaf_bin=b,
            level_widths=[int(len(l)) for l in levels[1:]], search_work=int(sum(len(l) for l in levels[1:])),
            target_state=sd.tobytes(), leaves=leaves.tobytes(), p_walk=pv.tobytes(), p_walk_naive=pn.tobytes(),
            leaf_ham_sd=(leaves != sd[None, :]).sum(1).astype(np.uint8).tobytes(),
            leaf_density=leaves.mean(1).astype(np.float32).tobytes(), leaf_activity=act.astype(np.float32).tobytes(),
            x0_index=int(xi[0]), p_walk_entropy_norm=float(-(pv[pv > 0] * np.log(pv[pv > 0])).sum() / np.log(L)) if L > 1 else 1.0,
        ))
    return found, tries


def main():
    ap = argparse.ArgumentParser(description='CAInvert leaf catalog (all preimages) for Exp 28')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--depth', type=int, required=True)
    ap.add_argument('--rule', type=int, default=110)
    ap.add_argument('--bins', default='1-1,2-3,4-15,16-63,64-128')
    ap.add_argument('--quota', type=int, default=4000)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--level-cap', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--time-limit-min', type=float, default=200.0)
    ap.add_argument('--max-tries-per-task', type=int, default=400)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    bins = [tuple(int(v) for v in s.split('-')) for s in a.bins.split(',')]
    os.makedirs(a.out_dir, exist_ok=True)
    counts = [0] * len(bins); rows, seen = [], set()
    t0, tries_total, task = time.time(), 0, 0
    with Pool(a.workers) as pool:
        while True:
            open_bins = [i for i, c in enumerate(counts) if c < a.quota]
            if not open_bins or (time.time() - t0) / 60 > a.time_limit_min:
                break
            jobs = [(a.seed * 1_000_003 + a.depth * 10_007 + task + j, a.width, a.depth, a.rule, bins, open_bins,
                     a.level_cap, a.max_tries_per_task, 50) for j in range(2 * a.workers)]
            task += len(jobs)
            for found, tries in pool.imap_unordered(_work, jobs):
                tries_total += tries
                for r in found:
                    if r['tree_id'] in seen or counts[r['leaf_bin']] >= a.quota:
                        continue
                    seen.add(r['tree_id']); counts[r['leaf_bin']] += 1; rows.append(r)
            print(f'  {(time.time() - t0) / 60:6.1f} min | tries {tries_total:,} | counts {counts}', flush=True)
    for r in rows:
        r['split'] = 'val' if int(r['tree_id'], 16) % 1000 < int(a.val_frac * 1000) else 'train'
        r['catalog_version'] = 'v0.2-leaves'
    rows.sort(key=lambda r: (r['leaf_bin'], r['tree_id']))
    out = os.path.join(a.out_dir, f'leaves_w{a.width}_d{a.depth}_seed{a.seed}.parquet')
    pq.write_table(pa.Table.from_pylist(rows), out)
    meta = dict(width=a.width, depth=a.depth, rule=a.rule, bins=bins, quota=a.quota, counts=counts, tries=tries_total,
                minutes=round((time.time() - t0) / 60, 1), level_cap=a.level_cap, val_frac=a.val_frac, n=len(rows))
    json.dump(meta, open(out.replace('.parquet', '.json'), 'w'), indent=1)
    print('[leaf_miner] done:', json.dumps(meta), '->', out)


if __name__ == '__main__':
    main()
