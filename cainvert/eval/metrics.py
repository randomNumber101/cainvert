"""Evaluation protocol: solve@k with the unbiased pass@k estimator, a null control, compute budgets and
stratified summaries over the exact difficulty descriptors.

Protocol (what the benchmark reports for a model)
-------------------------------------------------
For every task: one deterministic answer and M stochastic rollouts (test-time breadth).  A rollout
counts as *solved* iff it passes the ground-truth-free verifier (:mod:`cainvert.eval.verifier`).

* ``det``           exact-match rate of the deterministic answer
* ``solve@k``       P(at least one of k rollouts is a solution), unbiased estimate from M >= k rollouts
                    (Chen et al. 2021, "Evaluating Large Language Models Trained on Code")
* ``null@k``        the same estimate for a *null control*: the deterministic answer with as many random
                    bits flipped as the rollout changed -- how much of solve@k is mere variance
* ``budget``        compute per rollout in layer passes (``T * H * (L + 1)`` for the two-level recurrent
                    models of the reference baselines; pass your own for other architectures)

All numbers are reported per depth d, per (d, Lambda bin), per (d, number of hints) and per
(d, search-work quantile), because Lambda, hints and search work are correlated in the catalogs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def pass_at_k(c: np.ndarray, n: int, k: int) -> np.ndarray:
    """Per-task unbiased estimate of P(at least one success in k draws) from c successes in n draws."""
    c = np.asarray(c, dtype=np.int64)
    out = np.ones(c.shape, dtype=np.float64)
    m = (n - c) >= k
    for i in np.where(m)[0]:
        out[i] = 1.0 - np.prod(1.0 - k / np.arange(n - c[i] + 1, n + 1))
    return out


def null_control(det: np.ndarray, rollouts: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each rollout flip as many random bits of ``det`` as the rollout differs from ``det``; return
    whether that random perturbation hits the target.  det (N,W), rollouts (N,M,W), target (N,W) -> (N,M) bool."""
    det = np.asarray(det, dtype=np.uint8)
    roll = np.asarray(rollouts, dtype=np.uint8)
    N, M, W = roll.shape
    ham = (roll != det[:, None, :]).sum(-1)  # (N, M)
    rank = rng.random((N, M, W)).argsort(-1).argsort(-1)
    flip = rank < ham[:, :, None]
    null = det[:, None, :] ^ flip.astype(np.uint8)
    return (null == np.asarray(target, dtype=np.uint8)[:, None, :]).all(-1)


def layer_passes(T: int, H: int = 2, L: int = 2) -> int:
    """Compute budget of one rollout of a two-level recurrent model with T segments, H high-level and L
    low-level cycles per segment: every H cycle runs L low-level passes and one high-level pass."""
    return int(T) * int(H) * (int(L) + 1)


def work_quantile_bins(search_work: np.ndarray, depth: np.ndarray, q: int = 4) -> np.ndarray:
    """Quantile bin (0..q-1) of search_work within each depth."""
    sw, d = np.asarray(search_work), np.asarray(depth)
    out = np.zeros(len(sw), dtype=np.int64)
    for dd in np.unique(d):
        m = d == dd
        edges = np.quantile(sw[m], np.linspace(0, 1, q + 1)[1:-1]) if m.sum() > q else []
        out[m] = np.searchsorted(edges, sw[m], side="right")
    return out


def strata(meta: Dict[str, np.ndarray], min_n: int = 20, work_q: int = 4) -> Dict[str, np.ndarray]:
    """Boolean masks for 'all', 'd{d}', 'd{d}|bin{b}', 'd{d}|h{h}', 'd{d}|bin{b}|h{h}', 'd{d}|work{q}'.
    ``meta`` needs depth, leaf_bin, num_hints, search_work (each (N,))."""
    d = np.asarray(meta["depth"])
    out: Dict[str, np.ndarray] = {"all": np.ones(len(d), dtype=bool)}
    b = np.asarray(meta["leaf_bin"]) if "leaf_bin" in meta else None
    h = np.asarray(meta["num_hints"]) if "num_hints" in meta else None
    wq = work_quantile_bins(meta["search_work"], d, work_q) if "search_work" in meta else None
    for dd in np.unique(d):
        md = d == dd
        out[f"d{dd}"] = md
        if b is not None:
            for bb in np.unique(b[md]):
                mb = md & (b == bb)
                if mb.sum() >= min_n:
                    out[f"d{dd}|bin{bb}"] = mb
                    if h is not None:
                        for hh in np.unique(h[mb]):
                            mh = mb & (h == hh)
                            if mh.sum() >= min_n:
                                out[f"d{dd}|bin{bb}|h{hh}"] = mh
        if h is not None:
            for hh in np.unique(h[md]):
                mh = md & (h == hh)
                if mh.sum() >= min_n:
                    out[f"d{dd}|h{hh}"] = mh
        if wq is not None:
            for qq in np.unique(wq[md]):
                mq = md & (wq == qq)
                if mq.sum() >= min_n:
                    out[f"d{dd}|work{qq}"] = mq
    return out


def summarise(
    meta: Dict[str, np.ndarray],
    solved: np.ndarray,
    det_ok: Optional[np.ndarray] = None,
    null: Optional[np.ndarray] = None,
    gt_match: Optional[np.ndarray] = None,
    k_list: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    min_n: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Stratified summary.  solved (N, M) bool verifier passes per rollout; det_ok (N,) bool; null (N, M)
    bool; gt_match (N, M) bool (exact ground-truth hits, for the verifier false-positive check)."""
    solved = np.asarray(solved, dtype=bool)
    N, M = solved.shape
    c = solved.sum(1)
    cn = np.asarray(null, dtype=bool).sum(1) if null is not None else None
    out: Dict[str, Dict[str, float]] = {}
    for name, m in strata(meta, min_n=min_n).items():
        s: Dict[str, float] = {"n": int(m.sum()), "rollout_acc": float(solved[m].mean())}
        if det_ok is not None:
            s["det"] = float(np.asarray(det_ok, dtype=bool)[m].mean())
        if gt_match is not None:
            g = np.asarray(gt_match, dtype=bool)
            s["verifier_false_pos"] = float((solved[m] & ~g[m]).sum() / max(solved[m].sum(), 1))
        for k in k_list:
            if k > M:
                continue
            s[f"solve@{k}"] = float(pass_at_k(c[m], M, k).mean())
            if cn is not None:
                s[f"null@{k}"] = float(pass_at_k(cn[m], M, k).mean())
        out[name] = s
    return out


def evaluate_rollouts(
    records: Sequence[dict],
    rollouts: np.ndarray,
    det: Optional[np.ndarray] = None,
    k_list: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    rule: int = 110,
    seed: int = 0,
    min_n: int = 20,
) -> Dict[str, Dict[str, float]]:
    """One-call protocol: catalog ``records`` (see :func:`cainvert.storage.exact_catalog.load_records`),
    ``rollouts`` (N, M, W) candidate answers, optional ``det`` (N, W) deterministic answers."""
    from ..storage.exact_catalog import records_to_arrays
    from .verifier import residual

    a = records_to_arrays(records)
    roll = np.asarray(rollouts, dtype=np.uint8)
    solved = residual(roll, a["s_d"], a["hints"], a["depth"], rule) == 0
    gt = (roll == a["x0"][:, None, :]).all(-1)
    det_ok = null = None
    if det is not None:
        det = np.asarray(det, dtype=np.uint8)
        det_ok = (det == a["x0"]).all(-1)
        null = null_control(det, roll, a["x0"], np.random.default_rng(seed))
    meta = {k: a[k] for k in ("depth", "leaf_bin", "num_hints", "search_work")}
    return summarise(meta, solved, det_ok=det_ok, null=null, gt_match=gt, k_list=k_list, min_n=min_n)
