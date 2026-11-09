"""Ground-truth-free verifier for CAInvert candidates.

A candidate x for a task (S_D, H_0, d) is a solution iff f^d(x) = S_D and x agrees with every revealed
hint.  Because the hints of an exact catalog single out one preimage, a candidate that passes the
verifier IS the ground truth; ``residual`` counts how far a candidate is from passing (forward
mismatches + hint violations), which is what a search procedure can minimise without labels.

NumPy implementation (no torch needed) plus an optional torch twin for use inside evaluation loops.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def rule_table(rule: int = 110) -> np.ndarray:
    return np.array([(int(rule) >> i) & 1 for i in range(8)], dtype=np.uint8)


def eca_forward(x: np.ndarray, steps: np.ndarray | int = 1, rule: int = 110) -> np.ndarray:
    """One or more forward steps of an elementary CA on a periodic ring.

    ``x``: (W,) or (N, W) in {0,1}; ``steps``: int or (N,) per-row step counts (rows with fewer steps stop
    early).  Same neighbourhood order as :class:`cainvert.engines.eca_1d.ECA1DEngine` (left, centre, right)."""
    table = rule_table(rule)
    x = np.asarray(x, dtype=np.uint8)
    single = x.ndim == 1
    x = x[None, :] if single else x
    steps = np.broadcast_to(np.asarray(steps, dtype=np.int64), (x.shape[0],))
    out = x.copy()
    for s in range(int(steps.max()) if steps.size else 0):
        idx = (np.roll(out, 1, axis=-1) << 2) | (out << 1) | np.roll(out, -1, axis=-1)
        nxt = table[idx]
        act = (steps > s)[:, None]
        out = np.where(act, nxt, out)
    return out[0] if single else out


def residual(candidates: np.ndarray, s_d: np.ndarray, hints: np.ndarray, depths: np.ndarray | int, rule: int = 110) -> np.ndarray:
    """#cells where f^d(candidate) != S_D plus #revealed hints the candidate violates, per row.

    Shapes: candidates (N, W) or (N, M, W); s_d (N, W); hints (N, W) int8 (-1 = masked); depths (N,) or int.
    Returns (N,) or (N, M) int64."""
    cand = np.asarray(candidates, dtype=np.uint8)
    s_d = np.asarray(s_d, dtype=np.uint8)
    hints = np.asarray(hints, dtype=np.int8)
    N, W = s_d.shape
    depths = np.broadcast_to(np.asarray(depths, dtype=np.int64), (N,))
    multi = cand.ndim == 3
    M = cand.shape[1] if multi else 1
    flat = cand.reshape(N * M, W)
    fwd = eca_forward(flat, np.repeat(depths, M), rule)
    mism = (fwd != np.repeat(s_d, M, axis=0)).sum(-1)
    h = np.repeat(hints, M, axis=0)
    viol = ((h >= 0) & (flat.astype(np.int8) != h)).sum(-1)
    res = (mism + viol).astype(np.int64)
    return res.reshape(N, M) if multi else res


def is_solution(candidates: np.ndarray, s_d: np.ndarray, hints: np.ndarray, depths: np.ndarray | int, rule: int = 110) -> np.ndarray:
    """Boolean verifier pass (residual == 0)."""
    return residual(candidates, s_d, hints, depths, rule) == 0


def check_ground_truth(records, rule: int = 110) -> np.ndarray:
    """Sanity check for a list of catalog records: the stored x0 must pass the verifier (all True)."""
    from ..storage.exact_catalog import records_to_arrays
    a = records_to_arrays(records)
    return is_solution(a["x0"], a["s_d"], a["hints"], a["depth"], rule)


# ---------------------------------------------------------------------------------------------
# optional torch twin
# ---------------------------------------------------------------------------------------------
try:
    import torch

    def eca_forward_torch(x: "torch.Tensor", steps: "torch.Tensor", rule: int = 110) -> "torch.Tensor":
        """x (B, W) long/bool in {0,1}; steps (B,) long. Returns (B, W) long."""
        table = torch.tensor([(int(rule) >> i) & 1 for i in range(8)], device=x.device, dtype=torch.long)
        x = x.long()
        for s in range(int(steps.max().item()) if steps.numel() else 0):
            idx = (torch.roll(x, 1, -1) << 2) | (x << 1) | torch.roll(x, -1, -1)
            x = torch.where((steps > s).view(-1, 1), table[idx], x)
        return x

    def residual_torch(preds: "torch.Tensor", input_ids: "torch.Tensor", hint_mask: "torch.Tensor",
                       depths: "torch.Tensor", width: int, rule: int = 110) -> "torch.Tensor":
        """Residual from the model's own input layout: input_ids = [S_D][SEP][H_0] (B, 2W+1)."""
        s_d = input_ids[:, :width]
        hints = input_ids[:, width + 1:]
        fwd = eca_forward_torch(preds, depths, rule)
        res = (fwd != s_d).sum(-1)
        viol = ((preds != hints) & hint_mask).sum(-1)
        return (res + viol).long()
except ImportError:  # pragma: no cover
    pass
