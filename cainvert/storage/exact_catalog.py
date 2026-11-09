"""Loading exact (v0.2) catalogs: split selection, per-cell balancing, tensors.

An exact catalog is a directory of ``exact_w{W}_d{d}_seed{s}.parquet`` files (one depth per file) with
the columns written by :mod:`cainvert.mining.exact_miner`.  The important ones:

===================  ==========================================================================
column               meaning
===================  ==========================================================================
target_state         S_D, W bytes in {0,1}                      (model input, first track)
hints_s0             H_0, W int8 in {-1, 0, 1}, -1 = masked     (model input, second track)
ground_truth_s0      x0, W bytes in {0,1}                       (target; unique given S_D + H_0)
depth                d, number of backward steps
leaf_count           Lambda, exact number of d-step preimages of S_D
leaf_bin             index of the Lambda bin used for stratification
level_widths         [w_1, ..., w_d], exact number of k-step preimages (search frontier widths)
search_work          sum_k w_k, nodes an exact backward BFS has to expand
num_hints            |H_0|, number of revealed cells (greedy hitting set, unique solution)
hint_order           greedy hint positions first, then the remaining positions in random order
split                'train' / 'val' / 'test', keyed by the hash of S_D (schema >= v0.2.1)
===================  ==========================================================================

Model input/target format (same as v0.1)::

    input_ids = [S_D (W tokens in {0,1})] + [SEP = 3] + [H_0 (W tokens, MASK = 2)]   -> 2W + 1 tokens
    target    = x0 (W tokens in {0,1})
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

META_COLUMNS = ("tree_id, depth, leaf_count, effective_branching, num_hints, hint_density, mean_activity, "
                "leaf_bin, search_work, level_widths, hint_order, split")
DATA_COLUMNS = "target_state, ground_truth_s0, hints_s0"
SEP_TOKEN = 3
MASK_TOKEN = 2


def catalog_glob(catalog_path: str) -> str:
    """Directory -> recursive parquet glob; file or glob -> unchanged (DuckDB needs forward slashes)."""
    p = catalog_path.replace("\\", "/")
    if os.path.isdir(catalog_path):
        return p.rstrip("/") + "/**/*.parquet"
    return p


def available_splits(catalog_path: str) -> List[str]:
    import duckdb
    g = catalog_glob(catalog_path)
    rows = duckdb.connect().execute(f"SELECT DISTINCT split FROM read_parquet('{g}', union_by_name=True)").fetchall()
    return sorted(r[0] for r in rows)


def load_records(
    catalog_path: str,
    split: str,
    per_depth: Optional[int] = None,
    per_cell: Optional[int] = None,
    depths: Optional[Iterable[int]] = None,
    cell: str = "leaf_bin",
    extra_where: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return catalog rows of one split as dicts (bytes columns stay bytes).

    ``per_depth``: keep the first n rows per depth (ordered by tree_id, i.e. by the S_D hash -> random
    but deterministic).  ``per_cell``: keep the first n rows per (depth, ``cell``) instead, which gives
    a Lambda-balanced evaluation set.  Rows come back ordered by (depth, cell, tree_id).
    """
    import duckdb
    g = catalog_glob(catalog_path)
    if split not in available_splits(catalog_path):
        raise ValueError(f"split '{split}' not in catalog {catalog_path} (have {available_splits(catalog_path)}); "
                         f"schema v0.2.0 catalogs have no test split -> run cainvert resplit first")
    where = [f"split = '{split}'"]
    if depths is not None:
        where.append("depth IN (" + ",".join(str(int(d)) for d in depths) + ")")
    if extra_where:
        where.append(f"({extra_where})")
    w = " AND ".join(where)
    cols = f"{META_COLUMNS}, {DATA_COLUMNS}"
    if per_cell is not None:
        sql = f"""WITH r AS (SELECT {cols}, ROW_NUMBER() OVER (PARTITION BY depth, {cell} ORDER BY tree_id) rn
                  FROM read_parquet('{g}', union_by_name=True) WHERE {w})
                  SELECT * EXCLUDE (rn) FROM r WHERE rn <= {int(per_cell)} ORDER BY depth, {cell}, tree_id"""
    elif per_depth is not None:
        sql = f"""WITH r AS (SELECT {cols}, ROW_NUMBER() OVER (PARTITION BY depth ORDER BY tree_id) rn
                  FROM read_parquet('{g}', union_by_name=True) WHERE {w})
                  SELECT * EXCLUDE (rn) FROM r WHERE rn <= {int(per_depth)} ORDER BY depth, tree_id"""
    else:
        sql = f"SELECT {cols} FROM read_parquet('{g}', union_by_name=True) WHERE {w} ORDER BY depth, {cell}, tree_id"
    res = duckdb.connect().execute(sql)
    names = [c[0] for c in res.description]
    return [dict(zip(names, row)) for row in res.fetchall()]


def records_to_arrays(records: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Stack the per-row byte strings into arrays: s_d (N,W) uint8, x0 (N,W) uint8, hints (N,W) int8,
    plus depth, leaf_count, num_hints, leaf_bin, search_work as int arrays."""
    if not records:
        raise ValueError("no records")
    s_d = np.stack([np.frombuffer(r["target_state"], dtype=np.uint8) for r in records])
    x0 = np.stack([np.frombuffer(r["ground_truth_s0"], dtype=np.uint8) for r in records])
    hints = np.stack([np.frombuffer(r["hints_s0"], dtype=np.int8) for r in records])
    out = {"s_d": s_d, "x0": x0, "hints": hints}
    for k in ("depth", "leaf_count", "num_hints", "leaf_bin", "search_work"):
        out[k] = np.array([int(r[k]) for r in records], dtype=np.int64)
    return out


def encode_input(s_d: np.ndarray, hints: np.ndarray, sep_token: int = SEP_TOKEN, mask_token: int = MASK_TOKEN) -> np.ndarray:
    """[S_D] + [SEP] + [H_0 with -1 -> MASK] as int64 (works for (W,) and (N, W))."""
    s_d = np.asarray(s_d, dtype=np.int64)
    h = np.asarray(hints, dtype=np.int64)
    h = np.where(h < 0, mask_token, h)
    sep = np.full(s_d.shape[:-1] + (1,), sep_token, dtype=np.int64)
    return np.concatenate([s_d, sep, h], axis=-1)


def hints_with_k_revealed(record: Dict[str, Any], k: int) -> np.ndarray:
    """Hint vector with the first k positions of ``hint_order`` revealed (k >= num_hints keeps the
    solution unique; k < num_hints makes the task ambiguous on purpose)."""
    x0 = np.frombuffer(record["ground_truth_s0"], dtype=np.uint8)
    order = np.asarray(record["hint_order"], dtype=np.int64)
    h = np.full(x0.shape[0], -1, dtype=np.int8)
    h[order[:k]] = x0[order[:k]]
    return h


try:  # torch is optional
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    _TorchDataset = object  # type: ignore[misc,assignment]


class ExactCatalogDataset(_TorchDataset):
    """In-memory PyTorch dataset over records from :func:`load_records`.

    Items: ``input_ids`` (2W+1,) long, ``target`` (W,) long, ``hint_mask`` (W,) bool (True = revealed),
    ``depth`` () long, plus the scalar meta fields ``leaf_count``, ``num_hints``, ``leaf_bin``,
    ``search_work`` (python ints, collate to tensors) and ``tree_id`` (str).  ``self.records`` keeps the
    full rows (incl. ``level_widths`` / ``hint_order``) for joins at analysis time.
    ``reveal`` optionally overrides the number of revealed hints via ``hint_order``.
    """

    def __init__(self, records: Sequence[Dict[str, Any]], reveal: Optional[int] = None):
        if not _TORCH:
            raise ImportError("torch is required for ExactCatalogDataset")
        self.records = list(records)
        self.reveal = reveal

    @classmethod
    def from_catalog(cls, catalog_path: str, split: str, **kw) -> "ExactCatalogDataset":
        reveal = kw.pop("reveal", None)
        return cls(load_records(catalog_path, split, **kw), reveal=reveal)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        s_d = np.frombuffer(r["target_state"], dtype=np.uint8)
        x0 = np.frombuffer(r["ground_truth_s0"], dtype=np.uint8)
        h = hints_with_k_revealed(r, self.reveal) if self.reveal is not None else np.frombuffer(r["hints_s0"], dtype=np.int8)
        return {
            "input_ids": torch.from_numpy(encode_input(s_d, h)),
            "target": torch.from_numpy(x0.astype(np.int64)),
            "hint_mask": torch.from_numpy(h >= 0),
            "depth": torch.tensor(int(r["depth"]), dtype=torch.long),
            "leaf_count": int(r["leaf_count"]),
            "num_hints": int(np.sum(h >= 0)),
            "leaf_bin": int(r["leaf_bin"]),
            "search_work": int(r["search_work"]),
            "tree_id": r["tree_id"],
        }


def load_splits(
    catalog_path: str,
    train_per_depth: Optional[int] = None,
    eval_per_cell: Optional[int] = None,
    eval_split: str = "val",
    depths: Optional[Iterable[int]] = None,
) -> Tuple["ExactCatalogDataset", "ExactCatalogDataset"]:
    """(train dataset, eval dataset) -- eval balanced per (depth, Lambda bin).  Select checkpoints and
    hyper-parameters on ``eval_split='val'`` and report numbers on ``eval_split='test'``."""
    tr = ExactCatalogDataset.from_catalog(catalog_path, "train", per_depth=train_per_depth, depths=depths)
    ev = ExactCatalogDataset.from_catalog(catalog_path, eval_split, per_cell=eval_per_cell, depths=depths)
    return tr, ev
