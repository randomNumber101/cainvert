"""Reference solver: exact backward search with hints (the oracle baseline).

Expands the full preimage tree of S_D level by level with the exact 1D engine and keeps the leaf that
agrees with the hints.  Its cost in expanded nodes equals the catalog column ``search_work``
(sum of level widths), so a model's compute can be compared against the exact search it replaces.
Hints constrain x0 only, so they are applied at the leaves; intermediate levels cannot be pruned.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..engines.eca_1d import ECA1DEngine


def solve_exact(s_d: np.ndarray, hints: np.ndarray, depth: int, rule: int = 110, level_cap: Optional[int] = None,
                engine: Optional[ECA1DEngine] = None) -> Dict[str, object]:
    """Returns dict(solution (W,) uint8 or None, n_solutions, level_widths, search_work, capped)."""
    eng = engine or ECA1DEngine(rule)
    s_d = np.asarray(s_d, dtype=np.uint8)
    hints = np.asarray(hints, dtype=np.int8)
    level = s_d[None, :]
    widths = []
    for _ in range(int(depth)):
        nxt = [eng.get_preimages(s).preimages for s in level if eng.count_preimages(s)]
        if not nxt:
            return dict(solution=None, n_solutions=0, level_widths=widths + [0], search_work=int(sum(widths)), capped=False)
        level = np.unique(np.concatenate(nxt), axis=0)
        widths.append(int(len(level)))
        if level_cap is not None and len(level) > level_cap:
            return dict(solution=None, n_solutions=-1, level_widths=widths, search_work=int(sum(widths)), capped=True)
    rev = hints >= 0
    ok = (level[:, rev].astype(np.int8) == hints[rev][None, :]).all(1) if rev.any() else np.ones(len(level), dtype=bool)
    sols = level[ok]
    return dict(solution=sols[0] if len(sols) == 1 else None, n_solutions=int(len(sols)), level_widths=widths,
                search_work=int(sum(widths)), capped=False)


def solve_records(records, rule: int = 110, level_cap: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Run the reference solver on catalog records; returns arrays solved (N,) bool, matches_gt (N,) bool,
    search_work (N,) and the catalog's own search_work for comparison."""
    eng = ECA1DEngine(rule)
    solved, match, work, cat = [], [], [], []
    for r in records:
        s_d = np.frombuffer(r["target_state"], dtype=np.uint8)
        h = np.frombuffer(r["hints_s0"], dtype=np.int8)
        x0 = np.frombuffer(r["ground_truth_s0"], dtype=np.uint8)
        res = solve_exact(s_d, h, int(r["depth"]), rule, level_cap, eng)
        solved.append(res["solution"] is not None)
        match.append(res["solution"] is not None and bool(np.array_equal(res["solution"], x0)))
        work.append(res["search_work"])
        cat.append(int(r.get("search_work", -1)))
    return dict(solved=np.array(solved), matches_gt=np.array(match), search_work=np.array(work), catalog_search_work=np.array(cat))
