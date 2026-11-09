# CAInvert: a controlled benchmark for depth and search in recursive models

<p align="center">
  <a href="https://github.com/randomNumber101/cainvert"><img src="https://img.shields.io/badge/GitHub-Repo-blue.svg?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/randomNumber101/cainvert-benchmark"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="HuggingFace"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
</p>

CAInvert asks a model to **invert a 1D cellular automaton**: given the state $S_D$ after $D$ steps of
Wolfram's Rule 110 on a ring of $W$ cells, plus a few revealed cells of the start state, reconstruct the
start state $S_0$.  Every task comes with **exact** difficulty descriptors computed by enumerating the
complete backward tree: the depth $d$, the number of valid preimages $\Lambda$, the width of every level of
the backward search and the number of hints that make the answer unique.  Solutions are checked by a
ground-truth-free verifier (simulate forward, compare).  That makes CAInvert a testbed for questions such
as *does a model's recursion depth track the problem depth?* and *does test-time breadth (sampling) buy
what the search tree says it should?* for tiny recursive / looped models, without the confounds of
natural-language benchmarks.

**Version 0.2 (exact catalogs, this README) supersedes the v0.1 sampled catalogs**, which capped the
branching per node and therefore under-counted $\Lambda$ and could not guarantee unique solutions.
See *Legacy v0.1* at the end.

---

## The task

Rule 110 is a local, deterministic, **non-injective** map $f$ on binary rings.  Running it $d$ steps from
$S_0$ gives $S_D = f^d(S_0)$.  Going back is ambiguous: $S_D$ has $\Lambda \ge 0$ valid $d$-step preimages,
arranged in a backward tree whose level $k$ holds the $w_k$ valid $k$-step preimages.  To make the task
well-posed we reveal a few cells of $S_0$ (a *hint tape* $H_0$) chosen so that **exactly one** preimage
agrees with them.

```
input   x = [S_D]  [SEP]  [H_0]        2W + 1 tokens, vocabulary {0, 1, MASK=2, SEP=3}
target  y = S_0                        W tokens in {0, 1}
valid   f^d(y) == S_D  and  y agrees with every revealed hint      (checked without labels)
```

### Difficulty descriptors (per task, exact)

| column          | meaning                                                                                   |
|-----------------|-------------------------------------------------------------------------------------------|
| `depth`         | $d$, number of backward steps                                                             |
| `leaf_count`    | $\Lambda$, number of valid $d$-step preimages of $S_D$ (without hints)                   |
| `level_widths`  | $[w_1, \dots, w_d]$, number of valid $k$-step preimages (the frontier of an exact search)  |
| `search_work`   | $\sum_k w_k$, nodes an exact backward breadth-first search expands                        |
| `num_hints`     | number of revealed cells (greedy hitting set over all $\Lambda-1$ distractors)            |
| `hint_order`    | greedy hint positions first, then the remaining cells in random order: any prefix of length $h \ge$ `num_hints` is again a unique-solution hint set, shorter prefixes make the task ambiguous on purpose |
| `leaf_bin`      | $\Lambda$ bin used for stratification (1 / 2–3 / 4–15 / 16–63 / 64–255 / 256–1023 / 1024–4096) |
| `split`         | `train` / `val` / `test`, keyed by the hash of $S_D$                                      |

$\Lambda$, `search_work` and `num_hints` are correlated; the evaluation protocol therefore stratifies by
depth, $\Lambda$ bin, hint count and search-work quantile.

### How the catalogs are mined

1. Draw $S_0$ uniformly from $\{0,1\}^W$ and roll forward $d$ steps.  Keep the trajectory only if every
   state has 25–75 % live cells and at least 5 % of the cells change per step (no still lifes).
2. Enumerate the **complete** backward tree of $S_D$ level by level with the exact 1D preimage solver
   (de Bruijn transfer matrix, all preimages of a ring in $O(W)$ per preimage); trees with a level wider
   than `--level-cap` (20 000) are rejected.
3. The ground truth is the $S_0$ that was rolled forward; conditioned on $S_D$ it is a uniformly drawn
   leaf (up to the trajectory filter).  Hints are a greedy hitting set that separates it from all other
   leaves (greedy = near-minimal, not minimal).
4. Fill a quota per $(d, \Lambda \text{ bin})$ by rejection sampling.  The catalog is therefore
   **balanced over $\Lambda$ bins, not the natural $\Lambda$ distribution**.
5. One task per $(S_D, d)$; ids and the deterministic 70/15/15 split are keyed by the hash of $S_D$, so
   the same target never appears in two splits.

The v0.1 engine is not involved; counting and enumeration are exact (verified against brute force in
`tests/test_exact_miner.py`).  Note that counting 1D preimages is polynomial (transfer matrix), so the
benchmark is *not* about an intractable problem: it is about whether a bounded-depth model performs the
required serial computation and search, with the exact search as the reference.

---

## Installation

```bash
git clone https://github.com/randomNumber101/cainvert.git
cd cainvert
pip install -e .            # numpy, pyarrow, duckdb
pip install -e ".[torch]"   # + PyTorch dataset / torch verifier
pip install -e ".[hub]"     # + Hugging Face download/upload
pytest -q                   # exactness, hint uniqueness, split, protocol tests
```

## Quick start

```python
from cainvert.storage.exact_catalog import load_records, records_to_arrays, encode_input
from cainvert.eval import evaluate_rollouts, is_solution, solve_exact

# 1. load a Lambda-balanced evaluation set (300 tasks per (depth, Lambda bin))
test = load_records("data/catalogs/catalog_1d_w48_v021_exact", split="test", per_cell=300)
a = records_to_arrays(test)                 # s_d, x0, hints (N, W); depth, leaf_count, search_work (N,)
x = encode_input(a["s_d"], a["hints"])      # (N, 2W+1) model input

# 2. your model: one deterministic answer and M rollouts per task, shape (N, W) and (N, M, W) in {0,1}
det, rollouts = my_model(x), my_sampler(x, M=64)

# 3. the protocol: verifier-based solve@k, null control, strata by d / Lambda / hints / search work
report = evaluate_rollouts(test, rollouts, det=det, k_list=(1, 8, 64))
print(report["d3"]["solve@64"], report["d3|bin4"]["solve@64"], report["all"]["verifier_false_pos"])

# 4. the reference: exact backward search with hints (cost = search_work)
print(solve_exact(a["s_d"][0], a["hints"][0], a["depth"][0]))
```

PyTorch:

```python
from torch.utils.data import DataLoader
from cainvert.storage.exact_catalog import load_splits
train, val = load_splits("data/catalogs/catalog_1d_w48_v021_exact", train_per_depth=15000, eval_per_cell=300, eval_split="val")
for batch in DataLoader(train, batch_size=64, shuffle=True):
    batch["input_ids"], batch["target"], batch["hint_mask"], batch["depth"]   # (B,2W+1) (B,W) (B,W) (B,)
```

Select checkpoints and hyper-parameters on `val`; report on `test`.

## Evaluation protocol (what to report)

* **det** – exact-match rate of the deterministic answer, per depth.
* **solve@k** – probability that at least one of $k$ stochastic rollouts passes the verifier, estimated
  without bias from $M \ge k$ rollouts (the pass@k estimator of Chen et al., 2021).  Because hints make the
  solution unique, a verifier pass *is* the ground truth (`verifier_false_pos` must be 0).
* **null@k** – the same for a null control that flips as many random bits of the deterministic answer as
  the rollout changed: how much of solve@k is mere variance.
* **compute** – per rollout in layer passes (`cainvert.eval.layer_passes(T, H, L)` for two-level recurrent
  models) so that depth (more recursion steps $T$) and breadth (more rollouts $k$) can be traded at equal
  budget.
* **strata** – every number per depth $d$, per $(d, \Lambda\text{ bin})$, per $(d, \text{hints})$ and per
  $(d, \text{search-work quantile})$.
* **reference** – the exact backward search (`cainvert.eval.solve_exact`) solves every task; its cost is
  the catalog's `search_work`.

## Catalogs

| catalog                       | $W$ | depths | tasks / ($d$, bin) | schema       | status                               |
|-------------------------------|-----|--------|--------------------|--------------|--------------------------------------|
| `catalog_1d_w48_v021_exact`   | 48  | 1–6    | 3000 (d ≤ 4), 2000 (d 5–6); $\Lambda=1$ is rare at $d \ge 4$ | v0.2.1-exact | release candidate (Hugging Face upload pending) |
| `catalog_1d_w32_v021_exact`   | 32  | 1–4    | 3000; bin ≥ 1024 empty at $W=32$ | v0.2.1-exact | auxiliary |

Every parquet file ships with a JSON sidecar (mining arguments, bin counts, sha256 of the parquet,
split counts).  `cainvert verify --catalog <dir>` re-checks a catalog: unique ids, no target in two splits,
ground truth passes the verifier, the reference solver recovers exactly the ground truth, `search_work`
matches.

## Mining your own

```bash
# one depth per call (array job); quotas per Lambda bin; deterministic given --seed once quotas are full
cainvert mine-exact --width 48 --depth 3 --quota 3000 --workers 16 \
    --bins "1-1,2-3,4-15,16-63,64-255,256-1023,1024-4096" --out-dir data/catalogs/my_w48
cainvert verify --catalog data/catalogs/my_w48
# re-key a schema v0.2.0 catalog (ids/split by x0) to v0.2.1 (ids/split by S_D, duplicates dropped)
cainvert resplit --in data/catalogs/old_v02 --out data/catalogs/new_v021
```

Mining cost grows quickly with depth (W = 48, 16 workers: d = 1 in a minute, d = 4 in ~3 h, d = 6 in ~8 h).

## Package layout

```
cainvert/
├── engines/eca_1d.py        exact 1D preimage enumeration and counting (de Bruijn transfer matrix)
├── mining/exact_miner.py    exact, branching-controlled catalogs (schema v0.2.1)     `cainvert mine-exact`
├── mining/resplit.py        re-key v0.2.0 catalogs                                   `cainvert resplit`
├── storage/exact_catalog.py split loading, balanced eval sets, tensors, hint control
├── eval/verifier.py         ground-truth-free verifier (numpy; torch twin)
├── eval/metrics.py          solve@k, null control, compute budget, strata, one-call protocol
├── eval/reference.py        exact backward search with hints (oracle baseline)
├── cli/main.py              mine-exact | resplit | verify | inspect | mine (legacy)
└── (core/, mining/tree_miner.py, engines/sat_2d.py, storage/torch_dataset.py)   legacy v0.1
tests/                       brute-force exactness, hint uniqueness, split keying, protocol
```

## Legacy v0.1

Version 0.1 mined **sampled** trees (at most 16 preimages per node, level subsampling) for 1D Rule 110
($W \in \{81, 400, 900\}$) and 2D Game of Life ($9\times9$, $20\times20$, $30\times30$) and published
510k of them on Hugging Face under the configs `1d_w81`, `1d_w400`, `1d_w900`, `2d_*`.  In those files
`leaf_count` is a lower bound of $\Lambda$ and the hints only separate the ground truth from the
*sampled* leaves, so a verifier can accept other valid preimages.  They remain available for backward
compatibility (`cainvert mine`, `cainvert.storage.torch_dataset.CAInversionDataset`) but should not be used
for new results.

## License

MIT.  If you use CAInvert, please cite the repository (paper in preparation).
