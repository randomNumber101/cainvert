# CAInvert: Cellular Automata Inversion Reasoning Benchmark

<p align="center">
  <a href="https://github.com/randomNumber101/cainvert"><img src="https://img.shields.io/badge/GitHub-Repo-blue.svg?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/randomNumber101/cainvert-benchmark"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="HuggingFace"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg?logo=pytorch" alt="PyTorch"></a>
  <a href="https://duckdb.org/"><img src="https://img.shields.io/badge/DuckDB-0.9%2B-fff000.svg" alt="DuckDB"></a>
</p>

**CAInvert** is a high-throughput mining engine, PyTorch reasoning framework, and large-scale dataset portfolio for evaluating recursive neural architectures on **computationally irreducible inverse dynamics**.

Operating on **Wolfram Rule 110** (1D Turing-complete) and **Conway's Game of Life** (2D class 4), CAInvert reverses non-injective cellular automata dynamics by mining full **preimage branching trees** and synthesizing minimal spatial **disambiguation hint tapes** via greedy hitting-set algorithms.

---

## 🌟 Key Highlights

* 🌲 **500,000+ Stratified Preimage Trees**: Balanced benchmark partitions across scale parity $N \in \{81, 400, 900\}$ and depths $d \in \{1, 2, 3, 4, 6, 8, 10, 12\}$.
* ⚡ **High-Throughput Dual Engines**:
  * **1D Engine**: Exact transfer-matrix trellis dynamic programming in $O(16W)$ time per step.
  * **2D Engine**: SAT-based backward unrolling via PySAT with incremental clause caching.
* 🗄️ **Zero-Copy Parquet Catalogs via DuckDB**: Over 500k trees stored in only ~160 MB of sharded Parquet files. Sub-second SQL queries, predicate pushdowns, and zero RAM-bloat.
* 🔥 **Native PyTorch Dynamic Dataset**: On-the-fly SQL curriculum slicing ($d \in [1, 4]$ vs $d \in [6, 12]$) and ambiguity filtering directly into standard PyTorch `DataLoader` pipelines.
* 🔬 **The K-Step Wall Probe**: Empirically benchmarks continuous representation drift, revealing a strict negative correlation ($r = -0.7399$) between recursion depth and state fidelity.

---

## 📊 Dataset Portfolio Overview

The CAInvert dataset suite provides exact token length and complexity parity between 1D and 2D lattices ($N \in \{81, 400, 900\}$, corresponding to $9 \times 9$ Sudoku, $20 \times 20$ Maze, and $30 \times 30$ planning grids):

| Catalog Split | Dimension | Rule | Lattice Shape | Sequence Length ($N$) | Verified Trees | Shards (Size) | Target Depths ($d$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1d_w81`** | 1D | Rule 110 | $(81,)$ | $N = 81$ | **249,920** | 100 shards (32.5 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`1d_w400`** | 1D | Rule 110 | $(400,)$ | $N = 400$ | **149,920** | 60 shards (54.1 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`1d_w900`** | 1D | Rule 110 | $(900,)$ | $N = 900$ | **100,000** | 40 shards (73.9 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`2d_9x9`** | 2D | Life $B3/S23$ | $(9, 9)$ | $N = 81$ | **4,400** | 176 shards (1.5 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`2d_30x30`** | 2D | Life $B3/S23$ | $(30, 30)$ | $N = 900$ | **150+** | 6 shards (0.2 MB) | $1, 2, 3, 4, 6, 8$ |
| **Total** | - | - | - | - | **504,390+** | **382 shards (~162 MB)** | - |

---

## 🔬 Empirical Baseline: `ProbWTADynamic_S`

Evaluation of our best recursive Transformer architecture (`ProbWTADynamic_S`: dynamic macro-segments $S=d$, dual-state $H/L$ cycles, probabilistic $\epsilon$-WTA routing with $M=4$ candidates) on `1d_w81`:

<p align="center">
  <img src="https://raw.githubusercontent.com/randomNumber101/cainvert/main/docs/images/probwta_benchmark_dashboard.png" width="90%" alt="Benchmark Dashboard">
</p>

### Key Empirical Findings:
1. **Monotonic Representation Drift**:
   * Token Accuracy at $d=1$: **89.56%**
   * Token Accuracy at $d=2$: **75.02%**
   * Token Accuracy at $d=4$: **67.50%**
   * Token Accuracy at $d=12$: **57.55%** (approaching random baseline)
2. **K-Step Wall on Large Lattices ($W=81$)**:
   Because full sequence correctness across 81 bits scales as $(P_{\text{token}})^{81}$, exact sequence accuracy collapses from $2.0\%$ at $d=1$ to $0.0\%$ at $d \ge 2$, providing clear evidence that continuous latent loops require explicit discretization (VQ / discrete scratchpads) for deep recursion.
3. **Correlation Analysis**:
   * Correlation(Depth $d$, Token Accuracy): **$r = -0.7399$** (dominant performance destruction factor).
   * Correlation($\log \Lambda$, Token Accuracy): **$r = -0.5178$** (moderate ambiguity penalty).

---

## 🚀 Quickstart & Installation

```bash
git clone https://github.com/randomNumber101/cainvert.git
cd cainvert
pip install -e .
```

To install optional PyTorch and developer dependencies:
```bash
pip install -e ".[torch,dev]"
```

---

## 💡 Usage Examples

### 1. Dynamic Dataset Loading in PyTorch via DuckDB
No need to uncompress gigabytes of data into RAM. Construct custom training splits on the fly:

```python
from torch.utils.data import DataLoader
from cainvert.storage.torch_dataset import CAInversionDataset

# 1. Curriculum Training: Only Depths 1 to 4 with non-trivial ambiguity
train_dataset = CAInversionDataset(
    catalog_path="data/catalogs/catalog_1d_w81",
    depth_min=1,
    depth_max=4,
    leaf_min=8,                          # Filter out trivial deterministic preimages
    sql_filter="abs(hash(tree_id)) % 10 < 8"  # Deterministic 80% SQL train-split
)

# 2. Extrapolation Test Split: Depths 8 to 12
test_dataset = CAInversionDataset(
    catalog_path="data/catalogs/catalog_1d_w81",
    depth_min=8,
    depth_max=12,
    sql_filter="abs(hash(tree_id)) % 10 >= 8" # Deterministic 20% test-split
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for batch in train_loader:
    # batch['input_ids']: [B, 2N+1] Tensor -> [S_D] + [SEP=3] + [Hints_S0 (Mask=2)]
    # batch['target']:    [B, N]    Tensor -> Ground Truth S_0 in {0, 1}
    # batch['depth']:     [B]       Tensor -> Recursion depth d
    inputs, targets, depths = batch["input_ids"], batch["target"], batch["depth"]
```

### 2. High-Speed SQL Analytics with `TreeCatalog`
Inspect, filter, or aggregate precomputed tree properties with sub-second execution:

```python
from cainvert.storage.catalog import TreeCatalog

catalog = TreeCatalog("data/catalogs/catalog_1d_w81/*.parquet")

# Run arbitrary analytical SQL queries:
df = catalog.query("""
    SELECT 
        depth,
        COUNT(*) AS num_trees,
        AVG(leaf_count) AS avg_leaves,
        AVG(effective_branching) AS avg_branching,
        AVG(hint_density) AS avg_hint_density
    FROM catalog
    GROUP BY depth
    ORDER BY depth
""")
print(df)
```

### 3. Mine Preimage Trees Programmatically

```python
from cainvert.engines.eca_1d import ECA1DEngine
from cainvert.mining.tree_miner import TreeMiner

engine = ECA1DEngine(rule_number=110)
miner = TreeMiner(engine=engine, leaf_cap=512, max_nodes_per_level=32)

# Unroll a verified preimage tree of depth 6 on an 81-cell lattice:
tree = miner.mine_single_tree(grid_shape=(81,), target_depth=6)

print(f"Mined Tree ID: {tree.tree_id}")
print(f"Total Leaves: {tree.leaf_count} | Effective Branching: {tree.effective_branching:.2f}")
print(f"Minimal Disambiguating Hints Required: {len(tree.hint_indices)}")
```

---

## 🛠️ CLI Reference

CAInvert includes a unified command-line interface:

```bash
# Mine 1D Rule 110 trees with 8 CPU workers:
cainvert mine --dimension 1d --rule 110 --width 81 --depths 1,2,3,4,6,8,10,12 --total-samples 25000 --workers 8 --out-dir data/catalogs/my_catalog

# Mine 2D Conway Life trees via SAT:
cainvert mine --dimension 2d --rule-str B3/S23 --width 9 --height 9 --depths 1,2,3,4,6 --total-samples 5000 --workers 4 --out-dir data/catalogs/life_9x9

# Audit and inspect Parquet catalogs:
cainvert inspect --catalog data/catalogs/my_catalog
```

---

## 📦 Directory Structure

```
cainvert/
├── cainvert/
│   ├── core/            # TreeData schema, filters, greedy hitting set solver
│   ├── engines/         # Trellis DP (1D ECA) & PySAT CNF solver (2D Life)
│   ├── mining/          # TreeMiner, ParallelMiner, StratifiedBuckets
│   ├── storage/         # Parquet I/O, TreeCatalog (DuckDB), PyTorch Dataset
│   └── cli/             # Unified CLI commands
├── tests/               # Full pytest suite
├── docs/                # Figures, schema documentation
├── pyproject.toml       # Modern packaging configuration
└── LICENSE              # MIT License
```

---

## 📜 Citation

If you use CAInvert in your research, please cite:

```bibtex
@software{cainvert2026,
  author = {Khalil, M. and collaborators},
  title = {CAInvert: Cellular Automata Inversion Reasoning Benchmark for Recursive Models},
  year = {2026},
  url = {https://github.com/randomNumber101/cainvert}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
