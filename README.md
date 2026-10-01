# CAInvert: Cellular Automata Inversion Reasoning Benchmark

<p align="center">
  <a href="https://github.com/randomNumber101/cainvert"><img src="https://img.shields.io/badge/GitHub-Repo-blue.svg?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/randomNumber101/cainvert-benchmark"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="HuggingFace"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg?logo=pytorch" alt="PyTorch"></a>
  <a href="https://duckdb.org/"><img src="https://img.shields.io/badge/DuckDB-0.9%2B-fff000.svg" alt="DuckDB"></a>
</p>

**CAInvert** is a high-throughput mining engine, PyTorch reasoning framework, and large-scale dataset suite designed for evaluating recursive neural architectures and reasoning models on **computationally irreducible inverse dynamics**.

Operating on **Wolfram Rule 110** (1D Turing-complete) and **Conway's Game of Life** (2D class 4), CAInvert inverts cellular automata forward dynamics by mining full **preimage branching trees** and synthesizing minimal spatial **disambiguation hint tapes** via hitting-set algorithms.

The complete benchmark portfolio of **510,000+ verified preimage trees** is publicly available on [Hugging Face Datasets](https://huggingface.co/datasets/randomnumber101/cainvert-benchmark).

---

## 🎯 The Inversion Task

In forward cellular automata, state $S_0$ evolves deterministically through local transition rules to $S_D$ after $D$ time steps:
$$S_0 \xrightarrow{\mathcal{T}} S_1 \xrightarrow{\mathcal{T}} \dots \xrightarrow{\mathcal{T}} S_D$$

Because the local update rule $\mathcal{T}$ is **non-injective** (many distinct local configurations yield the same successor), inverting $S_D \to S_0$ is fundamentally ambiguous: a single state $S_D$ can possess multiple valid backward preimages, forming a branching tree of depth $D$ with $\Lambda$ leaves at $t=0$.

```
Forward:   S_0 --------------------------> S_D  (Deterministic, local)
                                            |
Inverse:   S_D ─── Branching Tree ───> {S_0^(1), S_0^(2), ..., S_0^(Λ)}  (Ambiguous)
                 + Minimal Hints H_0
                 ───────────────────> Unique True S_0                    (Well-posed)
```

To formulate a rigorous, well-posed prediction task for neural networks, CAInvert:
1. **Unrolls the full preimage tree** backwards from $S_D$ to depth $D$, discovering all $\Lambda$ valid ancestor states $\{S_0^{(1)}, \dots, S_0^{(\Lambda)}\}$.
2. **Selects a ground truth ancestor** $S_0^*$.
3. **Solves a minimal hitting set** over the leaf difference matrix to compute the smallest subset of spatial indices $H_{\text{indices}} \subset \{0, \dots, N-1\}$ whose values uniquely distinguish $S_0^*$ from all other $\Lambda - 1$ candidate preimages.
4. **Constructs the hint tape** $H_0 \in \{0, 1, \mathtt{MASK}\}^N$, where only the minimal disambiguating cell values are revealed and all other positions are masked.

### Model Input & Target Format

The neural network is tasked with reconstructing the complete initial configuration $S_0^*$ given only the final observation $S_D$ and the sparse spatial hints $H_0$:

$$\mathbf{x} = [S_D] \circ [\mathtt{SEP}] \circ [H_0] \quad \in \{0, 1, \mathtt{MASK}, \mathtt{SEP}\}^{2N+1}$$
$$\mathbf{y} = S_0^* \quad \in \{0, 1\}^N$$

* **Lattice Size $N$**: Total number of cells (e.g., $N=81$ for $1\text{D } W=81$ or $2\text{D } 9 \times 9$; $N=400$ for $1\text{D } W=400$ or $2\text{D } 20 \times 20$; $N=900$ for $1\text{D } W=900$ or $2\text{D } 30 \times 30$).
* **Input Sequence Length**: $2N + 1$ tokens, where:
  * Token values `0` and `1`: Cell binary states.
  * Token `2` (`MASK`): Unrevealed hint positions.
  * Token `3` (`SEP`): Spatial delimiter between $S_D$ and $H_0$.
* **Target Output**: The exact $N$-dimensional binary sequence $S_0^* \in \{0, 1\}^N$.

---

## 💡 Why CAInvert as a Reasoning Benchmark?

Evaluating multi-step reasoning in neural architectures (such as recursive Transformers, iterative latent models, and Chain-of-Thought systems) is often confounded by natural language ambiguities, dataset memorization, or loose definitions of "reasoning depth". CAInvert provides an ideal testbed:

### 1. Fine-Grained, Parametric Complexity Controls
Every sample in CAInvert comes annotated with exact topological and combinatorial difficulty metrics:
* **Recursion Depth ($d$)**: The exact number of backward deduction steps required to reach the origin ($d \in [1, 12]$).
* **Branching Factor ($b_{\text{eff}}$)**: The effective branching rate $b_{\text{eff}} = \Lambda^{1/d}$, measuring local ambiguity per step.
* **Leaf Ambiguity ($\Lambda$)**: The total combinatorial space of valid preimages ($1 \le \Lambda \le 512+$).
* **Hint Density ($\rho_H$)**: The exact ratio $|H_{\text{indices}}| / N$, quantifying the spatial constraint strength provided to the model.

### 2. Computational Irreducibility (No Shortcuts)
Both **Rule 110** (1D) and **Conway's Game of Life** (2D) are Class 4, Turing-complete cellular automata. By Wolfram's principle of computational irreducibility, there is no closed-form shortcut $f(S_D, d) \to S_0$ that bypasses simulating or constraint-solving the intermediate steps. The model cannot exploit algebraic tricks or superficial n-gram patterns.

### 3. Direct Structural Parallel to Classic Hard Benchmarks
CAInvert shares fundamental problem structures with standard reasoning benchmarks while offering precise parametric control:

| Benchmark | Structural Nature | Search / Inference Dynamics | What CAInvert Contributes |
| :--- | :--- | :--- | :--- |
| **Sudoku** | Local constraint satisfaction | Propagating row/col/box rules; branching on ambiguity; backtracking. | Same local constraint propagation and disambiguation, with customizable lattice dimensions ($N \in \{81, 400, 900\}$) and continuous depth control ($d$). |
| **Maze Hard** | Spatial pathfinding & reachability | Graph search, dead-end detection, backtracking from terminal state to start. | Backward unrolling directly mirrors backward search through state graphs; branching points correspond to path intersections. |
| **ARC (Abstraction & Reasoning Corpus)** | Spatial induction & transformation | Identifying geometric transformation primitives under spatial priors. | Clean grid-based spatial transformations without open-ended inductive ambiguity: the forward operator is fixed, isolating deductive reasoning capacity. |

---

## 📊 Dataset Portfolio Overview

The pre-mined benchmark is partitioned across exact scale parities between 1D and 2D lattices ($N \in \{81, 400, 900\}$):

| Catalog Split | Dimension | Rule | Lattice Shape | Sequence Tokens ($N$) | Verified Trees | Shards (Size) | Target Depths ($d$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1d_w81`** | 1D | Rule 110 | $(81,)$ | $N = 81$ | **249,920** | 100 shards (32.5 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`1d_w400`** | 1D | Rule 110 | $(400,)$ | $N = 400$ | **149,920** | 60 shards (54.1 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`1d_w900`** | 1D | Rule 110 | $(900,)$ | $N = 900$ | **100,000** | 40 shards (73.9 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`2d_9x9`** | 2D | Life $B3/S23$ | $(9, 9)$ | $N = 81$ | **6,000** | 192 shards (1.8 MB) | $1, 2, 3, 4, 6, 8, 10, 12$ |
| **`2d_20x20`** | 2D | Life $B3/S23$ | $(20, 20)$ | $N = 400$ | **2,200** | 411 shards (4.4 MB) | $1, 2, 3, 4, 6, 8, 10$ |
| **`2d_30x30`** | 2D | Life $B3/S23$ | $(30, 30)$ | $N = 900$ | **2,324** | 437 shards (8.0 MB) | $1, 2, 3, 4, 6, 8, 10$ |
| **Total** | - | - | - | - | **510,364** | **1,240 shards (174.8 MB)** | - |

All splits are hosted on [Hugging Face: randomnumber101/cainvert-benchmark](https://huggingface.co/datasets/randomnumber101/cainvert-benchmark).

---

## 📦 Package Architecture

```
cainvert/
├── cainvert/
│   ├── core/            # Core schemas (TreeData), tree filters, minimal hitting-set solver
│   ├── engines/         # 1D Trellis DP (O(16W)) & 2D PySAT CNF incremental unroller
│   ├── mining/          # TreeMiner, ParallelMiner worker pool, StratifiedBuckets
│   ├── storage/         # Zero-copy Parquet I/O, TreeCatalog (DuckDB), PyTorch Dataset
│   └── cli/             # CLI commands (cainvert mine, inspect, merge)
├── tests/               # Pytest suite with complete coverage
├── docs/                # Figures, architecture docs, and tutorials
├── pyproject.toml       # Modern packaging configuration
└── LICENSE              # MIT License
```

* **`cainvert.core`**: Defines the typed `TreeData` container, validation rules, and the greedy hitting-set algorithm that extracts the minimal spatial hint indices $H_{\text{indices}}$ to disambiguate the true root from competing leaves.
* **`cainvert.engines`**: High-performance forward and inverse solvers:
  * `ECA1DEngine`: Transfer-matrix dynamic programming over de Bruijn trellises, finding all 1D preimages in $O(16W)$ time per step.
  * `Life2DEngine`: SAT-based inverse formulation using incremental CNF encodings via PySAT (`cadical195`), with spatial boundary conditions and clause caching.
* **`cainvert.mining`**: Tree generation pipeline that samples target states, unrolls preimage trees level-by-level, prunes invalid paths, and stratifies trees across targeted $(d, b)$ buckets.
* **`cainvert.storage`**: High-throughput storage layer combining compact Parquet serialisation, DuckDB SQL querying, and zero-overhead PyTorch streaming datasets.
* **`cainvert.cli`**: Unified command-line interface for distributed mining and catalog management.

---

## 🚀 Installation

```bash
git clone https://github.com/randomNumber101/cainvert.git
cd cainvert
pip install -e .
```

To install optional PyTorch and Hugging Face Hub dependencies:
```bash
pip install -e ".[torch,hub,dev]"
```

---

## 📖 Usage Guide

### 1. Loading the Benchmark from Hugging Face

You can load any partition directly with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the 81-token 1D Rule 110 partition (249,920 trees)
ds_1d = load_dataset("randomnumber101/cainvert-benchmark", "1d_w81", split="train")

# Load the 2D Game of Life 9x9 partition
ds_2d = load_dataset("randomnumber101/cainvert-benchmark", "2d_9x9", split="train")

sample = ds_1d[0]
print(f"Tree ID: {sample['tree_id']}")
print(f"Target Depth: {sample['target_depth']}")
print(f"Leaf Count: {sample['leaf_count']}")
print(f"Hint Indices: {sample['hint_indices']}")
```

### 2. High-Performance SQL Analytics via DuckDB (Zero Download)

Because the dataset is stored in standard columnar Parquet, you can query metadata and filter splits directly over HTTP without downloading the entire repository into memory:

```python
import duckdb

conn = duckdb.connect()

# Query summary statistics directly from Hugging Face via remote Parquet
url = "https://huggingface.co/datasets/randomnumber101/cainvert-benchmark/resolve/main/data/1d_w81/*.parquet"

summary = conn.execute(f"""
    SELECT 
        depth,
        COUNT(*) AS num_trees,
        AVG(leaf_count) AS avg_leaves,
        AVG(effective_branching) AS avg_branching,
        AVG(hint_density) AS avg_hint_density
    FROM read_parquet('{url}')
    GROUP BY depth
    ORDER BY depth
""").df()

print(summary)
```

### 3. Dynamic PyTorch Training & Curriculum Slicing

Use `CAInversionDataset` to feed pre-tokenized $[S_D] + [\mathtt{SEP}] + [H_0] \to S_0$ tensors directly into your PyTorch training loop:

```python
from torch.utils.data import DataLoader
from cainvert.storage.torch_dataset import CAInversionDataset

# 1. Easy Curriculum Split: Depths 1 to 4 with non-trivial ambiguity
train_dataset = CAInversionDataset(
    catalog_path="data/catalogs/catalog_1d_w81",  # Local folder or glob
    depth_min=1,
    depth_max=4,
    leaf_min=4,                                  # Filter out trivial deterministic preimages
    sql_filter="abs(hash(tree_id)) % 10 < 8"     # Deterministic 80% SQL train split
)

# 2. Out-of-Distribution Depth Evaluation: Depths 8 to 12
eval_dataset = CAInversionDataset(
    catalog_path="data/catalogs/catalog_1d_w81",
    depth_min=8,
    depth_max=12,
    sql_filter="abs(hash(tree_id)) % 10 >= 8"    # Deterministic 20% eval split
)

loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for batch in loader:
    # batch['input_ids']: [B, 2N+1] Tensor -> [S_D] + [SEP=3] + [Hints_S0 (Mask=2)]
    # batch['target']:    [B, N]    Tensor -> Ground Truth S_0 in {0, 1}
    # batch['depth']:     [B]       Tensor -> Target recursion depth d
    inputs  = batch["input_ids"]
    targets = batch["target"]
    depths  = batch["depth"]
```

### 4. Mining Additional Preimage Trees

You can mine additional custom splits using the unified CLI:

```bash
# Mine 25,000 1D Rule 110 trees with 8 CPU workers:
cainvert mine --dimension 1d \
              --rule 110 \
              --width 81 \
              --depths 1,2,3,4,6,8,10,12 \
              --total-samples 25000 \
              --workers 8 \
              --out-dir data/catalogs/my_1d_catalog

# Mine 2D Conway Life trees via SAT:
cainvert mine --dimension 2d \
              --rule-str B3/S23 \
              --width 9 \
              --height 9 \
              --depths 1,2,3,4,6 \
              --total-samples 2000 \
              --workers 4 \
              --out-dir data/catalogs/my_2d_catalog

# Inspect and verify catalog distribution:
cainvert inspect --catalog data/catalogs/my_1d_catalog
```

Or mine programmatically within Python:

```python
from cainvert.engines.eca_1d import ECA1DEngine
from cainvert.mining.tree_miner import TreeMiner

engine = ECA1DEngine(rule_number=110)
miner = TreeMiner(engine=engine, leaf_cap=512, max_nodes_per_level=32)

# Mine a single tree of depth 6 on an 81-cell lattice:
tree = miner.mine_single_tree(grid_shape=(81,), target_depth=6)

print(f"Mined Tree ID: {tree.tree_id}")
print(f"Leaves: {tree.leaf_count} | Effective Branching: {tree.effective_branching:.2f}")
print(f"Disambiguating Hint Positions: {tree.hint_indices}")
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
