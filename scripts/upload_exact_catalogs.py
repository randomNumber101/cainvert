"""Upload exact (schema v0.2.1) catalogs to the Hugging Face dataset repo and write the dataset card.

    export HF_TOKEN=...            # never put the token into a file
    python scripts/upload_exact_catalogs.py \
        --catalog 1d_w48_exact=data/catalogs/catalog_1d_w48_v021_exact \
        --catalog 1d_w32_exact=data/catalogs/catalog_1d_w32_v021_exact \
        --repo-id randomNumber101/cainvert-benchmark [--dry-run]

Each catalog goes to ``data/<config>/`` (parquet + json sidecars).  The card lists every config with
``data_files`` so that ``datasets.load_dataset(repo, "<config>")`` works; the v0.1 configs are kept and
marked legacy.  Parquet sha256 values from the sidecars are written into the card.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

LEGACY = ["1d_w81", "1d_w400", "1d_w900", "2d_9x9", "2d_20x20", "2d_30x30"]

CARD_HEADER = """---
license: mit
pretty_name: CAInvert
tags:
- cellular-automata
- recursive-reasoning
- rule110
- benchmark
size_categories:
- 100K<n<1M
configs:
{configs}
---
"""

CARD_BODY = """# CAInvert – exact cellular-automata inversion catalogs

Invert Rule 110 on a ring of W cells: given S_D = f^d(S_0) and a hint tape H_0 (a few revealed cells
of S_0 that make the answer unique), reconstruct S_0.  Every task carries exact difficulty descriptors
from the complete backward tree.  Code, protocol and tests: https://github.com/randomNumber101/cainvert

## Configs (schema v0.2.1-exact)

{exact_table}

Columns: `tree_id`, `sd_hash`, `x0_hash`, `depth`, `leaf_count` (Λ, exact), `level_widths` (w_1..w_d),
`search_work` (Σ w_k), `num_hints`, `hint_order`, `leaf_bin`, `target_state` (S_D, W bytes),
`ground_truth_s0` (S_0, W bytes), `hints_s0` (W int8, -1 = masked), `split` (train/val/test keyed by the
hash of S_D), `effective_branching`, `hint_density`, `mean_activity`, `catalog_version`.

Model input: `[S_D] [SEP=3] [H_0 with -1 -> MASK=2]` (2W+1 tokens); target `S_0`.  Select on `val`,
report on `test`.  Evaluation protocol (verifier-based solve@k, null control, strata, reference solver):
`pip install cainvert` → `cainvert.eval`.

Sampling: S_0 uniform on {{0,1}}^W, trajectory filter (25–75 % live cells, ≥ 5 % cells changing per step),
complete backward enumeration (levels > 20 000 rejected), greedy hitting-set hints, quotas per
(depth, Λ bin) by rejection sampling (balanced over Λ bins, not the natural Λ distribution).

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", "1d_w48_exact", split="train")
row = ds[0]; row["depth"], row["leaf_count"], row["search_work"], row["split"]
```

## Legacy configs (v0.1, sampled trees)

`{legacy}` were mined with a per-node cap of 16 preimages and level subsampling: `leaf_count` is a
lower bound of Λ and hints only separate the ground truth from the *sampled* leaves.  Kept for
backward compatibility; do not use for new results.

## Checksums

{checksums}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", action="append", required=True, help="<config>=<local dir>, repeatable")
    ap.add_argument("--repo-id", default="randomNumber101/cainvert-benchmark")
    ap.add_argument("--card-out", default="docs/DATASET_CARD.md", help="where to also write the card locally")
    ap.add_argument("--dry-run", action="store_true", help="write the card, upload nothing")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("HF_TOKEN is not set (export HF_TOKEN=...)", file=sys.stderr)
        return 2

    cats = {}
    for spec in args.catalog:
        name, path = spec.split("=", 1)
        files = sorted(glob.glob(os.path.join(path, "exact_*.parquet")))
        if not files:
            print(f"no exact_*.parquet under {path}", file=sys.stderr)
            return 2
        metas = []
        for f in files:
            j = f[:-8] + ".json"
            metas.append(json.load(open(j)) if os.path.exists(j) else {})
        cats[name] = (path, files, metas)

    cfg_lines = []
    for name in list(cats) + LEGACY:
        cfg_lines.append(f"- config_name: {name}\n  data_files:\n  - split: train\n    path: data/{name}/*.parquet")
    rows, sums = [], []
    for name, (path, files, metas) in cats.items():
        n = sum(m.get("n", 0) for m in metas)
        depths = sorted(m.get("depth", 0) for m in metas)
        W = metas[0].get("width", "?")
        rows.append(f"| `{name}` | {W} | {depths[0]}–{depths[-1]} | {n:,} | {metas[0].get('schema_version', '?')} |")
        for f, m in zip(files, metas):
            sums.append(f"- `data/{name}/{os.path.basename(f)}`: sha256 `{m.get('parquet_sha256', '?')}` "
                        f"(n={m.get('n', '?')}, counts={m.get('counts', '?')})")
    table = "| config | W | depths | tasks | schema |\n|---|---|---|---|---|\n" + "\n".join(rows)
    card = CARD_HEADER.format(configs="\n".join(cfg_lines)) + CARD_BODY.format(
        exact_table=table, repo_id=args.repo_id, legacy="`, `".join(LEGACY), checksums="\n".join(sums))
    os.makedirs(os.path.dirname(args.card_out) or ".", exist_ok=True)
    with open(args.card_out, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"dataset card written to {args.card_out}")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    for name, (path, files, _) in cats.items():
        print(f"uploading {len(files)} parquet files (+ sidecars) from {path} -> data/{name}/")
        api.upload_folder(folder_path=path, path_in_repo=f"data/{name}", repo_id=args.repo_id,
                          repo_type="dataset", allow_patterns=["exact_*.parquet", "exact_*.json"])
    api.upload_file(path_or_fileobj=args.card_out, path_in_repo="README.md", repo_id=args.repo_id, repo_type="dataset")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
