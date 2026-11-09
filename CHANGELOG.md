# Changelog

## 0.2.0 (2026-11)

Exact catalogs and an evaluation protocol.

* `cainvert.mining.exact_miner` (schema **v0.2.1-exact**): complete backward enumeration (true Λ, level
  widths, search work), greedy hitting-set hints that are unique against *all* preimages, `hint_order`
  for hint-controlled experiments, Λ-bin quotas.  Ids and the 70/15/15 train/val/test split are keyed by
  the hash of the target state S_D, one task per (S_D, d).  Results are consumed in task order, so a
  catalog is a deterministic function of the seed once the quotas are full.  JSON sidecar with mining
  arguments, counts and the parquet sha256.
* `cainvert.mining.resplit`: re-key v0.2.0 catalogs (ids/split by x0; the same S_D could land in train and
  val with two ground truths) to v0.2.1 without re-mining.
* `cainvert.storage.exact_catalog`: split loading, Λ-balanced evaluation sets, tensors, `ExactCatalogDataset`,
  `hints_with_k_revealed`.
* `cainvert.eval`: ground-truth-free verifier (numpy + torch), unbiased solve@k, null control, compute budget
  in layer passes, strata (depth × Λ bin × hints × search-work quantile), `evaluate_rollouts`, exact
  reference solver.
* CLI: `cainvert mine-exact`, `cainvert resplit`, `cainvert verify`.
* Tests: brute-force exactness of the enumeration and of every level width, hint uniqueness, split keying,
  deterministic deduplication, protocol and reference solver.
* README rewritten: no claims of computational irreducibility or minimal hint sets; v0.1 marked legacy.
* `python-sat` is now an optional dependency (`[sat2d]`); the 2D Life engine is legacy.

## 0.1.0 (2026-09-30)

Initial release: sampled tree miner (capped branching), 1D Rule 110 and 2D Life catalogs on Hugging Face.
