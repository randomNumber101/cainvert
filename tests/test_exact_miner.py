"""Exact miner (schema v0.2.1): brute-force exactness, hint uniqueness, split keying, dedup, CLI."""
import itertools
import os

import numpy as np
import pyarrow.parquet as pq
import pytest

from cainvert.engines.eca_1d import ECA1DEngine
from cainvert.mining.exact_miner import (
    assign_split, exact_tree, greedy_hint_order, main as mine_main, state_hash, SCHEMA_VERSION,
)
from cainvert.mining.resplit import resplit_table


def brute_force_preimages(engine, s_d, depth):
    W = len(s_d)
    states = np.array(list(itertools.product([0, 1], repeat=W)), dtype=np.uint8)
    cur = states.copy()
    for _ in range(depth):
        cur = np.stack([engine.forward_step(s) for s in cur])
    return states[(cur == s_d[None, :]).all(1)]


@pytest.mark.parametrize("W,depth", [(8, 1), (10, 2), (12, 2), (10, 3)])
def test_exact_tree_matches_brute_force(W, depth):
    eng = ECA1DEngine(110)
    rng = np.random.RandomState(W * 100 + depth)
    checked = 0
    for _ in range(20):
        x0 = rng.randint(0, 2, W).astype(np.uint8)
        s_d = x0.copy()
        for _ in range(depth):
            s_d = eng.forward_step(s_d)
        widths, leaves = exact_tree(eng, s_d, depth, level_cap=10 ** 6)
        bf = brute_force_preimages(eng, s_d, depth)
        assert leaves is not None and len(widths) == depth
        assert sorted(map(bytes, leaves)) == sorted(map(bytes, bf))
        assert widths[-1] == len(bf)
        # level k width = number of k-step preimages
        for k in range(1, depth + 1):
            assert widths[k - 1] == len(brute_force_preimages(eng, s_d, k))
        checked += 1
    assert checked == 20


def test_greedy_hints_single_out_ground_truth():
    eng = ECA1DEngine(110)
    rng = np.random.RandomState(0)
    W, depth = 12, 2
    n_amb = 0
    for _ in range(30):
        x0 = rng.randint(0, 2, W).astype(np.uint8)
        s_d = x0.copy()
        for _ in range(depth):
            s_d = eng.forward_step(s_d)
        _, leaves = exact_tree(eng, s_d, depth, 10 ** 6)
        gt = np.where((leaves == x0[None, :]).all(1))[0]
        assert len(gt) == 1
        distract = np.delete(leaves, gt[0], axis=0)
        chosen, order = greedy_hint_order(x0, distract, rng)
        assert sorted(order) == list(range(W)) and order[:len(chosen)] == chosen
        if len(distract):
            n_amb += 1
            agree = (leaves[:, chosen] == x0[chosen][None, :]).all(1)
            assert agree.sum() == 1
            # every proper prefix of the greedy set leaves >= 2 candidates (greedy adds only when needed)
            if len(chosen) > 1:
                sub = chosen[:-1]
                assert (leaves[:, sub] == x0[sub][None, :]).all(1).sum() >= 2
        else:
            assert chosen == []
    assert n_amb > 0


def test_split_is_keyed_by_target_hash_and_balanced():
    rng = np.random.RandomState(1)
    hashes = [state_hash(rng.randint(0, 2, 32).astype(np.uint8)) for _ in range(20000)]
    splits = np.array([assign_split(h, 0.15, 0.15) for h in hashes])
    assert set(splits) == {"train", "val", "test"}
    assert abs((splits == "test").mean() - 0.15) < 0.01
    assert abs((splits == "val").mean() - 0.15) < 0.01
    # deterministic
    assert [assign_split(h) for h in hashes[:100]] == [assign_split(h) for h in hashes[:100]]
    assert assign_split("0" * 16) == "test" and assign_split("3" * 16) == "train"


def _row(sd, x0, W, d=1):
    return dict(sd_hash=state_hash(sd), x0_hash=state_hash(x0), depth=d, leaf_count=2, leaf_bin=1,
                level_widths=[2] * d, search_work=2 * d, num_hints=1, hint_order=list(range(W)),
                target_state=sd.tobytes(), ground_truth_s0=x0.tobytes(),
                hints_s0=np.full(W, -1, dtype=np.int8).tobytes(), mean_activity=0.3)


def test_resplit_drops_duplicate_targets_deterministically():
    W = 16
    rng = np.random.RandomState(3)
    sd = rng.randint(0, 2, W).astype(np.uint8)
    a, b = rng.randint(0, 2, W).astype(np.uint8), rng.randint(0, 2, W).astype(np.uint8)
    other = rng.randint(0, 2, W).astype(np.uint8)
    rows = [_row(sd, a, W), _row(sd, b, W), _row(other, a, W)]
    table, dropped = resplit_table([dict(r) for r in rows], W, 1, 110, 0.15, 0.15)
    assert dropped == 1 and table.num_rows == 2
    kept = {r["sd_hash"]: r for r in table.to_pylist()}
    assert kept[state_hash(sd)]["x0_hash"] == min(state_hash(a), state_hash(b))
    assert all(r["tree_id"].endswith(r["sd_hash"]) for r in table.to_pylist())
    assert all(r["catalog_version"] == SCHEMA_VERSION for r in table.to_pylist())
    # order of input rows must not matter
    table2, _ = resplit_table([dict(r) for r in rows[::-1]], W, 1, 110, 0.15, 0.15)
    assert table.to_pylist() == table2.to_pylist()


MINE_ARGS = ["--width", "12", "--depth", "1", "--quota", "4", "--workers", "2", "--bins", "1-3,4-15",
             "--time-limit-min", "0.5", "--max-tries-per-task", "200"]


def test_cli_mines_small_catalog(tmp_path):
    import json
    path = mine_main(MINE_ARGS + ["--out-dir", str(tmp_path / "cat")])
    t = pq.read_table(path).to_pylist()
    meta = json.load(open(path.replace(".parquet", ".json")))
    assert len(t) == sum(meta["counts"]) == meta["n"] >= 4
    assert meta["schema_version"] == SCHEMA_VERSION and len(meta["parquet_sha256"]) == 64
    assert len({r["sd_hash"] for r in t}) == len(t)
    assert len({r["tree_id"] for r in t}) == len(t)
    assert {r["split"] for r in t} <= {"train", "val", "test"}
    for r in t:
        assert state_hash(r["target_state"]) == r["sd_hash"] and r["tree_id"].endswith(r["sd_hash"])
        assert sum(r["level_widths"]) == r["search_work"] and r["level_widths"][-1] == r["leaf_count"]
        assert r["catalog_version"] == SCHEMA_VERSION
    # mining twice with the same seed gives the same catalog once the quotas were filled before the time limit
    path2 = mine_main(MINE_ARGS + ["--out-dir", str(tmp_path / "cat2")])
    if meta["quotas_complete"]:
        assert pq.read_table(path).to_pylist() == pq.read_table(path2).to_pylist()
