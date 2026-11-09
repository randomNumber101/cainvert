"""Evaluation protocol: verifier, pass@k, null control, strata, reference solver, loader."""
import numpy as np
import pyarrow.parquet as pq
import pytest

from cainvert.engines.eca_1d import ECA1DEngine
from cainvert.eval import (
    check_ground_truth, eca_forward, evaluate_rollouts, is_solution, layer_passes, null_control, pass_at_k,
    residual, solve_exact, solve_records, strata,
)
from cainvert.mining.exact_miner import main as mine_main
from cainvert.storage.exact_catalog import (
    available_splits, encode_input, hints_with_k_revealed, load_records, records_to_arrays,
)


@pytest.fixture(scope="module")
def small_catalog(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cat"))
    for d in (1, 2):
        mine_main(["--width", "12", "--depth", str(d), "--quota", "8", "--workers", "2", "--bins", "1-3,4-15",
                   "--time-limit-min", "0.5", "--max-tries-per-task", "300", "--out-dir", out])
    return out


def test_eca_forward_matches_engine():
    eng = ECA1DEngine(110)
    rng = np.random.RandomState(0)
    x = rng.randint(0, 2, (50, 20)).astype(np.uint8)
    one = eca_forward(x, 1)
    assert all(np.array_equal(one[i], eng.forward_step(x[i])) for i in range(50))
    steps = rng.randint(0, 4, 50)
    multi = eca_forward(x, steps)
    for i in range(50):
        y = x[i].copy()
        for _ in range(steps[i]):
            y = eng.forward_step(y)
        assert np.array_equal(multi[i], y)


def test_residual_and_is_solution():
    eng = ECA1DEngine(110)
    rng = np.random.RandomState(1)
    W = 16
    x0 = rng.randint(0, 2, (8, W)).astype(np.uint8)
    s_d = eca_forward(x0, 2)
    hints = np.full((8, W), -1, dtype=np.int8)
    hints[:, 0] = x0[:, 0]
    assert residual(x0, s_d, hints, 2).tolist() == [0] * 8
    wrong = x0.copy()
    wrong[:, 0] ^= 1  # violates the revealed hint
    r = residual(wrong, s_d, hints, 2)
    assert (r >= 1).all()
    assert is_solution(np.stack([x0, wrong], 1), s_d, hints, 2).tolist() == [[True, False]] * 8
    assert not is_solution(x0, s_d, hints, 3).all()  # wrong depth


def test_pass_at_k_values():
    assert pass_at_k(np.array([4]), 4, 1)[0] == 1.0
    assert pass_at_k(np.array([0]), 4, 2)[0] == 0.0
    assert pass_at_k(np.array([2]), 4, 1)[0] == pytest.approx(0.5)
    assert pass_at_k(np.array([2]), 4, 2)[0] == pytest.approx(1 - 1 / 6)  # C(2,2)/C(4,2)
    assert pass_at_k(np.array([1]), 4, 4)[0] == 1.0


def test_null_control_flips_same_hamming():
    rng = np.random.default_rng(0)
    det = np.zeros((3, 10), dtype=np.uint8)
    roll = np.zeros((3, 5, 10), dtype=np.uint8)
    roll[:, :, :2] = 1  # every rollout differs from det in 2 bits
    tgt = np.zeros((3, 10), dtype=np.uint8)
    hits = null_control(det, roll, tgt, rng)
    assert hits.shape == (3, 5) and not hits.any()  # flipping 2 bits of det never recovers det
    roll[:] = 0
    assert null_control(det, roll, tgt, rng).all()  # zero flips -> det == target


def test_layer_passes():
    assert layer_passes(4, 2, 2) == 24 and layer_passes(1, 1, 1) == 2


def test_strata_keys():
    meta = dict(depth=np.array([1] * 40 + [2] * 40), leaf_bin=np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20),
                num_hints=np.array([0] * 20 + [2] * 20 + [0] * 20 + [3] * 20), search_work=np.arange(80))
    s = strata(meta, min_n=10)
    assert "all" in s and "d1" in s and "d2|bin1" in s and "d1|h0" in s and "d2|bin1|h3" in s and "d1|work0" in s
    assert s["d2|bin1|h3"].sum() == 20


def test_catalog_ground_truth_and_reference_solver(small_catalog):
    recs = [r for s in available_splits(small_catalog) for r in load_records(small_catalog, s)]
    assert len(recs) >= 8 and {r["depth"] for r in recs} == {1, 2}
    assert check_ground_truth(recs).all()
    ref = solve_records(recs)
    assert ref["solved"].all() and ref["matches_gt"].all()
    assert (ref["search_work"] == ref["catalog_search_work"]).all()
    a = records_to_arrays(recs)
    assert a["s_d"].shape == (len(recs), 12) and encode_input(a["s_d"], a["hints"]).shape == (len(recs), 25)
    # revealing fewer hints than num_hints makes the task ambiguous for at least one tree
    amb = [r for r in recs if r["num_hints"] > 0]
    assert amb
    r = amb[0]
    h = hints_with_k_revealed(r, 0)
    res = solve_exact(np.frombuffer(r["target_state"], np.uint8), h, r["depth"])
    assert res["n_solutions"] == r["leaf_count"] >= 2


def test_evaluate_rollouts_protocol(small_catalog):
    recs = load_records(small_catalog, "train", depths=[2])
    a = records_to_arrays(recs)
    N, W = a["x0"].shape
    rng = np.random.default_rng(0)
    # rollouts: the truth in half of the rollouts, random garbage otherwise
    roll = np.where(rng.random((N, 8, 1)) < 0.5, a["x0"][:, None, :], rng.integers(0, 2, (N, 8, W), dtype=np.uint8))
    out = evaluate_rollouts(recs, roll, det=a["x0"], k_list=(1, 8), min_n=1)
    assert out["all"]["det"] == 1.0 and out["all"]["verifier_false_pos"] == 0.0
    assert 0.0 < out["all"]["solve@1"] < out["all"]["solve@8"] <= 1.0
    assert "d2" in out and out["d2"]["n"] == N


def test_loader_split_errors(small_catalog):
    with pytest.raises(ValueError):
        load_records(small_catalog, "nope")
    per_cell = load_records(small_catalog, "train", per_cell=1)
    cells = {(r["depth"], r["leaf_bin"]) for r in per_cell}
    assert len(per_cell) == len(cells)
