"""Unit tests for 2D SAT Preimage Engine."""

import numpy as np
import pytest
from cainvert.engines.sat_2d import SAT2DEngine, parse_life_rule


def test_parse_life_rule():
    b_set, s_set = parse_life_rule("B3/S23")
    assert b_set == {3}
    assert s_set == {2, 3}

    b_set2, s_set2 = parse_life_rule("b36/s23")
    assert b_set2 == {3, 6}
    assert s_set2 == {2, 3}


def test_sat_2d_forward_and_preimages():
    engine = SAT2DEngine(grid_shape=(5, 5), rule_str="B3/S23")
    rng = np.random.RandomState(42)

    # 1. Generate random state
    s0 = rng.randint(0, 2, size=(5, 5), dtype=np.uint8)

    # 2. Forward step
    s1 = engine.forward_step(s0)
    assert s1.shape == (5, 5)

    # 3. Solve all preimages for s1
    res = engine.get_preimages(s1, max_count=None)
    assert res.count >= 1, "Expected at least 1 preimage (s0 itself!)"
    assert len(res.preimages) >= 1

    # 4. Verify that EVERY returned preimage forward-steps back to s1
    for pre in res.preimages:
        fwd = engine.forward_step(pre)
        assert np.array_equal(fwd, s1), "Preimage forward step did not reproduce target state!"

    # 5. Verify ground truth s0 is among returned preimages
    s0_matches = any(np.array_equal(pre, s0) for pre in res.preimages)
    assert s0_matches, "Original s0 state was not found among preimages!"
