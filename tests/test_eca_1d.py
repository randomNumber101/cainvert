"""Unit tests for 1D ECA De Bruijn Engine."""

import numpy as np
import pytest
from cainvert.engines.eca_1d import ECA1DEngine


def test_eca_1d_forward_and_backward():
    engine = ECA1DEngine(rule_number=110)
    rng = np.random.RandomState(42)

    # 1. Random 1D state
    s0 = rng.randint(0, 2, size=16, dtype=np.uint8)

    # 2. Forward step
    s1 = engine.forward_step(s0)
    assert s1.shape == (16,)

    # 3. Solve preimages
    res = engine.get_preimages(s1)
    assert res.count >= 1
    assert len(res.preimages) >= 1

    # 4. Verify preimages map to target
    for pre in res.preimages:
        fwd = engine.forward_step(pre)
        assert np.array_equal(fwd, s1), "1D Preimage did not reproduce target!"

    # 5. Verify s0 presence
    assert any(np.array_equal(pre, s0) for pre in res.preimages)
