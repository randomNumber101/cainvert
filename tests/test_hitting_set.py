"""Unit tests for Greedy Minimal Hitting Set solver."""

import numpy as np
import pytest
from cainvert.core.hitting_set import solve_greedy_hitting_set, verify_disambiguation


def test_hitting_set_disambiguation_1d():
    rng = np.random.RandomState(42)
    N = 32
    gt = rng.randint(0, 2, size=N, dtype=np.uint8)

    # Generate 5 distractors, each differing in at least one bit
    distractors = []
    for _ in range(5):
        dis = gt.copy()
        flip_idx = rng.choice(N, size=2, replace=False)
        dis[flip_idx] ^= 1
        distractors.append(dis)
    distractors = np.array(distractors, dtype=np.uint8)

    indices, hint_vec, density = solve_greedy_hitting_set(gt, distractors)
    assert len(indices) > 0
    assert verify_disambiguation(hint_vec, gt, distractors)


def test_hitting_set_disambiguation_2d():
    rng = np.random.RandomState(42)
    H, W = 6, 6
    gt = rng.randint(0, 2, size=(H, W), dtype=np.uint8)

    distractors = []
    for _ in range(8):
        dis = gt.copy()
        dis[rng.randint(0, H), rng.randint(0, W)] ^= 1
        distractors.append(dis)
    distractors = np.array(distractors, dtype=np.uint8)

    indices, hint_vec, density = solve_greedy_hitting_set(gt, distractors)
    assert len(indices) > 0
    assert verify_disambiguation(hint_vec, gt, distractors)
