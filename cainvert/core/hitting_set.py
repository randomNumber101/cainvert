"""Greedy Minimal Hitting Set solver for hint vector generation."""

from typing import List, Tuple
import numpy as np


def solve_greedy_hitting_set(
    ground_truth: np.ndarray,
    distractors: np.ndarray,
) -> Tuple[List[int], np.ndarray, float]:
    """Find a minimal set of bit positions to disambiguate ground truth from all distractors.

    Formulation:
      Given ground truth S_0* and M-1 distractors S_0^(m), compute difference matrix:
        D[m, i] = (S_0*[i] != S_0^(m)[i])
      A hitting set H is an index subset such that for every m, there is at least one
      i in H with D[m, i] == 1.
      The greedy algorithm repeatedly chooses the bit position covering the maximum
      number of currently uncovered distractors.

    Args:
        ground_truth: Binary array of shape (N,) or (H, W).
        distractors: Binary array of shape (M-1, N) or (M-1, H, W).

    Returns:
        selected_indices: List of revealed bit positions (1D flattened indices, sorted).
        hint_vector: Array of shape matching ground_truth with values:
                     -1 for masked positions, ground_truth values at revealed positions.
        hint_density: Proportion of revealed bits (len(selected_indices) / total_cells).
    """
    orig_shape = ground_truth.shape
    gt_flat = np.asarray(ground_truth, dtype=np.uint8).flatten()
    total_cells = gt_flat.shape[0]

    if len(distractors) == 0 or distractors.shape[0] == 0:
        hint_vector = np.full(total_cells, -1, dtype=np.int8).reshape(orig_shape)
        return [], hint_vector, 0.0

    if distractors.ndim > 2 or (distractors.ndim == 2 and orig_shape != (total_cells,)):
        dis_flat = np.asarray(distractors, dtype=np.uint8).reshape(distractors.shape[0], -1)
    else:
        dis_flat = np.asarray(distractors, dtype=np.uint8)

    num_distractors = dis_flat.shape[0]

    # Difference matrix: True where distractor differs from ground truth
    D = (dis_flat != gt_flat[np.newaxis, :])  # Shape: (M-1, total_cells)

    # Sanity check: Ensure all distractors differ from ground truth in at least one bit
    has_diff = np.any(D, axis=1)
    if not np.all(has_diff):
        num_identical = int(np.sum(~has_diff))
        raise ValueError(
            f"Found {num_identical} distractor(s) identical to ground truth! "
            f"Ground truth cannot be uniquely disambiguated."
        )

    uncovered = np.ones(num_distractors, dtype=bool)
    selected_indices: List[int] = []

    while np.any(uncovered):
        scores = np.sum(D[uncovered, :], axis=0)
        best_col = int(np.argmax(scores))

        if scores[best_col] == 0:
            break

        selected_indices.append(best_col)
        uncovered[D[:, best_col]] = False

    selected_indices.sort()

    hint_flat = np.full(total_cells, -1, dtype=np.int8)
    for idx in selected_indices:
        hint_flat[idx] = gt_flat[idx]

    hint_density = float(len(selected_indices) / total_cells)
    return selected_indices, hint_flat.reshape(orig_shape), hint_density


def verify_disambiguation(
    hint_vector: np.ndarray,
    ground_truth: np.ndarray,
    distractors: np.ndarray,
) -> bool:
    """Verify that the hint vector matches ground truth and eliminates ALL distractors."""
    hint_flat = np.asarray(hint_vector, dtype=np.int8).flatten()
    gt_flat = np.asarray(ground_truth, dtype=np.uint8).flatten()
    if len(distractors) == 0 or distractors.shape[0] == 0:
        return True

    if distractors.ndim > 2 or (distractors.ndim == 2 and ground_truth.ndim > 1):
        dis_flat = np.asarray(distractors, dtype=np.uint8).reshape(distractors.shape[0], -1)
    else:
        dis_flat = np.asarray(distractors, dtype=np.uint8)

    revealed_mask = (hint_flat >= 0)
    revealed_indices = np.where(revealed_mask)[0]

    # Check ground truth match
    if not np.array_equal(hint_flat[revealed_mask], gt_flat[revealed_mask]):
        return False

    if dis_flat.shape[0] == 0:
        return True

    # For every distractor, at least one revealed bit must mismatch
    diffs = (dis_flat[:, revealed_indices] != gt_flat[revealed_indices])
    return bool(np.all(np.any(diffs, axis=-1)))
