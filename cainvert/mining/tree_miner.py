import hashlib
import time
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from ..core.types import TreeData
from ..core.filters import is_density_valid, is_trajectory_dynamic, compute_density
from ..core.hitting_set import solve_greedy_hitting_set, verify_disambiguation
from ..engines.base import BaseEngine


class TreeMiner:
    """Orchestrates tree unrolling, filtering, and disambiguation."""

    def __init__(
        self,
        engine: BaseEngine,
        leaf_cap: int = 1024,
        max_branch_per_node: int = 16,
        max_nodes_per_level: int = 256,
        min_density: float = 0.25,
        max_density: float = 0.75,
        min_step_activity: float = 0.05,
        dynamic_level_capacity: bool = True,
    ):
        self.engine = engine
        self.leaf_cap = leaf_cap
        self.max_branch_per_node = max_branch_per_node
        self.max_nodes_per_level = max_nodes_per_level
        self.min_density = min_density
        self.max_density = max_density
        self.min_step_activity = min_step_activity
        self.dynamic_level_capacity = dynamic_level_capacity

    def mine_single_tree(
        self,
        grid_shape: Tuple[int, ...],
        target_depth: int = 6,
        rng: Optional[np.random.RandomState] = None,
        tree_idx: int = 0,
        node_timeout_sec: Optional[float] = 1.0,
        existing_hashes: Optional[Set[str]] = None,
    ) -> Optional[TreeData]:
        """Mine a single verified preimage tree of given target_depth.

        Returns:
            TreeData if successfully mined and validated; None if rejected.
        """
        if rng is None:
            rng = np.random.RandomState()

        # 1. Sample random initial state
        x0 = rng.randint(0, 2, size=grid_shape, dtype=np.uint8)
        x0_hash = hashlib.sha256(x0.tobytes()).hexdigest()[:16]

        # Fast pre-flight deduplication check against historical and current catalog
        if existing_hashes is not None and x0_hash in existing_hashes:
            return None

        if not is_density_valid(x0, self.min_density, self.max_density):
            return None

        # 2. Roll forward deterministically
        trajectory = [x0]
        curr = x0
        for _ in range(target_depth):
            curr = self.engine.forward_step(curr)
            trajectory.append(curr)

        # 3. Filter for active, non-trivial dynamics
        is_dyn, _ = is_trajectory_dynamic(
            trajectory,
            min_density=self.min_density,
            max_density=self.max_density,
            min_step_activity=self.min_step_activity,
        )
        if not is_dyn:
            return None

        # 4. Backward BFS unrolling from root target X_d
        root = trajectory[-1]
        tree_levels: List[List[np.ndarray]] = [[root]]

        for step in range(1, target_depth + 1):
            cur_nodes = tree_levels[-1]
            gt_ancestor = trajectory[target_depth - step]
            gt_ancestor_flat = gt_ancestor.flatten()
            next_nodes: List[np.ndarray] = []

            # Dynamic geometric capacity for level k: ceil(L_max ^ (step / D))
            if self.dynamic_level_capacity:
                level_capacity = int(np.ceil(self.leaf_cap ** (step / float(target_depth))))
            else:
                level_capacity = self.leaf_cap

            prev_gt = trajectory[target_depth - step + 1]
            prev_gt_flat = prev_gt.flatten()

            gt_node_idx = None
            for idx, n in enumerate(cur_nodes):
                if np.array_equal(n.flatten(), prev_gt_flat):
                    gt_node_idx = idx
                    break

            # Order nodes so ground-truth ancestor is ALWAYS expanded first
            if gt_node_idx is not None:
                ordered_nodes = [cur_nodes[gt_node_idx]] + [cur_nodes[i] for i in range(len(cur_nodes)) if i != gt_node_idx]
            else:
                ordered_nodes = cur_nodes

            max_expand_nodes = min(len(ordered_nodes), self.max_nodes_per_level)
            if len(ordered_nodes) > max_expand_nodes:
                other_indices = list(range(1, len(ordered_nodes)))
                sampled = rng.choice(other_indices, size=max_expand_nodes - 1, replace=False).tolist()
                nodes_to_expand = [ordered_nodes[0]] + [ordered_nodes[i] for i in sampled]
            else:
                nodes_to_expand = ordered_nodes

            for node in nodes_to_expand:
                # Early stop if we have already accumulated enough candidates for this level
                if len(next_nodes) >= level_capacity:
                    break

                res = self.engine.get_preimages(
                    node,
                    max_count=self.max_branch_per_node,
                    timeout_seconds=node_timeout_sec,
                )
                if res.count == 0 or len(res.preimages) == 0:
                    # Garden of Eden side-branch: dead-ends naturally, main branch continues
                    continue
                next_nodes.extend(res.preimages)

            # Guarantee that gt_ancestor is strictly included in next_nodes
            gt_present = any(np.array_equal(n.flatten(), gt_ancestor_flat) for n in next_nodes)
            if not gt_present:
                next_nodes.append(gt_ancestor)

            # Deduplicate next_nodes
            raw_nodes = np.array(next_nodes, dtype=np.uint8)
            nodes_flat = raw_nodes.reshape(raw_nodes.shape[0], -1)
            _, unique_indices = np.unique(nodes_flat, axis=0, return_index=True)
            unique_nodes = [next_nodes[i] for i in unique_indices]

            # If unique_nodes exceeds level_capacity, subsample while preserving gt_ancestor
            if len(unique_nodes) > level_capacity:
                gt_idx = None
                for idx, n in enumerate(unique_nodes):
                    if np.array_equal(n.flatten(), gt_ancestor_flat):
                        gt_idx = idx
                        break
                other_indices = [i for i in range(len(unique_nodes)) if i != gt_idx]
                sampled_indices = rng.choice(other_indices, size=level_capacity - 1, replace=False).tolist()
                unique_nodes = [unique_nodes[gt_idx]] + [unique_nodes[i] for i in sampled_indices]

            tree_levels.append(unique_nodes)

        final_depth = len(tree_levels) - 1
        if final_depth < target_depth:
            return None

        # 5. Extract unique leaves and verify ground truth ancestor
        raw_leaves = np.array(tree_levels[-1], dtype=np.uint8)
        leaves_flat = raw_leaves.reshape(raw_leaves.shape[0], -1)
        unique_flat, unique_indices = np.unique(leaves_flat, axis=0, return_index=True)
        leaves = raw_leaves[unique_indices]

        target_gt = trajectory[target_depth - final_depth]
        gt_flat = target_gt.flatten()

        matches = np.where(np.all(unique_flat == gt_flat[np.newaxis, :], axis=1))[0]
        if len(matches) != 1:
            return None

        gt_leaf_idx = int(matches[0])
        distractors = np.delete(leaves, gt_leaf_idx, axis=0)

        # 6. Disambiguation via Minimal Hitting Set
        hint_indices, hint_vec, hint_density = solve_greedy_hitting_set(target_gt, distractors)
        if not verify_disambiguation(hint_vec, target_gt, distractors):
            return None

        leaf_count = len(leaves)
        effective_branching = float(leaf_count ** (1.0 / max(1, final_depth)))
        mean_activity = float(np.mean([
            np.mean(trajectory[i] != trajectory[i + 1])
            for i in range(len(trajectory) - 1)
        ]))

        tree_id = f"{self.engine.dimension}_{self.engine.rule_name}_d{final_depth}_L{leaf_count}_{tree_idx:06d}"
        tree_uid = f"{self.engine.dimension}_{self.engine.rule_name}_d{final_depth}_L{leaf_count}_{x0_hash}"

        return TreeData(
            tree_id=tree_id,
            tree_uid=tree_uid,
            x0_hash=x0_hash,
            dimension=self.engine.dimension,
            rule_name=self.engine.rule_name,
            grid_shape=grid_shape,
            depth=final_depth,
            leaf_count=leaf_count,
            effective_branching=effective_branching,
            root_target=root,
            ground_truth_s0=target_gt,
            ground_truth_leaf_idx=gt_leaf_idx,
            leaves_s0=leaves,
            trajectory=trajectory[:final_depth + 1],
            hint_indices=hint_indices,
            hints=hint_vec,
            hint_density=hint_density,
            mean_activity=mean_activity,
            metadata={
                "attempted_depth": target_depth,
                "num_hints": len(hint_indices),
            }
        )
