"""Active learning pool with fixed R_ref reference set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ActiveLearningPool:
    """
    Manages labeled L_t, unlabeled U_t, fixed reference R_ref, and train split.

    Initialization:
      - L0 is split into L0_train_seed (for initial labeled content) and R_ref (fixed forever).
      - R_ref indices are fixed at init and never queried for labeling.
      - L_t_train = L_t \\ R_ref (grows as batches are labeled).
    """

    X_pool: np.ndarray
    y_pool: np.ndarray  # ground truth (oracle); hidden for U_t until queried
    r_ref_indices: np.ndarray  # fixed subset of initial L0
    labeled_indices: np.ndarray
    unlabeled_indices: np.ndarray

    @classmethod
    def from_pool(
        cls,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        initial_labeled_fraction: float,
        r_ref_fraction: float,
        seed: int = 42,
    ) -> "ActiveLearningPool":
        rng = np.random.default_rng(seed)
        n = len(X_pool)
        perm = rng.permutation(n)

        n_l0 = max(2, int(n * initial_labeled_fraction))
        l0_indices = perm[:n_l0]
        u0_indices = perm[n_l0:]

        n_r_ref = max(1, int(len(l0_indices) * r_ref_fraction))
        r_ref_indices = l0_indices[:n_r_ref]

        return cls(
            X_pool=X_pool,
            y_pool=y_pool,
            r_ref_indices=np.sort(r_ref_indices),
            labeled_indices=np.sort(l0_indices),
            unlabeled_indices=np.sort(u0_indices),
        )

    @property
    def n_pool(self) -> int:
        return len(self.X_pool)

    def get_labeled(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X_pool[self.labeled_indices], self.y_pool[self.labeled_indices]

    def get_unlabeled(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X_pool[self.unlabeled_indices], self.y_pool[self.unlabeled_indices]

    def get_r_ref(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X_pool[self.r_ref_indices], self.y_pool[self.r_ref_indices]

    def get_l_train(self) -> tuple[np.ndarray, np.ndarray]:
        """L_t_train = L_t \\ R_ref (true labels only)."""
        train_mask = ~np.isin(self.labeled_indices, self.r_ref_indices)
        train_idx = self.labeled_indices[train_mask]
        return self.X_pool[train_idx], self.y_pool[train_idx]

    def get_l_train_indices(self) -> np.ndarray:
        train_mask = ~np.isin(self.labeled_indices, self.r_ref_indices)
        return np.sort(self.labeled_indices[train_mask])

    def query(self, indices: np.ndarray) -> None:
        """Move indices from U_t to L_t (oracle provides true labels from y_pool)."""
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return
        in_u = np.isin(indices, self.unlabeled_indices)
        if not np.all(in_u):
            bad = indices[~in_u]
            raise ValueError(f"Indices not in unlabeled pool: {bad}")
        if np.any(np.isin(indices, self.r_ref_indices)):
            raise ValueError("Cannot query R_ref indices")
        self.labeled_indices = np.sort(np.concatenate([self.labeled_indices, indices]))
        self.unlabeled_indices = np.sort(
            self.unlabeled_indices[~np.isin(self.unlabeled_indices, indices)]
        )

    def true_labels(self, indices: np.ndarray) -> np.ndarray:
        return self.y_pool[indices]

    def state_dict(self) -> dict:
        return {
            "labeled_indices": self.labeled_indices.copy(),
            "unlabeled_indices": self.unlabeled_indices.copy(),
        }

    def load_state(self, state: dict) -> None:
        self.labeled_indices = state["labeled_indices"].copy()
        self.unlabeled_indices = state["unlabeled_indices"].copy()
