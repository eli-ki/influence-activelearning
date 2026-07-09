"""Active learning pool with fixed R_ref reference set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_pool_fractions(
    initial_labeled_fraction: float,
    r_ref_fraction: float,
) -> None:
    if not 0.0 < initial_labeled_fraction < 1.0:
        raise ValueError(
            f"initial_labeled_fraction must be in (0, 1), got {initial_labeled_fraction}"
        )
    if not 0.0 < r_ref_fraction < 1.0:
        raise ValueError(f"r_ref_fraction must be in (0, 1), got {r_ref_fraction}")
    if initial_labeled_fraction + r_ref_fraction >= 1.0:
        raise ValueError(
            "initial_labeled_fraction + r_ref_fraction must be < 1.0 "
            f"(got {initial_labeled_fraction} + {r_ref_fraction})"
        )


@dataclass
class ActiveLearningPool:
    """
    Manages labeled L_t, unlabeled U_t, fixed reference R_ref, and train split.

    Initialization (disjoint three-way split of the pool P):
      - L0: initial labeled seed (trainable; grows as batches are queried).
      - R_ref: fixed reference set, sampled separately from P \\ L0.
      - U0: remaining unlabeled pool P \\ (L0 ∪ R_ref).

    L0, R_ref, and U0 are pairwise disjoint. R_ref is never trained on or queried.
    L_t_train = L_t (all labeled indices; R_ref is not part of L_t).
    """

    X_pool: np.ndarray
    y_pool: np.ndarray  # ground truth (oracle); hidden for U_t until queried
    r_ref_indices: np.ndarray  # fixed reference subset of P, disjoint from L_t
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
        _validate_pool_fractions(initial_labeled_fraction, r_ref_fraction)

        rng = np.random.default_rng(seed)
        n = len(X_pool)
        if n < 3:
            raise ValueError(f"Pool must have at least 3 samples to split L0/R_ref/U0, got {n}")

        perm = rng.permutation(n)

        n_l0 = max(2, int(n * initial_labeled_fraction))
        n_r_ref = max(1, int(n * r_ref_fraction))
        if n_l0 + n_r_ref >= n:
            raise ValueError(
                f"Pool too small for requested fractions: n={n}, "
                f"|L0|={n_l0}, |R_ref|={n_r_ref} (need at least one unlabeled point)"
            )

        l0_indices = perm[:n_l0]
        r_ref_indices = perm[n_l0 : n_l0 + n_r_ref]
        u0_indices = perm[n_l0 + n_r_ref :]

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
        """L_t_train: all queried labels (R_ref is disjoint and never included)."""
        return self.X_pool[self.labeled_indices], self.y_pool[self.labeled_indices]

    def get_l_train_indices(self) -> np.ndarray:
        return self.labeled_indices.copy()

    def query(self, indices: np.ndarray) -> None:
        """Move indices from U_t to L_t (oracle provides true labels from y_pool)."""
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return
        if np.any(np.isin(indices, self.r_ref_indices)):
            raise ValueError("Cannot query R_ref indices")
        in_u = np.isin(indices, self.unlabeled_indices)
        if not np.all(in_u):
            bad = indices[~in_u]
            raise ValueError(f"Indices not in unlabeled pool: {bad}")
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
