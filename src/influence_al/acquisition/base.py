"""Acquisition function protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from influence_al.data.pool import ActiveLearningPool
from influence_al.models.trainer import LGBMTrainer, TrainResult


@dataclass
class AcquisitionContext:
    """Shared state passed to acquisition functions each round."""

    pool: ActiveLearningPool
    trainer: LGBMTrainer
    model: Any  # M_t trained on L_t_train true labels only
    train_result: TrainResult
    config: dict = field(default_factory=dict)
    round_idx: int = 0

    def X_u(self) -> np.ndarray:
        X, _ = self.pool.get_unlabeled()
        return X

    def idx_u(self) -> np.ndarray:
        return self.pool.unlabeled_indices.copy()

    def X_l_train(self) -> np.ndarray:
        X, _ = self.pool.get_l_train()
        return X

    def y_l_train(self) -> np.ndarray:
        _, y = self.pool.get_l_train()
        return y

    def X_r_ref(self) -> np.ndarray:
        X, _ = self.pool.get_r_ref()
        return X

    def y_r_ref(self) -> np.ndarray:
        _, y = self.pool.get_r_ref()
        return y


class AcquisitionFunction(ABC):
    """Score unlabeled candidates; higher = more valuable to label."""

    @abstractmethod
    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        indices : array of pool indices for U_t
        scores : array of scores aligned with indices (higher = select first)
        """
        ...

    def select(
        self,
        ctx: AcquisitionContext,
        batch_size: int,
        batch_selector: Optional[Any] = None,
    ) -> np.ndarray:
        indices, scores = self.score(ctx)
        if batch_selector is not None:
            return batch_selector.select(
                indices=indices,
                scores=scores,
                X=ctx.pool.X_pool[indices],
                model=ctx.model,
                trainer=ctx.trainer,
                batch_size=batch_size,
            )
        order = np.argsort(scores)[::-1]
        return indices[order[:batch_size]]
