"""Oracle add-one-retrain calibration (evaluation only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from influence_al.data.pool import ActiveLearningPool
from influence_al.models.trainer import LGBMTrainer


@dataclass
class OracleResult:
    spearman_rho: float
    p_value: float
    n_evaluated: int
    proxy_scores: np.ndarray
    true_deltas: np.ndarray


class OracleCalibrator:
    """
    Compare proxy acquisition scores to true add-one-retrain benefit on R_ref.

    true_delta(u) = metric(R_ref | train(L_t_train ∪ {u})) - metric(R_ref | M_t)
    """

    def __init__(self, subset_size: int = 50, seed: int = 42):
        self.subset_size = subset_size
        self.seed = seed

    def calibrate(
        self,
        pool: ActiveLearningPool,
        trainer: LGBMTrainer,
        model,
        proxy_indices: np.ndarray,
        proxy_scores: np.ndarray,
        round_idx: int = 0,
    ) -> Optional[OracleResult]:
        if len(proxy_indices) == 0:
            return None

        rng = np.random.default_rng(self.seed + round_idx)
        n_eval = min(self.subset_size, len(proxy_indices))
        eval_pos = rng.choice(len(proxy_indices), n_eval, replace=False)
        eval_indices = proxy_indices[eval_pos]
        eval_proxy = proxy_scores[eval_pos]

        X_l, y_l = pool.get_l_train()
        X_ref, y_ref = pool.get_r_ref()
        baseline = trainer.score_metric(model, X_ref, y_ref)

        true_deltas = np.zeros(n_eval, dtype=np.float64)
        for i, idx in enumerate(eval_indices):
            x_u = pool.X_pool[idx : idx + 1]
            y_u = pool.y_pool[idx : idx + 1]
            X_aug = np.vstack([X_l, x_u])
            y_aug = np.concatenate([y_l, y_u])
            model_aug = trainer.fit(X_aug, y_aug).model
            after = trainer.score_metric(model_aug, X_ref, y_ref)
            true_deltas[i] = after - baseline

        if np.std(eval_proxy) < 1e-12 or np.std(true_deltas) < 1e-12:
            rho, pval = 0.0, 1.0
        else:
            rho, pval = spearmanr(eval_proxy, true_deltas)

        return OracleResult(
            spearman_rho=float(rho),
            p_value=float(pval),
            n_evaluated=n_eval,
            proxy_scores=eval_proxy,
            true_deltas=true_deltas,
        )
