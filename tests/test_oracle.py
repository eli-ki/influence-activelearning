"""Tests for oracle calibration."""

import numpy as np

from influence_al.data.pool import ActiveLearningPool
from influence_al.evaluation.oracle import OracleCalibrator
from influence_al.models.trainer import LGBMTrainer


def test_oracle_calibrate_runs():
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    pool = ActiveLearningPool.from_pool(X, y, 0.2, 0.25, seed=0)
    trainer = LGBMTrainer("classification", lgbm_params={"n_estimators": 20}, seed=0)
    X_tr, y_tr = pool.get_l_train()
    model = trainer.fit(X_tr, y_tr).model
    indices = pool.unlabeled_indices[:20]
    proxy = np.random.randn(20)
    oracle = OracleCalibrator(subset_size=10, seed=0)
    result = oracle.calibrate(pool, trainer, model, indices, proxy, round_idx=0)
    assert result is not None
    assert result.n_evaluated == 10
    assert -1.0 <= result.spearman_rho <= 1.0
