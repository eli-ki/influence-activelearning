"""Tests for baseline acquisition."""

import numpy as np

from influence_al.acquisition.baselines import RandomAcquisition, UncertaintyAcquisition
from influence_al.acquisition.base import AcquisitionContext
from influence_al.data.pool import ActiveLearningPool
from influence_al.models.trainer import LGBMTrainer


def test_random_scores_all_unlabeled():
    X = np.random.randn(80, 4)
    y = np.random.randint(0, 2, 80)
    pool = ActiveLearningPool.from_pool(X, y, 0.15, 0.2, seed=0)
    trainer = LGBMTrainer("classification", seed=0)
    X_tr, y_tr = pool.get_l_train()
    model = trainer.fit(X_tr, y_tr).model
    ctx = AcquisitionContext(pool=pool, trainer=trainer, model=model, train_result=None, config={"seed": 0})
    acq = RandomAcquisition()
    indices, scores = acq.score(ctx)
    assert len(indices) == len(pool.unlabeled_indices)
    assert len(scores) == len(indices)


def test_uncertainty_scores_finite():
    X = np.random.randn(60, 3)
    y = np.random.randint(0, 2, 60)
    pool = ActiveLearningPool.from_pool(X, y, 0.2, 0.2, seed=1)
    trainer = LGBMTrainer("classification", seed=1)
    X_tr, y_tr = pool.get_l_train()
    model = trainer.fit(X_tr, y_tr).model
    ctx = AcquisitionContext(pool=pool, trainer=trainer, model=model, train_result=None)
    indices, scores = UncertaintyAcquisition().score(ctx)
    assert np.all(np.isfinite(scores))
