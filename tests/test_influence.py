"""Integration test for influence acquisition (requires tree-influence)."""

import numpy as np
import pytest

from influence_al.acquisition.influence import TempModelInfluenceAcquisition
from influence_al.acquisition.base import AcquisitionContext
from influence_al.data.pool import ActiveLearningPool
from influence_al.models.trainer import LGBMTrainer

pytest.importorskip("tree_influence")


def test_influence_scorer_returns_scores():
    X = np.random.randn(80, 4)
    y = np.random.randint(0, 2, 80)
    pool = ActiveLearningPool.from_pool(X, y, 0.2, 0.25, seed=0)
    trainer = LGBMTrainer("classification", lgbm_params={"n_estimators": 30}, seed=0)
    X_tr, y_tr = pool.get_l_train()
    model = trainer.fit(X_tr, y_tr).model
    ctx = AcquisitionContext(
        pool=pool,
        trainer=trainer,
        model=model,
        train_result=None,
        config={"seed": 0},
        round_idx=0,
    )
    acq = TempModelInfluenceAcquisition(explainer_name="boostin", reference="r_ref")
    indices, scores = acq.score(ctx)
    assert len(indices) == len(pool.unlabeled_indices)
    assert len(scores) == len(indices)
    assert np.all(np.isfinite(scores))
