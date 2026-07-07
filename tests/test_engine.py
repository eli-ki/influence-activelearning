"""Integration test for AL engine (small dataset, few rounds)."""

import numpy as np

from influence_al.acquisition.baselines import RandomAcquisition
from influence_al.data.pool import ActiveLearningPool
from influence_al.loop.engine import ActiveLearningEngine, build_acquisition
from influence_al.models.trainer import LGBMTrainer


def test_engine_runs_few_rounds():
    X = np.random.randn(120, 5)
    y = np.random.randint(0, 2, 120)
    pool = ActiveLearningPool.from_pool(X, y, 0.15, 0.2, seed=0)
    trainer = LGBMTrainer("classification", lgbm_params={"n_estimators": 20}, seed=0)
    config = {
        "batch_size": 5,
        "n_rounds": 3,
        "oracle": {"enabled": False},
        "diversity": {"prefilter_multiplier": 3},
    }
    engine = ActiveLearningEngine(
        pool=pool,
        trainer=trainer,
        acquisition=RandomAcquisition(),
        X_test=X[:20],
        y_test=y[:20],
        config=config,
        method_name="random",
        dataset_name="synthetic",
        seed=0,
    )
    result = engine.run()
    assert len(result.learning_curve.test_metrics) >= 2


def test_build_acquisition_methods():
    cfg = {"influence": {"explainer": "boostin", "reference": "r_ref"}}
    for method in ["random", "uncertainty", "margin", "loss", "badge", "influence", "influence_r_pseudo"]:
        acq = build_acquisition(method, cfg)
        assert acq is not None
