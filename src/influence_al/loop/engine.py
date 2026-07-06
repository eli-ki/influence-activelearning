"""Active learning loop engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from influence_al.acquisition.base import AcquisitionContext, AcquisitionFunction
from influence_al.acquisition.diversity import BatchSelector, BatchSelectorConfig
from influence_al.data.pool import ActiveLearningPool
from influence_al.evaluation.metrics import LearningCurveResult
from influence_al.evaluation.oracle import OracleCalibrator
from influence_al.models.trainer import LGBMTrainer


@dataclass
class RunResult:
    method: str
    dataset: str
    seed: int
    learning_curve: LearningCurveResult
    round_logs: List[dict] = field(default_factory=list)


def build_acquisition(method: str, config: dict) -> AcquisitionFunction:
    from influence_al.acquisition.baselines import (
        BadgeAdaptedAcquisition,
        FeatureKMeansAcquisition,
        LossAcquisition,
        MarginAcquisition,
        RandomAcquisition,
        UncertaintyAcquisition,
    )
    from influence_al.acquisition.influence import TempModelInfluenceAcquisition
    from influence_al.acquisition.shapley import ShapleyPrefilterAcquisition

    inf_cfg = config.get("influence", {})
    if method == "influence":
        return TempModelInfluenceAcquisition(
            explainer_name=inf_cfg.get("explainer", "boostin"),
            aggregation=inf_cfg.get("aggregation", "sum"),
            pseudo_label_mode=inf_cfg.get("pseudo_label_mode", "argmax"),
            top_k_classes=inf_cfg.get("top_k_classes", 3),
            reference=inf_cfg.get("reference", "r_ref"),
            r_pseudo_size=inf_cfg.get("r_pseudo_size", 200),
            self_influence_filter=config.get("diversity", {}).get("self_influence_filter", False),
            self_influence_quantile=config.get("diversity", {}).get("self_influence_quantile", 0.9),
        )
    if method == "influence_shapley":
        sh_cfg = config.get("shapley", {})
        return ShapleyPrefilterAcquisition(
            prefilter_fraction=sh_cfg.get("prefilter_fraction", 0.2),
            knn_neighbors=sh_cfg.get("knn_neighbors", 5),
            top_k_classes=sh_cfg.get("top_k_classes", 10),
            influence_kwargs={
                "explainer_name": inf_cfg.get("explainer", "boostin"),
                "aggregation": inf_cfg.get("aggregation", "sum"),
                "pseudo_label_mode": inf_cfg.get("pseudo_label_mode", "argmax"),
                "reference": inf_cfg.get("reference", "r_ref"),
            },
        )
    builders = {
        "random": RandomAcquisition,
        "uncertainty": UncertaintyAcquisition,
        "margin": MarginAcquisition,
        "loss": LossAcquisition,
        "feature_kmeans": FeatureKMeansAcquisition,
        "badge": BadgeAdaptedAcquisition,
    }
    if method not in builders:
        raise ValueError(f"Unknown method: {method}")
    return builders[method]()


class ActiveLearningEngine:
    """Runs the full AL protocol from the v2 plan."""

    def __init__(
        self,
        pool: ActiveLearningPool,
        trainer: LGBMTrainer,
        acquisition: AcquisitionFunction,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: dict,
        method_name: str = "influence",
        dataset_name: str = "unknown",
        seed: int = 42,
    ):
        self.pool = pool
        self.trainer = trainer
        self.acquisition = acquisition
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.seed = seed

        div_cfg = config.get("diversity", {})
        self.batch_selector = BatchSelector(
            BatchSelectorConfig(
                prefilter_multiplier=div_cfg.get("prefilter_multiplier", 5),
                embedding=div_cfg.get("embedding", "features"),
                discard_negative=div_cfg.get("discard_negative", False),
            )
        )
        oracle_cfg = config.get("oracle", {})
        self.oracle = OracleCalibrator(
            subset_size=oracle_cfg.get("subset_size", 50),
            seed=seed,
        ) if oracle_cfg.get("enabled", True) else None

        self.batch_size = config.get("batch_size", 50)
        self.n_rounds = config.get("n_rounds", 10)
        self.use_batch_selector = method_name not in ("random",)

    def _eval_model(self):
        X_l, y_l = self.pool.get_l_train()
        return self.trainer.fit(X_l, y_l)

    def run(self) -> RunResult:
        n_labels_history: List[int] = []
        test_metrics: List[float] = []
        beats_random: List[bool] = []
        oracle_rhos: List[float] = []
        round_logs: List[dict] = []

        train_result = self._eval_model()
        model = train_result.model
        n_labels_history.append(len(self.pool.labeled_indices))
        test_metrics.append(self.trainer.score_metric(model, self.X_test, self.y_test))

        for round_idx in range(self.n_rounds):
            if len(self.pool.unlabeled_indices) == 0:
                break

            ctx = AcquisitionContext(
                pool=self.pool,
                trainer=self.trainer,
                model=model,
                train_result=train_result,
                config={**self.config, "seed": self.seed},
                round_idx=round_idx,
            )

            bs = min(self.batch_size, len(self.pool.unlabeled_indices))
            selector = self.batch_selector if self.use_batch_selector else None
            batch = self.acquisition.select(ctx, bs, batch_selector=selector)

            log: Dict[str, Any] = {
                "round": round_idx,
                "batch_size": len(batch),
                "n_labeled": len(self.pool.labeled_indices),
                "n_unlabeled": len(self.pool.unlabeled_indices),
            }

            if self.oracle is not None and hasattr(self.acquisition, "score"):
                indices, scores = self.acquisition.score(ctx)
                oracle_result = self.oracle.calibrate(
                    self.pool, self.trainer, model, indices, scores, round_idx
                )
                if oracle_result is not None:
                    log["oracle_spearman"] = oracle_result.spearman_rho
                    oracle_rhos.append(oracle_result.spearman_rho)

            if hasattr(self.acquisition, "last_diagnostics"):
                log["diagnostics"] = self.acquisition.last_diagnostics

            self.pool.query(batch)

            train_result = self._eval_model()
            model = train_result.model
            n_labels_history.append(len(self.pool.labeled_indices))
            test_metrics.append(self.trainer.score_metric(model, self.X_test, self.y_test))
            round_logs.append(log)

        curve = LearningCurveResult(
            method=self.method_name,
            dataset=self.dataset_name,
            seed=self.seed,
            n_labels=n_labels_history,
            test_metrics=test_metrics,
            beats_random_rounds=beats_random,
            oracle_spearman=oracle_rhos,
        )
        return RunResult(
            method=self.method_name,
            dataset=self.dataset_name,
            seed=self.seed,
            learning_curve=curve,
            round_logs=round_logs,
        )

    def run_with_random_baseline_comparison(self) -> RunResult:
        """Run main method and track per-round comparison vs one random batch (optional)."""
        return self.run()
