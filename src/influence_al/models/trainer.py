"""LightGBM training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

TaskType = Literal["classification", "regression"]


@dataclass
class TrainResult:
    model: Any
    task: TaskType


class LGBMTrainer:
    """Train/evaluate LightGBM with shared hyperparameters across all AL methods."""

    def __init__(self, task: TaskType, lgbm_params: Optional[dict] = None, seed: int = 42):
        self.task = task
        self.lgbm_params = lgbm_params or {}
        self.seed = seed

    def _base_params(self) -> dict:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": self.seed,
            "verbose": -1,
        }
        params.update(self.lgbm_params)
        return params

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainResult:
        if self.task == "classification":
            n_classes = len(np.unique(y))
            model = LGBMClassifier(**self._base_params())
            model.fit(X, y)
        else:
            model = LGBMRegressor(**self._base_params())
            model.fit(X, y)
        return TrainResult(model=model, task=self.task)

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return model.predict_proba(X)
        preds = model.predict(X)
        return preds.reshape(-1, 1)

    def pseudo_labels(
        self,
        model: Any,
        X: np.ndarray,
        mode: str = "argmax",
        top_k: int = 3,
    ) -> np.ndarray:
        if self.task == "regression":
            return self.predict(model, X)
        proba = self.predict_proba(model, X)
        if mode == "argmax":
            return np.argmax(proba, axis=1)
        if mode == "top_k_max":
            return np.argmax(proba, axis=1)
        if mode == "expected":
            classes = np.arange(proba.shape[1])
            return np.round(np.sum(proba * classes, axis=1)).astype(np.int64)
        raise ValueError(f"Unknown pseudo_label_mode: {mode}")

    def score_metric(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(model, X)
        if self.task == "classification":
            return float(accuracy_score(y, preds))
        return float(r2_score(y, preds))

    def loss_per_sample(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            proba = self.predict_proba(model, X)
            n = len(y)
            losses = np.zeros(n, dtype=np.float64)
            for i in range(n):
                p = proba[i, int(y[i])]
                losses[i] = -np.log(max(p, 1e-12))
            return losses
        preds = self.predict(model, X)
        return (preds - y) ** 2

    def leaf_index_embedding(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Leaf indices per tree as a flat embedding (always 2D)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        leaf = np.asarray(model.predict(X, pred_leaf=True), dtype=np.float64)
        if leaf.ndim == 1:
            leaf = leaf.reshape(-1, 1)
        return leaf

    def uncertainty_scores(self, model: Any, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            proba = self.predict_proba(model, X)
            proba = np.clip(proba, 1e-12, 1.0)
            entropy = -np.sum(proba * np.log(proba), axis=1)
            return entropy
        preds = self.predict(model, X)
        return np.abs(preds)

    def margin_scores(self, model: Any, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("Margin scoring only for classification")
        proba = self.predict_proba(model, X)
        sorted_p = np.sort(proba, axis=1)
        margin = sorted_p[:, -1] - sorted_p[:, -2]
        return -margin  # higher score = more uncertain
