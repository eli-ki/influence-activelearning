"""Phase 2: ADS-style Shapley pre-filter using tree_influence DShap/SubSample."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from influence_al.acquisition.base import AcquisitionContext, AcquisitionFunction
from influence_al.acquisition.influence import TempModelInfluenceAcquisition


def _get_shapley_explainer(name: str = "subsample"):
    name = name.lower()
    if name == "subsample":
        from tree_influence.explainers import SubSample
        return SubSample()
    if name == "dshap":
        from tree_influence.explainers import DShap
        return DShap()
    raise ValueError(f"Unknown shapley explainer: {name}")


class ShapleyPrefilterAcquisition(AcquisitionFunction):
    """
    ADS-style pipeline:
      1. Compute Data Shapley values on L_t_train (leaf embeddings).
      2. KNN regressor: embedding -> Shapley value.
      3. Predict Shapley for U_t; keep top fraction.
      4. Temp-model influence on survivors.
    """

    def __init__(
        self,
        prefilter_fraction: float = 0.2,
        knn_neighbors: int = 5,
        top_k_classes: int = 10,
        shapley_explainer: str = "subsample",
        influence_kwargs: Optional[dict] = None,
    ):
        self.prefilter_fraction = prefilter_fraction
        self.knn_neighbors = knn_neighbors
        self.top_k_classes = top_k_classes
        self.shapley_explainer = shapley_explainer
        self.influence_kwargs = influence_kwargs or {}
        self._influence = TempModelInfluenceAcquisition(**self.influence_kwargs)
        self._last_diagnostics: dict = {}

    @property
    def last_diagnostics(self) -> dict:
        d = self._last_diagnostics.copy()
        d.update(self._influence.last_diagnostics)
        return d

    def _leaf_embeddings(self, ctx: AcquisitionContext, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_l, y_l = ctx.X_l_train(), ctx.y_l_train()
        X_u = ctx.X_u()
        emb_l = ctx.trainer.leaf_index_embedding(model, X_l)
        emb_u = ctx.trainer.leaf_index_embedding(model, X_u)
        return emb_l, emb_u, y_l

    def _compute_shapley_values(
        self,
        ctx: AcquisitionContext,
        model,
        X_l: np.ndarray,
        y_l: np.ndarray,
    ) -> np.ndarray:
        try:
            explainer = _get_shapley_explainer(self.shapley_explainer)
            explainer.fit(model, X_l, y_l)
            if hasattr(explainer, "get_shapley_values"):
                return explainer.get_shapley_values()
            return explainer.get_self_influence(X_l, y_l)
        except Exception:
            losses = ctx.trainer.loss_per_sample(model, X_l, y_l)
            return -losses

    def _predict_unlabeled_shapley(
        self,
        ctx: AcquisitionContext,
        emb_l: np.ndarray,
        shapley_l: np.ndarray,
        emb_u: np.ndarray,
    ) -> np.ndarray:
        if ctx.trainer.task == "regression":
            reg = KNeighborsRegressor(n_neighbors=min(self.knn_neighbors, len(emb_l)))
            reg.fit(emb_l, shapley_l)
            return reg.predict(emb_u)

        n_classes = int(np.max(ctx.pool.y_pool)) + 1
        proba = ctx.trainer.predict_proba(ctx.model, ctx.X_u())
        top_k = min(self.top_k_classes, proba.shape[1])
        top_classes = np.argsort(proba, axis=1)[:, -top_k:]

        scores = np.zeros(len(emb_u), dtype=np.float64)
        for c in range(n_classes):
            mask_l = ctx.y_l_train() == c
            if mask_l.sum() < 2:
                continue
            reg = KNeighborsRegressor(n_neighbors=min(self.knn_neighbors, int(mask_l.sum())))
            reg.fit(emb_l[mask_l], shapley_l[mask_l])
            pred_c = reg.predict(emb_u)
            in_top = np.any(top_classes == c, axis=1)
            scores[in_top] = np.maximum(scores[in_top], pred_c[in_top])
        if np.all(scores == 0):
            reg = KNeighborsRegressor(n_neighbors=min(self.knn_neighbors, len(emb_l)))
            reg.fit(emb_l, shapley_l)
            scores = reg.predict(emb_u)
        return scores

    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        indices = ctx.idx_u()
        X_l, y_l = ctx.X_l_train(), ctx.y_l_train()
        emb_l, emb_u, _ = self._leaf_embeddings(ctx, ctx.model)
        shapley_l = self._compute_shapley_values(ctx, ctx.model, X_l, y_l)
        shapley_u = self._predict_unlabeled_shapley(ctx, emb_l, shapley_l, emb_u)

        n_keep = max(1, int(len(indices) * self.prefilter_fraction))
        keep_order = np.argsort(shapley_u)[::-1][:n_keep]
        keep_indices = indices[keep_order]

        sub_ctx = AcquisitionContext(
            pool=_SubPoolView(ctx.pool, keep_indices),
            trainer=ctx.trainer,
            model=ctx.model,
            train_result=ctx.train_result,
            config=ctx.config,
            round_idx=ctx.round_idx,
        )
        sub_indices, sub_scores = self._influence.score(sub_ctx)

        full_scores = np.full(len(indices), -np.inf, dtype=np.float64)
        for i, idx in enumerate(sub_indices):
            pos = np.where(indices == idx)[0]
            if len(pos):
                full_scores[pos[0]] = sub_scores[i]

        self._last_diagnostics = {
            "shapley_prefilter_n_keep": n_keep,
            "shapley_u_mean": float(np.mean(shapley_u)),
        }
        return indices, full_scores


class _SubPoolView:
    """Minimal view restricting unlabeled set to a subset for influence scoring."""

    def __init__(self, pool, subset_indices: np.ndarray):
        self._pool = pool
        self.unlabeled_indices = np.asarray(subset_indices, dtype=int)
        self.r_ref_indices = pool.r_ref_indices
        self.X_pool = pool.X_pool
        self.y_pool = pool.y_pool
        self.labeled_indices = pool.labeled_indices

    def get_l_train(self):
        return self._pool.get_l_train()

    def get_l_train_indices(self):
        return self._pool.get_l_train_indices()

    def get_r_ref(self):
        return self._pool.get_r_ref()

    def get_unlabeled(self):
        return self.X_pool[self.unlabeled_indices], self.y_pool[self.unlabeled_indices]
