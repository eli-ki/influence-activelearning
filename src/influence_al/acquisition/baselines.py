"""Baseline acquisition methods."""

from __future__ import annotations

import numpy as np

from influence_al.acquisition.base import AcquisitionContext, AcquisitionFunction


class RandomAcquisition(AcquisitionFunction):
    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        indices = ctx.idx_u()
        rng = np.random.default_rng(ctx.config.get("seed", 0) + ctx.round_idx)
        scores = rng.random(len(indices))
        return indices, scores


class UncertaintyAcquisition(AcquisitionFunction):
    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        indices = ctx.idx_u()
        X_u = ctx.X_u()
        scores = ctx.trainer.uncertainty_scores(ctx.model, X_u)
        return indices, scores


class MarginAcquisition(AcquisitionFunction):
    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        if ctx.trainer.task != "classification":
            return UncertaintyAcquisition().score(ctx)
        indices = ctx.idx_u()
        X_u = ctx.X_u()
        scores = ctx.trainer.margin_scores(ctx.model, X_u)
        return indices, scores


class LossAcquisition(AcquisitionFunction):
    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        indices = ctx.idx_u()
        X_u = ctx.X_u()
        pseudo = ctx.trainer.pseudo_labels(ctx.model, X_u)
        losses = ctx.trainer.loss_per_sample(ctx.model, X_u, pseudo)
        return indices, losses


class FeatureKMeansAcquisition(AcquisitionFunction):
    """Diversity-only: uniform scores; diversity handled entirely by BatchSelector."""

    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        indices = ctx.idx_u()
        scores = np.ones(len(indices), dtype=np.float64)
        return indices, scores


class BadgeAdaptedAcquisition(AcquisitionFunction):
    """
    GBDT analogue of BADGE: uncertainty pre-filtering + k-means++ on leaf embeddings.
    Scoring uses uncertainty; BatchSelector should use embedding=leaf_index.
    """

    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        return UncertaintyAcquisition().score(ctx)
