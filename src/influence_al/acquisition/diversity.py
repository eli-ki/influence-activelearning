"""Batch selection: pre-filter + k-means++ diversity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class BatchSelectorConfig:
    prefilter_multiplier: int = 5
    embedding: str = "features"  # features | leaf_index
    discard_negative: bool = False


def _as_2d_features(emb: np.ndarray, X_fallback: np.ndarray) -> np.ndarray:
    """Ensure k-means input is 2D; fall back to raw features if embedding is degenerate."""
    X_fb = np.atleast_2d(np.asarray(X_fallback, dtype=np.float64))
    emb = np.asarray(emb, dtype=np.float64)

    if emb.ndim == 1:
        if X_fb.ndim == 2 and X_fb.shape[0] == emb.shape[0] and X_fb.shape[1] > 1:
            emb = X_fb
        else:
            emb = emb.reshape(-1, 1)
    elif emb.ndim != 2:
        emb = emb.reshape(emb.shape[0], -1)

    if emb.shape[0] != X_fb.shape[0]:
        return X_fb

    # All-identical rows or zero vectors: use feature space for diversity
    if emb.shape[1] == 0 or np.allclose(emb, emb.ravel()[0]):
        if X_fb.shape[1] > 0 and not np.allclose(X_fb, X_fb.ravel()[0]):
            return X_fb

    if not np.isfinite(emb).all():
        return np.nan_to_num(X_fb, nan=0.0, posinf=0.0, neginf=0.0)

    return emb


class BatchSelector:
    """Shared diversity module for all acquisition methods."""

    def __init__(self, config: BatchSelectorConfig):
        self.config = config

    def _embeddings(
        self,
        X: np.ndarray,
        model: Any,
        trainer: Any,
    ) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.config.embedding == "leaf_index":
            try:
                return trainer.leaf_index_embedding(model, X)
            except Exception:
                return X
        return X

    def select(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        X: np.ndarray,
        model: Any,
        trainer: Any,
        batch_size: int,
    ) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        scores = np.asarray(scores, dtype=np.float64)
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[0] != len(indices):
            raise ValueError(
                f"Feature matrix rows ({X.shape[0]}) must match indices ({len(indices)})"
            )

        if len(indices) <= batch_size:
            order = np.argsort(scores)[::-1]
            return indices[order[:batch_size]]

        mask = np.ones(len(indices), dtype=bool)
        if self.config.discard_negative:
            mask = scores > 0
            if mask.sum() < batch_size:
                mask = np.ones(len(indices), dtype=bool)

        idx_m = indices[mask]
        sc_m = scores[mask]
        X_m = X[mask]

        n_prefilter = min(len(idx_m), self.config.prefilter_multiplier * batch_size)
        top_order = np.argsort(sc_m)[::-1][:n_prefilter]
        idx_pf = idx_m[top_order]
        sc_pf = sc_m[top_order]
        X_pf = X_m[top_order]

        if len(idx_pf) <= batch_size:
            return idx_pf

        emb = _as_2d_features(self._embeddings(X_pf, model, trainer), X_pf)

        n_clusters = min(batch_size, len(idx_pf))
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            random_state=0,
        )
        labels = kmeans.fit_predict(emb)

        selected = []
        for c in range(n_clusters):
            cluster_mask = labels == c
            if not np.any(cluster_mask):
                continue
            cluster_idx = np.where(cluster_mask)[0]
            best_local = cluster_idx[np.argmax(sc_pf[cluster_idx])]
            selected.append(idx_pf[best_local])

        if len(selected) < batch_size:
            remaining = np.setdiff1d(idx_pf, selected)
            remaining_scores = sc_pf[np.isin(idx_pf, remaining)]
            order = np.argsort(remaining_scores)[::-1]
            for i in order:
                if len(selected) >= batch_size:
                    break
                cand = remaining[i]
                if cand not in selected:
                    selected.append(cand)

        return np.array(selected[:batch_size], dtype=int)
