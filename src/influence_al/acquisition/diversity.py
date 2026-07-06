"""Batch selection: pre-filter + k-means++ diversity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class BatchSelectorConfig:
    prefilter_multiplier: int = 5
    embedding: str = "features"  # features | leaf_index
    discard_negative: bool = False


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
        if self.config.embedding == "leaf_index":
            return trainer.leaf_index_embedding(model, X)
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
        if len(indices) <= batch_size:
            order = np.argsort(scores)[::-1]
            return indices[order]

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

        emb = self._embeddings(X_pf, model, trainer)
        kmeans = KMeans(
            n_clusters=batch_size,
            init="k-means++",
            n_init=10,
            random_state=0,
        )
        labels = kmeans.fit_predict(emb)

        selected = []
        for c in range(batch_size):
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
