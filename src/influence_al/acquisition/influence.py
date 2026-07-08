"""Temp-model influence acquisition (core method)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from influence_al.acquisition.base import AcquisitionContext, AcquisitionFunction

INFLUENCE_METHODS = frozenset({"influence", "influence_shapley", "influence_r_pseudo"})


def validate_influence_dataset(
    y: np.ndarray,
    task: str,
    dataset_name: str,
) -> None:
    """BoostIn/LeafInfluence only support binary classification (labels 0 and 1)."""
    if task != "classification":
        raise ValueError(
            f"Influence-based acquisition requires a classification dataset, "
            f"but '{dataset_name}' is configured for {task}. "
            f"Use a non-influence method for regression."
        )
    classes = np.unique(y)
    if len(classes) != 2 or not np.array_equal(classes, [0, 1]):
        raise ValueError(
            f"Influence-based acquisition (BoostIn/LeafInfluence) requires exactly "
            f"two classes labeled 0 and 1, but dataset '{dataset_name}' has "
            f"{len(classes)} classes: {classes.tolist()}. "
            f"Use baseline methods (random, uncertainty, badge, margin, loss) on "
            f"multiclass ADS/RALIF datasets, or subsample two classes."
        )


def _get_explainer(name: str):
    try:
        name = name.lower()
        if name == "boostin":
            from tree_influence.explainers import BoostIn
            return BoostIn()
        if name == "leafinfluence":
            from tree_influence.explainers import LeafInfluence
            return LeafInfluence(n_jobs=1)
    except ImportError as e:
        raise ImportError(
            "tree-influence is required for influence-based acquisition. "
            "Install with: pip install 'tree-influence>=0.1.7' (Python 3.9–3.10)."
        ) from e
    raise ValueError(f"Unknown explainer: {name}")


def _aggregate_influence(
    influence_row: np.ndarray,
    ref_losses: Optional[np.ndarray],
    aggregation: str,
) -> float:
    if aggregation == "sum":
        return float(np.sum(influence_row))
    if aggregation == "mean":
        return float(np.mean(influence_row))
    if aggregation == "loss_weighted":
        if ref_losses is None:
            return float(np.sum(influence_row))
        weights = ref_losses / (ref_losses.sum() + 1e-12)
        return float(np.sum(influence_row * weights))
    raise ValueError(f"Unknown aggregation: {aggregation}")


class TempModelInfluenceAcquisition(AcquisitionFunction):
    """
    Per-round protocol:
      1. M_t already trained on L_t_train (in ctx.model).
      2. Pseudo-label U_t with M_t.
      3. Train M_tilde on L_t_train ∪ U_t (pseudo).
      4. Fit explainer on M_tilde; compute I = influence(train, R).
      5. score(u) = aggregate_r I[row(u), r] for u in U_t rows.
    """

    def __init__(
        self,
        explainer_name: str = "boostin",
        aggregation: str = "sum",
        pseudo_label_mode: str = "argmax",
        top_k_classes: int = 3,
        reference: str = "r_ref",
        r_pseudo_size: int = 200,
        self_influence_filter: bool = False,
        self_influence_quantile: float = 0.9,
    ):
        self.explainer_name = explainer_name
        self.aggregation = aggregation
        self.pseudo_label_mode = pseudo_label_mode
        self.top_k_classes = top_k_classes
        self.reference = reference
        self.r_pseudo_size = r_pseudo_size
        self.self_influence_filter = self_influence_filter
        self.self_influence_quantile = self_influence_quantile
        self._last_diagnostics: dict = {}

    @property
    def last_diagnostics(self) -> dict:
        return self._last_diagnostics

    def _reference_set(
        self, ctx: AcquisitionContext, X_u: np.ndarray, y_pseudo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.reference == "r_ref":
            return ctx.X_r_ref(), ctx.y_r_ref()
        if self.reference == "r_pseudo":
            n = min(self.r_pseudo_size, len(X_u))
            rng = np.random.default_rng(ctx.config.get("seed", 0) + ctx.round_idx)
            sel = rng.choice(len(X_u), n, replace=False)
            return X_u[sel], y_pseudo[sel]
        raise ValueError(f"Unknown reference: {self.reference}")

    def _build_temp_model(
        self,
        ctx: AcquisitionContext,
        X_u: np.ndarray,
        y_pseudo: np.ndarray,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        X_l, y_l = ctx.X_l_train(), ctx.y_l_train()
        X_train = np.vstack([X_l, X_u])
        y_train = np.concatenate([y_l, y_pseudo])
        result = ctx.trainer.fit(X_train, y_train)
        return result.model, X_train, y_train

    def score(self, ctx: AcquisitionContext) -> tuple[np.ndarray, np.ndarray]:
        indices = ctx.idx_u()
        X_u = ctx.X_u()

        y_pseudo = ctx.trainer.pseudo_labels(
            ctx.model,
            X_u,
            mode=self.pseudo_label_mode,
            top_k=self.top_k_classes,
        )

        model_tilde, X_train, y_train = self._build_temp_model(ctx, X_u, y_pseudo)

        X_ref, y_ref = self._reference_set(ctx, X_u, y_pseudo)

        explainer = _get_explainer(self.explainer_name)
        explainer.fit(model_tilde, X_train, y_train)
        influence = explainer.get_local_influence(X_ref, y_ref)

        n_l = len(ctx.pool.get_l_train_indices())
        n_u = len(indices)
        u_rows = influence[n_l : n_l + n_u, :]

        ref_losses = ctx.trainer.loss_per_sample(model_tilde, X_ref, y_ref)

        scores = np.array(
            [
                _aggregate_influence(u_rows[i], ref_losses, self.aggregation)
                for i in range(n_u)
            ],
            dtype=np.float64,
        )

        if self.self_influence_filter and n_u > 0:
            try:
                self_inf = explainer.get_self_influence(X_train, y_train)
                u_self = self_inf[n_l : n_l + n_u]
                threshold = np.quantile(u_self, self.self_influence_quantile)
                high_self = u_self >= threshold
                scores[high_self] *= 0.1
            except Exception:
                pass

        self._last_diagnostics = {
            "influence_matrix_shape": influence.shape,
            "n_l_train": n_l,
            "n_u": n_u,
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "n_positive": int(np.sum(scores > 0)),
            "reference": self.reference,
        }

        return indices, scores
