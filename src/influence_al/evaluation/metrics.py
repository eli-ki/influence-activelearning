"""Learning curve metrics and reliability checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class LearningCurveResult:
    method: str
    dataset: str
    seed: int
    n_labels: List[int]
    test_metrics: List[float]
    beats_random_rounds: List[bool]
    oracle_spearman: List[float]


def compute_learning_curve_stats(results: List[LearningCurveResult]) -> dict:
    """Aggregate multi-seed learning curves (aligns to shortest curve length)."""
    if not results:
        return {}
    method = results[0].method
    dataset = results[0].dataset
    min_len = min(len(r.test_metrics) for r in results)
    all_metrics = np.array([r.test_metrics[:min_len] for r in results])
    all_labels = results[0].n_labels[:min_len]
    return {
        "method": method,
        "dataset": dataset,
        "n_seeds": len(results),
        "n_labels": all_labels,
        "test_metric_mean": all_metrics.mean(axis=0).tolist(),
        "test_metric_std": all_metrics.std(axis=0).tolist(),
        "final_metric_mean": float(all_metrics[:, -1].mean()),
        "reliability_vs_random": float(
            np.mean([np.mean(r.beats_random_rounds) for r in results if r.beats_random_rounds])
        )
        if any(r.beats_random_rounds for r in results)
        else None,
        "oracle_spearman_mean": float(
            np.nanmean([np.nanmean(r.oracle_spearman) for r in results if r.oracle_spearman])
        )
        if any(r.oracle_spearman for r in results)
        else None,
    }


def labels_to_target_metric(
    n_labels: List[int],
    test_metrics: List[float],
    target_fraction: float = 0.95,
    full_supervision_metric: float | None = None,
) -> int | None:
    """Labels needed to reach target_fraction of full supervision performance."""
    if full_supervision_metric is None:
        full_supervision_metric = max(test_metrics)
    target = target_fraction * full_supervision_metric
    for n, m in zip(n_labels, test_metrics):
        if m >= target:
            return n
    return None
