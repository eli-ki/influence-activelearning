"""Label efficiency and robustness metrics for active learning studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def learning_curve_auc(n_labels: List[int], test_metrics: List[float]) -> float:
    """Area under the learning curve (trapezoid rule, normalized by label span)."""
    if len(n_labels) < 2 or len(test_metrics) < 2:
        return float(test_metrics[0]) if test_metrics else 0.0
    x = np.asarray(n_labels, dtype=float)
    y = np.asarray(test_metrics, dtype=float)
    span = max(x[-1] - x[0], 1.0)
    area = float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0))
    return area / span


def labels_at_target_per_run(
    run: dict,
    target_metric: float,
) -> Optional[int]:
    """Minimum labeled count to reach target_metric; None if never reached."""
    n_labels = run.get("n_labels", [])
    metrics = run.get("test_metrics", [])
    for n, m in zip(n_labels, metrics):
        if m >= target_metric:
            return int(n)
    return None


def compute_scenario_target_metric(all_runs_by_method: Dict[str, List[dict]], fraction: float = 0.95) -> float:
    """Target = fraction × best final test metric observed in the scenario."""
    finals = []
    for runs in all_runs_by_method.values():
        for r in runs:
            if r.get("test_metrics"):
                finals.append(r["test_metrics"][-1])
    if not finals:
        return 0.0
    return fraction * max(finals)


def summarize_method_runs(
    method: str,
    runs: List[dict],
    target_metric: float,
    baseline_runs: Optional[List[dict]] = None,
) -> dict:
    """Aggregate efficiency and reliability stats for one method in a scenario."""
    finals = [r["test_metrics"][-1] for r in runs if r.get("test_metrics")]
    labels_at = [labels_at_target_per_run(r, target_metric) for r in runs]
    labels_at_valid = [x for x in labels_at if x is not None]
    aucs = [
        learning_curve_auc(r.get("n_labels", []), r.get("test_metrics", []))
        for r in runs
        if r.get("test_metrics")
    ]

    oracle = []
    for r in runs:
        if r.get("oracle_spearman"):
            oracle.extend(r["oracle_spearman"])

    row: Dict[str, Any] = {
        "method": method,
        "n_seeds": len(runs),
        "final_metric_mean": float(np.mean(finals)) if finals else None,
        "final_metric_std": float(np.std(finals)) if finals else None,
        "target_metric": target_metric,
        "labels_at_target_mean": float(np.mean(labels_at_valid)) if labels_at_valid else None,
        "labels_at_target_std": float(np.std(labels_at_valid)) if labels_at_valid else None,
        "reach_rate": float(len(labels_at_valid) / len(runs)) if runs else 0.0,
        "curve_auc_mean": float(np.mean(aucs)) if aucs else None,
        "curve_auc_std": float(np.std(aucs)) if aucs else None,
        "oracle_spearman_mean": float(np.nanmean(oracle)) if oracle else None,
    }

    if baseline_runs and method != "random":
        wins = 0
        savings = []
        for r in runs:
            b = next((x for x in baseline_runs if x["seed"] == r["seed"]), None)
            if b is None:
                continue
            if r["test_metrics"][-1] > b["test_metrics"][-1]:
                wins += 1
            lt = labels_at_target_per_run(r, target_metric)
            lb = labels_at_target_per_run(b, target_metric)
            if lt is not None and lb is not None:
                savings.append(lb - lt)
        row["win_rate_vs_random"] = wins / len(runs) if runs else None
        row["label_savings_vs_random_mean"] = float(np.mean(savings)) if savings else None

    return row


def build_scenario_summary(
    results_by_method: Dict[str, dict],
    target_fraction: float = 0.95,
    baseline_method: str = "random",
) -> List[dict]:
    """Build summary rows for all methods in one scenario."""
    runs_by_method = {m: results_by_method[m]["runs"] for m in results_by_method}
    target = compute_scenario_target_metric(runs_by_method, target_fraction)
    baseline = runs_by_method.get(baseline_method)

    rows = []
    for method, payload in results_by_method.items():
        rows.append(
            summarize_method_runs(
                method,
                payload["runs"],
                target_metric=target,
                baseline_runs=baseline if method != baseline_method else None,
            )
        )
    return rows


def format_summary_table(rows: List[dict]) -> str:
    """Markdown table for scenario summary."""
    headers = [
        "method",
        "final_metric_mean",
        "labels_at_target_mean",
        "reach_rate",
        "win_rate_vs_random",
        "label_savings_vs_random_mean",
        "curve_auc_mean",
        "oracle_spearman_mean",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_label_efficiency_bars(rows: List[dict], output_path: Path, title: str = "") -> Path:
    """Bar chart: labels needed to reach target metric (lower is better)."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_rows = [r for r in rows if r.get("labels_at_target_mean") is not None]
    if not plot_rows:
        return output_path

    methods = [r["method"] for r in plot_rows]
    means = [r["labels_at_target_mean"] for r in plot_rows]
    stds = [r.get("labels_at_target_std") or 0.0 for r in plot_rows]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 0.9), 4.5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Labels to reach target")
    ax.set_title(title or "Label efficiency (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_robustness_comparison(
    scenario_summaries: Dict[str, List[dict]],
    output_path: Path,
    dataset: str = "",
) -> Path:
    """Grouped bars: final metric and labels-at-target across corruption scenarios."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = list(scenario_summaries.keys())
    methods = sorted({r["method"] for rows in scenario_summaries.values() for r in rows})
    if not scenarios or not methods:
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [
        ("final_metric_mean", "Final test metric (higher is better)", axes[0]),
        ("labels_at_target_mean", "Labels to target (lower is better)", axes[1]),
    ]

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    for m_idx, method in enumerate(methods):
        offset = (m_idx - (len(methods) - 1) / 2) * width
        for key, ylabel, ax in metrics:
            vals = []
            for sc in scenarios:
                row = next((r for r in scenario_summaries[sc] if r["method"] == method), None)
                v = row.get(key) if row else None
                vals.append(v if v is not None else np.nan)
            ax.bar(x + offset, vals, width, label=method, alpha=0.85)

    for key, ylabel, ax in metrics:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_ylabel(ylabel.split("(")[0].strip())
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="lower right")

    title = f"Robustness across scenarios"
    if dataset:
        title += f" — {dataset}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def compute_robustness_deltas(
    scenario_summaries: Dict[str, List[dict]],
    baseline_scenario: str = "clean",
) -> List[dict]:
    """Per-method degradation from clean → corrupted scenarios."""
    clean_rows = {r["method"]: r for r in scenario_summaries.get(baseline_scenario, [])}
    deltas = []
    for scenario, rows in scenario_summaries.items():
        if scenario == baseline_scenario:
            continue
        for row in rows:
            base = clean_rows.get(row["method"])
            if not base:
                continue
            final_drop = None
            if base.get("final_metric_mean") and row.get("final_metric_mean"):
                final_drop = row["final_metric_mean"] - base["final_metric_mean"]
            label_penalty = None
            if base.get("labels_at_target_mean") and row.get("labels_at_target_mean"):
                label_penalty = row["labels_at_target_mean"] - base["labels_at_target_mean"]
            deltas.append(
                {
                    "method": row["method"],
                    "scenario": scenario,
                    "final_metric_drop": final_drop,
                    "extra_labels_vs_clean": label_penalty,
                }
            )
    return deltas
