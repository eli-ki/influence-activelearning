"""Tests for robustness study reporting."""

import csv
from pathlib import Path

import pytest

from influence_al.evaluation.efficiency import (
    compute_robustness_deltas,
    plot_label_efficiency_bars,
    plot_robustness_comparison,
)
from influence_al.experiments.study import build_study_report


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_label_efficiency_bars(tmp_path):
    rows = [
        {
            "method": "influence",
            "labels_at_target_mean": 120.0,
            "labels_at_target_std": 10.0,
        },
        {
            "method": "random",
            "labels_at_target_mean": 180.0,
            "labels_at_target_std": 15.0,
        },
    ]
    out = plot_label_efficiency_bars(rows, tmp_path / "eff.png")
    assert out.exists()


def test_robustness_deltas_and_report(tmp_path):
    base_row = {
        "method": "influence",
        "n_seeds": 3,
        "final_metric_mean": 0.84,
        "labels_at_target_mean": 200,
        "reach_rate": 1.0,
        "curve_auc_mean": 0.8,
    }
    ood_row = {**base_row, "final_metric_mean": 0.79, "labels_at_target_mean": 280}
    _write_summary(tmp_path / "clean" / "summary.csv", [base_row, {**base_row, "method": "random"}])
    _write_summary(tmp_path / "ood_20" / "summary.csv", [ood_row, {**ood_row, "method": "random"}])

    summaries = {
        "clean": [base_row],
        "ood_20": [ood_row],
    }
    deltas = compute_robustness_deltas(summaries)
    assert len(deltas) == 1
    assert deltas[0]["final_metric_drop"] == pytest.approx(-0.05)
    assert deltas[0]["extra_labels_vs_clean"] == 80

    plot_robustness_comparison(summaries, tmp_path / "robustness.png", dataset="cifar10")
    assert (tmp_path / "robustness.png").exists()

    study = {"name": "test", "dataset": "cifar10", "target_fraction": 0.95}
    report = build_study_report(study, ["clean", "ood_20"], tmp_path)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "robustness_comparison.png" in text
