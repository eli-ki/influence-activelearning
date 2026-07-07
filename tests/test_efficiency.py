"""Tests for label-efficiency metrics."""

from influence_al.evaluation.efficiency import (
    build_scenario_summary,
    labels_at_target_per_run,
    learning_curve_auc,
    summarize_method_runs,
)


def _run(n_labels, metrics):
    return {"n_labels": n_labels, "test_metrics": metrics, "seed": 0}


def test_labels_at_target_per_run():
    run = _run([10, 20, 30], [0.7, 0.82, 0.85])
    assert labels_at_target_per_run(run, 0.8) == 20
    assert labels_at_target_per_run(run, 0.9) is None


def test_learning_curve_auc():
    auc = learning_curve_auc([10, 20, 30], [0.7, 0.8, 0.85])
    assert 0.7 < auc < 0.85


def test_summarize_method_runs_vs_random():
    influence = [
        {"seed": 0, "n_labels": [10, 20], "test_metrics": [0.7, 0.86], "oracle_spearman": [0.1]},
        {"seed": 1, "n_labels": [10, 20], "test_metrics": [0.72, 0.84], "oracle_spearman": [0.2]},
    ]
    random = [
        {"seed": 0, "n_labels": [10, 20], "test_metrics": [0.68, 0.82]},
        {"seed": 1, "n_labels": [10, 20], "test_metrics": [0.7, 0.83]},
    ]
    row = summarize_method_runs("influence", influence, target_metric=0.81, baseline_runs=random)
    assert row["win_rate_vs_random"] == 1.0
    assert row["label_savings_vs_random_mean"] == 0.0


def test_build_scenario_summary():
    results = {
        "influence": {
            "runs": [
                {"seed": 0, "n_labels": [10, 30], "test_metrics": [0.7, 0.9]},
            ]
        },
        "random": {
            "runs": [
                {"seed": 0, "n_labels": [10, 30], "test_metrics": [0.65, 0.85]},
            ]
        },
    }
    rows = build_scenario_summary(results, target_fraction=0.95)
    assert len(rows) == 2
    inf = next(r for r in rows if r["method"] == "influence")
    assert inf["target_metric"] == 0.95 * 0.9
