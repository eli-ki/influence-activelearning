"""Robustness and label-efficiency study runner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from influence_al.evaluation.efficiency import (
    build_scenario_summary,
    compute_robustness_deltas,
    format_summary_table,
    plot_label_efficiency_bars,
    plot_robustness_comparison,
)
from influence_al.experiments.run import (
    load_config,
    plot_learning_curves,
    run_multi_seed,
    stats_from_json,
)


def load_study_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_scenario_config(
    config_dir: Path,
    dataset: str,
    method: str,
    scenario: dict,
    study: dict,
) -> dict:
    cfg = load_config(config_dir, dataset, method)
    cfg["scenario_name"] = scenario["name"]
    cfg["pool_corruption"] = scenario.get("pool_corruption") or {}
    for key in ("batch_size", "n_rounds", "n_seeds", "output_dir"):
        if study.get(key) is not None:
            cfg[key] = study[key]
    if scenario.get("config"):
        cfg.update(scenario["config"])
    return cfg


def run_scenario(
    study: dict,
    scenario: dict,
    config_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> Dict[str, dict]:
    dataset = study["dataset"]
    methods = study["methods"]
    scenario_dir = Path(study.get("output_dir", "results/study")) / scenario["name"]
    scenario_dir.mkdir(parents=True, exist_ok=True)

    results_by_method: Dict[str, dict] = {}
    all_stats: List[dict] = []

    print(f"\n=== Scenario: {scenario['name']} ===")
    if scenario.get("description"):
        print(scenario["description"])

    for method in methods:
        m_cfg = merge_scenario_config(config_dir, dataset, method, scenario, study)
        m_cfg["output_dir"] = str(scenario_dir)
        out_path = scenario_dir / f"{dataset}_{method}.json"

        if dry_run:
            status = "exists" if out_path.exists() else "run"
            print(f"  [dry-run] {method} ({m_cfg.get('n_seeds', 3)} seeds) — {status}")
            continue

        if skip_existing and out_path.exists():
            try:
                with open(out_path, encoding="utf-8") as f:
                    out = json.load(f)
                print(f"  Skipping {method} (found {out_path.name})")
                results_by_method[method] = out
                if out.get("stats"):
                    all_stats.append(out["stats"])
                continue
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  Warning: could not load {out_path.name}, re-running: {exc}", file=sys.stderr)

        try:
            n_seeds = m_cfg.get("n_seeds", 3)
            out = run_multi_seed(dataset, method, m_cfg, n_seeds)
        except Exception as exc:
            print(f"  ERROR {method}: {exc}", file=sys.stderr)
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {out_path.name} (final={out['stats'].get('final_metric_mean', 0):.4f})")

        results_by_method[method] = out
        if out.get("stats"):
            all_stats.append(out["stats"])

    if not dry_run and len(all_stats) >= 2:
        plot_path = plot_learning_curves(
            all_stats,
            scenario_dir / f"{dataset}_comparison.png",
            title_suffix=scenario["name"],
        )
        print(f"  Plot: {plot_path}")

    if not dry_run and results_by_method:
        summary_rows = build_scenario_summary(
            results_by_method,
            target_fraction=study.get("target_fraction", 0.95),
            baseline_method=study.get("baseline_method", "random"),
        )
        _write_summary_csv(scenario_dir / "summary.csv", summary_rows)
        with open(scenario_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write(f"# {scenario['name']}\n\n")
            if scenario.get("description"):
                f.write(f"{scenario['description']}\n\n")
            f.write(format_summary_table(summary_rows))
            f.write("\n")
        plot_label_efficiency_bars(
            summary_rows,
            scenario_dir / "label_efficiency.png",
            title=f"Label efficiency — {scenario['name']}",
        )

    return results_by_method


def _write_summary_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summary_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    float_keys = {
        "final_metric_mean",
        "final_metric_std",
        "target_metric",
        "labels_at_target_mean",
        "labels_at_target_std",
        "reach_rate",
        "win_rate_vs_random",
        "label_savings_vs_random_mean",
        "curve_auc_mean",
        "curve_auc_std",
        "oracle_spearman_mean",
    }
    for row in rows:
        for key in float_keys:
            val = row.get(key)
            if val not in (None, "", "-"):
                try:
                    row[key] = float(val)
                except ValueError:
                    pass
        if row.get("n_seeds"):
            try:
                row["n_seeds"] = int(row["n_seeds"])
            except ValueError:
                pass
    return rows


def build_study_report(study: dict, scenario_names: List[str], output_root: Path) -> Path:
    scenario_summaries: Dict[str, List[dict]] = {}
    for name in scenario_names:
        csv_path = output_root / name / "summary.csv"
        if csv_path.exists():
            scenario_summaries[name] = _load_summary_csv(csv_path)

    if len(scenario_summaries) >= 2:
        plot_robustness_comparison(
            scenario_summaries,
            output_root / "robustness_comparison.png",
            dataset=study.get("dataset", ""),
        )
        deltas = compute_robustness_deltas(scenario_summaries)
        if deltas:
            _write_summary_csv(output_root / "robustness_deltas.csv", deltas)

    lines = [
        f"# Study: {study.get('name', 'study')}",
        "",
        f"Dataset: **{study['dataset']}**",
        f"Target metric: {study.get('target_fraction', 0.95) * 100:.0f}% of best final accuracy in each scenario.",
        "",
        "Metrics: **final_metric_mean**, **labels_at_target_mean** (label efficiency), "
        "**curve_auc_mean** (area under learning curve), **win_rate_vs_random**, "
        "**oracle_spearman_mean** (influence proxy calibration).",
        "",
        "## Scenarios",
        "",
    ]
    if (output_root / "robustness_comparison.png").exists():
        lines.append("![robustness](robustness_comparison.png)")
        lines.append("")
    for name in scenario_names:
        summary_path = output_root / name / "summary.md"
        if summary_path.exists():
            lines.append(f"### {name}")
            lines.append("")
            with open(summary_path, encoding="utf-8") as f:
                content = f.read()
            # Skip duplicate H1
            for line in content.splitlines():
                if line.startswith("# "):
                    continue
                lines.append(line)
            lines.append("")
        plot = output_root / name / f"{study['dataset']}_comparison.png"
        if plot.exists():
            lines.append(f"![{name} learning curves]({name}/{plot.name})")
            lines.append("")
        eff = output_root / name / "label_efficiency.png"
        if eff.exists():
            lines.append(f"![{name} label efficiency]({name}/{eff.name})")
            lines.append("")

    report_path = output_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run robustness / label-efficiency study (clean vs noisy pool)"
    )
    parser.add_argument(
        "--study",
        type=str,
        default="adult_robustness",
        help="Study config name under configs/studies/",
    )
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Subset of scenario names (default: all in study config)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip methods whose JSON output already exists (default: true)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild report.md from existing scenario summaries",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    study_path = config_dir / "studies" / f"{args.study}.yaml"
    study = load_study_config(study_path)
    output_root = Path(study.get("output_dir", f"results/study/{args.study}"))
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = study.get("scenarios", [])
    if args.scenarios:
        scenarios = [s for s in scenarios if s["name"] in args.scenarios]

    if args.report_only:
        names = [s["name"] for s in scenarios]
        report = build_study_report(study, names, output_root)
        print(f"Wrote {report}")
        return

    ran: List[str] = []
    for scenario in scenarios:
        run_scenario(
            study,
            scenario,
            config_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
        ran.append(scenario["name"])

    if not args.dry_run:
        report = build_study_report(study, ran, output_root)
        print(f"\nStudy report: {report}")


if __name__ == "__main__":
    main()
