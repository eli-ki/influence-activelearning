"""Experiment CLI and multi-seed harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import yaml

from influence_al.acquisition.influence import INFLUENCE_METHODS, validate_influence_dataset
from influence_al.data.datasets import load_dataset
from influence_al.data.noisy import inject_ood_rows
from influence_al.data.pool import ActiveLearningPool
from influence_al.evaluation.metrics import LearningCurveResult, compute_learning_curve_stats
from influence_al.loop.engine import ActiveLearningEngine, build_acquisition
from influence_al.models.trainer import LGBMTrainer

DEFAULT_COMPARE_METHODS = ["influence", "random", "uncertainty"]


def load_config(config_dir: Path, dataset: str, method: str) -> dict:
    with open(config_dir / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_path = config_dir / "datasets" / f"{dataset}.yaml"
    if dataset_path.exists():
        with open(dataset_path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f))
    method_path = config_dir / "methods" / f"{method}.yaml"
    if method_path.exists():
        with open(method_path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def run_single(
    dataset_name: str,
    method: str,
    seed: int,
    config: dict,
) -> dict:
    data = load_dataset(
        dataset_name,
        test_fraction=config.get("test_fraction", 0.2),
        seed=seed,
        max_pool_samples=config.get("max_pool_samples"),
    )
    X_pool, y_pool = data.X_pool.copy(), data.y_pool.copy()

    corruption = config.get("pool_corruption") or {}
    ood_frac = corruption.get("ood_fraction", 0.0)
    if ood_frac and ood_frac > 0:
        X_pool, y_pool = inject_ood_rows(X_pool, y_pool, ood_frac, seed=seed + 1000)

    if method in INFLUENCE_METHODS:
        validate_influence_dataset(y_pool, data.task, dataset_name)
    pool = ActiveLearningPool.from_pool(
        X_pool,
        y_pool,
        initial_labeled_fraction=config.get("initial_labeled_fraction", 0.05),
        r_ref_fraction=config.get("r_ref_fraction", 0.2),
        seed=seed,
    )
    trainer = LGBMTrainer(
        task=data.task,
        lgbm_params=config.get("lgbm", {}),
        seed=seed,
    )
    acquisition = build_acquisition(method, config)
    engine = ActiveLearningEngine(
        pool=pool,
        trainer=trainer,
        acquisition=acquisition,
        X_test=data.X_test,
        y_test=data.y_test,
        config=config,
        method_name=method,
        dataset_name=dataset_name,
        seed=seed,
    )
    if method == "badge":
        config.setdefault("diversity", {})["embedding"] = "leaf_index"
    result = engine.run()
    return {
        "method": method,
        "dataset": dataset_name,
        "seed": seed,
        "task": data.task,
        "scenario": config.get("scenario_name"),
        "pool_corruption": corruption,
        "n_labels": result.learning_curve.n_labels,
        "test_metrics": result.learning_curve.test_metrics,
        "oracle_spearman": result.learning_curve.oracle_spearman,
        "round_logs": result.round_logs,
    }


def run_multi_seed(
    dataset_name: str,
    method: str,
    config: dict,
    n_seeds: int | None = None,
) -> dict:
    n_seeds = n_seeds or config.get("n_seeds", 5)
    base_seed = config.get("seed", 42)
    results = []
    failed_seeds = []
    for i in range(n_seeds):
        seed = base_seed + i
        try:
            results.append(run_single(dataset_name, method, seed, config))
        except Exception as exc:
            failed_seeds.append(seed)
            print(f"ERROR: {method} seed {seed} failed: {exc}", file=sys.stderr)
    if not results:
        raise RuntimeError(f"All seeds failed for method '{method}': {failed_seeds}")
    if failed_seeds:
        print(f"Warning: {method} completed {len(results)}/{n_seeds} seeds", file=sys.stderr)
    stats = compute_learning_curve_stats(
        [
            LearningCurveResult(
                method=r["method"],
                dataset=r["dataset"],
                seed=r["seed"],
                n_labels=r["n_labels"],
                test_metrics=r["test_metrics"],
                beats_random_rounds=[],
                oracle_spearman=r.get("oracle_spearman", []),
            )
            for r in results
        ]
    )
    return {"runs": results, "stats": stats}


def plot_learning_curves(
    all_stats: List[dict],
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for stats in all_stats:
        x = stats["n_labels"]
        y = stats["test_metric_mean"]
        ax.plot(x, y, marker=".", markersize=4, linewidth=1.5, label=stats["method"])
        std = stats.get("test_metric_std")
        if std and max(std) > 0:
            arr = np.array(y)
            std_arr = np.array(std)
            ax.fill_between(x, arr - std_arr, arr + std_arr, alpha=0.15)

    ax.set_xlabel("Number of labeled samples")
    ax.set_ylabel("Test metric (accuracy / R²)")
    if all_stats:
        title = f"Active learning — {all_stats[0].get('dataset', '')}"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_curve_summary(stats: dict) -> None:
    method = stats.get("method", "?")
    x = stats.get("n_labels", [])
    y = stats.get("test_metric_mean", [])
    if not x or not y:
        return
    print(f"  [{method}] learning curve (n_labels -> test_metric):")
    step = max(1, len(x) // 6)
    for i in range(0, len(x), step):
        print(f"    {x[i]:4d} -> {y[i]:.4f}")
    if (len(x) - 1) % step != 0:
        print(f"    {x[-1]:4d} -> {y[-1]:.4f}  (final)")


def stats_from_json(path: Path) -> dict | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("stats") or None


def plot_from_results_dir(output_dir: Path, dataset: str, methods: List[str] | None = None) -> Path | None:
    methods = methods or DEFAULT_COMPARE_METHODS
    all_stats = []
    for method in methods:
        path = output_dir / f"{dataset}_{method}.json"
        if not path.exists():
            print(f"Missing {path}", file=sys.stderr)
            continue
        stats = stats_from_json(path)
        if stats:
            all_stats.append(stats)
    if len(all_stats) < 2:
        print("Need at least 2 method JSON files to plot.", file=sys.stderr)
        return None
    return plot_learning_curves(all_stats, output_dir / f"{dataset}_comparison.png")


def resolve_compare_methods(args: argparse.Namespace) -> tuple[List[str], bool]:
    if args.compare is not None:
        if len(args.compare) == 0:
            return DEFAULT_COMPARE_METHODS, True
        return args.compare, True
    return [args.method], False


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-influence active learning experiments")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--method", type=str, default="influence")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Labels queried per AL round")
    parser.add_argument("--n-rounds", type=int, default=None, help="Number of active learning rounds")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="Compare methods (default: influence random uncertainty)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate comparison PNG from existing JSON (no re-run)",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    cfg = load_config(config_dir, args.dataset, args.method)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.n_rounds is not None:
        cfg["n_rounds"] = args.n_rounds
    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    methods, compare_mode = resolve_compare_methods(args)

    if args.plot_only:
        plot_path = plot_from_results_dir(output_dir, args.dataset, methods if compare_mode else None)
        if plot_path:
            print(f"Saved plot: {plot_path}")
        return

    all_stats = []
    failed_methods: List[str] = []

    for method in methods:
        m_cfg = load_config(config_dir, args.dataset, method)
        m_cfg.update({k: v for k, v in cfg.items() if k not in m_cfg or k in ("output_dir", "seed", "batch_size", "n_rounds")})
        if args.seed is not None:
            m_cfg["seed"] = args.seed
        if args.n_seeds is not None:
            m_cfg["n_seeds"] = args.n_seeds
        if args.batch_size is not None:
            m_cfg["batch_size"] = args.batch_size
        if args.n_rounds is not None:
            m_cfg["n_rounds"] = args.n_rounds

        try:
            n_seeds = m_cfg.get("n_seeds", 5)
            if n_seeds > 1:
                out = run_multi_seed(args.dataset, method, m_cfg, n_seeds)
            else:
                seed = args.seed or m_cfg.get("seed", 42)
                out = {"runs": [run_single(args.dataset, method, seed, m_cfg)], "stats": {}}
                out["stats"] = compute_learning_curve_stats(
                    [
                        LearningCurveResult(
                            method=method,
                            dataset=args.dataset,
                            seed=r["seed"],
                            n_labels=r["n_labels"],
                            test_metrics=r["test_metrics"],
                            beats_random_rounds=[],
                            oracle_spearman=r.get("oracle_spearman", []),
                        )
                        for r in out["runs"]
                    ]
                )
        except Exception as exc:
            failed_methods.append(method)
            print(f"ERROR: method '{method}' failed: {exc}", file=sys.stderr)
            continue

        out_path = (output_dir / f"{args.dataset}_{method}.json").resolve()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {out_path}")
        if out.get("stats"):
            all_stats.append(out["stats"])
            print(f"  final metric mean: {out['stats'].get('final_metric_mean'):.4f}")
            if method in INFLUENCE_METHODS and out["stats"].get("oracle_spearman_mean") is not None:
                print(f"  oracle spearman mean: {out['stats']['oracle_spearman_mean']:.3f}")
            print_curve_summary(out["stats"])

    if failed_methods:
        print(f"Warning: failed methods: {failed_methods}", file=sys.stderr)

    if compare_mode:
        # Prefer in-memory stats; fall back to JSON on disk (e.g. after partial runs)
        if len(all_stats) < 2:
            all_stats = []
            for method in methods:
                path = output_dir / f"{args.dataset}_{method}.json"
                stats = stats_from_json(path) if path.exists() else None
                if stats:
                    all_stats.append(stats)
        if len(all_stats) >= 2:
            plot_path = plot_learning_curves(all_stats, output_dir / f"{args.dataset}_comparison.png")
            print(f"Saved plot: {plot_path}")
        else:
            print(
                "Warning: fewer than 2 methods have results; no comparison plot. "
                "Re-run with --plot-only once more JSON files exist.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
