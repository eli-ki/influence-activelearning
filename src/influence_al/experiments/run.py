"""Experiment CLI and multi-seed harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

from influence_al.data.datasets import load_dataset
from influence_al.data.pool import ActiveLearningPool
from influence_al.evaluation.metrics import compute_learning_curve_stats
from influence_al.loop.engine import ActiveLearningEngine, build_acquisition
from influence_al.models.trainer import LGBMTrainer


def load_config(config_dir: Path, dataset: str, method: str) -> dict:
    with open(config_dir / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    method_path = config_dir / "methods" / f"{method}.yaml"
    if method_path.exists():
        with open(method_path) as f:
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
    )
    pool = ActiveLearningPool.from_pool(
        data.X_pool,
        data.y_pool,
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
    for i in range(n_seeds):
        seed = base_seed + i
        results.append(run_single(dataset_name, method, seed, config))
    stats = compute_learning_curve_stats(
        [
            __import__("influence_al.evaluation.metrics", fromlist=["LearningCurveResult"]).LearningCurveResult(
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


def plot_learning_curves(all_stats: List[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for stats in all_stats:
        ax.plot(
            stats["n_labels"],
            stats["test_metric_mean"],
            label=f"{stats['method']} ({stats['dataset']})",
        )
        if stats.get("test_metric_std"):
            arr = np.array(stats["test_metric_mean"])
            std = np.array(stats["test_metric_std"])
            ax.fill_between(stats["n_labels"], arr - std, arr + std, alpha=0.2)
    ax.set_xlabel("Number of labeled samples")
    ax.set_ylabel("Test metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-influence active learning experiments")
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--method", type=str, default="influence")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="Run multiple methods and plot together",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    cfg = load_config(config_dir, args.dataset, args.method)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = args.compare if args.compare else [args.method]
    all_stats = []

    for method in methods:
        m_cfg = load_config(config_dir, args.dataset, method)
        m_cfg.update({k: v for k, v in cfg.items() if k not in m_cfg or k in ("output_dir", "seed")})
        if args.seed is not None:
            m_cfg["seed"] = args.seed
        if args.n_seeds is not None:
            m_cfg["n_seeds"] = args.n_seeds

        if args.n_seeds or m_cfg.get("n_seeds", 1) > 1:
            out = run_multi_seed(args.dataset, method, m_cfg, m_cfg.get("n_seeds"))
        else:
            seed = args.seed or m_cfg.get("seed", 42)
            out = {"runs": [run_single(args.dataset, method, seed, m_cfg)], "stats": {}}
            from influence_al.evaluation.metrics import LearningCurveResult
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

        out_path = output_dir / f"{args.dataset}_{method}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {out_path}")
        if out.get("stats"):
            all_stats.append(out["stats"])
            print(f"  final metric mean: {out['stats'].get('final_metric_mean')}")
            if out["stats"].get("oracle_spearman_mean") is not None:
                print(f"  oracle spearman mean: {out['stats']['oracle_spearman_mean']:.3f}")

    if len(all_stats) > 1:
        plot_path = output_dir / f"{args.dataset}_comparison.png"
        plot_learning_curves(all_stats, plot_path)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
