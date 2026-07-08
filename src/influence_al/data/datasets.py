"""Unified dataset loaders: tabular benchmarks and ADS/RALIF vision benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from influence_al.data.tabular_datasets import TABULAR_DATASET_REGISTRY, load_tabular_dataset
from influence_al.data.vision_datasets import VISION_DATASET_REGISTRY, _subsample, load_vision_dataset

TaskType = Literal["classification", "regression"]


@dataclass
class DatasetBundle:
    """Train pool features/labels and held-out test set."""

    X_pool: np.ndarray
    y_pool: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    task: TaskType
    name: str
    feature_names: Optional[list[str]] = None
    n_classes: Optional[int] = None
    papers: tuple[str, ...] = ()


DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    name: (task, "tabular") for name, (task, _) in TABULAR_DATASET_REGISTRY.items()
}
DATASET_REGISTRY.update(
    {name: ("classification", "vision") for name in VISION_DATASET_REGISTRY}
)


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)


def _scale_features(X_pool: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_pool = scaler.fit_transform(X_pool)
    X_test = scaler.transform(X_test)
    X_pool = np.nan_to_num(X_pool, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_pool.astype(np.float64), X_test.astype(np.float64)


def load_dataset(
    name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_pool_samples: int | None = None,
    max_test_samples: int | None = None,
    data_root: str | Path | None = None,
) -> DatasetBundle:
    """Load a tabular or vision benchmark into pool P and held-out test T."""
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")

    task_kind, source = DATASET_REGISTRY[name]

    if source == "tabular":
        X_pool, y_pool, X_test, y_test, task, feature_names = load_tabular_dataset(
            name,
            test_fraction=test_fraction,
            seed=seed,
            max_pool_samples=max_pool_samples,
        )
        n_classes = int(np.max(y_pool)) + 1 if task == "classification" else None
        return DatasetBundle(
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            task=task,
            name=name,
            feature_names=feature_names,
            n_classes=n_classes,
        )

    meta = VISION_DATASET_REGISTRY[name]
    root = Path(data_root) if data_root is not None else None
    X_pool, y_pool, X_test, y_test = load_vision_dataset(
        name,
        data_root=root,
        max_pool_samples=max_pool_samples,
        seed=seed,
    )
    X_pool, X_test = _scale_features(X_pool, X_test)
    X_test, y_test = _subsample(X_test, y_test, max_test_samples, seed + 1)

    n_classes = meta.get("classes")
    if n_classes is None:
        n_classes = int(np.max(y_pool)) + 1

    return DatasetBundle(
        X_pool=X_pool,
        y_pool=y_pool.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        task="classification",
        name=name,
        feature_names=None,
        n_classes=int(n_classes),
        papers=tuple(meta.get("papers", ())),
    )
