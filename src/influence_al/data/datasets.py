"""Dataset loaders for ADS / RALIF vision classification benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from influence_al.data.vision_datasets import VISION_DATASET_REGISTRY, load_vision_dataset, _subsample

TaskType = Literal["classification"]


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


DATASET_REGISTRY = {name: ("classification", "vision") for name in VISION_DATASET_REGISTRY}


def _subsample_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_test_samples: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _subsample(X_test, y_test, max_test_samples, seed)


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)


def _scale_features(
    X_pool: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    """Load an ADS / RALIF benchmark using official train/test splits.

    Images are flattened to feature vectors for LightGBM. ``test_fraction`` is
    ignored — each benchmark uses its canonical held-out test set.
    """
    del test_fraction  # official splits only

    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")

    meta = VISION_DATASET_REGISTRY[name]
    root = Path(data_root) if data_root is not None else None
    X_pool, y_pool, X_test, y_test = load_vision_dataset(
        name,
        data_root=root,
        max_pool_samples=max_pool_samples,
        seed=seed,
    )
    X_pool, X_test = _scale_features(X_pool, X_test)
    X_test, y_test = _subsample_test(X_test, y_test, max_test_samples, seed + 1)

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
