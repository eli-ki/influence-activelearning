"""Unified dataset loaders: tabular benchmarks and ADS/RALIF vision benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_iris,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

TaskType = Literal["classification", "regression"]

TABULAR_DATASET_REGISTRY: dict[str, tuple[TaskType, str]] = {
    "iris": ("classification", "sklearn"),
    "breast_cancer": ("classification", "sklearn"),
    "credit_g": ("classification", "openml"),
    "phoneme": ("classification", "openml"),
    "adult": ("classification", "openml"),
    "diabetes": ("regression", "sklearn"),
    "california_housing": ("regression", "sklearn"),
}

VISION_DATASET_NAMES = (
    "cifar10",
    "cifar100",
    "cinic10",
    "tiny_imagenet",
    "svhn",
    "svhn_extra",
    "fashion_mnist",
    "inaturalist",
)


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
DATASET_REGISTRY.update({name: ("classification", "vision") for name in VISION_DATASET_NAMES})


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)


def _encode_labels(y: np.ndarray) -> np.ndarray:
    if y.dtype.kind in ("i", "u", "f") and len(np.unique(y)) > 20:
        return y.astype(np.float64)
    le = LabelEncoder()
    return le.fit_transform(y).astype(np.int64)


def _feature_names(data: object) -> Optional[list[str]]:
    names = getattr(data, "feature_names", None)
    if names is None:
        return None
    return [str(name) for name in names]


def _load_openml(name: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    openml_map = {
        "credit_g": "credit-g",
        "phoneme": "phoneme",
        "adult": "adult",
    }
    data = fetch_openml(openml_map.get(name, name), version=1, as_frame=True, parser="auto")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc.fit_transform(data.data.astype(str)).astype(np.float64)
    y = data.target.values

    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def _load_tabular_dataset(
    name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_pool_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TaskType, Optional[list[str]]]:
    """Return (X_pool, y_pool, X_test, y_test, task, feature_names)."""
    name = name.lower()
    if name not in TABULAR_DATASET_REGISTRY:
        raise ValueError(f"Unknown tabular dataset '{name}'")

    task, source = TABULAR_DATASET_REGISTRY[name]

    if source == "sklearn":
        if name == "iris":
            data = load_iris()
        elif name == "breast_cancer":
            data = load_breast_cancer()
        elif name == "diabetes":
            data = load_diabetes()
        elif name == "california_housing":
            data = fetch_california_housing()
        else:
            raise ValueError(name)
        X, y = data.data, data.target
        feature_names = _feature_names(data)
    else:
        X, y = _load_openml(name, max_samples=max_pool_samples)
        feature_names = None

    if max_pool_samples and source == "sklearn" and len(X) > max_pool_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_pool_samples, replace=False)
        X, y = X[idx], y[idx]

    y = _encode_labels(y)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )

    scaler = StandardScaler()
    X_pool = scaler.fit_transform(X_pool)
    X_test = scaler.transform(X_test)
    X_pool = np.nan_to_num(X_pool, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        X_pool.astype(np.float64),
        y_pool,
        X_test.astype(np.float64),
        y_test,
        task,
        feature_names,
    )


def _scale_features(X_pool: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_pool = scaler.fit_transform(X_pool)
    X_test = scaler.transform(X_test)
    X_pool = np.nan_to_num(X_pool, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_pool.astype(np.float64), X_test.astype(np.float64)


def _load_vision_dataset(
    name: str,
    seed: int,
    max_pool_samples: int | None,
    max_test_samples: int | None,
    data_root: str | Path | None,
) -> DatasetBundle:
    try:
        from influence_al.data.vision_datasets import VISION_DATASET_REGISTRY, _subsample, load_vision_dataset
    except ImportError as exc:
        raise ImportError(
            "Vision datasets require influence_al.data.vision_datasets and torchvision. "
            "Sync vision_datasets.py to your workspace or install with: pip install -e '.[vision]'"
        ) from exc

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

    _task_kind, source = DATASET_REGISTRY[name]

    if source == "tabular":
        X_pool, y_pool, X_test, y_test, task, feature_names = _load_tabular_dataset(
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

    return _load_vision_dataset(
        name,
        seed=seed,
        max_pool_samples=max_pool_samples,
        max_test_samples=max_test_samples,
        data_root=data_root,
    )
