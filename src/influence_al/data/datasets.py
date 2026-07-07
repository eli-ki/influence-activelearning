"""Dataset loaders for tabular classification and regression benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

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


DATASET_REGISTRY = {
    "iris": ("classification", "sklearn"),
    "breast_cancer": ("classification", "sklearn"),
    "credit_g": ("classification", "openml"),
    "phoneme": ("classification", "openml"),
    "diabetes": ("regression", "sklearn"),
    "california_housing": ("regression", "sklearn"),
}


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
    }
    data = fetch_openml(openml_map.get(name, name), version=1, as_frame=True, parser="auto")
    X = data.data.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float64)
    y = data.target.values
    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def load_dataset(
    name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_pool_samples: int | None = None,
) -> DatasetBundle:
    """Load a benchmark dataset; split into pool P and test T."""
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    task, source = DATASET_REGISTRY[name]

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
        X, y, test_size=test_fraction, random_state=seed, stratify=y if task == "classification" else None
    )

    scaler = StandardScaler()
    X_pool = scaler.fit_transform(X_pool)
    X_test = scaler.transform(X_test)

    return DatasetBundle(
        X_pool=X_pool.astype(np.float64),
        y_pool=y_pool,
        X_test=X_test.astype(np.float64),
        y_test=y_test,
        task=task,
        name=name,
        feature_names=feature_names,
    )
