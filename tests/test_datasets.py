"""Dataset loader tests."""

import os
from pathlib import Path

import pytest

from influence_al.data.datasets import list_datasets, load_dataset


@pytest.mark.parametrize("name", list_datasets())
def test_registry_lists_ads_ralif_datasets(name):
    assert name in {
        "cifar10",
        "cifar100",
        "cinic10",
        "tiny_imagenet",
        "svhn",
        "svhn_extra",
        "fashion_mnist",
        "inaturalist",
    }


def _fashion_mnist_cached() -> bool:
    root = Path(os.environ.get("INFLUENCE_AL_DATA_DIR", "data"))
    return (root / "FashionMNIST").exists()


def test_load_fashion_mnist_subsampled():
    pytest.importorskip("torchvision")
    if not _fashion_mnist_cached():
        pytest.skip("Fashion-MNIST not cached locally")
    data = load_dataset(
        "fashion_mnist",
        seed=42,
        max_pool_samples=256,
        max_test_samples=512,
    )
    assert data.task == "classification"
    assert data.X_pool.shape[0] == 256
    assert data.X_pool.shape[1] == 28 * 28
    assert data.X_test.shape[0] == 512
    assert data.n_classes == 10
    assert "RALIF" in data.papers
