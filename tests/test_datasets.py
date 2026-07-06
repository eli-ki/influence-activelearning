"""Dataset loader tests."""

from influence_al.data.datasets import load_dataset


def test_load_breast_cancer_feature_names():
    data = load_dataset("breast_cancer", seed=42)
    assert data.feature_names is not None
    assert len(data.feature_names) == data.X_pool.shape[1]
    assert data.feature_names[0] == "mean radius"
