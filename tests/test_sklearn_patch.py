"""Tests for scikit-learn compatibility shims."""

import numpy as np
import sklearn.preprocessing as preprocessing

from influence_al.compat.sklearn_patch import patch_sklearn_one_hot_encoder


def test_one_hot_encoder_accepts_legacy_sparse_kwarg():
    patch_sklearn_one_hot_encoder()
    OneHotEncoder = preprocessing.OneHotEncoder
    y = np.array([[0], [1], [2]])
    encoder = OneHotEncoder(categories=[list(range(3))], sparse=False, dtype=np.float64)
    encoded = encoder.fit_transform(y)
    assert encoded.shape == (3, 3)
    assert isinstance(encoder, preprocessing._encoders.OneHotEncoder)
