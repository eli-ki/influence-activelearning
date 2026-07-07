"""Tests for batch selection."""

import numpy as np

from influence_al.acquisition.diversity import BatchSelector, BatchSelectorConfig


class DummyTrainer:
    def leaf_index_embedding(self, model, X):
        return X


def test_batch_selector_handles_1d_degenerate_embedding():
    n = 100
    indices = np.arange(n)
    scores = np.zeros(n)
    scores[:10] = 1.0
    X = np.random.randn(n, 5)
    selector = BatchSelector(BatchSelectorConfig(prefilter_multiplier=3, embedding="features"))
    batch = selector.select(indices, scores, X, model=None, trainer=DummyTrainer(), batch_size=10)
    assert len(batch) == 10

    n = 100
    indices = np.arange(n)
    scores = np.random.randn(n)
    X = np.random.randn(n, 4)
    selector = BatchSelector(BatchSelectorConfig(prefilter_multiplier=3))
    batch = selector.select(indices, scores, X, model=None, trainer=DummyTrainer(), batch_size=10)
    assert len(batch) == 10
    assert len(np.unique(batch)) == 10
