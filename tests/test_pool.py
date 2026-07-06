"""Tests for ActiveLearningPool."""

import numpy as np
import pytest

from influence_al.data.pool import ActiveLearningPool


def test_pool_init_and_r_ref_fixed():
    X = np.random.randn(200, 5)
    y = np.random.randint(0, 3, 200)
    pool = ActiveLearningPool.from_pool(X, y, initial_labeled_fraction=0.1, r_ref_fraction=0.2, seed=0)
    assert len(pool.labeled_indices) + len(pool.unlabeled_indices) == 200
    assert len(pool.r_ref_indices) >= 1
    assert np.all(np.isin(pool.r_ref_indices, pool.labeled_indices))
    r_ref_at_init = pool.r_ref_indices.copy()

    u_idx = pool.unlabeled_indices[:5]
    pool.query(u_idx)
    assert np.array_equal(pool.r_ref_indices, r_ref_at_init)
    assert len(pool.labeled_indices) == 20 + 5


def test_l_train_excludes_r_ref():
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    pool = ActiveLearningPool.from_pool(X, y, initial_labeled_fraction=0.2, r_ref_fraction=0.25, seed=1)
    train_idx = pool.get_l_train_indices()
    assert not np.any(np.isin(train_idx, pool.r_ref_indices))
    X_tr, y_tr = pool.get_l_train()
    assert len(X_tr) == len(pool.labeled_indices) - len(pool.r_ref_indices)


def test_cannot_query_r_ref():
    X = np.random.randn(50, 2)
    y = np.zeros(50)
    pool = ActiveLearningPool.from_pool(X, y, initial_labeled_fraction=0.2, r_ref_fraction=0.5, seed=0)
    with pytest.raises(ValueError):
        pool.query(pool.r_ref_indices[:1])
