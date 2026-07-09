"""Tests for ActiveLearningPool."""

import numpy as np
import pytest

from influence_al.data.pool import ActiveLearningPool, _validate_pool_fractions


def test_pool_init_disjoint_partition():
    X = np.random.randn(200, 5)
    y = np.random.randint(0, 3, 200)
    pool = ActiveLearningPool.from_pool(X, y, initial_labeled_fraction=0.1, r_ref_fraction=0.05, seed=0)

    n_l0 = len(pool.labeled_indices)
    n_r_ref = len(pool.r_ref_indices)
    n_u0 = len(pool.unlabeled_indices)
    assert n_l0 + n_r_ref + n_u0 == 200
    assert n_l0 >= 2
    assert n_r_ref >= 1
    assert n_u0 >= 1

    assert not np.any(np.isin(pool.r_ref_indices, pool.labeled_indices))
    assert not np.any(np.isin(pool.r_ref_indices, pool.unlabeled_indices))
    assert not np.any(np.isin(pool.labeled_indices, pool.unlabeled_indices))

    r_ref_at_init = pool.r_ref_indices.copy()
    u_idx = pool.unlabeled_indices[:5]
    pool.query(u_idx)
    assert np.array_equal(pool.r_ref_indices, r_ref_at_init)
    assert len(pool.labeled_indices) == n_l0 + 5
    assert len(pool.unlabeled_indices) == n_u0 - 5


def test_l_train_equals_labeled():
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    pool = ActiveLearningPool.from_pool(X, y, initial_labeled_fraction=0.2, r_ref_fraction=0.1, seed=1)
    train_idx = pool.get_l_train_indices()
    assert np.array_equal(train_idx, pool.labeled_indices)
    assert not np.any(np.isin(train_idx, pool.r_ref_indices))
    X_tr, y_tr = pool.get_l_train()
    assert len(X_tr) == len(pool.labeled_indices)


def test_cannot_query_r_ref():
    X = np.random.randn(50, 2)
    y = np.zeros(50)
    pool = ActiveLearningPool.from_pool(X, y, initial_labeled_fraction=0.2, r_ref_fraction=0.1, seed=0)
    with pytest.raises(ValueError, match="Cannot query R_ref"):
        pool.query(pool.r_ref_indices[:1])


def test_validate_pool_fractions_rejects_overlap_budget():
    with pytest.raises(ValueError, match="must be < 1.0"):
        _validate_pool_fractions(0.6, 0.5)


def test_pool_too_small_for_fractions():
    X = np.random.randn(5, 2)
    y = np.zeros(5)
    with pytest.raises(ValueError, match="Pool too small"):
        ActiveLearningPool.from_pool(X, y, initial_labeled_fraction=0.8, r_ref_fraction=0.15, seed=0)
