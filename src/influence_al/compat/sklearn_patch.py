"""Shims for scikit-learn API changes used by tree-influence."""

from __future__ import annotations

import inspect
import sys


def patch_sklearn_one_hot_encoder() -> None:
    """
    tree-influence calls OneHotEncoder(..., sparse=False), removed in sklearn 1.4.
    Map legacy `sparse` to `sparse_output` when needed.
    """
    import sklearn.preprocessing as preprocessing

    if getattr(preprocessing, "_influence_al_sparse_patched", False):
        return

    base_cls = preprocessing.OneHotEncoder
    base_params = inspect.signature(base_cls.__init__).parameters
    if "sparse" in base_params:
        preprocessing._influence_al_sparse_patched = True
        return

    class _LegacySparseOneHotEncoder:
        def __new__(cls, **kwargs):
            sparse = kwargs.pop("sparse", None)
            if sparse is not None:
                kwargs.setdefault("sparse_output", sparse)
            return base_cls(**kwargs)

    preprocessing.OneHotEncoder = _LegacySparseOneHotEncoder
    preprocessing._influence_al_sparse_patched = True

    util_module = sys.modules.get("tree_influence.explainers.parsers.util")
    if util_module is not None:
        util_module.OneHotEncoder = _LegacySparseOneHotEncoder
