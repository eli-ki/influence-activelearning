"""Compatibility shims for third-party dependencies."""

from influence_al.compat.sklearn_patch import patch_sklearn_one_hot_encoder

__all__ = ["patch_sklearn_one_hot_encoder"]
