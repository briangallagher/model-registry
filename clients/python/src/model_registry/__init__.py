# src/model_registry/__init__.py

"""Shim module to expose kubeflow.model_registry under flat namespace."""

from kubeflow.model_registry import ModelRegistry, __version__

__all__ = ["ModelRegistry", "__version__"]