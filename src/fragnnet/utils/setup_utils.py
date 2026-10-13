"""Model and dataset class lookup by ``model_type``.

Thin wrappers over :mod:`fragnnet.registry`, kept so existing scripts and tests that import
these names from here keep working. The registry imports a class only when it is requested.
"""

from fragnnet.registry import get_dataset_cls, get_model_cls

__all__ = ["get_dataset_cls", "get_model_cls"]
