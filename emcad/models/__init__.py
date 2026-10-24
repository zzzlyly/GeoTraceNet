from __future__ import annotations

from importlib import import_module

__all__ = ["builtin_models"]


def __getattr__(name: str):
    if name == "builtin_models":
        module = import_module("emcad.models.builtin")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
