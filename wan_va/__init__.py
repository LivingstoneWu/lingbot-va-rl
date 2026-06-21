# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

__all__ = ["configs", "distributed", "modules", "rl"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
