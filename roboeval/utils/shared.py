"""Shared utils."""
from __future__ import annotations

import inspect
from importlib import import_module
from pkgutil import walk_packages
from types import ModuleType
from typing import Optional, Any, Type


def find_class_in_module(module: ModuleType, class_name: str) -> Optional[Type]:
    """Find a class by its name in a module."""
    return _find_member_in_module(module, class_name, inspect.isclass)


def find_constant_in_module(module: ModuleType, constant_name: str) -> Optional[Any]:
    """Find a constant by its name in a module."""
    return _find_member_in_module(module, constant_name)


def _find_member_in_module(
    module: ModuleType, member_name: str, predicate=None
) -> Optional[Any]:
    if hasattr(module, "__path__"):
        for loader, name, is_pkg in walk_packages(
            module.__path__, module.__name__ + "."
        ):
            import_module(name)
    members = inspect.getmembers(module, predicate)
    for name, value in members:
        if name == member_name:
            return value
    parent_name = f"{module.__name__}."
    submodules = [
        submodule
        for _, submodule in inspect.getmembers(module, inspect.ismodule)
        if submodule.__name__.startswith(parent_name)
    ]
    for submodule in submodules:
        cls = _find_member_in_module(submodule, member_name, predicate)
        if cls:
            return cls
    return None
