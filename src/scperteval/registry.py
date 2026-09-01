"""A minimal decorator registry for the pluggable building blocks."""

from __future__ import annotations

from collections.abc import Callable


class Registry:
    """Maps a name to a function plus optional metadata, populated by decoration.

    Parameters
    ----------
    kind : str
        Human-readable label used in error messages (e.g. ``"de-method"``).

    Example
    -------
    >>> from scperteval.registry import Registry
    >>> MY_REG = Registry("example")
    >>> @MY_REG.register("double", description="multiply by 2")
    ... def double(x):
    ...     return x * 2
    >>> MY_REG["double"](3)
    6
    >>> MY_REG.meta("double")
    {'description': 'multiply by 2'}
    >>> MY_REG.names()
    ['double']
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, tuple[Callable, dict]] = {}

    def register(self, name: str, **meta) -> Callable:
        """Decorator that registers a function under ``name`` with optional metadata."""

        def deco(fn: Callable) -> Callable:
            self._items[name] = (fn, meta)
            return fn

        return deco

    def add(self, name: str, fn: Callable, **meta) -> None:
        """Register a function under ``name`` without using the decorator form."""
        self._items[name] = (fn, meta)

    def __getitem__(self, name: str) -> Callable:
        """Return the function registered under ``name``."""
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._items[name][0]

    def meta(self, name: str) -> dict:
        """Return the metadata dict registered alongside ``name``."""
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._items[name][1]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        """Sorted list of registered names."""
        return sorted(self._items)
