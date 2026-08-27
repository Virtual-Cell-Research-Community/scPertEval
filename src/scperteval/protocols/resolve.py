"""Resolve protocol specs (the ``-p`` DSL) into concrete :class:`~scperteval.types.Protocol` objects.

Shared by the CLI and the Python API so both accept the exact same spec language: ``"all"``,
a group name (``pseudobulk``/``distributional``/``de``), a bare protocol name, or a tunable
protocol with a value (``name=value``, e.g. ``mse_top_k=30``).
"""

from __future__ import annotations

import warnings
from importlib.util import find_spec

from ..types import Protocol
from .table import GROUPS, PROTOCOLS, TABLE

#: Modules each optional extra installs, for ``Protocol.requires_extra`` availability checks.
EXTRA_MODULES: dict[str, tuple[str, ...]] = {"sinkhorn": ("torch", "geomloss")}


def _extra_installed(extra: str) -> bool:
    """Whether every module of an optional extra is importable."""
    return all(find_spec(module) is not None for module in EXTRA_MODULES[extra])


def available(p: Protocol) -> bool:
    """Whether ``p``'s optional dependencies are installed (always ``True`` if it needs none)."""
    return p.requires_extra is None or _extra_installed(p.requires_extra)


def _install_hint(extra: str) -> str:
    return f"install them with:  pip install 'scperteval[{extra}]'"


def _concrete(p: Protocol) -> Protocol:
    """A tunable protocol at its default value; a fixed protocol unchanged."""
    return p.resolve(p.param.default) if p.parameterised else p  # type: ignore[union-attr]


def _select(candidates: list[Protocol], token: str) -> list[Protocol]:
    """Concretise a bulk selection, dropping protocols whose optional extra is missing.

    ``all`` and the group names are *bulk* selectors, so a missing optional dependency skips
    the affected protocols with a warning rather than failing the run — otherwise a base
    install could not run ``-p all`` at all. Naming a protocol explicitly still raises.
    """
    usable = [p for p in candidates if available(p)]
    skipped = [p for p in candidates if not available(p)]
    if skipped:
        extras = sorted({p.requires_extra for p in skipped if p.requires_extra})
        names = ", ".join(p.name for p in skipped)
        warnings.warn(
            f"{token!r} skipped {len(skipped)} protocol(s) needing optional dependencies: {names}. "
            f"To include them, {'; '.join(_install_hint(e) for e in extras)}",
            stacklevel=2,
        )
    return [_concrete(p) for p in usable]


def _check_available(p: Protocol) -> None:
    """Raise a clean, actionable error when an explicitly named protocol can't run."""
    if not available(p):
        assert p.requires_extra is not None  # available() is only False when an extra is required
        raise ValueError(
            f"protocol {p.name!r} needs the optional '{p.requires_extra}' dependencies "
            f"({', '.join(EXTRA_MODULES[p.requires_extra])}); {_install_hint(p.requires_extra)}"
        )


def _resolve_token(token: str) -> list[Protocol]:
    if token == "all":
        return _select(list(TABLE), token)
    if token in GROUPS:
        return _select([p for p in TABLE if p.group == token], token)
    if "=" in token:  # a tunable protocol with a value, e.g. mse_top_k=30
        name, _, value = token.partition("=")
        p = PROTOCOLS.get(name)
        if p is None or not p.parameterised:
            raise ValueError(f"unknown tunable protocol {name!r}; try `scperteval list protocols`")
        _check_available(p)
        return [p.resolve(p.param.cast(value))]  # type: ignore[union-attr]
    p = PROTOCOLS.get(token)
    if p is None:
        raise ValueError(f"unknown protocol {token!r}; try `scperteval list protocols`")
    _check_available(p)
    return [_concrete(p)]


def resolve_protocols(specs: list[str]) -> list[Protocol]:
    """Resolve protocol specs to a de-duplicated list of concrete protocols."""
    out: list[Protocol] = []
    for spec in specs:
        for token in spec.split(","):
            token = token.strip()
            if token:
                out += _resolve_token(token)
    by_name: dict[str, Protocol] = {}
    for p in out:
        by_name.setdefault(p.name, p)
    return list(by_name.values())
