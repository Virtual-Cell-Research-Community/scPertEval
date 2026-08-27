"""The optional `sinkhorn` extra: the metric itself, and how a base install degrades.

torch/geomloss are an optional extra, so two behaviours need guarding: the Sinkhorn metric
must actually work when the extra *is* installed (CI installs it via
``envs.hatch-test.features``), and every entry point must degrade cleanly when it is not —
bulk selectors skip, explicit requests raise an actionable error.
"""

from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager
from importlib.util import find_spec

import numpy as np
import pytest

from scperteval.protocols import metrics as M
from scperteval.protocols import resolve
from scperteval.protocols.resolve import resolve_protocols

_EXTRA_MODULES = ("torch", "geomloss")
has_sinkhorn = all(find_spec(m) is not None for m in _EXTRA_MODULES)
needs_sinkhorn = pytest.mark.skipif(not has_sinkhorn, reason="requires the optional `sinkhorn` extra")


@contextmanager
def hidden_modules(*names: str):
    """Make ``import <name>`` raise ModuleNotFoundError, as on an install without the extra."""
    real_import = builtins.__import__
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in names}

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in names:
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}", name=name.split(".")[0])
        return real_import(name, *args, **kwargs)

    for key in saved:
        del sys.modules[key]
    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


@contextmanager
def warnings_as_errors():
    """Turn any warning into a failure, to assert a code path stays silent."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture
def without_extra(monkeypatch):
    """Report torch/geomloss as absent to the resolver's availability check."""
    real_find_spec = resolve.find_spec
    monkeypatch.setattr(resolve, "find_spec", lambda name: None if name in _EXTRA_MODULES else real_find_spec(name))


# --- the metric itself (only meaningful with the extra installed) ---


@needs_sinkhorn
def test_sinkhorn_w2_is_zero_for_identical_populations():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    assert M.sinkhorn_w2(x, x.copy(), None) == pytest.approx(0.0, abs=1e-3)


@needs_sinkhorn
def test_sinkhorn_w2_grows_with_separation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    near = M.sinkhorn_w2(x, x + 1.0, None)
    far = M.sinkhorn_w2(x, x + 5.0, None)
    assert 0.0 < near < far


@needs_sinkhorn
def test_sinkhorn_w2_is_symmetric():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(40, 6)).astype(np.float32), rng.normal(size=(50, 6)).astype(np.float32)
    assert M.sinkhorn_w2(x, y, None) == pytest.approx(M.sinkhorn_w2(y, x, None), rel=1e-4)


def test_sinkhorn_w2_returns_nan_for_an_empty_population():
    # the empty guard short-circuits before the optional import, so this holds either way
    empty = np.zeros((0, 4), dtype=np.float32)
    assert np.isnan(M.sinkhorn_w2(empty, np.ones((3, 4), dtype=np.float32), None))


def test_sinkhorn_w2_without_the_extra_raises_an_actionable_error():
    x = np.ones((3, 4), dtype=np.float32)
    with hidden_modules(*_EXTRA_MODULES), pytest.raises(ModuleNotFoundError, match=r"scperteval\[sinkhorn\]"):
        M.sinkhorn_w2(x, x, None)


# --- protocol selection on an install without the extra ---


def _names(specs):
    return [p.name for p in resolve_protocols(specs)]


def test_all_skips_sinkhorn_and_warns_when_the_extra_is_missing(without_extra):
    with pytest.warns(UserWarning, match=r"scperteval\[sinkhorn\]"):
        names = _names(["all"])
    assert not [n for n in names if n.startswith("sinkhorn_w2")]
    assert "energy_distance_pca_k=50" in names  # the rest of the table still resolves


def test_group_selection_skips_sinkhorn_when_the_extra_is_missing(without_extra):
    with pytest.warns(UserWarning, match="sinkhorn_w2_top_k"):
        names = _names(["distributional"])
    assert names and not [n for n in names if n.startswith("sinkhorn_w2")]


def test_selection_without_sinkhorn_protocols_does_not_warn(without_extra):
    with warnings_as_errors():
        assert _names(["pseudobulk"])


@pytest.mark.parametrize("spec", ["sinkhorn_w2_pca_k", "sinkhorn_w2_pca_k=30"])
def test_naming_sinkhorn_explicitly_raises_when_the_extra_is_missing(without_extra, spec):
    with pytest.raises(ValueError, match=r"scperteval\[sinkhorn\]"):
        resolve_protocols([spec])


@needs_sinkhorn
def test_all_includes_sinkhorn_when_the_extra_is_installed():
    assert [n for n in _names(["all"]) if n.startswith("sinkhorn_w2")]
