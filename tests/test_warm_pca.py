"""The warm/prepare path fits each requested PCA size once, and does so stably.

Covers the ``prepare`` hook seam on the space registry. sklearn's PCA is not basis-stable across
``n_components`` (the solver switches, and randomized SVD is not nested), so each requested
``pca_<k>`` must get its own fit — a smaller ``pca_k`` may never be sliced out of a larger fit, or
its result silently changes and desyncs anything already projected through the old basis. The tests
check that every requested size is fit (deterministically, independent of iteration order), that a
given ``pca_k`` resolves to the same basis regardless of whether a larger fit also exists, and that
correctness does not depend on the hook (lazy use with no warm still works).
"""

from __future__ import annotations

import numpy as np
from conftest import make_cfg, make_dataset

from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.protocols.resolve import resolve_protocols


def _spy_fit_pca(ctx):
    """Record the ``n_components`` of every ``_fit_pca`` call on ``ctx``."""
    calls: list[int] = []
    orig = ctx._fit_pca

    def spy(n_components):
        calls.append(n_components)
        return orig(n_components)

    ctx._fit_pca = spy  # instance attribute shadows the bound method
    return calls


def _ctx(ng=120):
    # ng large enough that pca_50 and pca_100 are both valid (k <= min(n_cells, n_genes)) and,
    # with ng=120, that pca_50 uses the randomized solver while pca_100 uses full — the non-nested
    # regime where slicing pca_50 out of the pca_100 fit would change its result.
    cfg = make_cfg()
    return Context(Dataset(make_dataset(ng=ng), cfg), cfg)


def test_warm_fits_each_requested_size():
    """Two PCA dims requested (pca_50 + pca_100): each size is fit once, none reused by slicing."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    protocols = resolve_protocols(["energy_distance_pca_k=50", "unbiased_mmd_median_pca_k=100"])
    ctx.warm(protocols)

    assert sorted(calls) == [50, 100], f"expected one fit per size, got {calls}"

    # Exercising the projections for both dims must not trigger any further fit (both are cached).
    ctx.reference_projection("pca_50")
    ctx.reference_projection("pca_100")
    assert sorted(calls) == [50, 100], f"projection triggered a refit: {calls}"


def test_warm_is_order_independent():
    """Every requested size is fit regardless of which spec comes first."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    ctx.warm(resolve_protocols(["unbiased_mmd_median_pca_k=100", "energy_distance_pca_k=50"]))
    assert sorted(calls) == [50, 100]


def test_pca_50_basis_stable_across_a_larger_fit():
    """A given ``pca_k`` resolves to the same basis whether or not a larger fit also exists.

    Regression for the refit-desync bug: fitting ``pca_100`` must not change what ``pca_50``
    (and its cached reference projection) yields.
    """
    alone = _ctx()
    proj_alone = alone.reference_projection("pca_50")

    with_larger = _ctx()
    with_larger.pca(100)  # a larger fit exists first
    proj_after = with_larger.reference_projection("pca_50")  # must still use pca_50's own basis

    assert proj_alone.shape == proj_after.shape == (len(alone.reference().cells), 50)
    assert np.allclose(proj_alone, proj_after), "pca_50 desynced after a larger fit was created"


def test_lazy_pca_without_warm_is_correct():
    """Correctness must not depend on the hook: no warm, projections still compute."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    proj = ctx.reference_projection("pca_50")
    assert proj.shape[0] == len(ctx.reference().cells)
    assert proj.shape[1] == 50
    assert np.isfinite(proj).all()
    # A single lazy fit happened (floored at 50), and it was not pre-warmed.
    assert calls == [50]


def test_single_dim_case_fits_once_at_floor():
    """The common single-dimension case still fits exactly once (no added work)."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    ctx.warm(resolve_protocols(["energy_distance_pca_k=50"]))
    ctx.reference_projection("pca_50")
    assert calls == [50]
