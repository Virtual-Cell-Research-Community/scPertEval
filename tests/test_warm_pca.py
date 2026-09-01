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

from scperteval.blocks.spaces import SPACES
from scperteval.blocks.spaces.helpers import fitted_pca, pca_for
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.protocols.resolve import resolve_protocols


def _fits(ctx):
    """The PCA fit sizes on ``ctx``, one entry per size actually fit.

    Read off the shared cache rather than by instrumenting the fit: a cached computation runs
    its body only on a miss, so one memo entry per size *is* one fit per size.
    """
    body = fitted_pca.__wrapped__  # the undecorated function the cache keys on
    return sorted(params[0] for fn, params in ctx._store.memo if fn is body)


def _ctx(ng=120):
    # ng large enough that pca_50 and pca_100 are both valid (k <= min(n_cells, n_genes)) and,
    # with ng=120, that pca_50 uses the randomized solver while pca_100 uses full — the non-nested
    # regime where slicing pca_50 out of the pca_100 fit would change its result.
    cfg = make_cfg()
    # Instances are created on demand, so register the sizes these tests look up by name --
    # otherwise a test only passes when some earlier test in the same process resolved pca_k first.
    SPACES.instance("pca", 50)
    SPACES.instance("pca", 100)
    return Context(Dataset(make_dataset(ng=ng), cfg), cfg)


def test_warm_fits_each_requested_size():
    """Two PCA dims requested (pca_50 + pca_100): each size is fit once, none reused by slicing."""
    ctx = _ctx()
    protocols = resolve_protocols(["energy_distance_pca_k=50", "unbiased_mmd_median_pca_k=100"])
    ctx.warm(protocols)

    assert _fits(ctx) == [50, 100], f"expected one fit per size, got {_fits(ctx)}"

    # Exercising the projections for both dims must not trigger any further fit (both are cached).
    ctx.reference_projection("pca_50")
    ctx.reference_projection("pca_100")
    assert _fits(ctx) == [50, 100], f"projection triggered a refit: {_fits(ctx)}"


def test_warm_is_order_independent():
    """Every requested size is fit regardless of which spec comes first."""
    ctx = _ctx()
    ctx.warm(resolve_protocols(["unbiased_mmd_median_pca_k=100", "energy_distance_pca_k=50"]))
    assert _fits(ctx) == [50, 100]


def test_pca_50_basis_stable_across_a_larger_fit():
    """A given ``pca_k`` resolves to the same basis whether or not a larger fit also exists.

    Regression for the refit-desync bug: fitting ``pca_100`` must not change what ``pca_50``
    (and its cached reference projection) yields.
    """
    alone = _ctx()
    proj_alone = alone.reference_projection("pca_50")

    with_larger = _ctx()
    pca_for(with_larger, 100)  # a larger fit exists first
    proj_after = with_larger.reference_projection("pca_50")  # must still use pca_50's own basis

    assert proj_alone.shape == proj_after.shape == (len(alone.reference().cells), 50)
    assert np.allclose(proj_alone, proj_after), "pca_50 desynced after a larger fit was created"


def test_lazy_pca_without_warm_is_correct():
    """Correctness must not depend on the hook: no warm, projections still compute."""
    ctx = _ctx()
    proj = ctx.reference_projection("pca_50")
    assert proj.shape[0] == len(ctx.reference().cells)
    assert proj.shape[1] == 50
    assert np.isfinite(proj).all()
    # A single lazy fit happened (floored at 50), and it was not pre-warmed.
    assert _fits(ctx) == [50]


def test_single_dim_case_fits_once_at_floor():
    """The common single-dimension case still fits exactly once (no added work)."""
    ctx = _ctx()
    ctx.warm(resolve_protocols(["energy_distance_pca_k=50"]))
    ctx.reference_projection("pca_50")
    assert _fits(ctx) == [50]
