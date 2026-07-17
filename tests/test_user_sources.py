"""Runtime user sources (``prepare(sources=)``) as controls and as ``center_on`` centering variants."""

from __future__ import annotations

import numpy as np
import pytest

import scperteval as sp

PREP = dict(subsample=400, seed=0, min_cells=10, workers=1)


def _centroid(adata) -> np.ndarray:
    """A sensible (finite, non-degenerate) 1-D baseline: the dataset's global mean."""
    return np.asarray(adata.X, dtype=np.float64).mean(0)


def _finite(agg: dict) -> bool:
    return bool(np.isfinite(list(agg.values())).all())


# --- center_on mints a named variant -----------------------------------------


def test_center_on_user_source_mints_named_variant(dataset_adata):
    prep = sp.prepare(dataset_adata, "pearson", sources={"myref": _centroid(dataset_adata)}, **PREP)
    r = sp.calibrate(prep, "pearson", center_on="myref")
    assert list(r.per_perturbation["protocol"].unique()) == ["pearson_center_myref"]  # variant, not "pearson"
    assert _finite(r.aggregate)


def test_center_on_builtin_source(dataset_adata):
    # a built-in centroid source proves center_on is not user-source-specific (global_mean isn't the
    # default negative, so centering on it doesn't zero the negative control out).
    prep = sp.prepare(dataset_adata, "pearson", **PREP)
    r = sp.calibrate(prep, "pearson", center_on="global_mean")
    assert list(r.per_perturbation["protocol"].unique()) == ["pearson_center_global_mean"]
    assert _finite(r.aggregate)


def test_center_on_in_score_mode(dataset_adata, predictions_factory):
    prep = sp.prepare(dataset_adata, "pearson", sources={"myref": _centroid(dataset_adata)}, **PREP)
    r = sp.score(prep, "pearson", predictions_factory(dataset_adata), center_on="myref")
    assert list(r.per_perturbation["protocol"].unique()) == ["pearson_center_myref"]
    assert _finite(r.aggregate)


# --- user sources as controls (via #24's runtime override) -------------------


def test_user_centroid_as_negative_control(dataset_adata):
    prep = sp.prepare(dataset_adata, "pearson", sources={"myref": _centroid(dataset_adata)}, **PREP)
    r = sp.calibrate(prep, "pearson", negative="myref")
    assert _finite(r.aggregate)
    assert list(r.per_perturbation["negative"].unique()) == ["myref"]  # recorded as the resolved negative


def test_user_cells_as_control_for_population_protocol(dataset_adata):
    cells = np.asarray(dataset_adata.X, dtype=np.float64)[:40]  # a (n, G) population
    prep = sp.prepare(dataset_adata, "energy_distance_pca_k", sources={"mycells": cells}, **PREP)
    r = sp.calibrate(prep, "energy_distance_pca_k", negative="mycells")
    assert _finite(r.aggregate)
    assert list(r.per_perturbation["negative"].unique()) == ["mycells"]


# --- handles are isolated; the global registry is never mutated --------------


def test_sources_do_not_leak_across_handles(dataset_adata):
    from scperteval.sources import SOURCES

    g = dataset_adata.n_vars
    before = set(SOURCES.names())
    p1 = sp.prepare(dataset_adata, [], sources={"s": np.ones(g)}, **PREP)
    p2 = sp.prepare(dataset_adata, [], sources={"s": np.full(g, 5.0)}, **PREP)
    assert set(SOURCES.names()) == before  # global registry untouched
    assert "s" not in SOURCES
    assert p1._sources["s"][0](None, None)[0] == 1.0 and p2._sources["s"][0](None, None)[0] == 5.0


def test_source_array_is_copied_not_aliased(dataset_adata):
    g = dataset_adata.n_vars
    src = np.asfortranarray(np.arange(2 * g, dtype=np.int32).reshape(2, g))  # non-contiguous, wrong dtype
    prep = sp.prepare(dataset_adata, [], sources={"c": src}, **PREP)
    stored = prep._sources["c"][0](None, None)
    src[0, 0] = 999  # mutating the caller's array must not touch the stored copy
    assert stored[0, 0] != 999.0
    assert stored.dtype == np.float64 and stored.flags["C_CONTIGUOUS"]


# --- registration validation errors ------------------------------------------


@pytest.mark.parametrize(
    ("sources", "match"),
    [
        (lambda g: {"x": np.ones(g + 1)}, "genes but the dataset has"),
        (lambda g: {"x": np.array([np.nan] + [0.0] * (g - 1))}, "non-finite"),
        (lambda g: {"x": np.ones(g, dtype=np.complex128)}, "real-valued numeric"),
        (lambda g: {"control": np.ones(g)}, "shadows a built-in"),
        (lambda g: {"auto": np.ones(g)}, "'auto' is reserved"),
    ],
)
def test_source_validation_errors(dataset_adata, sources, match):
    with pytest.raises((ValueError, TypeError), match=match):
        sp.prepare(dataset_adata, [], sources=sources(dataset_adata.n_vars), **PREP)


# --- center_on validation errors ---------------------------------------------


def test_center_on_rejects_population_protocol(dataset_adata):
    prep = sp.prepare(dataset_adata, "energy_distance_pca_k", sources={"myref": _centroid(dataset_adata)}, **PREP)
    with pytest.raises(ValueError, match="only applies to centroid"):
        sp.calibrate(prep, "energy_distance_pca_k", center_on="myref")


def test_center_on_rejects_de_protocol(dataset_adata):
    prep = sp.prepare(dataset_adata, "de_auprc", sources={"myref": _centroid(dataset_adata)}, **PREP)
    with pytest.raises(ValueError, match="only applies to centroid"):
        sp.calibrate(prep, "de_auprc", center_on="myref")


def test_center_on_rejects_already_centered_base(dataset_adata):
    prep = sp.prepare(dataset_adata, "pearson_ctrl", sources={"myref": _centroid(dataset_adata)}, **PREP)
    with pytest.raises(ValueError, match="already centers on"):
        sp.calibrate(prep, "pearson_ctrl", center_on="myref")


def test_center_on_rejects_cells_source(dataset_adata):
    cells = np.asarray(dataset_adata.X, dtype=np.float64)[:40]
    prep = sp.prepare(dataset_adata, "pearson", sources={"mycells": cells}, **PREP)
    with pytest.raises(ValueError, match="must be a centroid"):
        sp.calibrate(prep, "pearson", center_on="mycells")


def test_center_on_rejects_unknown_source(dataset_adata):
    prep = sp.prepare(dataset_adata, "pearson", **PREP)
    with pytest.raises(ValueError, match="not registered"):
        sp.calibrate(prep, "pearson", center_on="nope")
