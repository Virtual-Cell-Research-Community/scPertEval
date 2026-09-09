"""compare(): scoring a query population against every reference perturbation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scperteval as sp
from scperteval.dataset import to_dense

PREP = dict(subsample=400, seed=0, min_cells=10, workers=1)

#: One protocol per representation, plus a weighted one (it reads ``ctx.current_pert``) and a
#: per-perturbation space (its gene panel comes from the reference).
REPRESENTATIVE = ["pearson_ctrl", "mse_top_k=50", "energy_distance_pca_k=50", "de_auprc", "wmse_exp2"]


@pytest.fixture
def prep(dataset_adata):
    return sp.prepare(dataset_adata, "all", **PREP)


def cells_of(prepared, pert):
    """A perturbation's own cells, dense — what compare() uses for a query named from the handle."""
    return np.asarray(to_dense(prepared._ds.cells(pert)), dtype=np.float64)


def as_adata(dataset_adata, cells, labels):
    a = ad.AnnData(np.asarray(cells, dtype=np.float32))
    a.var_names = list(dataset_adata.var_names)
    a.obs["perturbation"] = labels
    return a


# --- shape and inputs --------------------------------------------------------


def test_returns_query_by_reference_frame(prep):
    perts = [str(p) for p in prep._ds.perturbations]
    frame = sp.compare(prep, "pearson_ctrl", "pertA")
    assert list(frame.index) == ["pertA"]
    assert list(frame.columns) == perts
    assert (frame.index.name, frame.columns.name) == ("query", "reference")
    assert frame.to_numpy().dtype == np.float64


def test_query_input_forms_agree(prep, dataset_adata):
    """A name, a bare array and a labelled AnnData describing the same cells score identically."""
    cells = cells_of(prep, "pertA")
    by_name = sp.compare(prep, "pearson_ctrl", ["pertA"])
    by_array = sp.compare(prep, "pearson_ctrl", cells, query_origin="pertA")
    by_adata = sp.compare(prep, "pearson_ctrl", as_adata(dataset_adata, cells, ["pertA"] * len(cells)))
    assert np.allclose(by_name.to_numpy(), by_array.to_numpy())
    assert np.allclose(by_name.to_numpy(), by_adata.to_numpy())
    assert list(by_array.index) == ["query"]  # a bare array has no identity of its own


def test_multiple_queries_and_reference_subset(prep):
    frame = sp.compare(prep, "mse", ["pertA", "pertB"], references=["pertB", "pertA"])
    assert list(frame.index) == ["pertA", "pertB"]
    assert list(frame.columns) == ["pertB", "pertA"]  # caller's order, not the handle's
    # each query scores 0 (perfect) against its own reference and worse against the other
    assert frame.loc["pertA", "pertA"] == pytest.approx(0.0)
    assert frame.loc["pertB", "pertB"] == pytest.approx(0.0)
    assert frame.loc["pertA", "pertB"] > 0


def test_anndata_with_several_labels(prep, dataset_adata):
    a, b = cells_of(prep, "pertA"), cells_of(prep, "pertB")
    labels = ["pertA"] * len(a) + ["pertB"] * len(b)
    frame = sp.compare(prep, "mse", as_adata(dataset_adata, np.vstack([a, b]), labels))
    assert list(frame.index) == ["pertA", "pertB"]
    assert frame.loc["pertA", "pertA"] == pytest.approx(0.0)


def test_gene_order_is_aligned_for_anndata(prep, dataset_adata):
    """An AnnData query is aligned by gene name, so a shuffled one scores the same."""
    cells = cells_of(prep, "pertA")
    plain = as_adata(dataset_adata, cells, ["pertA"] * len(cells))
    shuffled = plain[:, np.random.default_rng(0).permutation(plain.n_vars)].copy()
    assert np.allclose(
        sp.compare(prep, "pearson_ctrl", plain).to_numpy(),
        sp.compare(prep, "pearson_ctrl", shuffled).to_numpy(),
    )


# --- retrieval behaves ------------------------------------------------------


@pytest.mark.parametrize("protocol", REPRESENTATIVE)
def test_every_representation_retrieves_its_own_reference(prep, protocol):
    """A query taken straight from the handle scores best against its own reference."""
    row = sp.compare(prep, protocol, "pertA").loc["pertA"]
    better = max if sp.api._single_protocol(protocol).better == "higher" else min
    assert better(row.items(), key=lambda kv: kv[1])[0] == "pertA"


# --- equivalence with score() ------------------------------------------------


@pytest.mark.parametrize("protocol", REPRESENTATIVE)
def test_matches_score_on_the_same_pairing(prep, dataset_adata, protocol):
    """compare(q, references=[q]) is score()'s value for q under a perfect prediction."""
    perts = [str(p) for p in prep._ds.perturbations]
    blocks = [cells_of(prep, p) for p in perts]
    labels = [p for p, c in zip(perts, blocks) for _ in range(len(c))]
    pred = as_adata(dataset_adata, np.vstack(blocks), labels)

    scored = sp.score(prep, protocol, pred).per_perturbation
    expected = float(scored.loc[scored["perturbation"] == "pertA", "raw_prediction"].iloc[0])
    got = float(sp.compare(prep, protocol, "pertA", references=["pertA"]).iloc[0, 0])
    assert got == expected


# --- the space is the reference's -------------------------------------------


def test_space_is_fit_on_the_reference(prep):
    """Each column uses its *reference's* gene panel, not one panel fit on the query.

    Both sides of every comparison are cut to the reference's top-k genes. If the panel were fit
    once on the query instead, every column would share it and the whole row would change — so
    this reproduces the per-reference panel by hand and requires an exact match.
    """
    from scperteval.blocks.spaces import SPACES

    proto = sp.api._single_protocol("mse_top_k=50")
    ctx = prep._run_context(protocols=[proto.name], calibrator="score", truth="gt_all_cells")
    query = cells_of(prep, "pertA").mean(0)

    row = sp.compare(prep, "mse_top_k=50", "pertA").loc["pertA"]
    for ref in row.index:
        ctx.current_pert = ref
        panel = SPACES.meta(proto.space)["select"](ctx, ref)  # the reference's own genes
        gt = ctx.centroid(ref, "gt_all_cells", proto.centering)[panel]
        assert row[ref] == pytest.approx(float(np.mean((gt - query[panel]) ** 2)))

    # and the panels really do differ between references, so the check above has teeth
    panels = {ref: tuple(SPACES.meta(proto.space)["select"](ctx, ref)) for ref in row.index}
    assert len(set(panels.values())) > 1


# --- query_origin ------------------------------------------------------------


def test_query_origin_changes_the_de_background(prep):
    """Excluding the query's own perturbation is not a no-op, and differs from excluding nothing."""
    cells = cells_of(prep, "pertA")
    with_origin = sp.compare(prep, "de_auprc", cells, query_origin="pertA")
    without = sp.compare(prep, "de_auprc", cells)
    assert not np.allclose(with_origin.to_numpy(), without.to_numpy())


def test_query_de_is_constant_across_references(prep):
    """The query's DE vector is computed once per call, against a background fixed by origin.

    Scored against one reference or against all of them, the query side must be identical — which
    it can only be if the background does not follow the reference.
    """
    cells = cells_of(prep, "pertA")
    full = sp.compare(prep, "de_auprc", cells, query_origin="pertA")
    for ref in full.columns:
        one = sp.compare(prep, "de_auprc", cells, references=[ref], query_origin="pertA")
        assert float(one.iloc[0, 0]) == float(full.loc["query", ref])


def test_query_named_from_handle_defaults_to_its_own_origin(prep):
    cells = cells_of(prep, "pertA")
    assert np.allclose(
        sp.compare(prep, "de_auprc", "pertA").to_numpy(),
        sp.compare(prep, "de_auprc", cells, query_origin="pertA").to_numpy(),
    )


def test_query_origin_mapping_for_several_queries(prep):
    frame = sp.compare(prep, "de_auprc", ["pertA", "pertB"], query_origin={"pertA": "pertB"})
    single = sp.compare(prep, "de_auprc", cells_of(prep, "pertA"), query_origin="pertB")
    assert np.allclose(frame.loc["pertA"].to_numpy(), single.to_numpy()[0])


# --- errors ------------------------------------------------------------------


def test_rejects_unknown_reference(prep):
    with pytest.raises(ValueError, match="references not in the prepared dataset"):
        sp.compare(prep, "mse", "pertA", references=["nope"])


def test_rejects_unknown_query_name(prep):
    with pytest.raises(ValueError, match="queries not in the prepared dataset"):
        sp.compare(prep, "mse", "nope")


def test_rejects_wrong_gene_count(prep):
    with pytest.raises(ValueError, match="genes but the dataset has"):
        sp.compare(prep, "mse", np.zeros((5, 3)))


def test_rejects_non_finite_query(prep):
    cells = cells_of(prep, "pertA").copy()
    cells[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sp.compare(prep, "mse", cells)


def test_rejects_1d_query(prep):
    with pytest.raises(ValueError, match="must be a 2-D"):
        sp.compare(prep, "mse", np.zeros(len(prep._ds.var_names)))


def test_rejects_dataset_scope_protocol(prep):
    with pytest.raises(ValueError, match="dataset-scope protocol"):
        sp.compare(prep, "rank", "pertA")


def test_rejects_unknown_origin(prep):
    with pytest.raises(ValueError, match="would exclude nothing"):
        sp.compare(prep, "de_auprc", cells_of(prep, "pertA"), query_origin="nope")


def test_rejects_scalar_origin_for_several_queries(prep):
    with pytest.raises(ValueError, match="mapping instead"):
        sp.compare(prep, "mse", ["pertA", "pertB"], query_origin="pertA")


def test_rejects_non_prepared_handle(dataset_adata):
    with pytest.raises(TypeError, match="compare\\(\\) takes a handle from prepare\\(\\)"):
        sp.compare(dataset_adata, "mse", "pertA")


def test_reserved_query_source_name(dataset_adata):
    with pytest.raises(ValueError, match="reserved"):
        sp.prepare(dataset_adata, [], sources={"_compare_query": np.zeros(dataset_adata.n_vars)}, **PREP)


# --- the query is reduced once per call --------------------------------------


def test_query_is_reduced_once_not_per_reference(prep):
    """A centroid protocol gets the query already pseudobulked, so no averaging happens per column.

    ``compare()`` exists to do the query-side work once; for the centroid path that means the
    registered source hands back a 1-D centroid, leaving ``Context.centroid`` nothing to average.
    """
    from scperteval.api import _query_source

    cells = cells_of(prep, "pertA")
    fn, meta = _query_source(cells, sp.api._single_protocol("pearson_ctrl"))
    assert meta == {"provides": "centroid", "cacheable": False}
    assert np.allclose(fn(None, "any-perturbation"), cells.mean(0))

    fn, meta = _query_source(cells, sp.api._single_protocol("energy_distance_pca_k=50"))
    assert meta == {"provides": "cells", "cacheable": False}
    assert fn(None, "any-perturbation").shape == cells.shape


def test_query_source_ignores_the_perturbation_it_is_asked_for(prep):
    """The query is constant across references — that is what makes one reduction enough."""
    from scperteval.api import _query_source

    fn, _ = _query_source(cells_of(prep, "pertA"), sp.api._single_protocol("mse"))
    assert fn(None, "pertA") is fn(None, "pertD")


# --- the query never touches the shared cache --------------------------------


def test_query_is_never_written_to_the_handle_cache(prep):
    """A query is per-call data: nothing derived from it may outlive the call on the handle."""
    sp.compare(prep, "de_auprc", cells_of(prep, "pertA"), query_origin="pertA")
    keys = repr(list(prep._store.memo))
    assert "_compare_query" not in keys


def test_center_on_mints_the_variant(prep, dataset_adata):
    vec = np.asarray(dataset_adata.X).mean(0)
    prepared = sp.prepare(dataset_adata, "all", sources={"myvec": vec}, **PREP)
    frame = sp.compare(prepared, "pearson", "pertA", center_on="myvec")
    assert frame.shape == (1, len(prepared._ds.perturbations))
    assert not np.allclose(frame.to_numpy(), sp.compare(prepared, "pearson", "pertA").to_numpy())


# --- out_dir ------------------------------------------------------------------


def test_out_dir_writes_the_matrix_round_trip(prep, tmp_path):
    """The CSV must reproduce the returned frame, query labels included."""
    frame = sp.compare(prep, "pearson_ctrl", ["pertA", "pertB"], out_dir=str(tmp_path))
    written = list(tmp_path.glob("*__compare.csv"))
    assert len(written) == 1
    back = pd.read_csv(written[0], index_col=0)
    assert list(back.index) == list(frame.index)
    assert list(back.columns) == list(frame.columns)
    np.testing.assert_allclose(back.to_numpy(), frame.to_numpy())


def test_out_dir_no_collision_across_protocols(prep, tmp_path):
    sp.compare(prep, "mse", "pertA", out_dir=str(tmp_path))
    sp.compare(prep, "pearson_ctrl", "pertA", out_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*__compare.csv"))) == 2


def test_out_dir_records_the_center_on_variant(prep, dataset_adata, tmp_path):
    """A minted `center_on` variant must reach the filename, not the base protocol name."""
    prepared = sp.prepare(dataset_adata, "all", sources={"myvec": np.asarray(dataset_adata.X).mean(0)}, **PREP)
    sp.compare(prepared, "pearson", "pertA", center_on="myvec", out_dir=str(tmp_path))
    assert list(tmp_path.glob("*pearson_center_myvec*__compare.csv"))


def test_no_out_dir_writes_nothing(prep, tmp_path):
    sp.compare(prep, "mse", "pertA")
    assert list(tmp_path.iterdir()) == []
