"""Feature spaces: the catalog rules, instance registration, and composition.

The rules are plain functions, so most tests call them directly; the registry tests cover
turning a catalog entry into a registered instance.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import anndata as ad
import numpy as np
import pytest
from conftest import make_cfg, make_dataset

from scperteval.blocks.spaces import OPS, SPACES
from scperteval.blocks.spaces.catalog import full, heg, hvg, perturbed_genes
from scperteval.blocks.spaces.helpers import targeted_genes
from scperteval.caching import cached
from scperteval.calibrators import CALIBRATORS
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.predictions import PredictionSet
from scperteval.protocols.table import PROTOCOLS
from scperteval.runner import run_protocol


def test_heg_picks_highest_control_expression_genes():
    rng = np.random.default_rng(0)
    ng = 10
    ctrl_means = np.array([1.0, 5.0, 2.0, 9.0, 0.5, 3.0, 8.0, 4.0, 7.0, 6.0])
    ctrl = rng.poisson(ctrl_means, size=(300, ng)).astype(np.float32)
    pert = rng.poisson(ctrl_means, size=(80, ng)).astype(np.float32)
    adata = ad.AnnData(np.vstack([ctrl, pert]))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 300 + ["pertA"] * 80

    cfg = make_cfg(min_cells=10)
    ctx = Context(Dataset(adata, cfg), cfg)
    # the 3 highest control-expressed genes are g3 (9.0), g6 (8.0), g8 (7.0), in that order
    assert list(np.argsort(-ctrl_means)[:3]) == [3, 6, 8]
    assert heg(ctx, 3).tolist() == [3, 6, 8]


def test_hvg_picks_highest_dispersion_genes():
    # scanpy's "seurat" flavor bins genes by mean expression (20 bins by default) before
    # z-scoring dispersion within each bin, so this needs enough genes spread across a
    # realistic mean range for the binning to be meaningful -- a handful of genes gives
    # near-empty, degenerate bins.
    rng = np.random.default_rng(0)
    ng, n_ctrl, n_top = 200, 800, 5
    base_means = rng.uniform(0.5, 20.0, size=ng)
    base = rng.poisson(base_means, size=(n_ctrl, ng)).astype(np.float32)
    overdispersed = set(rng.choice(ng, size=n_top, replace=False).tolist())
    # mean-preserving variance inflation: a 0.1x/10x scale mixture around the same base mean
    # (p_lo chosen so p_lo*0.1 + (1 - p_lo)*10 == 1, i.e. E[scale] == 1).
    lo, hi = 0.1, 10.0
    p_lo = (hi - 1.0) / (hi - lo)
    for g in overdispersed:
        scale = rng.choice([lo, hi], size=n_ctrl, p=[p_lo, 1 - p_lo])
        base[:, g] = rng.poisson(base_means[g] * scale).astype(np.float32)
    pert = rng.poisson(base_means, size=(80, ng)).astype(np.float32)
    # scanpy's "seurat" flavor expects log1p'd input (it internally un-logs via expm1), matching
    # real pipeline data (see docs/user-guide/datasets.md) -- raw counts would be meaningless.
    adata = ad.AnnData(np.log1p(np.vstack([base, pert])))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * n_ctrl + ["pertA"] * 80

    cfg = make_cfg(min_cells=10)
    ctx = Context(Dataset(adata, cfg), cfg)
    assert set(hvg(ctx, n_top).tolist()) == overdispersed


def test_perturbed_gene_indices_matches_var_names_and_skips_non_gene_labels():
    rng = np.random.default_rng(0)
    ng = 6
    adata = ad.AnnData(rng.poisson(1.0, size=(40, ng)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(ng)]
    # g1: single-gene perturbation; g2+g4: combo (+-delimited, see docs/user-guide/datasets.md);
    # drugX: a non-gene treatment label that doesn't match any var_names entry (skipped).
    adata.obs["perturbation"] = ["control"] * 10 + ["g1"] * 10 + ["g2+g4"] * 10 + ["drugX"] * 10

    cfg = make_cfg(min_cells=5)
    idx = targeted_genes(Context(Dataset(adata, cfg), cfg))
    assert sorted(idx.tolist()) == [1, 2, 4]
    assert np.issubdtype(idx.dtype, np.integer)  # float indices would fail to index X


def test_perturbed_gene_indices_raises_when_no_label_is_a_gene():
    """A drug/compound dataset has no targeted genes -- fail loudly, don't score an empty panel."""
    rng = np.random.default_rng(0)
    ng = 6
    adata = ad.AnnData(rng.poisson(1.0, size=(30, ng)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 10 + ["drugX"] * 10 + ["drugY"] * 10

    cfg = make_cfg(min_cells=5)
    with pytest.raises(ValueError, match="no perturbation label matches a gene"):
        targeted_genes(Context(Dataset(adata, cfg), cfg))


def _ctx_10_genes():
    """10 genes; heg(3) picks {3, 6, 8}; perturbations target {0, 3, 5} (g3 overlaps)."""
    rng = np.random.default_rng(0)
    ng = 10
    ctrl_means = np.array([1.0, 5.0, 2.0, 9.0, 0.5, 3.0, 8.0, 4.0, 7.0, 6.0])
    ctrl = rng.poisson(ctrl_means, size=(300, ng)).astype(np.float32)
    a = rng.poisson(ctrl_means, size=(40, ng)).astype(np.float32)
    b = rng.poisson(ctrl_means, size=(40, ng)).astype(np.float32)
    adata = ad.AnnData(np.log1p(np.vstack([ctrl, a, b])))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 300 + ["g3+g0"] * 40 + ["g5"] * 40

    cfg = make_cfg(min_cells=10)
    return Context(Dataset(adata, cfg), cfg)


@pytest.mark.parametrize(
    ("op", "label", "expected"),
    [
        (OPS.union, "union", [0, 3, 5, 6, 8]),  # {3, 6, 8} | {0, 3, 5}
        (OPS.intersection, "intersection", [3]),  # {3, 6, 8} & {0, 3, 5}
        (OPS.difference, "difference", [6, 8]),  # {3, 6, 8} - {0, 3, 5}
    ],
)
def test_combine_subsets_applies_the_set_operation(op, label, expected):
    ctx = _ctx_10_genes()
    name = SPACES.combine_subsets(
        op, SPACES.instance("heg", 3), SPACES.instance("perturbed_genes"), name=f"heg3_{label}_pert"
    )
    assert SPACES.meta(name)["select"](ctx, "g5").tolist() == expected


def test_combine_subsets_canonicalises_a_slice_so_full_composes_as_a_complement():
    """full returns a slice; combining must still yield integer positions."""
    ctx = _ctx_10_genes()
    name = SPACES.combine_subsets(OPS.difference, SPACES.instance("full"), SPACES.instance("heg", 3), name="not_heg3")
    # full returns a slice; combining must still yield integer positions
    assert SPACES.meta(name)["select"](ctx, "g5").tolist() == [0, 1, 2, 4, 5, 7, 9]  # 10 genes minus {3, 6, 8}


def test_combine_subsets_nests_to_any_depth():
    """A composition is itself a subset, so it can be composed again."""
    ctx = _ctx_10_genes()
    inner = SPACES.combine_subsets(
        OPS.difference, SPACES.instance("heg", 3), SPACES.instance("perturbed_genes"), name="heg3_not_pert"
    )
    outer = SPACES.combine_subsets(OPS.union, inner, SPACES.instance("hvg", 4), name="nested")

    got = SPACES.meta(outer)["select"](ctx, "g5")
    assert set(got.tolist()) == {6, 8} | set(hvg(ctx, 4).tolist())


def test_combine_subsets_derives_per_pert_from_its_operands():
    """The composite varies by perturbation exactly when one of its operands does."""
    both_global = SPACES.combine_subsets(
        OPS.union, SPACES.instance("heg", 3), SPACES.instance("perturbed_genes"), name="global_pair"
    )
    with_per_pert = SPACES.combine_subsets(
        OPS.union, SPACES.instance("heg", 3), SPACES.instance("top", 5), name="mixed_pair"
    )
    assert SPACES.meta(both_global)["global_space"] is True  # nothing varies -> project once
    assert SPACES.meta(with_per_pert)["global_space"] is False  # top_5 varies -> re-project


def test_combine_subsets_rejects_transforms_and_unknown_names():
    with pytest.raises(ValueError, match="no genes to combine"):
        SPACES.combine_subsets(OPS.union, SPACES.instance("pca"), SPACES.instance("heg", 3), name="with_pca")
    with pytest.raises(KeyError, match="unknown space"):
        SPACES.combine_subsets(OPS.union, "heg_99999", SPACES.instance("heg", 3), name="with_typo")


def test_a_rule_that_does_not_name_pert_is_never_given_one():
    """Not naming ``pert`` is how a space declares it is dataset-wide -- and makes it unreachable."""

    @SPACES.subset("no_pert", default=3, description="ignores the perturbation")
    def no_pert(ctx, k):
        return np.arange(k)

    assert SPACES.catalog and not next(s for s in SPACES.catalog() if s.name == "no_pert").per_pert
    ctx = _ctx_10_genes()
    # the rule is called without pert, so a body reaching for it would raise NameError, not
    # silently receive a stale one
    assert SPACES.meta(SPACES.instance("no_pert"))["select"](ctx, "g5").tolist() == [0, 1, 2]


def test_perturbed_and_hvgs_unions_hvg_with_the_targeted_genes():
    ctx = _ctx_10_genes()  # 10 genes, so hvg(8192) degrades to all of them
    got = SPACES.meta("perturbed_and_hvgs")["select"](ctx, "g5")
    assert set(got.tolist()) == set(hvg(ctx, 8192).tolist()) | set(perturbed_genes(ctx).tolist())


def test_catalog_lists_definitions_and_says_what_each_takes():
    labels = {s.name: s.label for s in SPACES.catalog()}
    assert labels["heg"] == "heg_<k>"  # parameter name read from the rule's signature
    assert labels["degs"] == "degs_<padj>"
    assert labels["full"] == "full"  # trailing default => takes no parameter
    assert labels["perturbed_and_hvgs"] == "perturbed_and_hvgs"


def test_instance_registers_on_demand_and_guards_its_value():
    assert SPACES.instance("heg") == "heg_1000"  # the catalog default
    assert SPACES.instance("heg", 250) == "heg_250"
    assert SPACES.instance("perturbed_genes") == "perturbed_genes"
    assert SPACES.meta("heg_250")["global_space"] is True
    assert SPACES.meta(SPACES.instance("top", 5))["global_space"] is False  # per_pert
    with pytest.raises(KeyError, match="unknown space"):
        SPACES.instance("nope")
    with pytest.raises(TypeError, match="takes no parameter"):
        SPACES.instance("full", 5)
    # Distinct values that format to the same name must not silently share one registration.
    SPACES.instance("degs", 0.05)
    with pytest.raises(ValueError, match="already registered with value"):
        SPACES.instance("degs", 0.05000000001)


def test_instance_is_thread_safe_for_a_not_yet_registered_value():
    """Many threads racing to register the same brand-new value must all agree on the outcome."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as ex:
        keys = list(ex.map(lambda _: SPACES.instance("heg", 4321), range(20)))
    assert keys == ["heg_4321"] * 20
    assert SPACES.meta("heg_4321")["value"] == 4321


def test_a_global_spaces_selection_runs_once_per_store_not_once_per_call():
    """A per_pert=False space's rule must run once per prepared dataset (shared CacheStore),
    reused across every fresh Context a verb call builds -- not once per (pert, source) call."""
    calls = []

    @SPACES.subset("test_count_calls", description="counts how many times its rule runs")
    def test_count_calls(ctx):
        calls.append(1)
        return slice(None)

    ctx1 = _ctx_10_genes()
    ctx2 = Context(ctx1.ds, ctx1.cfg, store=ctx1._store)  # a second, fresh Context, same store
    key = SPACES.instance("test_count_calls")
    for ctx, pert in [(ctx1, "g3+g0"), (ctx1, "g5"), (ctx2, "g3+g0")]:
        SPACES[key](ctx.ds.cells(pert), ctx, pert)
    assert len(calls) == 1


def test_a_per_perturbation_spaces_selection_still_runs_every_call():
    """per_pert=True is untouched by that caching -- it must still see every call's pert."""
    calls = []

    @SPACES.subset("test_count_pert_calls", description="counts how many times its rule runs")
    def test_count_pert_calls(ctx, pert):
        calls.append(pert)
        return slice(None)

    ctx = _ctx_10_genes()
    key = SPACES.instance("test_count_pert_calls")
    for pert in ("g3+g0", "g5", "g3+g0"):
        SPACES[key](ctx.ds.cells(pert), ctx, pert)
    assert calls == ["g3+g0", "g5", "g3+g0"]  # every call, not deduplicated


def test_a_rule_taking_a_parameter_must_declare_a_default():
    with pytest.raises(TypeError, match="needs a default"):

        @SPACES.subset("bad", description="no default for k")
        def bad(ctx, pert, k):
            return slice(None)


def test_full_is_a_view_not_a_copy():
    ctx = _ctx_10_genes()
    assert full(ctx) == slice(None)
    cells = ctx.ds.cells("g5")
    assert SPACES["full"](cells, ctx, "g5").shape == cells.shape


def test_space_runs_end_to_end_through_the_runner(cfg_factory):
    """The whole path: warm() -> cached reference projection -> per-perturbation scoring."""
    rng = np.random.default_rng(0)
    ng, n_ctrl, n_pert = 80, 300, 120
    genes = [f"g{i}" for i in range(ng)]
    parts = [rng.poisson(1.0, (n_ctrl, ng)).astype(np.float32)]
    labels = ["control"] * n_ctrl
    for lab, block in {"g0": range(0, 6), "g1+g2": range(15, 21), "g3": range(30, 36)}.items():
        x = rng.poisson(1.0, (n_pert, ng)).astype(np.float32)
        x[:, list(block)] += 6.0
        parts.append(x)
        labels += [lab] * n_pert
    adata = ad.AnnData(np.log1p(np.vstack(parts)))
    adata.var_names = genes
    adata.obs["perturbation"] = labels

    panel = SPACES.instance("perturbed_and_hvgs")
    proto = replace(PROTOCOLS["energy_distance_top_k"], name="ed_panel", space=panel, param=None)

    cfg = cfg_factory(truth="gt_all_cells", calibrator="score")
    ds = Dataset(adata, cfg)
    ctx = Context(ds, cfg)
    sub = adata[np.asarray(adata.obs["perturbation"]).astype(str) != "control"].copy()
    pred = ad.AnnData(np.asarray(sub.X, dtype=np.float32), obs=sub.obs.copy())
    pred.var_names = genes
    ctx.predictions = PredictionSet(pred, ds, cfg)

    ctx.warm([proto])
    assert ("reference_projection", (panel,)) in ctx._store.memo  # global space: projected once, shared
    agg, rows, _ = run_protocol(proto, ctx, CALIBRATORS["score"])
    assert len(rows) == 3
    assert np.isfinite(agg["mean"])


def test_importing_the_package_defines_the_catalog():
    """The catalog only exists because importing the package imports the rules that declare it.

    Checked in a fresh interpreter: this module imports ``catalog`` directly, which would define
    the spaces for the whole pytest session and hide a missing import in ``__init__``.
    """
    probe = (
        "from scperteval.blocks.spaces import SPACES;"
        "names = [s.name for s in SPACES.catalog()];"
        "assert 'heg' in names, names;"
        "assert SPACES['full'] is not None;"
        "print(len(names))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) >= 8


def test_a_per_perturbation_transform_is_not_registered_as_shared():
    """A transform that varies by perturbation must re-project the reference, like a subset does.

    Otherwise the reference is built once at ``pert=None`` and shared, while each candidate is
    built with its own ``pert`` -- comparing two populations in different feature spaces.
    """

    @SPACES.transform("pertdep", default=2, description="varies by perturbation")
    def pertdep(X, ctx, pert, k):
        return np.asarray(X)[:, :k]

    assert next(s for s in SPACES.catalog() if s.name == "pertdep").per_pert
    assert SPACES.meta(SPACES.instance("pertdep", 2))["global_space"] is False


@pytest.mark.parametrize(
    "rule",
    [
        lambda ctx, k, pert: None,  # pert after the parameter
        lambda ctx, k, pert=None: None,  # ...and defaulted, which would read as dataset-wide
        lambda ctx, a, b: None,  # two parameters
    ],
)
def test_a_rule_of_the_wrong_shape_is_rejected_at_registration(rule):
    """`pert` comes first when present; anything else would be silently mis-classified."""
    with pytest.raises(TypeError, match="does not match a space rule's shape"):
        SPACES.subset("badshape", default=1, description="…")(rule)


def test_a_name_cannot_be_redefined():
    """Silently replacing a definition would leave instances scoring on genes it no longer describes."""
    with pytest.raises(ValueError, match="already defined"):
        SPACES.subset("heg", default=7, description="an impostor")(lambda ctx, k: None)
    with pytest.raises(ValueError, match="already defined"):
        SPACES.combine_subsets(OPS.union, SPACES.instance("heg", 3), SPACES.instance("perturbed_genes"), name="heg")


@pytest.mark.parametrize("names", [(), ("heg_3",)])
def test_combine_subsets_needs_at_least_two_spaces(names):
    """One operand is an alias, not a combination; zero used to fail deep in the scoring loop."""
    with pytest.raises(ValueError, match="at least two"):
        SPACES.combine_subsets(OPS.union, *names, name="too_few")


def test_combine_subsets_rejects_an_operator_that_is_not_a_set_operation():
    with pytest.raises(ValueError, match="unknown op"):
        SPACES.combine_subsets(np.add, SPACES.instance("heg", 3), SPACES.instance("hvg", 4), name="bad_op")


def test_a_cached_helper_cannot_be_registered_as_a_space():
    """`@cached` belongs on the helper a rule calls, not stacked on the rule itself."""
    with pytest.raises(TypeError, match=r"must start with \(ctx"):

        @SPACES.subset("stacked", default=3, description="…")
        @cached
        def stacked(scope, k):
            return np.arange(k)


def test_a_space_that_selects_nothing_is_refused():
    """An empty panel makes every metric return nan; better to fail at the space."""
    ctx = _ctx_10_genes()
    name = SPACES.combine_subsets(
        OPS.intersection, SPACES.instance("heg", 1), SPACES.instance("hvg", 1), name="disjoint"
    )
    if SPACES.meta(name)["select"](ctx, "g5").size == 0:  # the panels don't overlap on this data
        with pytest.raises(ValueError, match="selected no genes"):
            SPACES[name](ctx.ds.cells("g5"), ctx, "g5")


@pytest.mark.parametrize("value", [0, -5, -0.5])
def test_a_space_parameter_must_be_positive(value):
    """Negative values mean three different things across the spaces, none of them useful."""
    with pytest.raises(ValueError, match="positive number"):
        SPACES.instance("heg", value)


def test_the_inherited_registry_entry_points_are_closed():
    """One way to define a space, so none can exist outside the catalog the listing reads."""
    with pytest.raises(TypeError, match=r"@SPACES\.subset"):
        SPACES.register("old_style", description="…")
    with pytest.raises(TypeError, match=r"@SPACES\.subset"):
        SPACES.add("old_style", lambda X, ctx, pert: None, description="…")


def test_a_cache_rejects_values_from_a_different_scope():
    """A CacheStore belongs to one prepare(); sharing one across seeds would serve stale values."""
    cfg = make_cfg(seed=0)
    ctx = Context(Dataset(make_dataset(), cfg), cfg)
    ctx.scope()  # records the scope
    other = Context(ctx.ds, make_cfg(seed=123), store=ctx._store)
    with pytest.raises(ValueError, match="different dataset, seed, or subsample"):
        other.scope()
