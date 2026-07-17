"""The native Python API — evaluate one protocol, or compute one DE method, at a time.

Usage is always **prepare, then run**: build a reusable :func:`prepare` handle for a dataset (read
+ index once, precompute the declared protocols' spaces), then call :func:`calibrate` /
:func:`score` / :func:`de` on that handle — each evaluates a single protocol or DE method and
returns in-memory results (pandas). Many calls share the handle's dataset and caches (no reload),
and are safe to run concurrently: each call builds its own lightweight context over the shared,
thread-safe cache, so nothing is mutated across calls.

Re-exported at the package root, e.g. ``scperteval.calibrate(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pandas as pd

from . import io
from .blocks.de import DE_METHODS
from .context import CacheStore, Context
from .dataset import Dataset
from .predictions import PredictionSet
from .protocols.resolve import resolve_protocols
from .runner import compute_de, run_all
from .sources import SOURCES
from .types import Protocol, RunConfig

if TYPE_CHECKING:  # annotation-only; keeps ``import scperteval`` from eagerly importing anndata
    from collections.abc import Callable

    from anndata import AnnData

__all__ = [
    "DatasetDEResults",
    "EvalResult",
    "Prepared",
    "calibrate",
    "de",
    "prepare",
    "score",
]

#: The calibration outputs selectable from :func:`calibrate` (closed set).
CalibratorName = Literal["drf", "bds"]
#: The DE backends selectable from :func:`de` / ``de_method`` — mirrors the ``DE_METHODS`` registry's
#: built-ins (kept in sync by ``tests/test_api.py::test_de_method_literal_matches_registry``).
DEMethodName = Literal["t-test", "MWU", "t-test_overestim_var"]


# --------------------------------------------------------------------------- result types


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating one protocol on one dataset.

    Attributes
    ----------
    aggregate : dict[str, float]
        The protocol's summary statistics — ``{"mean": …, "median": …}`` for ``drf``/``score``,
        ``{"bds": …}`` for ``bds``.
    per_perturbation : pandas.DataFrame
        One row per perturbation (raw control values + the calibrated score, or the raw metric) —
        the same layout the CLI writes to CSV.
    """

    aggregate: dict[str, float]
    per_perturbation: pd.DataFrame

    def __repr__(self) -> str:
        col = self.per_perturbation.get("perturbation")
        n = col.nunique() if col is not None else 0
        return f"EvalResult(aggregate={self.aggregate}, perturbations={n})"


class DatasetDEResults(NamedTuple):
    """Per-gene differential expression across the whole dataset, for one method.

    Both frames are indexed by perturbation with genes as columns.
    """

    #: Test statistic per (perturbation, gene).
    statistic: pd.DataFrame
    #: Benjamini-Hochberg adjusted p-value per (perturbation, gene).
    pvalue_adj: pd.DataFrame


# --------------------------------------------------------------------------- prepared handle


class Prepared:
    """A reusable, prepared dataset: build once with :func:`prepare`, pass to many verb calls.

    Holds the read-and-indexed dataset (resident in memory), a shared thread-safe cache, and the
    immutable prepare-time configuration. Each :func:`calibrate` / :func:`score` / :func:`de` call
    builds its own lightweight context over this handle, so the handle itself is never mutated —
    sequential *and* concurrent calls against one handle are safe. Treat it as opaque; its
    internals are not part of the public API.
    """

    __slots__ = ("_cfg", "_ds", "_sources", "_store")

    def __init__(
        self,
        ds: Dataset,
        store: CacheStore,
        cfg: RunConfig,
        sources: dict[str, tuple[Callable, dict]] | None = None,
    ):
        self._ds = ds
        self._store = store
        self._cfg = cfg
        self._sources = sources or {}  # per-handle runtime user sources ({name: (callable, meta)})

    def _run_context(self, **overrides) -> Context:
        """A fresh per-call context sharing this handle's dataset + cache, with per-call config."""
        return Context(self._ds, replace(self._cfg, **overrides), store=self._store, user_sources=self._sources)

    def __repr__(self) -> str:
        return f"Prepared(name={Path(self._cfg.dataset).stem!r}, perturbations={len(self._ds.perturbations)})"


# --------------------------------------------------------------------------- helpers


def _display_name(dataset, name: str | None) -> str:
    """The label threaded into ``cfg.dataset`` (drives summary headers and output filenames)."""
    if name is not None:
        return name
    if isinstance(dataset, (str, Path)):
        return str(dataset)
    return "dataset"


def _to_dataset(dataset, cfg: RunConfig) -> Dataset:
    """Build a :class:`~scperteval.dataset.Dataset` from a path or an in-memory AnnData."""
    if isinstance(dataset, (str, Path)):
        return Dataset.load(str(dataset), cfg)
    return Dataset(dataset, cfg)  # an AnnData (referenced, never mutated)


def _to_predictions(predictions, ds: Dataset, cfg: RunConfig) -> PredictionSet:
    """Build a :class:`~scperteval.predictions.PredictionSet` from a path or an AnnData."""
    if isinstance(predictions, (str, Path)):
        return PredictionSet.load(str(predictions), ds, cfg)
    return PredictionSet(predictions, ds, cfg)


def _require_prepared(prepared, verb: str) -> None:
    if not isinstance(prepared, Prepared):
        raise TypeError(
            f"{verb}() takes a handle from prepare(); got {type(prepared).__name__}. "
            f"Call prepare(dataset, protocols) first, then pass the result here."
        )


def _single_protocol(protocol: str):
    """Resolve one protocol spec to exactly one concrete protocol (error otherwise)."""
    if not isinstance(protocol, str):
        raise TypeError(f"protocol must be a single protocol name (str), not {type(protocol).__name__}")
    protos = resolve_protocols([protocol])
    if len(protos) != 1:
        raise ValueError(
            f"the API evaluates one protocol per call; {protocol!r} resolves to {len(protos)} "
            f"protocols (pass a single name, e.g. 'pearson_ctrl' or 'mse_top_k=30')"
        )
    return protos[0]


def _check_de_method(method: str) -> None:
    if method not in DE_METHODS:
        raise ValueError(f"unknown DE method {method!r}; available: {', '.join(DE_METHODS.names())}")


def _stamp() -> str:
    # Microsecond resolution: verbs share one handle and may run concurrently, so two writes of the
    # same protocol to one out_dir must get distinct filenames rather than silently overwriting.
    return datetime.now().strftime("%Y-%m-%dT%H%M%S%f")


def _const_source(array: np.ndarray):
    """A source callable that returns its stored constant array for any perturbation."""

    def fn(ctx, pert):
        return array

    return fn


def _validate_sources(sources, ds: Dataset) -> dict[str, tuple[Callable, dict]]:
    """Validate + register the ``prepare(sources=...)`` user sources for one handle.

    Each ``{name: array}`` becomes a per-handle source (``cacheable=False``): a 1-D ``(G,)`` array
    is a ``"centroid"`` and a 2-D ``(n, G)`` array is ``"cells"``. Values are copied to a contiguous
    ``float64`` array (never aliasing the caller's memory). ``G`` is checked against the dataset's
    gene count, but **not** gene order — see :func:`prepare`.
    """
    if not sources:
        return {}
    n_genes = len(ds.var_names)
    out: dict[str, tuple[Callable, dict]] = {}
    for name, array in sources.items():
        if name == "auto":
            raise ValueError("user source name 'auto' is reserved (the control-override sentinel); rename it")
        if name in SOURCES:
            raise ValueError(
                f"user source {name!r} shadows a built-in source ({', '.join(SOURCES.names())}); rename it"
            )
        if not isinstance(array, np.ndarray):
            raise TypeError(f"user source {name!r} must be a numpy array, got {type(array).__name__}")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"user source {name!r} must be numeric, got dtype {array.dtype}")
        if array.ndim == 1:
            provides, g = "centroid", array.shape[0]
        elif array.ndim == 2:
            provides, g = "cells", array.shape[1]
        else:
            raise ValueError(f"user source {name!r} must be 1-D (centroid) or 2-D (cells), got shape {array.shape}")
        if g != n_genes:
            raise ValueError(
                f"user source {name!r} has {g} genes but the dataset has {n_genes}; "
                f"columns must be in adata.var_names order"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"user source {name!r} has non-finite values (NaN/inf); all entries must be finite")
        data = np.array(array, dtype=np.float64, order="C")  # contiguous float64 copy, never aliasing caller memory
        out[name] = (_const_source(data), {"provides": provides, "cacheable": False})
    return out


def _apply_center_on(proto: Protocol, center_on: str, ctx: Context) -> Protocol:
    """Mint a named centering variant ``<base>_center_<center_on>`` from an un-centred centroid protocol.

    Centering is protocol identity, so a custom-vector baseline is recorded in the protocol name
    rather than silently overriding a catalog protocol. ``center_on`` must name a registered
    centroid source (user or built-in).
    """
    if proto.representation != "centroid":
        raise ValueError(f"center_on only applies to centroid protocols; {proto.name!r} is {proto.representation!r}")
    if proto.centering is not None:
        raise ValueError(
            f"center_on requires an un-centered protocol; {proto.name!r} already centers on {proto.centering!r}"
        )
    if not ctx.has_source(center_on):
        raise ValueError(
            f"center_on source {center_on!r} is not registered; valid sources: {', '.join(ctx.source_names())}"
        )
    provides = ctx.source_meta(center_on).get("provides")
    if provides != "centroid":
        raise ValueError(
            f"center_on source {center_on!r} provides {provides!r}, but a centering baseline must be a centroid (1-D)"
        )
    return replace(proto, centering=center_on, name=f"{proto.name}_center_{center_on}")


# --------------------------------------------------------------------------- public functions


def prepare(
    dataset: str | Path | AnnData,
    protocols: str | list[str],
    *,
    subsample: int = 8192,
    seed: int = 42,
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    sources: dict[str, np.ndarray] | None = None,
) -> Prepared:
    """Prepare a dataset for evaluation — the required first step.

    Reads and indexes the dataset once (held in memory) and **precomputes the declared protocols'
    feature spaces** (e.g. PCA, fit once at the largest requested ``k``) plus the shared reference
    sample, deterministically. The returned :class:`Prepared` handle is then passed to
    :func:`calibrate` / :func:`score` / :func:`de`, which reuse its dataset and caches — including
    across concurrent calls. Differential expression is **not** precomputed here; it is computed at
    the verb call under that call's DE method (and cached per method on the handle).

    Parameters
    ----------
    dataset : str or pathlib.Path or anndata.AnnData
        A preprocessed ``.h5ad`` path, or an in-memory AnnData. If an AnnData, the handle holds a
        reference to it (not a copy) — do not mutate it while the handle is in use, or results
        become inconsistent.
    protocols : str or list of str
        The protocol(s) you intend to evaluate — used to precompute their spaces up front (pass
        ``"all"`` for the whole catalog, or ``[]`` if you only need :func:`de` / no spaces). A verb
        may still run a protocol not declared here; its space is then computed on first use.
    subsample, seed, min_cells, perturbation_key, control_label, workers, name
        Dataset/run knobs fixed for the handle; see :class:`~scperteval.types.RunConfig`.
    sources : dict[str, numpy.ndarray], optional
        Runtime **user sources** registered on this handle (never on the global registry, so they
        don't leak across handles). Each ``{name: array}`` becomes a reusable, constant-across-
        perturbations source: a 1-D ``(G,)`` array is a centroid, a 2-D ``(n_cells, G)`` array is a
        cell population. Use them as controls (``negative=``/``positive=`` on :func:`calibrate`) or
        as a centering baseline (``center_on=`` on :func:`calibrate`/:func:`score`). Arrays must be
        numeric and all-finite, with ``G`` equal to the dataset's gene count. **Gene-order caveat:**
        columns are assumed to be in ``adata.var_names`` order — validation checks the gene *count*
        but cannot check the *order*, so a mis-ordered vector silently compares the wrong genes.

    Returns
    -------
    Prepared
        An opaque, reusable handle.
    """
    specs = [protocols] if isinstance(protocols, str) else list(protocols)
    protos = resolve_protocols(specs) if specs else []
    cfg = RunConfig(
        dataset=_display_name(dataset, name),
        protocols=[p.name for p in protos],
        subsample=subsample,
        seed=seed,
        min_cells=min_cells,
        perturbation_key=perturbation_key,
        control_label=control_label,
        workers=workers,
    )
    ds = _to_dataset(dataset, cfg)
    user_sources = _validate_sources(sources, ds)  # fail fast on bad user sources, before warming
    ctx = Context(ds, cfg)
    ctx.warm(protos)  # precompute declared spaces + reference (method-independent); no DE
    return Prepared(ctx.ds, ctx._store, cfg, user_sources)


def calibrate(
    prepared: Prepared,
    protocol: str,
    *,
    de_method: DEMethodName = "t-test",
    calibrator: CalibratorName = "drf",
    positive: str = "auto",
    negative: str = "auto",
    center_on: str | None = None,
    out_dir: str | Path | None = None,
) -> EvalResult:
    """Calibrate one protocol against the built-in positive/negative controls (DRF or BDS).

    Parameters
    ----------
    prepared : Prepared
        A handle from :func:`prepare`.
    protocol : str
        A single protocol spec — a name (``"pearson_ctrl"``) or a tunable one (``"mse_top_k=30"``).
    de_method : str, optional
        DE backend for any DE-dependent part of the protocol (default ``"t-test"``).
    calibrator : {"drf", "bds"}, optional
        Which calibrator to apply (default ``"drf"``).
    positive, negative : str, optional
        Override the protocol's control sources (``"auto"`` defers to the protocol). A registered
        user source (from ``prepare(sources=...)``) is accepted here.
    center_on : str, optional
        Center the (un-centred, centroid) protocol on a named centroid source's baseline. Because
        centering is protocol identity, this **mints a named variant** ``<protocol>_center_<name>``
        (never a silent override); the variant name flows into ``EvalResult`` and any CSV. ``name``
        may be a user source or a built-in centroid (e.g. ``"all_perturbed_mean"``).
    out_dir : str or pathlib.Path, optional
        If given, also write the per-perturbation CSV there (as the CLI does).

    Returns
    -------
    EvalResult
        ``.aggregate`` (the protocol's summary stats) and ``.per_perturbation`` (the detail table).
    """
    _require_prepared(prepared, "calibrate")
    if calibrator not in ("drf", "bds"):
        raise ValueError(
            f"calibrate calibrator must be 'drf' or 'bds', not {calibrator!r} (use score() for predictions)"
        )
    _check_de_method(de_method)
    proto = _single_protocol(protocol)
    ctx = prepared._run_context(
        protocols=[proto.name],
        de_method=de_method,
        calibrator=calibrator,
        positive=positive,
        negative=negative,
        out_dir=str(out_dir) if out_dir is not None else "results",
    )
    if center_on is not None:
        proto = _apply_center_on(proto, center_on, ctx)
        ctx.cfg.protocols = [proto.name]  # keep summary/CSV labels in sync with the minted variant
    aggregates, rows, _ = run_all(ctx.cfg, [proto], ctx)
    if out_dir is not None:
        io.write_rows(ctx.cfg, rows, _stamp())
    return EvalResult(aggregate=aggregates[proto.name], per_perturbation=io.rows_frame(ctx.cfg, rows))


def score(
    prepared: Prepared,
    protocol: str,
    predictions: str | Path | AnnData,
    *,
    de_method: DEMethodName = "t-test",
    center_on: str | None = None,
    out_dir: str | Path | None = None,
) -> EvalResult:
    """Score model predictions against ground truth for one protocol.

    Parameters
    ----------
    prepared : Prepared
        A handle from :func:`prepare` (the ground-truth dataset).
    protocol : str
        A single protocol spec (see :func:`calibrate`).
    predictions : str or pathlib.Path or anndata.AnnData
        Predicted cells — the same genes and perturbation labels as the dataset.
    de_method : str, optional
        DE backend for any DE-dependent part of the protocol (default ``"t-test"``).
    center_on : str, optional
        Center on a named centroid source's baseline, minting a ``<protocol>_center_<name>`` variant
        (see :func:`calibrate`).
    out_dir : str or pathlib.Path, optional
        If given, also write the per-perturbation CSV there.

    Returns
    -------
    EvalResult
        ``.aggregate`` (mean/median raw metric) and ``.per_perturbation`` (the detail table).
    """
    _require_prepared(prepared, "score")
    _check_de_method(de_method)
    proto = _single_protocol(protocol)
    ctx = prepared._run_context(
        protocols=[proto.name],
        de_method=de_method,
        calibrator="score",
        truth="gt_all_cells",
        out_dir=str(out_dir) if out_dir is not None else "results",
    )
    if center_on is not None:
        proto = _apply_center_on(proto, center_on, ctx)
        ctx.cfg.protocols = [proto.name]  # keep summary/CSV labels in sync with the minted variant
    ctx.predictions = _to_predictions(predictions, ctx.ds, ctx.cfg)
    aggregates, rows, _ = run_all(ctx.cfg, [proto], ctx)
    if out_dir is not None:
        io.write_rows(ctx.cfg, rows, _stamp())
    return EvalResult(aggregate=aggregates[proto.name], per_perturbation=io.rows_frame(ctx.cfg, rows))


def de(
    prepared: Prepared,
    method: DEMethodName = "t-test",
    *,
    out_dir: str | Path | None = None,
) -> DatasetDEResults:
    """Compute per-gene differential expression (ground truth vs all-perturbed) for one method.

    Parameters
    ----------
    prepared : Prepared
        A handle from :func:`prepare`.
    method : str, optional
        The DE backend (default ``"t-test"``). Different methods reuse the same prepared dataset,
        cached separately — no reload.
    out_dir : str or pathlib.Path, optional
        If given, also write the HDF5 export there (as the CLI does).

    Returns
    -------
    DatasetDEResults
        ``.statistic`` and ``.pvalue_adj`` DataFrames (perturbations × genes).
    """
    _require_prepared(prepared, "de")
    _check_de_method(method)
    ctx = prepared._run_context(de_method=method, out_dir=str(out_dir) if out_dir is not None else "results")
    ctx._ensure_reference_sums()
    statistic, pvalue_adj = compute_de(ctx)
    perts = list(ctx.perturbations)
    genes = [str(g) for g in ctx.ds.var_names]
    result = DatasetDEResults(
        statistic=pd.DataFrame(statistic, index=perts, columns=genes),
        pvalue_adj=pd.DataFrame(pvalue_adj, index=perts, columns=genes),
    )
    if out_dir is not None:
        io.write_de(ctx.cfg, ctx.ds.var_names, ctx.perturbations, {method: (statistic, pvalue_adj)}, _stamp())
    return result
