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

import pandas as pd

from . import io
from .blocks.de import DE_METHODS
from .context import CacheStore, Context
from .dataset import Dataset
from .predictions import PredictionSet
from .protocols.resolve import resolve_protocols
from .runner import compute_de, run_all
from .types import RunConfig

if TYPE_CHECKING:  # annotation-only; keeps ``import scperteval`` from eagerly importing anndata
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

    __slots__ = ("_cfg", "_ds", "_store")

    def __init__(self, ds: Dataset, store: CacheStore, cfg: RunConfig):
        self._ds = ds
        self._store = store
        self._cfg = cfg

    def _run_context(self, **overrides) -> Context:
        """A fresh per-call context sharing this handle's dataset + cache, with per-call config."""
        return Context(self._ds, replace(self._cfg, **overrides), store=self._store)

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
    ctx = Context(_to_dataset(dataset, cfg), cfg)
    ctx.warm(protos)  # precompute declared spaces + reference (method-independent); no DE
    return Prepared(ctx.ds, ctx._store, cfg)


def calibrate(
    prepared: Prepared,
    protocol: str,
    *,
    de_method: DEMethodName = "t-test",
    calibrator: CalibratorName = "drf",
    positive: str = "auto",
    negative: str = "auto",
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
        Override the protocol's control sources (``"auto"`` defers to the protocol).
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
