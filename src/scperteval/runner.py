"""Runs one protocol over every perturbation and applies the chosen calibrator."""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .types import Calibrator, MetricOutcome, Protocol

#: Generic (positive, negative) control defaults keyed on ``representation`` — the base tier of
#: control resolution, used unless a protocol declares a default or a run overrides one.
GENERIC_CONTROLS = {
    "centroid": ("interpolated", "all_perturbed_mean"),
    "population": ("tech_dup", "all_perturbed"),
    "de": ("tech_dup", "all_perturbed"),
}


def _n_workers(cfg) -> int:
    """Resolve the worker-thread count (``0`` = auto: CPU count minus 2, capped at 16)."""
    return cfg.workers if cfg.workers > 0 else max(1, min(16, (os.cpu_count() or 2) - 2))


def _resolve_candidates(p: Protocol, cfg) -> dict:
    """Map each calibrator input (candidate name) to a source name.

    Two-tier resolution for ``positive`` / ``negative``: an explicit run override (``cfg``) wins,
    else the protocol's declared default, else the representation's generic default. ``prediction``
    is always the model-prediction source (used by the ``score`` calibrator).
    """
    gen_pos, gen_neg = GENERIC_CONTROLS[p.representation]
    return {
        "positive": cfg.positive if cfg.positive != "auto" else (p.default_positive or gen_pos),
        "negative": cfg.negative if cfg.negative != "auto" else (p.default_negative or gen_neg),
        "prediction": "prediction",
    }


def run_protocol(p: Protocol, ctx, calibrator: Calibrator):
    """Run one protocol over every perturbation and apply the calibrator.

    ``p.scope`` chooses the execution path: ``"perturbation"``-scope protocols run in a
    thread pool (one perturbation at a time); ``"dataset"``-scope protocols collect all
    perturbations' datapoints first, then call the metric once.

    Parameters
    ----------
    p : ~scperteval.types.Protocol
        The concrete (non-parameterised) protocol to evaluate.
    ctx : ~scperteval.context.Context
        Per-run context holding the dataset, caches, and building blocks.
    calibrator : ~scperteval.types.Calibrator
        Calibrator that converts raw positive/negative control scores into a final value.

    Returns
    -------
    aggregates : dict
        Aggregate scores across perturbations (e.g. ``{"mean": 0.42, "median": 0.38}``).
    rows : list of dict
        Per-perturbation records with raw control values and the calibrated score.
    seconds : float
        Wall-clock time for this protocol.
    """
    candidates = _resolve_candidates(p, ctx.cfg)
    needed = {name: candidates[name] for name in calibrator.requires}
    _check_sources(p, needed, ctx)
    ctx.candidates = candidates  # the resolved negative drives _de_view's neg_reference choice
    start = perf_counter()
    perts, raws_list = _evaluate_protocol_raw(p, ctx, needed)
    seconds = perf_counter() - start
    agg, rows = _finalize(p, calibrator, perts, raws_list, needed)
    return agg, rows, seconds


def run_all(cfg, protocols, ctx):
    """Run every protocol over the dataset and collect results (no printing or file I/O).

    Selects the calibrator from ``cfg.calibrator``, warms the context once over *all* protocols
    (so shared singletons are precomputed before the parallel loop), then runs each protocol.
    Shared by the CLI's ``calibrate``/``score`` commands and the Python API.

    Parameters
    ----------
    cfg : ~scperteval.types.RunConfig
        Resolved run options; ``cfg.calibrator`` selects the calibrator.
    protocols : list of ~scperteval.types.Protocol
        The concrete protocols to evaluate.
    ctx : ~scperteval.context.Context
        Per-run context (in ``score`` mode its ``predictions`` must already be attached).

    Returns
    -------
    aggregates : dict
        ``{protocol_name: aggregate_dict}`` across perturbations.
    rows : list of dict
        Per-perturbation records for every protocol (raw controls + calibrated score).
    timed : list of tuple
        ``(protocol, seconds)`` wall-clock timing for each protocol.
    """
    from .calibrators import CALIBRATORS

    calibrator = CALIBRATORS[cfg.calibrator]
    _warn_declared_overrides(cfg, protocols)
    # nothing else is running yet to oversubscribe, unlike the per-perturbation loop below
    # (which is why scperteval otherwise pins BLAS/OpenMP to 1 thread, see __init__.py) -- so
    # warm()'s precompute can safely use more than one
    with threadpool_limits(limits=_n_workers(cfg)):
        ctx.warm(protocols)
    aggregates, rows, timed = {}, [], []
    for p in protocols:
        agg, proto_rows, seconds = run_protocol(p, ctx, calibrator)
        aggregates[p.name] = agg
        rows += proto_rows
        timed.append((p, seconds))
    return aggregates, rows, timed


def _warn_declared_overrides(cfg, protocols) -> None:
    """One consolidated warning when a run override replaces a protocol's *declared* default.

    Overriding a merely *generic* default is normal experimentation and stays silent; only a
    declared deviation (e.g. ``rank``'s ``global_mean``) is loud when replaced.
    """
    parts = []
    for role in ("positive", "negative"):
        override = getattr(cfg, role)
        if override == "auto":
            continue
        by_declared: dict[str, list[str]] = {}
        for p in protocols:
            declared = getattr(p, f"default_{role}")
            if declared is not None and declared != override:
                by_declared.setdefault(declared, []).append(p.name)
        if by_declared:
            groups = "; ".join(f"{', '.join(names)} (was {decl!r})" for decl, names in by_declared.items())
            parts.append(f"{role}={override!r} overrides the declared default for {groups}")
    if parts:
        warnings.warn("; ".join(parts), stacklevel=2)


def _as_outcome(result) -> MetricOutcome:
    """Normalize a raw metric return value (float or MetricOutcome) into one MetricOutcome."""
    if isinstance(result, MetricOutcome):
        return result
    value = float(result)
    return MetricOutcome(value, None if np.isfinite(value) else "metric_undefined")


def _calibrated_outcome(
    calibrator: Calibrator, raw_outcomes: dict[str, MetricOutcome], protocol: Protocol
) -> MetricOutcome:
    """Propagate raw metric undefinedness before applying the calibrator."""
    for outcome in raw_outcomes.values():
        if outcome.undefined_reason is not None or not np.isfinite(outcome.value):
            return MetricOutcome(float("nan"), outcome.undefined_reason or "metric_undefined")
    numeric = {name: outcome.value for name, outcome in raw_outcomes.items()}
    result = calibrator.per_pert(numeric, protocol)
    if isinstance(result, MetricOutcome):
        return result
    value = float(result)
    return MetricOutcome(value, None if np.isfinite(value) else "calibrator_undefined")


def _finalize(p, calibrator, perts, raws_list, needed):
    """Per-perturbation rows + the aggregate, from each perturbation's raw control outcomes.

    Each row records the resolved control *source* names (``needed``), their raw metric
    *values* (``raw_{name}``), any ``undefined_reason``, and the calibrated score.
    """
    per_pert = [_calibrated_outcome(calibrator, raw_outcomes, p) for raw_outcomes in raws_list]
    rows = [
        {
            "protocol": p.name,
            "perturbation": pert,
            **needed,
            **{f"raw_{name}": outcome.value for name, outcome in raws.items()},
            "undefined_reason": calibrated.undefined_reason or np.nan,
            calibrator.name: calibrated.value,
        }
        for pert, raws, calibrated in zip(perts, raws_list, per_pert)
    ]
    return calibrator.aggregate(np.asarray([outcome.value for outcome in per_pert], dtype=float)), rows


def _evaluate_per_perturbation(p: Protocol, ctx, candidates: dict[str, str]):
    """Evaluate candidate sources independently per perturbation."""

    def work(pert):
        ctx.current_pert = pert
        gt = ctx.view(pert, ctx.cfg.truth, p)
        return {name: _as_outcome(p.metric(gt, ctx.view(pert, src, p), ctx)) for name, src in candidates.items()}

    perts = ctx.perturbations
    with ThreadPoolExecutor(max_workers=_n_workers(ctx.cfg)) as pool:
        raws_list = list(pool.map(work, perts))
    return perts, raws_list


def _evaluate_dataset(p: Protocol, ctx, candidates: dict[str, str]):
    """Evaluate dataset-scope candidate sources once and unpack one outcome per perturbation."""
    perts = ctx.perturbations

    def collect(source):
        out = []
        for pert in perts:
            ctx.current_pert = pert
            out.append(ctx.view(pert, source, p))
        return out

    gt = collect(ctx.cfg.truth)
    scores = {name: p.metric(gt, collect(src), ctx) for name, src in candidates.items()}
    raws_list = [{name: _as_outcome(scores[name][i]) for name in candidates} for i in range(len(perts))]
    return perts, raws_list


def _evaluate_protocol_raw(p: Protocol, ctx, candidates: dict[str, str]):
    """Evaluate one protocol's raw (uncalibrated) metric on ``candidates`` according to ``p.scope``."""
    if p.scope == "dataset":
        return _evaluate_dataset(p, ctx, candidates)
    return _evaluate_per_perturbation(p, ctx, candidates)


def _validate_de_source(ctx, source: str, role: str) -> None:
    """Validate that a DE target or reference source is a known cell-providing source."""
    if not ctx.has_source(source):
        raise ValueError(f"unknown DE {role} source {source!r}; valid sources: {', '.join(ctx.source_names())}")
    provides = ctx.source_meta(source).get("provides")
    if provides != "cells":
        raise ValueError(f"DE {role} source {source!r} provides {provides!r}, expected 'cells'")
    if source == "prediction" and ctx.predictions is None:
        raise ValueError("DE source 'prediction' requires predictions")


def resolve_de_perturbations(ctx, perturbations=None) -> list[str]:
    """Validate an optional ordered perturbation subset against the prepared dataset."""
    if perturbations is None:
        return list(ctx.perturbations)
    selected = list(perturbations)
    if not selected:
        raise ValueError("DE perturbation selection is empty")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate perturbation in DE selection")
    unknown = sorted(set(selected) - set(ctx.perturbations))
    if unknown:
        raise ValueError(f"unknown perturbation(s) in DE selection: {', '.join(unknown)}")
    return selected


def compute_de(ctx, *, target: str | None = None, reference: str = "all_perturbed", perturbations=None):
    """Per-gene DE matrices for ``ctx.cfg.de_method``, for export.

    Returns ``(statistic, pvalue_adj)`` matrices (perturbations x genes) for the
    requested target-vs-reference comparison under the context's DE method.

    ``target=None`` uses ``ctx.cfg.truth``.
    """
    target = ctx.cfg.truth if target is None else target
    _validate_de_source(ctx, target, "target")
    _validate_de_source(ctx, reference, "reference")
    selected = resolve_de_perturbations(ctx, perturbations)

    def work(pert):
        d = ctx.de(pert, target, reference)
        return d.statistic, d.pvalue_adj

    with ThreadPoolExecutor(max_workers=_n_workers(ctx.cfg)) as pool:
        res = list(pool.map(work, selected))
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def _check_sources(p: Protocol, candidates: dict, ctx) -> None:
    """Validate the resolved control sources: known, and single-cell where the representation requires it.

    Resolution goes through ``ctx`` so per-handle user sources (from ``prepare(sources=...)``) are
    recognised alongside the built-in registry. Centroid protocols accept both cells and centroid
    sources (cells are pooled into a centroid), so only ``population``/``de`` protocols demand
    ``provides="cells"`` — a user centroid vector used there still hits the clear error below.
    """
    for name, src in candidates.items():
        if not ctx.has_source(src):
            raise ValueError(
                f"{p.name}: unknown {name} control source {src!r}; valid sources: {', '.join(ctx.source_names())}"
            )
        if p.representation in ("population", "de") and ctx.source_meta(src).get("provides") != "cells":
            raise ValueError(
                f"{p.name}: {name} source {src!r} provides "
                f"{ctx.source_meta(src).get('provides')!r}, but a single-cell protocol needs cells"
            )
