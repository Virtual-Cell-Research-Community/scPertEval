"""Runs one protocol over every perturbation and applies the chosen calibrator."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .sources import SOURCES
from .types import Calibrator, Protocol


def _n_workers(cfg) -> int:
    """Resolve the worker-thread count (``0`` = auto: CPU count minus 2, capped at 16)."""
    return cfg.workers if cfg.workers > 0 else max(1, min(16, (os.cpu_count() or 2) - 2))


def _resolve_candidates(p: Protocol, cfg) -> dict:
    """Map each calibrator input (candidate name) to a source name.

    ``positive`` / ``negative`` come from the protocol (or a CLI override); ``prediction``
    is always the model-prediction source (used by the ``score`` calibrator).
    """
    return {
        "positive": cfg.positive if cfg.positive != "auto" else p.positive,
        "negative": cfg.negative if cfg.negative != "auto" else p.negative,
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
    _check_sources(p, needed)
    run = _run_dataset if p.scope == "dataset" else _run_per_perturbation
    return run(p, ctx, calibrator, needed)


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


def _finalize(p, calibrator, perts, raws_list):
    """Per-perturbation rows + the aggregate, from each perturbation's raw control values."""
    per_pert = [calibrator.per_pert(raws, p) for raws in raws_list]
    rows = [
        {
            "protocol": p.name,
            "perturbation": pert,
            **{f"raw_{name}": raws[name] for name in raws},
            calibrator.name: value,
        }
        for pert, raws, value in zip(perts, raws_list, per_pert)
    ]
    return calibrator.aggregate(np.asarray(per_pert, dtype=float)), rows


def _run_per_perturbation(p: Protocol, ctx, calibrator: Calibrator, needed: dict):
    """Score one perturbation at a time (across a thread pool), gt vs each control."""

    def work(pert):
        ctx.current_pert = pert
        gt = ctx.view(pert, ctx.cfg.truth, p)
        return {name: p.metric(gt, ctx.view(pert, src, p), ctx) for name, src in needed.items()}

    perts = ctx.perturbations
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=_n_workers(ctx.cfg)) as pool:
        raws_list = list(pool.map(work, perts))
    seconds = perf_counter() - start
    agg, rows = _finalize(p, calibrator, perts, raws_list)
    return agg, rows, seconds


def _run_dataset(p: Protocol, ctx, calibrator: Calibrator, needed: dict):
    """Score dataset-scope protocols by handing the metric all perturbations at once.

    Build every perturbation's gt and control datapoints, call the metric once on the full
    lists, then read off each perturbation's score. Perturbations are treated as a single
    group (these datasets are single-covariate); drf instead ranks within each covariate group.
    """
    perts = ctx.perturbations

    def collect(source):
        out = []
        for pert in perts:
            ctx.current_pert = pert
            out.append(ctx.view(pert, source, p))
        return out

    start = perf_counter()
    gt = collect(ctx.cfg.truth)
    scores = {name: p.metric(gt, collect(src), ctx) for name, src in needed.items()}
    seconds = perf_counter() - start

    raws_list = [{name: float(scores[name][i]) for name in needed} for i in range(len(perts))]
    agg, rows = _finalize(p, calibrator, perts, raws_list)
    return agg, rows, seconds


def compute_de(ctx):
    """Per-gene DE matrices for ``ctx.cfg.de_method``, for export.

    Returns ``(statistic, pvalue_adj)`` matrices (perturbations x genes) for the
    ground-truth-vs-all-perturbed differential expression under the context's DE method.
    """

    def work(pert):
        d = ctx.de(pert, ctx.cfg.truth, "all_perturbed")
        return d.statistic, d.pvalue_adj

    with ThreadPoolExecutor(max_workers=_n_workers(ctx.cfg)) as pool:
        res = list(pool.map(work, ctx.perturbations))
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def _check_sources(p: Protocol, candidates: dict) -> None:
    if p.representation in ("population", "de"):
        for name, src in candidates.items():
            if SOURCES.meta(src).get("provides") != "cells":
                raise ValueError(
                    f"{p.name}: {name} source {src!r} provides "
                    f"{SOURCES.meta(src).get('provides')}, but a single-cell protocol needs cells"
                )
