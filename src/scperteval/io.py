"""Human-readable summary plus a per-perturbation CSV named with dataset + time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _print_summary(cfg, aggregates: dict, calibrator, protocols, controls: dict) -> None:
    """Print a formatted table of aggregate scores (and resolved controls) for every protocol."""
    name = Path(cfg.dataset).stem
    print(f"\n{name} · {cfg.de_method} · subsample={cfg.subsample} · seed={cfg.seed} · calibrator={cfg.calibrator}\n")
    agg_keys = sorted({k for v in aggregates.values() for k in v})
    header = f"{'protocol':26s} {'representation':14s} {'space':9s} {'+/- controls':33s} " + " ".join(
        f"{k:>9s}" for k in agg_keys
    )
    print(header)
    print("-" * len(header))
    for p in protocols:
        vals = aggregates.get(p.name, {})
        cells = " ".join(f"{vals.get(k, float('nan')):>9.3f}" for k in agg_keys)
        c = controls[p.name]
        ctrl = f"+{c['positive']}/-{c['negative']}"
        print(f"{p.name:26s} {p.representation:14s} {p.space:9s} {ctrl:33s} {cells}")
    print()


def rows_frame(cfg, rows: list) -> pd.DataFrame:
    """Per-perturbation rows plus run-provenance columns, as a DataFrame.

    Shared by the CLI (which writes it to CSV) and the Python API (which returns it), so
    both produce identical per-perturbation tables.
    """
    df = pd.DataFrame(rows)
    for col, val in (
        ("dataset", Path(cfg.dataset).stem),
        ("de_method", cfg.de_method),
        ("subsample", cfg.subsample),
        ("seed", cfg.seed),
    ):
        df[col] = val
    return df


def write_rows(cfg, rows: list, timestamp: str) -> Path:
    """Write per-perturbation rows (raw controls + calibrated score) to a timestamped CSV.

    A single-protocol run (the Python API's one-protocol-per-call) puts the protocol name in the
    filename so per-protocol calls to one ``out_dir`` don't collide; a multi-protocol CLI run
    (one CSV holds every protocol) keeps the plain name.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.protocols[0]}__" if len(cfg.protocols) == 1 else ""
    path = out_dir / f"{Path(cfg.dataset).stem}__{tag}{timestamp}__{cfg.calibrator}.csv"
    rows_frame(cfg, rows).to_csv(path, index=False)
    return path


def _write_timing(cfg, timed: list, timestamp: str) -> Path:
    """Write per-protocol wall-clock seconds (one row per protocol)."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": Path(cfg.dataset).stem,
            "protocol": p.name,
            "representation": p.representation,
            "space": p.space,
            "seconds": seconds,
        }
        for p, seconds in timed
    ]
    path = out_dir / f"{Path(cfg.dataset).stem}__{timestamp}__timing.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_de(cfg, genes, perturbations, results: dict, timestamp: str) -> Path:
    """Write per-gene DE (statistic + adjusted p) per method to an HDF5 file.

    Layout: ``genes``, ``perturbations``, and one group per method holding
    ``statistic`` and ``pvalue_adj`` matrices (perturbations x genes).
    """
    import h5py

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(cfg.dataset).stem}__{cfg.de_method}__{timestamp}__de.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("genes", data=np.asarray(genes, dtype="S"))
        f.create_dataset("perturbations", data=np.asarray(perturbations, dtype="S"))
        for method, (stat, padj) in results.items():
            g = f.create_group(method)
            g.create_dataset("statistic", data=stat)
            g.create_dataset("pvalue_adj", data=padj)
    return path
