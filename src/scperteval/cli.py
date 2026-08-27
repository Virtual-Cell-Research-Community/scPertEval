"""scPertEval command-line interface."""

from __future__ import annotations

import argparse
from datetime import datetime

from . import io
from .blocks.de import DE_METHODS
from .blocks.spaces import SPACES
from .calibrators import CALIBRATORS
from .context import Context
from .dataset import Dataset
from .predictions import PredictionSet
from .protocols.resolve import _concrete, _resolve_token, available, resolve_protocols  # noqa: F401 (re-exported)
from .protocols.table import TABLE
from .runner import _resolve_candidates, compute_de, run_all
from .sources import SOURCES
from .types import RunConfig


def _evaluate(cfg: RunConfig, protocols, ctx, quiet: bool) -> None:
    """Run every protocol over the dataset, print the summary, and write the CSV.

    Shared by ``calibrate`` and ``score`` (prediction vs ground truth); they differ only in
    how ``ctx`` is built and which calibrator ``cfg.calibrator`` selects.
    """
    aggregates, rows, timed = run_all(cfg, protocols, ctx)
    if not quiet:
        controls = {p.name: _resolve_candidates(p, cfg) for p in protocols}
        io._print_summary(cfg, aggregates, CALIBRATORS[cfg.calibrator], protocols, controls)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = io.write_rows(cfg, rows, stamp)
    print(f"-> {path}")
    if cfg.profile:
        print(f"-> {io._write_timing(cfg, timed, stamp)}")


def cmd_calibrate(args) -> None:
    """Run the ``calibrate`` command: score protocols against built-in controls (DRF/BDS)."""
    protocols = resolve_protocols(args.protocols or ["all"])
    cfg = RunConfig(
        dataset=args.dataset,
        protocols=[p.name for p in protocols],
        de_method=args.de_method,
        subsample=args.subsample,
        seed=args.seed,
        positive=args.positive,
        negative=args.negative,
        calibrator=args.calibrator,
        out_dir=args.out_dir,
        workers=args.workers,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        min_cells=args.min_cells,
        profile=args.profile,
    )
    ctx = Context(Dataset.load(cfg.dataset, cfg), cfg)
    _evaluate(cfg, protocols, ctx, args.quiet)


def cmd_score(args) -> None:
    """Run the ``score`` command: score predictions against ground truth, per protocol."""
    protocols = resolve_protocols(args.protocols or ["all"])
    cfg = RunConfig(
        dataset=args.dataset,
        protocols=[p.name for p in protocols],
        de_method=args.de_method,
        subsample=args.subsample,
        seed=args.seed,
        calibrator="score",
        out_dir=args.out_dir,
        workers=args.workers,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        min_cells=args.min_cells,
        profile=args.profile,
        predictions=args.predictions,
        truth="gt_all_cells",
    )
    assert cfg.predictions is not None  # required positional on the score subcommand
    ds = Dataset.load(cfg.dataset, cfg)
    ctx = Context(ds, cfg)
    ctx.predictions = PredictionSet.load(cfg.predictions, ds, cfg)
    _evaluate(cfg, protocols, ctx, args.quiet)


def cmd_de(args) -> None:
    """Run the ``de`` command: export per-gene differential expression for one method to HDF5."""
    cfg = RunConfig(
        dataset=args.dataset,
        protocols=[],
        de_method=args.method,
        subsample=args.subsample,
        seed=args.seed,
        out_dir=args.out_dir,
        workers=args.workers,
        min_cells=args.min_cells,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
    )
    ctx = Context(Dataset.load(cfg.dataset, cfg), cfg)
    ctx._ensure_reference_sums()
    statistic, pvalue_adj = compute_de(ctx)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = io.write_de(cfg, ctx.ds.var_names, ctx.perturbations, {args.method: (statistic, pvalue_adj)}, stamp)
    if not args.quiet:
        print(f"-> {path}  ({len(ctx.perturbations)} perturbations, method={args.method})")


def cmd_list(args) -> None:
    """Run the ``list`` command: print the available building blocks of one category."""

    def reg(registry, fmt):
        return [fmt(n, registry.meta(n)) for n in registry.names()]

    if args.what == "protocols":
        default_cfg = RunConfig(dataset="", protocols=[])  # no override: shows the resolved default controls

        def descr(p):
            scope = "" if p.scope == "perturbation" else f", {p.scope}-wide"
            knob = f"{p.param.name}=…" if p.parameterised else f"space={p.space}"
            c = _resolve_candidates(p, default_cfg)
            return f"{p.group}, {p.representation}{scope}, {knob}, controls +{c['positive']}/-{c['negative']}"

        def extra_note(p):
            if p.requires_extra is None:
                return ""
            state = "installed" if available(p) else "NOT installed"
            return f"  [needs '{p.requires_extra}' extra — {state}]"

        lines = [f"{p.name:24s} ({descr(p)}){extra_note(p)}" for p in TABLE]
    elif args.what == "de-methods":
        lines = reg(DE_METHODS, lambda n, m: f"{n:10s} — {m.get('description', '')}")
    elif args.what == "spaces":
        lines = reg(SPACES, lambda n, m: f"{n:10s} — {m.get('description', '')}")
    elif args.what == "sources":
        lines = reg(SOURCES, lambda n, m: f"{n:14s} ({m.get('provides')}) — {m.get('description', '')}")
    elif args.what == "calibrators":
        lines = [f"{n:6s} — {c.description}" for n, c in CALIBRATORS.items()]
    else:
        raise AssertionError(f"unexpected list target: {args.what!r}")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the ``scperteval`` argument parser (extracted so tests can inspect defaults)."""
    parser = argparse.ArgumentParser(prog="scperteval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    calibrate = sub.add_parser("calibrate", help="calibrate protocols against positive/negative controls (DRF/BDS)")
    calibrate.add_argument("dataset", help="preprocessed .h5ad")
    calibrate.add_argument(
        "-p",
        "--protocols",
        action="append",
        default=[],
        help="comma-separated names, a group (pseudobulk|distributional|de), or 'all'",
    )
    calibrate.add_argument(
        "--de-method",
        choices=DE_METHODS.names(),
        default="t-test",
        help="DE backend for EVERY DE-dependent unit in the run: the interpolated "
        "positive control, the top_k/degs spaces, the de_* protocols, and the WMSE weights",
    )
    calibrate.add_argument("--subsample", type=int, default=8192)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--positive", default="auto")
    calibrate.add_argument("--negative", default="auto")
    calibrate.add_argument(
        "--calibrator",
        default="drf",
        choices=[n for n, c in CALIBRATORS.items() if "prediction" not in c.requires],
        help="how per-perturbation values are calibrated (drf/bds)",
    )
    calibrate.add_argument("--out-dir", default="results")
    calibrate.add_argument("--workers", type=int, default=0, help="threads (0 = auto)")
    calibrate.add_argument("--perturbation-key", default="perturbation")
    calibrate.add_argument("--control-label", default="control")
    calibrate.add_argument("--min-cells", type=int, default=30, help="skip perturbations with fewer cells")
    calibrate.add_argument("--profile", action="store_true", help="also write a per-protocol wall-clock timing table")
    calibrate.add_argument("--quiet", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    score = sub.add_parser("score", help="score model predictions against ground truth (real cells), per protocol")
    score.add_argument("dataset", help="preprocessed .h5ad — the ground truth (real cells)")
    score.add_argument("predictions", help="predicted .h5ad — same genes and perturbation labels")
    score.add_argument(
        "-p",
        "--protocols",
        action="append",
        default=[],
        help="comma-separated names, a group (pseudobulk|distributional|de), or 'all'",
    )
    score.add_argument(
        "--de-method",
        choices=DE_METHODS.names(),
        default="t-test",
        help="DE backend for every DE-dependent unit (the top_k/degs spaces, the de_* protocols, and the WMSE weights)",
    )
    score.add_argument(
        "--subsample",
        type=int,
        default=8192,
        help="cells in the all-perturbed reference sample (the ground truth itself is never subsampled)",
    )
    score.add_argument("--seed", type=int, default=42)
    score.add_argument("--out-dir", default="results")
    score.add_argument("--workers", type=int, default=0, help="threads (0 = auto)")
    score.add_argument("--perturbation-key", default="perturbation")
    score.add_argument("--control-label", default="control")
    score.add_argument("--min-cells", type=int, default=30, help="skip perturbations with fewer cells")
    score.add_argument("--profile", action="store_true", help="also write a per-protocol wall-clock timing table")
    score.add_argument("--quiet", action="store_true")
    score.set_defaults(func=cmd_score)

    de = sub.add_parser("de", help="write per-gene DE (statistic + adj p) for one method to HDF5")
    de.add_argument("dataset", help="preprocessed .h5ad")
    de.add_argument(
        "--method",
        choices=DE_METHODS.names(),
        default="t-test",
        help="DE method to compute (GT first-half vs all-perturbed)",
    )
    de.add_argument("--subsample", type=int, default=8192)
    de.add_argument("--seed", type=int, default=42)
    de.add_argument("--out-dir", default="results")
    de.add_argument("--workers", type=int, default=0)
    de.add_argument("--min-cells", type=int, default=30)
    de.add_argument("--perturbation-key", default="perturbation")
    de.add_argument("--control-label", default="control")
    de.add_argument("--quiet", action="store_true")
    de.set_defaults(func=cmd_de)

    lst = sub.add_parser("list", help="list available building blocks")
    lst.add_argument("what", choices=["protocols", "de-methods", "spaces", "sources", "calibrators"])
    lst.set_defaults(func=cmd_list)

    return parser


def main(argv=None) -> None:
    """Parse arguments and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ValueError as e:  # e.g. an unknown protocol spec — a clean CLI error, not a traceback
        raise SystemExit(str(e)) from e
