"""Control/reference sources: each yields a perturbation's cells or pseudobulk centroid.

A source's positive/negative role is chosen at the CLI; the registry just produces
the data. ``provides`` ("cells" or "centroid") drives the runner's compatibility
check and how the context turns a source into a view. ``description`` is shown by
``scperteval list sources``.
"""

from __future__ import annotations

import numpy as np

from .dataset import to_dense
from .registry import Registry

SOURCES = Registry("source")
"""Registry of control/reference sources; keys are source names (e.g. ``"gt_half"``).

Use :meth:`~scperteval.registry.Registry.register` to add a custom source::

    from scperteval.sources import SOURCES

    @SOURCES.register("my_source", provides="cells", description="…")
    def src_my_source(ctx, pert):
        ...                            # cells or a centroid for `pert`
        return cells

Set ``provides`` to ``"cells"`` or ``"centroid"`` to match what the function returns.
"""


@SOURCES.register(
    "gt_half",
    provides="cells",
    description="ground truth — the first half of a perturbation's cells (calibration truth)",
)
def src_gt_half(ctx, pert):
    """Ground-truth cells: the first half of the perturbation's cells."""
    return ctx.ds.cells(pert, half="first")


@SOURCES.register(
    "gt_all_cells",
    provides="cells",
    description="ground truth — all of a perturbation's real cells (prediction-scoring truth)",
)
def src_gt_all_cells(ctx, pert):
    """Ground-truth cells: all of the perturbation's real cells."""
    return ctx.ds.cells(pert)


@SOURCES.register(
    "prediction",
    provides="cells",
    cacheable=False,  # cells come from the per-call predictions, not the dataset — never cache in the shared store
    description="model-predicted cells for the perturbation, from the --predictions h5ad",
)
def src_prediction(ctx, pert):
    """Model-predicted cells for the perturbation, gene-aligned to the dataset."""
    return ctx.predictions.cells(pert)


@SOURCES.register(
    "tech_dup",
    provides="cells",
    description="technical duplicate — the held-out second half (single-cell positive control)",
)
def src_tech_dup(ctx, pert):
    """Technical-duplicate cells: the perturbation's held-out second half."""
    return ctx.ds.cells(pert, half="second")


@SOURCES.register("control", provides="cells", description="non-targeting control cells")
def src_control(ctx, pert):
    """Non-targeting control cells (subsampled)."""
    return ctx.ds.control_cells(ctx.cfg.subsample)


@SOURCES.register(
    "all_perturbed",
    provides="cells",
    description="all-perturbed reference sample, leave-one-out (single-cell negative control)",
)
def src_all_perturbed(ctx, pert):
    """All-perturbed reference cells, with the target perturbation removed."""
    return ctx.reference().subset(pert)


@SOURCES.register(
    "all_perturbed_mean",
    provides="centroid",
    description="all-perturbed mean, excluding the target — leave-one-out "
    "(pseudobulk sibling of all_perturbed; pseudobulk negative control)",
)
def src_all_perturbed_mean(ctx, pert):
    """All-perturbed pseudobulk mean, excluding the target perturbation."""
    return ctx.ds.all_perturbed_mean_except(pert)


@SOURCES.register(
    "global_mean",
    provides="centroid",
    description="mean of all perturbations — shared baseline for the ranking protocols",
)
def src_global_mean(ctx, pert):
    """Pseudobulk mean over all perturbations (no target exclusion)."""
    return ctx.ds.all_perturbed_mean()


@SOURCES.register(
    "interpolated",
    provides="centroid",
    description="interpolated duplicate — DE-weighted blend of the held-out half and "
    "the dataset mean (pseudobulk positive control)",
)
def src_interpolated(ctx, pert):
    """DE-weighted blend toward the held-out replicate, else the all-perturbed mean.

    Implements the interpolated duplicate positive control of :cite:p:`Miller_2025`.
    Alpha = 1 - adjusted p per gene (from the run's DE method, vs all other perturbed cells
    (leave-one-out)); blend toward the held-out replicate where the gene is significant, else
    toward the all-perturbed mean.
    """
    tech = np.asarray(to_dense(ctx.ds.cells(pert, half="second"))).mean(0)
    alpha = np.nan_to_num(1.0 - ctx.de(pert, "tech_dup", "all_perturbed").pvalue_adj, nan=0.0)
    return alpha * tech + (1.0 - alpha) * ctx.ds.all_perturbed_mean_except(pert)
