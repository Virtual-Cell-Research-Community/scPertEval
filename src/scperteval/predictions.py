"""Model-predicted cells, loaded from a separate .h5ad and aligned to the dataset's genes.

In prediction-scoring mode the ground truth comes from the real dataset and the candidate
comes from here. The prediction file must contain exactly the dataset's genes (same set,
any order) and the same perturbation column; columns are reordered to the dataset's gene
order so every metric's positional ``gt - prediction`` comparison lines up.
"""

from __future__ import annotations

import numpy as np

from .dataset import Dataset
from .types import RunConfig


def _align_genes(pred_genes: np.ndarray, ds_genes: np.ndarray) -> np.ndarray:
    """Indices that reorder the prediction's genes into the dataset's gene order.

    Errors (naming what's wrong) unless the two gene sets are identical -- metrics compare
    gene vectors positionally, so a mismatch would silently compare the wrong genes.
    """
    pred_set, ds_set = set(map(str, pred_genes)), set(map(str, ds_genes))
    missing = [g for g in map(str, ds_genes) if g not in pred_set]
    extra = [g for g in map(str, pred_genes) if g not in ds_set]
    if missing or extra:

        def show(names):
            return ", ".join(names[:10]) + (f", … (+{len(names) - 10} more)" if len(names) > 10 else "")

        parts = []
        if missing:
            parts.append(f"predictions are missing {len(missing)} of the dataset's genes: {show(missing)}")
        if extra:
            parts.append(f"predictions have {len(extra)} genes not in the dataset: {show(extra)}")
        raise ValueError("prediction/dataset gene mismatch — " + "; ".join(parts))
    pos = {str(g): i for i, g in enumerate(pred_genes)}
    return np.array([pos[str(g)] for g in ds_genes], dtype=int)


class PredictionSet:
    """Predicted cells per perturbation, gene-aligned to a :class:`Dataset`."""

    def __init__(self, adata, ds: Dataset, cfg: RunConfig):
        self.cfg = cfg
        self.adata = adata
        self._reorder = _align_genes(np.asarray(adata.var_names), ds.var_names)
        self.pert = np.asarray(adata.obs[cfg.perturbation_key]).astype(str)

    @classmethod
    def load(cls, path: str, ds: Dataset, cfg: RunConfig) -> PredictionSet:
        """Load a prediction ``.h5ad`` and gene-align it to ``ds``."""
        from .io import read_h5ad

        return cls(read_h5ad(path, "predictions"), ds, cfg)

    def cells(self, pert: str) -> np.ndarray:
        """Predicted cells for one perturbation, columns in the dataset's gene order."""
        idx = np.where(self.pert == pert)[0]
        if len(idx) == 0:
            raise ValueError(
                f"predictions contain no cells for perturbation {pert!r} "
                f"(it is evaluated in the dataset but absent from the prediction file)"
            )
        return self.adata.X[idx][:, self._reorder]
