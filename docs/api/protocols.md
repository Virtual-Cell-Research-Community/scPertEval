# Protocols reference

## Protocol table

```{eval-rst}
.. module:: scperteval.protocols.table
.. currentmodule:: scperteval.protocols.table

.. autosummary::
    :toctree: ../generated

    TABLE
    PROTOCOLS
    GROUPS
```

## Metrics

Full reference for the built-in metric functions. Each function implements one protocol's
core computation: it receives two arrays (or `PerturbationDEResult` objects) and returns a scalar score.

```{eval-rst}
.. module:: scperteval.protocols.metrics
.. currentmodule:: scperteval.protocols.metrics

.. automodule:: scperteval.protocols.metrics
   :no-members:
   :no-index:

.. autosummary::
    :toctree: ../generated

    pearson
    spearman
    cosine
    squared_pearson
    r2
    clipped_r2
    explained_variance
    mse
    mae
    l2
    weighted_mse
    weighted_pearson
    weighted_r2
    energy_distance
    unbiased_mmd_median
    sinkhorn_w2
    rank_retrieval
    nir
    de_auprc
    de_auroc
    de_overlap
    de_overlap_at_truth_size
    de_direction_concordance
    de_spearman
    de_recall
    de_precision
    de_f1
    de_macro_f1
    de_jaccard
```
