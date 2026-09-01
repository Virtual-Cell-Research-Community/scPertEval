# Extension API

Reference for the building blocks used to extend scPertEval with custom protocols,
DE backends, feature spaces, control sources, and calibrators.

## Context

```{eval-rst}
.. module:: scperteval.context
.. currentmodule:: scperteval.context

.. autosummary::
    :toctree: ../generated

    Context
```

## Registry

```{eval-rst}
.. module:: scperteval.registry
.. currentmodule:: scperteval.registry

.. autosummary::
    :toctree: ../generated

    Registry
```

## Caching

```{eval-rst}
.. module:: scperteval.caching
.. currentmodule:: scperteval.caching

.. autosummary::
    :toctree: ../generated

    cached
    DatasetScope
```

## Feature spaces

```{eval-rst}
.. module:: scperteval.blocks.spaces
.. currentmodule:: scperteval.blocks.spaces

.. automodule:: scperteval.blocks.spaces
   :no-members:
   :no-index:

.. autosummary::
    :toctree: ../generated

    SPACES
    SpaceRegistry
    Space
```

## DE backends

```{eval-rst}
.. module:: scperteval.blocks.de
.. currentmodule:: scperteval.blocks.de

.. automodule:: scperteval.blocks.de
   :no-members:
   :no-index:

.. autosummary::
    :toctree: ../generated

    DE_METHODS
    bh
    ttest_from_moments
    de_ttest
    de_ttest_overestim
    de_mwu
```

## Control sources

```{eval-rst}
.. module:: scperteval.sources
.. currentmodule:: scperteval.sources

.. automodule:: scperteval.sources
   :no-members:
   :no-index:

.. autosummary::
    :toctree: ../generated

    SOURCES
```

Add entries here to register a new source; see [Add a control source](../user-guide/building-blocks.md#add-a-control-source).

## Predictions

```{eval-rst}
.. module:: scperteval.predictions
.. currentmodule:: scperteval.predictions

.. autosummary::
    :toctree: ../generated

    PredictionSet
```

`scperteval.predictions.PredictionSet` — model-predicted cells loaded from a `.h5ad` and
gene-aligned to the dataset, used by the `score` command.
