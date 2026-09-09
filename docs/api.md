# API reference

```{toctree}
:hidden:

api/api
api/architecture
api/types
api/protocols
api/extensions
api/io
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`code;1em;` Python API
:link: api/api
:link-type: doc

The public functions: `prepare`, `calibrate`, `score`, `compare`, `cross_calibrate`, `de`, and their result types
`Prepared` / `EvalResult` / `DatasetDEResults`.
:::

:::{grid-item-card} {octicon}`workflow;1em;` Architecture
:link: api/architecture
:link-type: doc

How datasets, `Context`, the building-block registries, and `Protocol`/`Calibrator` fit
together at runtime.
:::

:::{grid-item-card} {octicon}`cpu;1em;` Core types & runner
:link: api/types
:link-type: doc

`RunConfig`, `Protocol`, `Calibrator`, `PerturbationDEResult`, `Param`, `run_protocol`
:::

:::{grid-item-card} {octicon}`checklist;1em;` Protocols
:link: api/protocols
:link-type: doc

Built-in metric functions: `pearson`, `mse`, `de_auprc`, `rank_retrieval`, …
:::

:::{grid-item-card} {octicon}`plug;1em;` Extension API
:link: api/extensions
:link-type: doc

`Registry`, `SPACES`, `DE_METHODS`, `SOURCES`, `PredictionSet`, `Context`
:::

:::{grid-item-card} {octicon}`database;1em;` Dataset
:link: api/io
:link-type: doc

`Dataset`, `to_dense`, `write_rows`, `write_de`, …
:::

::::

## Protocols

- {obj}`~scperteval.protocols.table.TABLE` — list of all `Protocol` objects.
- {obj}`~scperteval.protocols.table.PROTOCOLS` — `{name: Protocol}` dict.
- {obj}`~scperteval.protocols.table.GROUPS` — sorted list of group names.

```{eval-rst}
.. protocol-table::
```
