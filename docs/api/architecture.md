# Architecture

This page traces how one `scperteval` run flows through the pieces described in the
[API reference](../api): the dataset is loaded, a {class}`~scperteval.context.Context`
caches shared computations, each {class}`~scperteval.types.Protocol` reads a view of the
data through the registered building blocks, and a {class}`~scperteval.types.Calibrator`
turns raw metric values into a score.

```{mermaid} ../_mermaid/architecture.mmd
:caption: How a run flows from the dataset through the Context, the building-block registries, and a Protocol/Calibrator pair to a score.
```

## Dataset and Context

{meth}`~scperteval.dataset.Dataset.load` reads a preprocessed `.h5ad` (downloaded from
`gs://scperteval/processed/`, see [Datasets](../user-guide/datasets)), splits each
perturbation's cells into two reproducible halves, and exposes cell/centroid accessors. In
scoring mode, a second file is loaded via `--predictions` into a
{class}`~scperteval.predictions.PredictionSet`, gene-aligned to the dataset.

Both feed a {class}`~scperteval.context.Context` — the per-run engine, instantiated once
and passed to every metric. It lazily builds and caches the shared, expensive computations
(the fitted PCA, DE moments, the leave-one-out all-perturbed reference sample) and turns a
`(perturbation, source)` pair into the exact view a protocol needs.

## Building blocks

Three registries, keyed by name and looked up through the `Context`:

- **Spaces** ({obj}`~scperteval.blocks.spaces.SPACES`) — a transform applied to the gene
  axis before scoring. Each is a decorated rule in `blocks/spaces/catalog.py`: gene subsets
  (`full`, `top_<k>`, `degs_<padj>`, `heg_<k>`, `hvg_<k>`, `perturbed_genes`,
  `perturbed_and_hvgs`) and one transform, `pca_<k>`. Definitions become registered instances via
  {meth}`~scperteval.blocks.spaces.SpaceRegistry.instance`; subsets fold together with
  {meth}`~scperteval.blocks.spaces.SpaceRegistry.combine_subsets`.
- **DE backends** ({obj}`~scperteval.blocks.de.DE_METHODS`) — differential-expression
  methods sharing one {class}`~scperteval.types.PerturbationDEResult` interface: `t-test`
  ({func}`~scperteval.blocks.de.de_ttest`, default, moment-based),
  `t-test_overestim_var` ({func}`~scperteval.blocks.de.de_ttest_overestim`,
  {func}`scanpy.tl.rank_genes_groups`'s conservative variant), `MWU`
  ({func}`~scperteval.blocks.de.de_mwu`, Mann-Whitney U / Cliff's delta).
- **Sources** ({obj}`~scperteval.sources.SOURCES`) — where a perturbation's cells or
  centroid come from: the ground truth (`gt_half`, `gt_all_cells`), `control`,
  `prediction`, and the built-in positive/negative controls used by calibration
  (`tech_dup`, `all_perturbed`, `all_perturbed_mean`, `global_mean`, `interpolated`).

See [Building blocks](../user-guide/building-blocks) for how to register a new one.

## RunConfig, Protocol, Calibrator

- {class}`~scperteval.types.RunConfig` — the resolved CLI options for one run: which
  protocols to run, the DE backend, the ground-truth label (`truth`), and (in scoring mode)
  the `predictions` path.
- {class}`~scperteval.types.Protocol` — pairs a pure `metric` function with
  `representation` (the shape of one perturbation's datapoint: `centroid`, `population`, or
  `de`) and `scope` (whether the metric is called once per perturbation or once over the
  whole dataset). See [Protocols](../user-guide/protocols) for how protocols are defined
  and looked up.
- {class}`~scperteval.types.Calibrator` — turns raw per-control metric values into a final
  score: `drf` (Dynamic Range Fraction) and `bds` (Bound Discrimination Score) for
  calibration mode, `score` for prediction-scoring mode ({obj}`~scperteval.calibrators.CALIBRATORS`).

## Outputs

{func}`~scperteval.runner.run_protocol` runs every protocol over every perturbation and,
driven by the chosen `Calibrator`, writes per-perturbation rows plus aggregates to a
`scores.csv`. The `scperteval de` command is a separate path:
{func}`~scperteval.runner.compute_de` computes per-gene DE directly from the
`Context` — independent of any `Protocol` or `Calibrator` — and writes it to `de.h5`.
