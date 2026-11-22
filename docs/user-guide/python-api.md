# Python API

scPertEval can be driven two ways: the [command-line interface](usage) (`scperteval …`) and
this native **Python API**. Both run the exact same engine over the same
[input data](usage.md#input-data); the CLI writes result files, while the Python API returns
in-memory [pandas](https://pandas.pydata.org/) objects — ideal for notebooks and scripts.

Install with `pip install scperteval` (or, from this repo,
`pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"`),
then:

```python
import scperteval as sp
```

Everything you need is at the package root; you don't build a `RunConfig` or touch
`Context`.

## Prepare first, then run

The API is always **prepare, then run**. You first call {func}`~scperteval.api.prepare` to
read and index a dataset once (and precompute the feature spaces the protocols you name will
need), which returns a reusable {class}`~scperteval.api.Prepared` handle. You then pass that
handle to {func}`~scperteval.api.calibrate`, {func}`~scperteval.api.score`, or
{func}`~scperteval.api.de` — each evaluates **one** protocol (or computes **one** DE method)
and returns its result in memory.

```python
prep = sp.prepare("data/wessels23.h5ad", ["pearson_ctrl", "mse", "de_auprc"])
```

`dataset` accepts either a path to a `.h5ad` **or** an in-memory
{class}`~anndata.AnnData`, so there's no need to write a temporary file in a notebook. The
second argument accepts the same protocol-spec language as the CLI's `-p`: `"all"`, a group
(`"pseudobulk"` / `"distributional"` / `"de"`), a name, or a tunable protocol as
`name=value` — either a sequence or a comma-separated string. Python callers may also pass
concrete {class}`~scperteval.types.Protocol` objects directly. Pass `[]` if you only need
{func}`~scperteval.api.de`. Prepare-time knobs (`subsample`, `seed`, `min_cells`,
`perturbation_key`, `control_label`, `target_key`, `workers`, `name`, `sources`) are fixed on the
handle.

One handle is **reusable and thread-safe**: run as many `calibrate` / `score` / `de` calls
against it as you like — they share its dataset and caches (no reload), and are safe to run
concurrently.

```python
a = sp.calibrate(prep, "pearson_ctrl")  # reuses the shared dataset + caches
b = sp.calibrate(prep, "mse")
c = sp.de(prep, "t-test")
```

### Concrete protocols, composable spaces & `truth_availability()`

Python callers can pass concrete {class}`~scperteval.types.Protocol` objects and composable
{mod}`scperteval.blocks.spaces` objects directly to `prepare()`, `calibrate()`, `score()`, and
`truth_availability()` without mutating the curated CLI catalog (`Protocol`, `MetricOutcome`,
`PROTOCOLS`, `spaces`, and `metrics` are all lazily re-exported at the package root):

```python
from functools import partial
import scperteval as sp

# Centroid protocol on control-referenced top-50 DEGs (excluding the perturbed target gene)
deg50 = sp.spaces.degs(padj=0.05, top_k=50, reference="control") - sp.spaces.own_target()
proto = sp.Protocol("pearson_deg50_no_target", sp.metrics.pearson, space=deg50)

# Centered centroid on HVGs (calibrate() automatically uses control_mean as negative)
pert_hvg = sp.Protocol("pearson_pert_hvg2k", sp.metrics.pearson, centering="all_perturbed_mean", space=sp.spaces.hvg(2000))

# Population distance on HVG-PCA (representation="population", better="lower", perfect=0.0 inferred)
hvg_pca = sp.spaces.pca(50).on(sp.spaces.hvg(2000))
edist = sp.Protocol("edist_hvg_pca", sp.metrics.energy_distance, space=hvg_pca)

# Dataset-scope retrieval on HVGs (scope="dataset" and default_negative="global_mean" inferred)
nir_hvg = sp.Protocol("nir_hvg2k", sp.metrics.nir, space=sp.spaces.hvg(2000))

# DE protocol: space sets the gene universe; metric kwargs set the DEG cutoff/cap/ranking
de_auprc = sp.Protocol(
    "de_auprc_deg50_no_target",
    partial(sp.metrics.de_auprc, padj=0.05, top_k=50),
    space=sp.spaces.full() - sp.spaces.own_target(),
    reference="control",
)
```

Built-in metrics in `sp.metrics` (including `functools.partial` wrappers) automatically populate
`representation`, `scope`, `better`, and `perfect`: `representation` and `scope` are enforced
against contradictions, while `better` and `perfect` may be overridden. Custom concrete `Protocol`
names must be unique and cannot shadow a curated catalog name in `sp.PROTOCOLS` (e.g. `"mse"`,
`"nir"`, `"de_auprc"`).

**Spaces vs. DE metric parameters:**
- For **`centroid` / `population`** protocols, put DEG selection (and its `reference`) in
  `space=sp.spaces.degs(...)` or `sp.spaces.top(...)`. For DE-weighted centroid metrics
  (`weighted_mse`, `weighted_pearson`, `weighted_r2`), pass `reference=...` via `partial(...)`;
  weights automatically slice to any gene-subset `space`.
- For **`de`** protocols, use `space` for the **gene universe** (`full()`, `hvg(2000)`,
  `full() - own_target()`), put `reference` on `Protocol`, and pass DEG cutoff/cap/ranking
  (`padj`, `top_k`, `rank_by`) to the `de_*` metric via `partial`. `rank_by` accepts
  `"significance"` (adjusted p-value, tie-broken by `|statistic|`) or `"absolute_effect"`
  (`|statistic|`); note that `de_auprc`, `de_auroc`, and `de_overlap` default to
  `"absolute_effect"` while the other `de_*` metrics and `sp.spaces.degs()` default to
  `"significance"`.

**Built-in `sp.spaces` factories:**
- Gene subsets (compose with `|`, `&`, `-`): `full()`, `hvg(k=2000)`, `heg(k=1000)`,
  `top(k=50, *, reference="all_perturbed")`,
  `degs(padj=0.05, *, top_k=None, rank_by="significance", reference="all_perturbed")`,
  `perturbed_genes()`, `own_target()`.
- Transforms (restrict to a dataset-wide subset via `.on(subset)`): `pca(k=50)`.

**Built-in `sp.metrics` functions:**
- **Centroid (`perturbation`):** `pearson`, `spearman`, `cosine`, `squared_pearson`, `r2`,
  `clipped_r2`, `explained_variance`, `mse`, `mae`, `l2`, and DE-weighted `weighted_mse`,
  `weighted_pearson`, `weighted_r2` (`exp=2.0, *, reference="all_perturbed"`).
- **Centroid (`dataset`):** `rank_retrieval(transpose=False)`, `nir`.
- **Population (`perturbation`):** `energy_distance`, `unbiased_mmd_median`, `sinkhorn_w2(blur=0.05)`.
- **DE (`perturbation`):** `de_auprc`, `de_auroc`, `de_overlap(k=50)`, `de_overlap_at_truth_size`,
  `de_direction_concordance`, `de_spearman(degs_only=False)`, `de_recall`, `de_precision`,
  `de_f1`, `de_macro_f1`, `de_jaccard` (supporting `padj`, `top_k`, `rank_by` via `partial`).

**Custom user metrics** remain plain `(gt, prediction, ctx) -> float | sp.MetricOutcome` functions
— pass `representation` explicitly (plus `better` and `perfect` when calling `calibrate()`):

```python
def relative_l1(gt, prediction, ctx):
    denom = float(np.sum(np.abs(gt)))
    if denom == 0.0:
        return sp.MetricOutcome(float("nan"), "zero_ground_truth_norm")
    return float(np.sum(np.abs(gt - prediction)) / denom)

custom = sp.Protocol("relative_l1", relative_l1, representation="centroid", better="lower", perfect=0.0)
```

If perturbation labels are not gene symbols (e.g. namespaced labels like `"CRISPRi_TP53"`), pass
`target_key="target_gene"` to `prepare()` so `sp.spaces.own_target()` can resolve them.

**Auditing ground-truth availability:** {func}`~scperteval.api.truth_availability` accepts a single
protocol **or** a sequence (unlike `calibrate`/`score`, which evaluate one protocol per call) and
returns a DataFrame with columns `["protocol", "perturbation", "ground_truth_defined",
"undefined_reason", "self_score"]`:

```python
avail = sp.truth_availability(prep, [de_auprc, proto], truth="gt_all_cells")  # or truth="gt_half" for calibration
```

## Calibrate

{func}`~scperteval.api.calibrate` calibrates **one** protocol against the built-in
positive/negative controls (DRF or BDS) — the programmatic equivalent of
{func}`scperteval calibrate <scperteval.api.calibrate>`:

```python
res = sp.calibrate(prep, "pearson_ctrl", de_method="t-test")

res.aggregate  # {"mean": …, "median": …} — the DRF summary for this protocol
res.per_perturbation  # DataFrame: raw control values + the calibrated DRF, one row per perturbation
```

`calibrate` takes a **single** protocol — a curated name (`"pearson_ctrl"`), a tunable spec
(`"mse_top_k=30"`), or a concrete `Protocol` object; it does not accept `"all"` or a group. Pass
`calibrator="bds"` for the Bound Discrimination Score instead of DRF. Any perturbation whose raw or
calibrated score is undefined is recorded as `NaN` with a machine-readable `undefined_reason`.

### Your own vectors as sources

{func}`~scperteval.api.prepare` accepts `sources={"name": array}` to register **runtime user
sources** on the handle (never on the global registry, so they don't leak across handles). A 1-D
`(G,)` array is a centroid; a 2-D `(n_cells, G)` array is a cell population. Arrays are validated
(numeric, all-finite, `G` = the dataset's gene count) and copied. **Columns are assumed to be in
`adata.var_names` order** — only the count is checked, so a mis-ordered vector silently compares the
wrong genes.

A registered source is reusable in any slot that accepts its shape:

```python
prep = sp.prepare(adata, "pearson", sources={"my_baseline": vec})  # vec is a (G,) centroid
sp.calibrate(prep, "pearson", negative="my_baseline")  # ... as a control
sp.calibrate(prep, "pearson", center_on="my_baseline")  # ... as a centering baseline
```

`center_on` centers an **un-centred, centroid** protocol on a named centroid source (user or
built-in, e.g. `"global_mean"`). Because centering is protocol identity, this **mints a named
variant** `<protocol>_center_<name>` — recorded in `EvalResult` and any CSV — rather than silently
overriding the catalog protocol. It also works on {func}`~scperteval.api.score`. (The CLI can't pass
arrays, so `center_on` is Python-API-only.)

{func}`~scperteval.api.calibrate` returns an {class}`~scperteval.api.EvalResult`:

- `.aggregate` — a `dict` of the calibrator's summary stats (`mean`/`median` for DRF,
  `bds` for BDS).
- `.per_perturbation` — a {class}`~pandas.DataFrame`, one row per perturbation, with columns
  `protocol`, `perturbation`, the resolved control source(s) (`positive`/`negative` or
  `prediction`), `raw_*`, `undefined_reason`, and the calibrated/raw score (`drf`/`bds`/`score`).

By default nothing is written; pass `out_dir="results"` to also write the CLI-style CSV.

## Score

{func}`~scperteval.api.score` scores a model's predictions against ground truth for **one**
protocol — the equivalent of {func}`scperteval score <scperteval.api.score>`. Note the
argument order: `(prepared, protocol, predictions)`. Predictions accept a path or an
in-memory {class}`~anndata.AnnData`, and must have the same genes and perturbation labels as
the dataset:

```python
res = sp.score(prep, "pearson", "predictions.h5ad")
res.aggregate  # {"mean": …, "median": …} — the raw metric summary
res.per_perturbation  # DataFrame with a `score` column per perturbation
```

## Differential expression

{func}`~scperteval.api.de` computes per-gene differential expression for **one** method — the
equivalent of {func}`scperteval de <scperteval.api.de>` (defaulting to `target="gt_half"` vs
`reference="all_perturbed"`):

```python
d = sp.de(prep, "t-test")

# All observed cells vs non-targeting controls, for a chosen subset of perturbations
d_ctrl = sp.de(prep, "t-test", target="gt_all_cells", reference="control", perturbations=["pertA"])

d.statistic  # DataFrame: perturbations × genes (the test statistic)
d.pvalue_adj  # DataFrame: perturbations × genes (BH-adjusted p-values)
```

{func}`~scperteval.api.de` returns a {class}`~scperteval.api.DatasetDEResults`, a
`NamedTuple` of two DataFrames (`statistic` and BH-adjusted `pvalue_adj`; raw unadjusted p-values
are not returned) that also unpacks directly:

```python
statistic, pvalue_adj = sp.de(prep, "MWU")
```

Different DE methods reuse the same prepared dataset (each cached separately, no reload).

See the [Python API reference](../api/api) for full signatures.
