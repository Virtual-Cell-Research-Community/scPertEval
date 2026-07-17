# Protocols

## Look up an evaluation protocol

Two files define each protocol:

- **[`src/scperteval/protocols/metrics.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/protocols/metrics.py)** — the metric, as a
  pure function of the ground truth and a `prediction` (the candidate being scored — a positive
  or negative control under `calibrate`, or your model's output under `score`). e.g. `mse`,
  `mmd`, `de_auprc`:

  ```python
  def mse(gt, prediction, ctx):
      return float(np.mean((gt - prediction) ** 2))
  ```

- **[`src/scperteval/protocols/table.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/protocols/table.py)** — one row wiring that function
  to its data: the data representation it receives (`representation`), feature space,
  reference centering, positive/negative controls, which direction is `better`
  (`"higher"`/`"lower"`), and the `perfect` score:

  ```python
  Protocol("mse", M.mse, representation="centroid",
           positive="interpolated", negative="all_perturbed_mean", better="lower", perfect=0.0)
  ```

The next section breaks these arguments down while building one up from scratch.

## Create a protocol

A protocol is two things: a pure metric **function** and a one-line **spec** that wires it
to data and scoring. We'll ease in — the simplest possible protocol first, then the spec
broken down, then a few richer examples.

### Start simple

Here is a complete new protocol: mean absolute error on the standard pseudobulk profiles.

1. Add a pure function to [`src/scperteval/protocols/metrics.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/protocols/metrics.py):

   ```python
   def mae(gt, prediction, ctx):
       return float(np.mean(np.abs(gt - prediction)))
   ```

   Every metric function has this signature. `gt` is one perturbation's ground-truth
   profile; `prediction` is the candidate being compared against it (under `calibrate`, scPertEval
   calls the function once for the positive control and once for the negative; under `score`, once
   with your model's prediction). `ctx` is the dataset context, needed by only a few metrics —
   ignore it otherwise. Return a single number.

2. Add a row to [`src/scperteval/protocols/table.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/protocols/table.py):

   ```python
   Protocol("mae", M.mae, representation="centroid",
            positive="interpolated", negative="all_perturbed_mean",
            better="lower", perfect=0.0)
   ```

Run it with `scperteval calibrate data.h5ad -p mae`. That is the whole protocol: MAE between each
perturbation's pseudobulk profile and its positive and negative controls, scored as
lower-is-better toward a perfect of 0.

### The spec

That row is the spec; parameters include:

| argument | meaning |
|---|---|
| `name` | selects the protocol on the CLI (`-p mae`) |
| `representation` | the shape of each datapoint your function receives (see below) |
| `scope` | `"perturbation"` (default) or `"dataset"` — how many perturbations at once (see below) |
| `space` | which features to score — `full` (default), or a feature space like `top_50` |
| `centering` | a baseline subtracted before scoring, e.g. `"ctrl"` (default: none) |
| `positive` / `negative` | the two control sources to compare |
| `better` | `"higher"` or `"lower"` — which direction is an improvement |
| `perfect` | the value a flawless prediction attains |
| `param` | optional — a parameter family (`top_k`, `pca_k`, `degs_padj`, `overlap_k`) that makes the protocol tunable from the CLI; omit for a fixed protocol |

**`representation`** decides the *shape* of each datapoint — the format `gt` and
`prediction` arrive in — so you never deal with sampling, references, or projection yourself:

| `representation` | a datapoint is |
|---|---|
| `centroid` | a 1-D pseudobulk vector (one value per gene) |
| `population` | a `(cells × genes)` matrix |
| `de` | a `PerturbationDEResult` (for the ground truth) / per-gene `\|statistic\|` ranking (for a prediction) |

**`scope`** is the independent companion axis — *how many* perturbations the metric sees at once:

| `scope` | the metric is called |
|---|---|
| `perturbation` (default) | once per perturbation — gets that perturbation's `(gt, prediction)` datapoints and returns a scalar |
| `dataset` | once for the whole dataset — gets the **list** of every perturbation's `gt` and `prediction` datapoints and returns one score per perturbation (e.g. a retrieval `rank`) |

The two compose freely: `rank` is just `representation="centroid", scope="dataset"`; a
distributional retrieval metric would be `representation="population", scope="dataset"`.

Many rows repeat the same wiring, so the top of `table.py` predefines the common
combinations as plain dicts. You then unpack one into a row with `**` (Python's
keyword-expansion syntax) to avoid retyping it:

```python
_PB = dict(group="pseudobulk", positive="interpolated", negative="all_perturbed_mean")
_LOWER = dict(better="lower", perfect=0.0)
```

With those, the `mae` row above is exactly `Protocol("mae", M.mae,
representation="centroid", **_PB, **_LOWER)` — same protocol, less repetition. You'll see
these bundles reused throughout the table.

### Building blocks — the palette

The values those arguments take — feature spaces, control sources, DE methods, calibrators
— are registered building blocks. `scperteval list <category>` shows what's available
in each, with descriptions:

**Feature spaces** (the `space` argument)

```bash
$ scperteval list spaces
degs_0.05  — ground-truth DEGs at adjusted p < 0.05, per perturbation
full       — all genes, no transform
pca_50     — top 50 principal components (fit on the dataset)
top_50     — top 50 genes by ground-truth effect size, per perturbation
```

`top_<k>` / `pca_<k>` / `degs_<padj>` are parameterised families (the defaults are shown);
a protocol template picks the value. If the space you need isn't here, see
[Add a feature space](building-blocks.md#add-a-feature-space).

**DE methods** (the `--de-method` choice)

```bash
$ scperteval list de-methods
MWU        — Mann-Whitney U / Cliff's delta effect size (via illico)
t-test     — Welch's t-test (default) — moment-based and fast
```

Chosen with `--de-method`; it applies to **every** DE-dependent unit (the `interpolated`
positive control, the `top_k`/`degs` spaces, the `de_*` protocols, and the WMSE weights).
To add another, see [Add a DE method](building-blocks.md#add-a-de-method).

**Control sources** (the `positive` / `negative` arguments)

```text
$ scperteval list sources
all_perturbed  (cells) — all-perturbed reference sample, leave-one-out (single-cell negative control)
all_perturbed_mean (centroid) — all-perturbed mean, excluding the target — leave-one-out (pseudobulk sibling of all_perturbed; pseudobulk negative control)
control        (cells) — non-targeting control cells
global_mean    (centroid) — mean of all perturbations — shared baseline for the ranking protocols
gt_all_cells   (cells) — ground truth — all of a perturbation's real cells (prediction-scoring truth)
gt_half        (cells) — ground truth — the first half of a perturbation's cells (calibration truth)
interpolated   (centroid) — interpolated duplicate — DE-weighted blend of the held-out half and the dataset mean (pseudobulk positive control)
prediction     (cells) — model-predicted cells for the perturbation, from the --predictions h5ad
tech_dup       (cells) — technical duplicate — the held-out second half (single-cell positive control)
```

Each `provides` cells or a pseudobulk `centroid`. Use via `positive=`/`negative=` (or
`--positive`/`--negative`). The truth source is chosen by the command, not by a protocol:
`calibrate` uses `gt_half` (holding the other half out as the positive control), while `score`
uses `gt_all_cells` and compares it to `prediction`. To add another, see
[Add a control source](building-blocks.md#add-a-control-source).

**Calibrators** (the `--calibrator` choice)

```bash
$ scperteval list calibrators
drf    — Dynamic Range Fraction — mean/median over perturbations (Miller et al. 2025)
bds    — Bound Discrimination Score — fraction of perturbations the positive control wins (SBB 2026)
score  — raw metric of a prediction vs ground truth — mean/median over perturbations (prediction-scoring mode)
```

`drf`/`bds` are chosen with `calibrate --calibrator`; `score` is selected automatically by the
`score` command. To add another, see [Add a calibrator](building-blocks.md#add-a-calibrator).

### More examples

With the spec and the palette in hand, richer protocols are just different combinations.

**Same wiring, different metric.** Cosine distance on pseudobulk reuses the bundles wholesale:

```python
def cosine(gt, prediction, ctx):
    return 1.0 - float(gt @ prediction / (np.linalg.norm(gt) * np.linalg.norm(prediction)))
```

```python
Protocol("cosine", M.cosine, representation="centroid", **_PB, **_LOWER)
```

**Restrict to a feature space.** Set `space` to score only some genes — e.g. MAE on the
top-50 DEGs:

```python
Protocol("mae_top50", M.mae, representation="centroid", space="top_50", **_PB, **_LOWER)
```

**Expose the space as a knob (parameterised).** To make `k` adjustable per invocation, add a
`param` to the same `Protocol(...)` row — nothing else changes. The row's name carries the
parameter, and the value is supplied on the CLI:

```python
Protocol("mae_top_k", M.mae, representation="centroid", param=top_k, **_PB, **_LOWER)
```

Then `scperteval calibrate data.h5ad -p mae_top_k=30` (or `mae_top_k` for the default `k=50`). The
families are `top_k` (top-k DEGs), `pca_k` (k PCs), and `degs_padj` (DEGs at adjusted
p < padj) for the space, and `overlap_k` to feed an integer straight to the metric.

**A metric over cells, not profiles.** Switch `representation` to `population` and your
function receives `(cells × genes)` matrices; pair it with the single-cell controls:

```python
def my_mmd(gt, prediction, ctx):      # gt, prediction are (cells × genes)
    ...
```

```python
Protocol("my_mmd_top50", M.my_mmd, representation="population", space="top_50",
         positive="tech_dup", negative="all_perturbed", better="lower", perfect=0.0)
```

This changes two pieces at once — the `representation` (so the function sees cells) and the controls
(the single-cell positive/negative) — which is the general pattern for a distributional
protocol.

By now you've seen every moving part: the function, the spec, the building blocks the spec
draws on, fixed and parameterised spaces, and switching the representation the function
sees. Most new metrics are some combination of these.
