# Reproducing evaluation protocols from the literature

{cite:t}`AhlmannEltze_2025` conclude that deep-learning perturbation-response predictors do not yet
outperform simple linear baselines. {cite:t}`Miller_2025` benchmark a similar class of models on
overlapping datasets and conclude the opposite: they do. So why the disagreement?

The answer lies in the protocols chosen for evaluation, namely:
1. The representation space: whether every gene is kept or solely the highly expressed, highly variable or
differentially expressed ones or whether perturbed cells are centered on some reference population.
1. Metrics: whether they are weighted or involve centering cells onto some reference population.
Or also whether they account for non-linearities, or are rank based.

`scPertEval` exists to make that whole axis explicit and swappable. This tutorial aims to reimplement
both papers' evaluations and cast light on their disagreement.

> [!WARNING]
> - It is not a faithful re-run of either paper's original codebase, but any diverging decisions are documented and explained.
> - For simplicity, the focus is on *unseen single-gene* predictions, but one
> can re-adapt the tutorial to *double-gene* perturbation. Please see the [Contributing](../contributing.md) section
> to open an issue if you would like us to implement such a tutorial, or to contribute one via a PR.
> - For time considerations, the benchmark is only run on the `Replogle22k562 essential` dataset.
> The same protocol is followed for other datasets.

For details on models and training pipeline, see the [Models](models.md) subsection.

Below, we discuss the papers' differences in detail.

## Ahlmann-Eltze, Huber & Anders 2025

{cite:t}`AhlmannEltze_2025` benchmark:

- 2 **Purpose-built perturbation prediction models**: GEARS and scGPT
- 3 **Single-cell foundation models** not originally designed for the task: Geneformer, scBERT and
UCE.
They are repurposed by plugging a linear decoder onto their cell embeddings.
  
All of them compete against:

- a **Mean** baseline (the average training pseudobulk)
- a **linear** baseline (ridge regression on PCA-derived gene/perturbation embeddings)

The data splitting strategy reuses GEARS' `simulation` split pipeline: a random 25% gene hold-out
repeated for only **2** seeds.

Evaluation is restricted to the 1,000 most highly-expressed genes in the control condition — a
fixed panel chosen independently of any model's predictions. The headline metric, $\mathrm{Pearson_\Delta}$,
is the Pearson correlation between predicted and observed change from control; a companion
raw $\ell_2$ distance is reported alongside it. To decide whether a model "outperforms" the Mean
baseline, the paper bootstraps (10,000 resamples) a 95% CI on its per-perturbation relative error
and requires that CI to sit entirely below 1, a bar most deep-learning models don't clear, hence
the paper's title.

## Miller et al. 2025

{cite:t}`Miller_2025` benchmark:

- 4 **Purpose-built perturbation prediction models**: scGPT, GEARS, scLAMBDA and PRESAGE
- 4 **Frozen gene embeddings with an MLP probe trained on top**: Geneformer, GenePT, ESM2 and scGPT

All of them compete against {cite:t}`AhlmannEltze_2025`'s Mean and linear
baselines and add two positive controls of their own: a **technical duplicate** (a held-out half
of a perturbation's own cells) and an **interpolated duplicate** (a per-gene blend of that duplicate
and the mean baseline, weighted by DEG significance).

Across 14 datasets spanning 9 studies and 10 cell lines, they filtered to at least 12 cells per perturbation,
200 genes per cell and 3 cells per gene, normalized to a 10k target sum and log1p-transformed,
restricted to the top-8192 HVGs plus any perturbed genes.
> Note: this preparation already inspired our own in [Tutorial 2: Preparing a dataset](../notebooks/02_preparing_a_dataset.ipynb).

Their splitting strategy is more robust: every perturbation is held out exactly once under a genuine 5-fold
cross-validation, seeded at 42, with a model retrained per fold and metrics pooled over all folds'
test predictions.

Before trusting any metric's verdict, Miller et al. calibrate it first: the **Dynamic Range Fraction (DRF)**
measures how much of the gap between a positive control (the interpolated
duplicate) and a negative control (the mean baseline) a metric actually resolves, and only metrics
that clear that bar are trusted to rank models at all. Their own metric family — $\mathrm{MSE}$, $\mathrm{wMSE}$,
$\mathrm{Pearson_\Delta}$ and $R^2_\Delta$, each centered against a control- or
all-perturbed-mean baseline and over all genes,
DEGs, or a DE-effect-size-weighted variant, plus $\mathrm{NIR}$ (Normalized Inverse Rank),
comes to 13 metrics in total.
Restricted to the well-calibrated metrics, a paired one-sided Student's t-test and Wilcoxon signed-rank test
(Bonferroni-corrected), backed by the same bootstrap-CI convention Ahlmann-Eltze et al. use, show
that deep-learning models do separate from the baselines — hence the paper's title.

## What we adopt

Neither protocol is wrong — they answer different questions about the same predictions.
However, {cite:t}`Miller_2025` or follow-on {cite}`Vollenweider_2026` argue that some protocols
are better calibrated and should be preferred.

We therefore show we can reimplement protocols from both {cite:t}`AhlmannEltze_2025` and
{cite:t}`Miller_2025` and easily explore their calibration properties with `scPertEval` while
also scoring model predictions.

For our tutorial, we adopt Miller's split — genuine, full-coverage k-fold cross-validation, where every
perturbation is held out exactly once — over Ahlmann-Eltze's two-seed repeated subsampling, which
pools a couple of random draws rather than covering the dataset.
And before trusting any ranking a metric produces, we calibrate it first, exactly as both follow-on papers do:
check its Dynamic Range Fraction ({cite}`Miller_2025`) and Bound Discrimination Score ({cite}`Vollenweider_2026`)
before drawing any conclusion from it.
> Perturbation Discrimination Score (PDS) from {cite}`Vollenweider_2026` could also be added. You can express interest
> by checking the [Contributing](../contributing.md) section.

The metrics under test follow the same logic: Ahlmann-Eltze et al.'s 2 (restricted to their
1,000-gene control-expression panel), Miller et al.'s own 13, and one further metric from
{cite}`Vollenweider_2026`'s follow-up — the **Weighted Pearson Delta** (in both Ctrl and
PerturbMean form, like every other delta metric here), which extends Miller's DE-effect-size
weighting — already applied to R²Δ — to Pearson Δ too:

| Metric | Gene set | scPertEval protocol |
|---|---|---|
| MSE | all genes | `mse` |
| MSE | DEG ($p_\text{adj} < 0.05$) | `mse_degs_padj` |
| WMSE | DE-weighted | `wmse_exp2` |
| $\ell_2$ ({cite}`AhlmannEltze_2025`) | top-1,000 control-expressed | `l2_heg_k` |
| PearsonΔCtrl | all genes | `pearson_ctrl` |
| PearsonΔCtrl | DEG ($p_\text{adj} < 0.05$) | `pearson_ctrl_degs_padj` |
| PearsonΔCtrl | DE-weighted ({cite}`Vollenweider_2026`) | `weighted_pearson_ctrl_exp2` |
| PearsonΔCtrl ({cite}`AhlmannEltze_2025`) | top-1,000 control-expressed | `pearson_ctrl_heg_k` |
| PearsonΔPerturbMean | all genes | `pearson_pert` |
| PearsonΔPerturbMean | DEG ($p_\text{adj} < 0.05$) | `pearson_pert_degs_padj` |
| PearsonΔPerturbMean | DE-weighted ({cite}`Vollenweider_2026`) | `weighted_pearson_pert_exp2` |
| R²ΔCtrl | all genes | `r2_ctrl` |
| R²ΔCtrl | DEG ($p_\text{adj} < 0.05$) | `r2_ctrl_degs_padj` |
| R²ΔCtrl | DE-weighted | `weighted_r2_ctrl_exp2` |
| R²ΔPerturbMean | all genes | `r2_pert` |
| R²ΔPerturbMean | DEG ($p_\text{adj} < 0.05$) | `r2_pert_degs_padj` |
| R²ΔPerturbMean | DE-weighted | `weighted_r2_pert_exp2` |
| NIR | all genes | `nir` |

You can now get details on models and training pipeline in the [Models](models.md) subsection.
