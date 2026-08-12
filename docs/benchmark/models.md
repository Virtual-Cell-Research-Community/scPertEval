# Models introduction and training detail

Six models are benchmarked for single-perturbation gene-expression response prediction,
spanning:

- Purpose-built architectures **trained from scratch** on the target dataset: GEARS,
PRESAGE, scLAMBDA, STATE
- A single-cell **foundation model fine-tuned** for the task: scGPT
- A **tabular foundation model** applied zero-shot with no perturbation-specific
training at all: TabICLv2.

> [!NOTE]
> - All six are trained and evaluated with a 5-fold cross-validation split on the `Replogle22k562 essential` dataset.
> - Additionally, where a model's own vendored code needed a patch, that patch is a
> compatibility or performance fix verified to leave behavior unchanged unless flagged otherwise below.

All of it — training scripts, patches, data prep — lives in [scPertEval-models](https://github.com/Virtual-Cell-Research-Community/scPertEval-models/tree/main). There, each model trains in its own
isolated `pixi` environment, offering a more reproducible and modern alternative to `conda`, while also being
simpler to deploy than the `docker` setup used in {cite:t}`Miller_2025`'s codebase. Also, unlike Miller et al.'s paper, no model is reimplemented or wrapped, which eases code review of any modifications.

## Data preparation

scPertEval's own generic prep (turning a raw perturb-seq `.h5ad` into log-normalized
expression plus a `perturbation` label per cell) is covered in
[Preparing a dataset](../notebooks/02_preparing_a_dataset.ipynb); `replogle22k562` is that
tutorial's "easy case," needing little beyond normalization. `models/data/prepare_data.py`
builds on that with the parts of {cite}`Miller_2025`'s own protocol that scPertEval's generic prep
doesn't already share, so every model trains on the same, benchmark-consistent population:

- **Drop unmeasured targets**: perturbations whose own target gene isn't itself a measured
  feature (GEARS/scGPT crash on these at predict time otherwise).
- **Drop GEARS-uncovered targets**: perturbations targeting a gene missing from GEARS' own
  bundled GO-annotation reference (5 of 1,789 genes for `replogle22k562`).
- **Drop scLAMBDA-uncovered targets**: perturbations targeting a gene missing from scLAMBDA's
  own GenePT gene-embedding reference (14 of 1,789 genes). Left in, these crash every run
  unconditionally, since scLAMBDA builds its embedding dict from every perturbed gene in the
  dataset up front, not per fold.
- **Per-perturbation downsampling**: non-control conditions capped at the (rounded) mean cell
  count over the surviving population, control capped at 8,192 cells (Miller's own value).
- **Minimum-cells filters**: perturbations below 12 cells (pre-downsampling) or 4 cells
  (post-downsampling) are dropped; both are a no-op for `replogle22k562` specifically.
  Kept as explicit checks for datasets where they'd matter.
- **HVG panel**: restricted to the top-8,192 highly-variable genes, unioned with every
  remaining perturbed gene, so no perturbation ever loses its own target.

`models/data/prepare_split.py` then replaces GEARS' own one-shot `simulation` split (a single
random ~25%-gene draw) with a genuine 5-fold, full-coverage split over perturbed genes: every
gene is in the test fold exactly once, gene-level rather than cell- or condition-level. The same
split is relabeled per model's own expected format (GEARS'/scGPT's condition strings, PRESAGE's
bare gene symbols) but is otherwise identical, so all six models and the three baselines are
scored on the same held-out genes.

## Deep-learning models

### GEARS

{cite}`Gears_2024` predicts a perturbation's effect through two graph neural networks: one
encodes each gene from a Pearson-correlation co-expression graph built on the training data,
the other encodes each possible perturbation from a Gene Ontology pathway-similarity graph. This is the
mechanism that lets GEARS extrapolate to genes never perturbed during training, since their
embedding is anchored to pathway-similar genes that were. A perturbation's active gene
embeddings are summed and passed through an MLP (arity-agnostic, though our single-gene
benchmark never exercises more than one term), added to each gene's co-expression embedding,
and decoded to a scalar effect per gene by a gene-specific linear layer; a second pass through
a shared cross-gene MLP lets that effect account for secondary effects elsewhere in the
transcriptome before the prediction is added to a control cell's expression. Training combines
an autofocus loss (an elevated-exponent MSE that automatically upweights genes with larger
true effects) with a direction-aware term penalizing sign mismatches.

We run GEARS largely as shipped, with one deliberate protocol change: rather than its own
`simulation` split (a 25%-gene holdout, the split {cite}`AhlmannEltze_2025` reused), we feed in
scPertEval's own 5-fold split. Architecture and training hyperparameters are left at GEARS' own
defaults, and GEARS already tracks and predicts from its best validation-loss checkpoint
natively. The remaining changes are non-modeling fixes to GEARS' own code: atomic, race-safe
downloads for its shared GO/co-expression reference files (needed once several SLURM-array
folds hit a cold cache concurrently), a couple of performance rewrites of its data-prep and
evaluation functions verified to produce identical output, and compatibility shims for
library-version drift.

### PRESAGE

{cite}`Presage_2025` is a gene-embedding regressor trained on pseudobulked data, one row per
perturbation rather than per cell. It combines gene embeddings from roughly 40 heterogeneous knowledge
sources (node2vec over graphs such as GO, Reactome and STRING; PCA over tabular sources such as
ESM, BioGPT and other Perturb-seq datasets), aligns each source's embedding into a shared space
through source-specific MLPs, pools them with a combination of a global learned weighting and a
graph-attention mechanism, and maps the pooled result to a predicted log-fold-change with a
final linear layer.

We reproduce PRESAGE's own official configuration for the K562-essential dataset verbatim (the
same knowledge sources, embedding size, attention weight and softmax temperature) rather than
rerunning the paper's own per-dataset hyperparameter search, since PRESAGE ships that config
directly and re-deriving it independently would only risk drift from the source of truth.
PRESAGE's own training loop already validates every epoch and predicts from its best
checkpoint; the fixes we made to its code are bug fixes rather than protocol changes: a
vectorized differential-expression computation (the shipped one only loads a precomputed cache
for PRESAGE's own benchmark datasets, and is silently empty for any other dataset), preserving
the best checkpoint on disk (the original code deletes it after loading), and adding the control
mean back before scoring: PRESAGE predicts that log-fold-change rather than absolute expression,
and every model needs to be scored in the same output space for the comparison to be fair — the
same conversion Miller et al.'s own benchmarking code applies for the same reason.

### scGPT

{cite}`scGPT_2024` is a transformer foundation model pretrained on tens of millions of cells,
treating each gene as a token whose embedding sums a gene-identity embedding, an
expression-value embedding, and a condition embedding; pretraining uses a generative attention
mask that lets a query gene attend only to already-"known" genes, iteratively expanding that
known set, since single-cell data has no natural sequence order for a causal mask to exploit.
For perturbation prediction specifically, the paper fine-tunes on control-cell-in,
perturbed-cell-out pairs with a binary perturbed/not-perturbed token appended to every gene
position, using log1p expression directly rather than the binned values used elsewhere in the
model.

We fine-tune scGPT's whole-human pretrained checkpoint (`scGPT_human`), the checkpoint scGPT's
own official perturbation-prediction tutorial loads and recommends, rather than the
organ-specific "scGPT blood" checkpoint the paper itself used for its own Adamson/Replogle
experiments. Fine-tuning hyperparameters (learning rate, decay schedule, epoch count) match that
same tutorial, including validating every epoch and predicting from the best checkpoint by
validation Pearson correlation. The remaining changes to scGPT's own code are compatibility
shims for library-version drift and performance rewrites of its data-prep functions, verified to
produce identical output.

### scLAMBDA

{cite}`scLambda_2024` is a conditional VAE: a cell's expression is modeled as generated from a
low-dimensional basal-state latent (encoded from the cell itself) plus a "salient" perturbation
representation, encoded from a fixed pretrained gene embedding for the perturbation's target
gene. The two are summed and decoded jointly, trained with a standard ELBO. Two extra
mechanisms keep the perturbation signal from leaking into the basal latent: an auxiliary
decoder that reconstructs the gene embedding back out of the salient representation, and an
adversarial mutual-information penalty between the two latents estimated via a MINE-style
critic; a further adversarial step perturbs each training example's gene embedding toward
higher reconstruction error, to keep the model from overfitting to the sparse set of
perturbations actually observed.

We supply GenePT's precomputed gene embeddings (the same choice scLAMBDA's own reference demo
makes) and otherwise run the library's model class unmodified, at its own defaults. No patch
exists for scLAMBDA in this repo; the only wrapper code around it enforces a 10-epoch floor (the
library only ever checkpoints its best model every 10 epochs, so fewer would leave no
checkpoint to predict from) and scrapes its own printed training log for metrics.

### STATE

{cite}`State_2025` treats perturbation response as a property of a *set* of cells rather than a
single cell: cells sharing a context, perturbation and batch are grouped into fixed-size sets,
paired with a same-size control set, and passed through a transformer that applies bidirectional
self-attention across the cells in the set (no positional encoding, since cell order is
arbitrary) conditioned on the perturbation. The transformer's output is added residually to the
input, so it only has to learn the perturbation effect, and training minimizes the Maximum Mean
Discrepancy between the predicted and observed *distributions* of cells in a set, rather than a
per-cell reconstruction loss. STATE also ships a separate, self-supervised cell-embedding model
(SE) that can supply ST's cell representations; we don't use it, training ST directly in HVG
expression space instead.

We run the official `state` CLI, training from scratch per fold (STATE fetches no
pretrained checkpoint at our settings) on a GPT2-backed transformer (`model=pertsets`) over
2,000 HVGs. Hidden size is left at `pertsets`'s own shipped default (328), which differs from
the paper's own published Table 3 value for this cell line (128): a full-scale diagnostic on our
data trains markedly worse at 128, and 328 also matches STATE's own publicly released checkpoint
for this dataset, so the paper's table is treated as stale here rather than a target to
reproduce. Cell-set size, by contrast, is knocked down from `pertsets`'s own default (512) to 64,
matching Arc's reference Colab and its real checkpoints for this cell line rather than either
the tool's default or the paper's table (32); the smaller value trains to a meaningfully worse
validation loss on our data too. The one patch applied restores an
early-stopping callback the installed library version doesn't ship, and forces single-process
data loading at prediction time to avoid a deadlock.

### TabICLv2

{cite}`TabFM_2026` applies TabICL {cite}`TabICLv2_2026` (a Prior-Fitted Network pretrained
purely on synthetic tabular tasks, with no biological pretraining at all) to perturbation
prediction with no gradient update whatsoever: the pretrained weights stay frozen, and
"training" a fold means only assembling an in-context example set from that fold's
perturbations. Since TabICL is a univariate regressor, the paper decomposes the gene-expression
response (like PRESAGE, a log-fold-change from control, not absolute expression) into around 128
principal components
and fits one componentwise in-context regression per component, using a perturbation's
PCA-reduced multi-modal gene embedding (the same embeddings PRESAGE builds) as its input feature
vector, then reconstructs the full response via inverse PCA.

We reuse PRESAGE's precomputed multi-modal embeddings as TabICL's per-perturbation features:
the paper's "multi-modal only" variant, not the "multi-modal + cross-dataset Perturb-seq"
variant it also reports getting an extra boost from, since we don't have those other datasets
processed through PRESAGE's own pipeline. PCA rank is left at the library's default of 128, and
scPertEval's own 5-fold split supplies the in-context example set directly. TabPFN, which the
source paper found comparable on some datasets, was dropped in favor of TabICL for tooling
reasons: its checkpoint requires an interactive license-acceptance step with no batch-friendly
path in our environment. As with PRESAGE, we add the control mean back to that reconstructed
log-fold-change before scoring: it predicts delta natively, and every model needs to be scored
in the same output space for the comparison to be fair.

## Training diagnostics

Per-fold training/validation curves for five of the six models (TabICL has no epoch-based
training loop to log, since it's a frozen, zero-shot in-context regressor), from the actual
`replogle22k562` runs
(source: [`figures/training_curves.png`](https://github.com/Virtual-Cell-Research-Community/scPertEval-models/blob/main/figures/training_curves.png)
in `scPertEval-models`):

![Per-model, per-fold training and validation curves](https://raw.githubusercontent.com/Virtual-Cell-Research-Community/scPertEval-models/main/figures/training_curves.png)

Shown for transparency about each model's own default training behavior, not as a cross-model
comparison: each y-axis is a different, model-native quantity. All five plateau well before their training budget
ends. Red dots mark the epoch/step actually saved as each fold's checkpoint, read from each
model's own selection logic rather than assumed to be the plotted curve's own best point.
PRESAGE, scGPT, scLAMBDA and STATE all select their checkpoint by the same quantity plotted
here, so their red dots sit at each curve's own optimum. GEARS is the exception: its checkpoint
is chosen by validation MSE over its DE-gene subset, a different column from the plain
validation MSE plotted here, so its red dots can land somewhere that doesn't look locally best
on this specific curve, a mismatch between what's shown and what was actually optimized, not an
error in either.

Resource usage and stopping behavior for the same runs (source:
[`figures/training_summary.html`](https://github.com/Virtual-Cell-Research-Community/scPertEval-models/blob/main/figures/training_summary.html)):

<div class="pst-scrollable-table-container"><table class="table">
  <thead><tr><th>Model</th><th>Params</th><th>GPU</th><th>Folds</th><th>Avg Elapsed</th><th>Avg MaxRSS</th><th>Avg Progress</th><th>Early Stopped</th></tr></thead>
  <tbody>
    <tr style="background-color:var(--pst-color-table-row-zebra-low-bg)"><td rowspan="2">gears</td><td rowspan="2">3.4M</td><td>h200</td><td>0,2,3,4</td><td>1h 18m</td><td>60.3G</td><td>20/20 epochs</td><td>no</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-low-bg)"><td>l40s</td><td>1</td><td>2h 6m</td><td>62.3G</td><td>20/20 epochs</td><td>no</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-high-bg)"><td rowspan="1">presage</td><td rowspan="1">9.1M</td><td>h200</td><td>0,1,2,3,4</td><td>0h 3m</td><td>35.0G</td><td>18/1000 epochs</td><td>yes</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-low-bg)"><td rowspan="2">scgpt</td><td rowspan="2">51.9M</td><td>h200</td><td>0,2,3,4</td><td>2h 1m</td><td>55.1G</td><td>15/15 epochs</td><td>no</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-low-bg)"><td>l40s</td><td>1</td><td>5h 10m</td><td>57.2G</td><td>15/15 epochs</td><td>no</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-high-bg)"><td rowspan="2">sclambda</td><td rowspan="2">13.1M</td><td>h200</td><td>1,2,3,4</td><td>0h 16m</td><td>43.9G</td><td>200/200 epochs</td><td>no</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-high-bg)"><td>l40s</td><td>0</td><td>0h 13m</td><td>44.2G</td><td>200/200 epochs</td><td>no</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-low-bg)"><td rowspan="1">state</td><td rowspan="1">38.3M</td><td>h200</td><td>0,1,2,3,4</td><td>1h 3m</td><td>23.0G</td><td>~1466/10000 steps</td><td>yes</td></tr>
    <tr style="background-color:var(--pst-color-table-row-zebra-high-bg)"><td rowspan="1">tabicl</td><td rowspan="1">28.5M</td><td>h200</td><td>0,1,2,3,4</td><td>0h 46m</td><td>23.5G</td><td>n/a</td><td>n/a</td></tr>
  </tbody>
</table></div>

## Baselines

Three baselines require no training:

- **No-change** predicts, for every perturbation, exactly the dataset's control-cell mean, a
  floor representing zero perturbation effect.
- **Mean** predicts each fold's held-out perturbations from the average of that fold's own
  training perturbations' means only ({cite}`Miller_2025`'s $\mu_\text{all}$), deliberately
  scoped to the fold's train split, since averaging in perturbations from the test fold would
  leak information no real trained model has access to.
- **Interpolated** predicts each fold's held-out perturbations from the average of that fold's own
  training perturbations' means interpolated with the technical duplicate.
  
  $$
  \mu_{p, \mathrm{interpolated}} = \alpha \odot \mu_{p, \mathrm{technical-duplicate}} + (1 - \alpha) \odot \mu_{all}
  $$

  Where $\alpha = 1 - p_{\mathrm{DEGs}}$.
  
  The interpolated baseline is an oracle baseline: it has access to private
  information — the $\mathrm{technical-duplicate}$ — and is therefore considered difficult to beat.

  It is scoped to the fold's train split — both the training mean and the $p_{\mathrm{DEGs}}$ computation — to preserve independence relative to other perturbations in the test set.
- **Linear** reproduces {cite}`AhlmannEltze_2025`'s ridge-regression baseline directly from
  their own published script: PCA-embed each fold's training pseudobulk (control included) into
  a shared gene/perturbation embedding $G$, fit a bilinear ridge map $W$ between $G$ and the
  training perturbations' own embedding $P$, and predict a held-out perturbation from its own
  row $\hat P$ in that same embedding.

  $$
  \hat{Y} = G W \hat{P}^\top + b
  $$

  Where $G$ is that shared PCA embedding and $\hat P$ is a held-out perturbation's own row in
  it — perturbations here are named after their own target gene, so a perturbation's embedding
  doubles as that gene's own embedding. $b = \mu_{\mathrm{control}} + \bar\delta$: the fold's
  control-cell mean plus the training perturbations' own mean deviation from it. Both $G$/$W$
  and $b$ are fit only on the fold's training perturbations, with control itself entered as one
  more training perturbation carrying a zero-vector embedding — a detail
  {cite}`AhlmannEltze_2025`'s own reference script encodes but its published methods text
  simplifies away as "$b$ = row means of $Y_{\mathrm{train}}$".

You can now get details on the benchmark implementation in the [Tutorial 4: Benchmarking](../notebooks/04_benchmark_implementation.ipynb) subsection.
