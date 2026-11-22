# Datasets

Seven preprocessed single-cell perturbation datasets are hosted in a
[Requester Pays](https://cloud.google.com/storage/docs/requester-pays) Google Cloud Storage bucket:

```
gs://scperteval/processed/
```

Data-transfer charges are billed to a Google Cloud project supplied by the requester. See
[Downloading](#downloading) for prerequisites and commands.

## Available datasets

| Dataset | Cell line | Reference |
|---|---|---|
| `replogle22k562` | K562 | {cite}`Replogle_2022` |
| `replogle22rpe1` | RPE1 | {cite}`Replogle_2022` |
| `nadig25hepg2` | HepG2 | {cite}`Nadig_2025` |
| `nadig25jurkat` | Jurkat | {cite}`Nadig_2025` |
| `wessels23` | THP-1 | {cite}`Wessels_2023` |
| `arch1` | H1 hESC | {cite}`Roohani_2025` |
| `kaden25rpe1` | RPE1 | {cite}`Kaden_2025` |

- `replogle22k562` / `replogle22rpe1` — CRISPRi Perturb-seq targeting ~2,057 essential genes, in
  K562 (chronic myeloid leukemia) and RPE1 (non-cancerous retinal pigment epithelial) cells
  respectively. This is the paper's *essential-gene* panel, not its much larger genome-wide K562
  screen (~9,866 genes, "K562_gwps") — that screen isn't part of scPertEval's datasets.
- `nadig25hepg2` / `nadig25jurkat` — companion genome-scale CRISPRi Perturb-seq screens
  extending the same essential-gene protocol to HepG2 (liver) and Jurkat (T-cell) lines.
- `wessels23` — Cas13 combinatorial Perturb-seq (CaRPool-seq) in THP-1 cells (an AML model),
  knocking down pairs of myeloid differentiation regulators (26 regulators, 158 gene-pairs); the
  only dataset here with two-gene combination knockdowns.
- `arch1` — the Virtual Cell Challenge benchmark: CRISPRi perturbations in H1 human embryonic
  stem cells. scPertEval's copy is the challenge's released training split (150 perturbations);
  the public/private test splits used for the challenge leaderboard aren't included.
- `kaden25rpe1` — a transcription-factor perturbation screen in RPE1 cells probing induction of
  fibroblast-like transcriptional states — reprogramming-oriented, unlike the essentiality
  screens above.

### Summary statistics

Cell and perturbation counts per dataset, split by single-gene vs. two-gene (combination)
perturbations. Counts exclude control cells.

| Dataset | Cells | Control cells | Perturbed cells (single) | Perturbed cells (double) | Unique perts (single) | Unique perts (double) |
|---|---:|---:|---:|---:|---:|---:|
| `replogle22k562` | 308,646 | 10,691 | 297,955 | 0 | 1,971 | 0 |
| `replogle22rpe1` | 240,774 | 11,485 | 229,289 | 0 | 2,016 | 0 |
| `nadig25hepg2` | 133,757 | 4,976 | 128,781 | 0 | 1,818 | 0 |
| `nadig25jurkat` | 258,202 | 12,013 | 246,189 | 0 | 2,137 | 0 |
| `wessels23` | 28,490 | 424 | 0 | 28,066 | 0 | 157 |
| `arch1` | 221,273 | 38,176 | 183,097 | 0 | 150 | 0 |
| `kaden25rpe1` | 850,225 | 42,233 | 807,992 | 0 | 1,836 | 0 |

:::{note}
Every dataset except `wessels23` consists entirely of single-gene knockdowns/knockouts.
The hosted `wessels23` contains only **combination** perturbations — its single-gene
perturbations (used as additive-model building blocks) are excluded from evaluation targets,
hence 0 in the table above. To recover the single-gene cells too, prepare `wessels23` yourself
from the raw file instead of using the hosted copy — the
[Preparing a dataset](../tutorials.md) tutorial keeps both singles and combos.
:::

## Format

A scPertEval input is an ordinary AnnData (`.h5ad`). scPertEval reads only three things and
recomputes everything else (differential expression, controls, baselines) in memory:

| Where | What | Required |
|---|---|---|
| `adata.X` | **log-normalised** expression, cells × genes — `normalize_total(target_sum=1e4)` then `log1p`; sparse `float32` is ideal | yes |
| `adata.obs["perturbation"]` | one **perturbation label per cell**; all non-targeting control cells share the single label `control` | yes |
| `adata.var_names` | **gene names** (the `var` index) | yes |

The `obs` column name and the control label are configurable — `--perturbation-key` and
`--control-label` (defaults `perturbation` and `control`). Nothing else is read.

:::{note}
`X` must be **log-normalised**, not raw counts — scPertEval does not normalise for you. If your
file is *already* normalised, don't normalise it again.
:::

**Trimming.** Everything scPertEval doesn't read should be dropped: a redundant raw-count layer
(often the single biggest source of bloat — roughly doubling the file), embeddings in `obsm`,
graphs and precomputed results in `uns`, and surplus `obs`/`var` columns. Keeping only `X`, the
`perturbation` column, and the gene names — written sparse, `float32`, gzip-compressed — makes
files markedly smaller and faster to load. The hosted files above are trimmed this way.

## Preparing your own dataset

Starting from a raw perturb-seq `.h5ad`, three steps get you to the format above:

1. **Clean `obs["perturbation"]`** — one label per cell, all controls collapsed to `control`,
   any guide/plasmid suffixes stripped, combinations joined with `+`, and cells with a missing
   label dropped.
2. **Log-normalise `X`** — `normalize_total(target_sum=1e4)` + `log1p`, with light QC filtering
   (e.g. `filter_cells(min_genes=200)`, `filter_genes(min_cells=3)`).
3. **Trim** — keep only `X`, `obs["perturbation"]` and the gene names; write sparse `float32`,
   gzip-compressed.

Only step 1 really differs between datasets. The
[Preparing a dataset](../tutorials.md) tutorial works through it end-to-end on three real
datasets, each showing a different wrinkle:

| Dataset | Cleanup needed in step 1 |
|---|---|
| `replogle22k562` | none — labels are already clean gene symbols; just normalise + trim |
| `adamson16` | strip the plasmid suffix (`SEC61A1_pDS031` → `SEC61A1`); route parenthesised `(mod)` constructs to `control` |
| `wessels23` | join combination guides with `+` (`IKZF1_SMARCD1` → `IKZF1+SMARCD1`) |

:::{note}
Combinations are **relabelled, not removed** — `GENE1+GENE2` is a perturbation like any other.
Whether a study evaluates combos, singles, or both is a downstream modelling choice (for
`wessels23`, combos are the targets and singles are additive-model building blocks).
:::

## Downloading

### Prerequisites

1. A Google account.
2. A Google Cloud project with billing enabled (see Google Cloud's guides on
   [creating a project](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
   and [enabling billing](https://cloud.google.com/billing/docs/how-to/modify-project)).
3. The [gcloud CLI](https://cloud.google.com/sdk/docs/install), authenticated via:

   ```bash
   gcloud auth login
   ```

Replace `YOUR_PROJECT_ID` in the commands below with your Google Cloud project ID (run
`gcloud projects list` to list accessible projects).

### Commands

```bash
# list available files
gcloud storage ls --billing-project=YOUR_PROJECT_ID gs://scperteval/processed/

# download a single file
gcloud storage cp --billing-project=YOUR_PROJECT_ID gs://scperteval/processed/replogle22k562_processed_complete.h5ad .

# download all files
gcloud storage cp --billing-project=YOUR_PROJECT_ID "gs://scperteval/processed/*_processed_complete.h5ad" .
```

To download over authenticated HTTPS on macOS or Linux:

```bash
curl --fail -L \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "x-goog-user-project: YOUR_PROJECT_ID" \
  -o replogle22k562_processed_complete.h5ad \
  https://storage.googleapis.com/scperteval/processed/replogle22k562_processed_complete.h5ad
```

Then pass the path directly to the CLI:

```bash
scperteval calibrate ./replogle22k562_processed_complete.h5ad -p all --de-method t-test
```
