# Changelog

## Unreleased

### Feature spaces — breaking (Python API)

The feature-space system was reworked. **The CLI is unaffected** — every subcommand, flag, and
protocol name behaves as before, and scores are unchanged. The entries below only affect code
that defines or names a space directly.

**Naming a space.** The per-family factories are gone; one call replaces them.

| before | now |
|---|---|
| `top_space(50)` | `SPACES.instance("top", 50)` |
| `degs_space(0.05)` | `SPACES.instance("degs", 0.05)` |
| `pca_space(50)` | `SPACES.instance("pca", 50)` |

A `Protocol` that pins a space must name a registered one. Spaces are created on demand rather
than all at import, so a bare string no longer resolves on its own:

```python
Protocol("mae_top50", M.mae, space="top_50", ...)                    # before
Protocol("mae_top50", M.mae, space=SPACES.instance("top", 50), ...)  # now
```

**Defining a space.** `@SPACES.register` and `SPACES.add` no longer accept spaces (both raise
with a pointer). Use `@SPACES.subset` for a gene subset, or `@SPACES.transform` for a space that
replaces the gene axis. A rule declares that it varies by perturbation *by naming a `pert`
argument* — there is no flag:

```python
@SPACES.subset("mito", default=20, description="top {v} mitochondrial genes")
def mito(ctx, k):                     # dataset-wide
    ...

@SPACES.subset("mine", default=50, description="top {v} genes for this perturbation")
def mine(ctx, pert, k):               # per-perturbation
    ...
```

Rules take one of four shapes — `(ctx)`, `(ctx, k)`, `(ctx, pert)`, `(ctx, pert, k)` — and any
other shape is rejected at registration.

`register_de_space` was removed; a DE-derived space is now an ordinary rule. `Context.pca(k)` is
gone — use `pca_for(ctx, k)` from `scperteval.blocks.spaces.helpers`.

**Space parameters must be positive numbers.** `top_k=-5` previously selected the *weakest* genes
and `degs_padj=-1` selected none; both now raise.

### Feature spaces — new

- New spaces: `heg_<k>` (highest-expressed genes), `hvg_<k>` (most variable genes),
  `perturbed_genes` (genes a perturbation targets), and `perturbed_and_hvgs` (the HVG ∪
  perturbed-genes panel of Miller et al. 2025).
- `SPACES.combine_subsets(op, *names, name=...)` builds a new space from existing ones by union,
  intersection, or difference. Whether the result varies by perturbation is derived from its
  operands.
- `@cached` and `DatasetScope` in `scperteval.caching` — compute a dataset-level value once per
  prepared dataset and reuse it.
- `precompute=` on `@SPACES.transform`, for setup heavy enough to want doing before the parallel
  scoring loop.

### Feature spaces — behaviour

- `scperteval list spaces` lists what each space *takes* (`heg_<k>`, with its default) rather than
  one already-created instance, and marks the spaces that vary by perturbation.
- A space that selects no genes raises instead of scoring `nan`.
- Naming a space that doesn't exist reports which spaces are available.

## 0.1.0

First release of scPertEval — reference implementations of single-cell perturbation evaluation
protocols, usable from the command line and as a native Python API.

scPertEval accompanies [Schäfer et al. (2026), *Towards Principled Evaluation of Single-Cell
Perturbation Prediction Models*](https://doi.org/10.64898/2026.07.23.740433).

### Features

- **22 evaluation protocols** in one declarative table, spanning three groups: `pseudobulk`
  (Pearson/MSE/weighted-MSE variants over full, top-k and DEG feature spaces, plus
  cross-perturbation retrieval rank), `distributional` (unbiased MMD, energy distance, and
  Sinkhorn W2 over top-k/PCA spaces), and `de` (AUPRC, AUROC, top-k overlap).
- **Three actions over the same catalog** — `calibrate` (score a protocol against empirical
  positive/negative controls per perturbation, reporting DRF or BDS), `score` (score model
  predictions against ground truth), and `de` (export per-gene differential expression).
- **Two calibrators**: Dynamic Range Fraction (DRF) and Bound Discrimination Score (BDS).
- **Native Python API** (`prepare` / `calibrate` / `score` / `de`) that reads and indexes a
  dataset once and reuses it across calls, alongside the `scperteval` CLI.
- **Extension points** for new protocols, metrics, feature spaces, DE methods, control sources,
  and calibrators — each a registered function plus one table row.

### Packaging

- `torch` and `geomloss` are an optional `sinkhorn` extra rather than base dependencies, so
  `pip install scperteval` stays light. Bulk protocol selections (`-p all`, `-p distributional`)
  skip the Sinkhorn protocols with a warning when the extra is absent; naming one explicitly
  raises an error that points at `pip install "scperteval[sinkhorn]"`.
- Ships a `py.typed` marker, so downstream users get the package's type hints.
- The version is derived from the git tag by `hatch-vcs`; see `RELEASE.md`.

### A note on the dependency floors

The lower bounds in `pyproject.toml` are bisected, not guessed — each is the oldest version the
full test suite passes on, and the `floors` CI job reinstalls exactly those pins on every push so
they cannot drift. Both ends of the supported range are verified: the declared floors
(anndata 0.12.7 / numpy 2.0 / pandas 2.2.2 / torch 2.4, Python 3.11) and current releases
(anndata 0.13 / numpy 2.4 / pandas 3.0 / torch 2.13, Python 3.14) both pass, which is why
dependencies are specified as open-ended floors with no upper caps.

What sets the floors, in case someone tries to lower them:

- **`numpy>=2`** is the dominant constraint. Under numpy 1.x, `illico`'s asymptotic-Wilcoxon
  path (the `MWU` DE method) derives its test count as a float and formats it with `:,d`,
  raising `Unknown format code 'd' for object of type 'float'`.
- **`h5py>=3.11`** and **`scikit-learn>=1.5`** follow from that: earlier releases ship numpy-1.x
  ABI wheels and abort at import with `numpy.dtype size changed`.
- **`torch>=2.4`** for the same reason — 2.1–2.3 compute correct results against numpy 2 but
  bury every run in `A module that was compiled using NumPy 1.x` warnings.
- **`anndata>=0.12.7`** — below it the suite cannot write its own h5ad fixtures (anndata refuses
  `pd.arrays.StringArray` without `settings.allow_write_nullable_strings`), and at 0.11.x and
  older it is a genuine runtime break: `illico` reaches for
  `anndata._core.sparse_dataset._CSCDataset`, which does not exist yet.
- **`pandas>=2.2.2`** and **`scipy>=1.13`** are inherited from `illico`'s own requirements;
  anything lower is unsatisfiable. Note `illico` 0.6.0's declared `anndata>=0.10.8` is looser
  than what it actually needs, which is why scPertEval pins these from testing rather than
  relying on the transitive constraint.
