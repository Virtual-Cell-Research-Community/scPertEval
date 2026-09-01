# Changelog

## Unreleased

### Added

- 16 protocols reproducing the Miller et al. 2025 and Ahlmann-Eltze et al. 2025 evaluations, on
  five new metrics (`r2`, `l2`, `weighted_pearson`, `weighted_r2`, `nir`) — 38 protocols in total,
  up from 22. Run `scperteval list protocols` for the catalog.
- Four gene-subset spaces: `heg_<k>` (highest-expressed), `hvg_<k>` (most variable),
  `perturbed_genes` (genes a perturbation targets), and `perturbed_and_hvgs`.
- `SPACES.combine_subsets(op, *names, name=...)` builds a space from existing ones by union,
  intersection, or difference.
- `@cached` and `DatasetScope` in `scperteval.caching`, for computing a dataset-level value once
  per prepared dataset.
- A Model Benchmark tutorial walking through a full evaluation with the package.

### Changed — results may differ

- Protocols centred on the perturbed mean (`pearson_pert*`, `r2_pert*`, `weighted_*_pert_exp2`)
  now score against the mean of *all* control cells rather than a capped random subsample. Scores
  shift slightly on datasets holding more than `--subsample` controls.
- `r2` and `weighted_r2` are unbounded below rather than floored at -1, so a prediction that
  ignores the perturbation is no longer indistinguishable from a merely poor one.

### Changed — breaking (Python API only; the CLI is unaffected)

Only code that defines or names a feature space directly needs updating.

- The per-space factories are gone: use `SPACES.instance("top", 50)` in place of `top_space(50)`,
  and likewise for `degs_space` / `pca_space`. `register_de_space` is removed, and `Context.pca(k)`
  becomes `pca_for(ctx, k)` from `scperteval.blocks.spaces.helpers`.
- A `Protocol` must name a space that exists: `space=SPACES.instance("top", 50)`, not
  `space="top_50"` — spaces are now created on demand rather than all at import.
- Define a space with `@SPACES.subset` or `@SPACES.transform` rather than `@SPACES.register`; a
  rule declares that it varies by perturbation by naming a `pert` argument, and must take one of
  `(ctx)`, `(ctx, k)`, `(ctx, pert)`, `(ctx, pert, k)`.
- A space parameter must be a positive number; `top_k=-5` previously scored the *weakest* genes.
- A space that selects no genes now raises instead of scoring `nan`.

### Housekeeping

`scperteval list spaces` shows what each space takes (`heg_<k>`) rather than one instance and
marks those that vary by perturbation; unknown space names report what is available; feature
spaces moved to a package (`scperteval.blocks.spaces`); citation and docstring corrections.

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
