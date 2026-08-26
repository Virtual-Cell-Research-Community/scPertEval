# Changelog

## Unreleased

Accumulating notes for the next release. Entries land here as PRs merge.

### Feature spaces — breaking (Python API)

The feature-space system was reworked. **The CLI is unaffected** — every subcommand, flag, and
protocol name behaves as before. The changes below only affect code that defines or names a space
directly.

**Naming a space.** The per-family factories are gone; one call replaces them.

| before | now |
|---|---|
| `top_space(50)` | `SPACES.instance("top", 50)` |
| `degs_space(0.05)` | `SPACES.instance("degs", 0.05)` |
| `pca_space(50)` | `SPACES.instance("pca", 50)` |

A `Protocol` that pins a space must name a registered one. Spaces are now created on demand
rather than all at import, so a bare string no longer resolves on its own:

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

`register_de_space` was removed; a DE-derived space is now an ordinary rule.

**Other removals.** `Context.pca(k)` is gone — use `pca_for(ctx, k)` from
`scperteval.blocks.spaces.helpers`.

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

- `scperteval list spaces` now lists what each space *takes* (`heg_<k>`, with its default) rather
  than one already-created instance, and marks the spaces that vary by perturbation.
- A space that selects no genes now raises instead of scoring `nan`.
- Naming a space that doesn't exist reports which spaces are available.
- Scores are unchanged: verified identical to the previous release across 22 protocols and 1,958
  perturbation rows on a 65k-cell dataset.
