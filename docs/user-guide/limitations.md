# Known limitations

Open problems worth knowing about before you build on a result. These are stated rather than
worked around: each one has plausible fixes that are wrong in a different way, so scPertEval
does not pick one silently.

(per-perturbation-spaces)=

## Per-perturbation spaces are not comparable across a `compare()` row

{func}`~scperteval.api.compare` scores one query against many references, and its
[feature space](building-blocks.md) is a property of the **reference** — that is the contract, and
it is the right one for a pairwise reading: *"how well does this query match reference `r`, judged
on the genes that define `r`?"*

But spaces come in two kinds:

- **Dataset-wide** — `full`, `hvg_<k>`, `heg_<k>`, `pca_<k>`. Every perturbation gets the same
  genes, so a whole row is computed in one space. **Unaffected by this limitation.**
- **Per-perturbation** — `top_<k>`, `degs_<padj>`, and anything composed from them. The genes are
  selected from the perturbation in hand, which under `compare()` is the reference.

With a per-perturbation space, every column of a row is computed in a *different* space:

```text
Q vs A  →  A's genes
Q vs B  →  B's genes
Q vs C  →  C's genes
```

Each pairwise value is individually meaningful. But the values along a row are **not on a common
scale**, so ranking them compares more than similarity. Two distinct mechanisms:

- **Panel composition** — affects `top_<k>` *and* `degs_<padj>`. A strongly perturbed reference's
  top genes carry larger effects than a weak one's, so its column tends to produce larger values
  regardless of how similar the query actually is.
- **Panel size** — affects `degs_<padj>` only. `degs` is a *p-value threshold*, not a fixed count:
  it keeps every gene with `pvalue_adj < padj`, so one reference may contribute 200 genes and
  another 50. `top_<k>` is fixed-*k* and immune to this second mechanism.

### Why it is not simply fixed

| approach | problem |
| --- | --- |
| fit the panel on the reference (what `compare()` does) | values are not comparable across a row |
| fit the panel on the query instead | asks a different question, and biases toward the query |
| restrict retrieval to dataset-wide spaces | loses per-perturbation targeting entirely |
| normalise each reference's column | plausible, but a property of the *readout*, not the space — so it belongs to the caller |

### What to do about it

If you rank, correlate or threshold values **along a row** of a `compare()` frame, prefer a
dataset-wide space, or normalise each reference's column against a null before comparing.

If you use a per-perturbation space anyway, check whether a reference's position in the row
tracks something about the reference rather than the query — its perturbation strength, or its
panel size — independently of which query is being scored. A row-wise result that mostly reflects
the references is this artifact, not a finding.

Values **down a column** (one reference, many queries) are unaffected: they share a space.
