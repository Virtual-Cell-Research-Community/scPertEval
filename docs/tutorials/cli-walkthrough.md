---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# CLI walkthrough

This tutorial runs scPertEval end-to-end from the command line on a tiny synthetic
dataset, so you can follow along without downloading anything. Every code cell below is
executed when the docs are built, so the output you see is real.

For a real benchmark you would point the CLI at a preprocessed `.h5ad` (see the sample
datasets linked from the [README](https://github.com/Virtual-Cell-Research-Community/scPertEval#readme)).

## Set up a dataset

We build a 60-gene toy dataset with a control population and four perturbations, each with
its own distinct block of up-regulated genes, and save it as an `.h5ad`:

```{code-cell} python
import pathlib
import tempfile

import anndata as ad
import numpy as np

rng = np.random.default_rng(0)
ng, n_ctrl, n_pert = 60, 150, 120
de_genes = {"pertA": range(0, 6), "pertB": range(15, 21), "pertC": range(30, 36), "pertD": range(45, 51)}

parts = [rng.poisson(1.0, (n_ctrl, ng)).astype(np.float32)]
labels = ["control"] * n_ctrl
for name, genes in de_genes.items():
    x = rng.poisson(1.0, (n_pert, ng)).astype(np.float32)
    x[:, list(genes)] += 6.0  # up-regulate this perturbation's marker genes
    parts.append(x)
    labels += [name] * n_pert

adata = ad.AnnData(np.vstack(parts))
adata.var_names = [f"g{i}" for i in range(ng)]
adata.obs["perturbation"] = labels

workdir = pathlib.Path(tempfile.mkdtemp())
data_path = workdir / "toy.h5ad"
adata.write_h5ad(data_path)

print(adata)
print("\nperturbations:", adata.obs["perturbation"].value_counts().to_dict())
```

## Calibrate protocols against built-in controls

`calibrate` scores each protocol against a positive and a negative control and reports a
calibrated **DRF** (or **BDS**) value — a measure of whether the protocol can tell real
perturbation signal from an uninformative baseline. From a shell you would run:

```bash
scperteval calibrate toy.h5ad -p pearson_ctrl,mse --output drf
```

The Python entry point takes the same arguments as a list, which is what we use here so the
run executes inside the docs build:

```{code-cell} python
from scperteval.cli import main

main(["calibrate", str(data_path), "-p", "pearson_ctrl,mse", "--out-dir", str(workdir)])
```

Each run also writes a per-perturbation CSV. Let's load it:

```{code-cell} python
import pandas as pd

drf_csv = next(workdir.glob("*__drf.csv"))
pd.read_csv(drf_csv).head()
```

## Export differential expression

The `de` command runs the ground-truth differential-expression step on its own and writes
the per-gene results to HDF5 — handy for inspecting the signal a protocol is scored against:

```bash
scperteval de toy.h5ad --methods t-test
```

```{code-cell} python
main(["de", str(data_path), "--methods", "t-test", "--out-dir", str(workdir)])
[p.name for p in workdir.glob("*__de.h5")]
```

## Where to next

- **Score predictions against ground truth** with `scperteval score dataset.h5ad predictions.h5ad` — see [Scoring predictions](../user-guide/scoring.md).
- List the available building blocks with `scperteval list protocols` (also `de-methods`, `spaces`, `sources`, `calibrators`).
- Read [Calibration](../user-guide/calibration.md) for what DRF and BDS actually measure.
