# Calibration

scPertEval assesses each **evaluation protocol** — a representation $\phi$ paired with a
metric $d$ — for its **separability**: can it reliably distinguish a perturbation's true
response from an uninformative baseline?

Let $s(\mathcal{X}, \mathcal{Y}) = d(\phi(\mathcal{X}), \phi(\mathcal{Y}))$ denote the
protocol-induced score, where **smaller values indicate better agreement** (similarity scores
are converted beforehand).

## Controls

For each perturbation $a$, every protocol is evaluated against two empirical controls:

- **Positive control** $s_{\text{pos}}^{(a)}$ — the best realistic score: comparing the
  observed cells against a **technical duplicate** (a held-out replicate); for pseudobulk
  protocols, an **interpolated duplicate** is used for stability.
- **Negative control** $s_{\text{neg}}^{(a)}$ — an uninformative baseline: comparing against
  the **all-perturbed reference excluding $a$** (full mean for pseudobulk; a subsample of
  8 192 cells by default for single-cell distances, configurable with `--subsample`).
Ideally $s_{\text{pos}}^{(a)} < s_{\text{neg}}^{(a)}$.

## Dynamic Range Fraction (DRF)

DRF asks: **how much of the available signal range does the protocol actually recover?**

$$
\operatorname{DRF}(a)
= \frac{s_{\text{neg}}^{(a)} - s_{\text{pos}}^{(a)}}{s_{\text{neg}}^{(a)} - s_{\text{optim}} + \xi}
$$

where $s_{\text{optim}}$ is the protocol's ideal score (0 for distance metrics; set by the
`perfect` field in the protocol spec) and $\xi > 0$ is a small stabilising constant.
The numerator is the recovered gap (how much the positive control beats the negative);
the denominator is the total available dynamic range from baseline down to optimal.

| $\operatorname{DRF}(a)$ | Meaning |
|---|---|
| $= 1$ | positive control achieves the optimal score |
| $= 0$ | positive and negative controls score equally |
| $< 0$ | positive control performs *worse* than the uninformative baseline |

`--calibrator drf` reports the mean/median of $\operatorname{DRF}(a)$ across perturbations.
Introduced by {cite}`Miller_2025`.

## Bound Discrimination Score (BDS)

BDS asks a simpler, binary question: **for what fraction of perturbations does the protocol
get the ordering right?**

$$
\operatorname{BDS}
= \frac{1}{|\mathcal{P}|}
  \sum_{a \in \mathcal{P}}
  \mathbf{1}\!\left[s_{\text{pos}}^{(a)} < s_{\text{neg}}^{(a)}\right]
$$

It records whether the positive control beats the negative, but not by how much.
A protocol with low BDS cannot distinguish a technical replicate from a random reference;
its scores should not be trusted regardless of their magnitude.

`--calibrator bds` reports this fraction. Introduced by {cite}`Vollenweider_2026`.

## DRF vs BDS

The two scores are complementary. BDS checks the **sign** — does
$s_{\text{pos}}^{(a)} < s_{\text{neg}}^{(a)}$? DRF checks the **magnitude** — how far along
the full dynamic range is that gap? A protocol can have high BDS (ordering consistently
correct) yet low DRF (margin negligible relative to what is achievable). Use both together:
BDS as a pass/fail gate on directionality, DRF as a quantitative measure of signal recovery.

## In practice

```bash
scperteval calibrate data/wessels23.h5ad -p pearson_ctrl,unbiased_mmd_median_pca_k=20,de_overlap_k=10 --de-method t-test
```

<!-- TODO: replace with real output from wessels23.h5ad -->
| protocol | DRF (mean) | BDS |
|---|---|---|
| `pearson_ctrl` | … | … |
| `unbiased_mmd_median_pca_k=20` | … | … |
| `de_overlap_k=10` | … | … |

A protocol with BDS < 0.5 cannot reliably order its controls — its scores should not be trusted
regardless of magnitude. A protocol with high BDS but low DRF is directionally correct but
recovers little of the available signal range.

→ See [Usage](usage.md) for the full CLI reference, all options, and `--help` output.
