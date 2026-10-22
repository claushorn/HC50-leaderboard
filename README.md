# Ridge Regression on ESM2 embeddings for the QMAP HC50 benchmark

The [QMAP](https://github.com/anthol42/QMAP) HC50 leaderboard carries a single entry: a
linear probe on ESM2 embeddings at min/mean/max Pearson *r* = −0.18 / 0.07 / 0.29. That
number reads as "hemolysis regression is unsolved."

It is an estimator artifact. `eval_prev_works/HemoLinear/main.py` fits
`sklearn.linear_model.LinearRegression` — unregularized OLS — on 1280-dimensional
mean-pooled ESM2-650M embeddings with **752** training rows after the homology train mask.
ESM2 dimensions are strongly collinear, so OLS there is close to ill-posed, and the
instability is what the leaderboard's negative minimum records.

Replacing `LinearRegression` with ridge — penalty chosen by 5-fold CV on the training pool
only, features standardized on train statistics, **nothing else changed** — moves the mean
**4.6×** and turns the worst split from −0.179 to +0.200.

## Results

QMAP's own filter chain, its five predefined splits, and its own `compute_metrics`.

| model | split 0 | 1 | 2 | 3 | 4 | min | mean | max |
|---|---|---|---|---|---|---|---|---|
| `esm2_ols` — HemoLinear, reproduced | −0.179 | 0.111 | 0.293 | 0.038 | 0.032 | −0.179 | **0.059** | 0.293 |
| *published README row* | | | | | | *−0.180* | *0.070* | *0.290* |
| `esm2_ridge` — identical embeddings | 0.232 | 0.367 | 0.300 | 0.250 | 0.200 | **0.200** | **0.270** | 0.367 |
| `desc_ridge` — 11 biophysical descriptors | 0.198 | 0.079 | 0.014 | 0.092 | 0.058 | 0.014 | 0.088 | 0.198 |
| `desc_esm_ridge` — concatenated | 0.230 | 0.363 | 0.298 | 0.249 | 0.200 | 0.200 | 0.268 | 0.363 |

Full per-split metrics, including Spearman, RMSE and R², are in
[`results/hc50_reproduction.csv`](results/hc50_reproduction.csv).

### Read the control row first

`esm2_ols` is not a competing model. It is the harness's control: it reproduces the
published row to 0.059 vs 0.070 and −0.179 vs −0.180. **If that row did not land on the
published numbers, nothing else in the table would mean anything**, the difference below
it could be any of a dozen pipeline details rather than the estimator. `run()` therefore
reports it alongside a `check_reproduction()` verdict, and the comparison that matters is
`esm2_ols` → `esm2_ridge` with every other component held fixed.

### What this does and does not claim

- **Does:** the published HC50 ceiling is set by the estimator, not by the ESM2
  representation. One line of code accounts for a 4.6× difference in mean *r*.
- **Does not:** claim hemolysis regression is solved. 0.270 is still modest. Replicate
  assays in DBAASP agree to roughly ±0.34 log₁₀, which puts the attainable ceiling near
  *r* ≈ 0.88, so there remains about an order of magnitude of headroom, and the HC50
  target also silently pools assays performed on different target cells.
- **Does not:** claim the biophysical descriptors are competitive. They are not (0.088).
  They are included because the concatenation (0.268) scores no better than the embeddings
  alone (0.270), which shows the embeddings already subsume them, the fair representation
  comparison the OLS row could not support.

Lambda is selected by inner CV on the training pool. The benchmark's test split is never
consulted during selection, so ridge has no information advantage over the OLS baseline.

## Reproduce

```bash
git clone https://github.com/claushorn/HC50-leaderboard
cd HC50-leaderboard
uv run python -m hemoridge.reproduce
```

Downloads the QMAP dataset through `qmap-benchmark`, embeds ~1.1k unique sequences with
ESM2-650M (cached in `data/cache/`, so only the first run pays for it; CUDA, Apple MPS and
CPU are all handled), fits all four models on all five splits, and writes
`results/hc50_reproduction.csv`.

Roughly 10 minutes on a laptop GPU for the embedding pass, seconds per model thereafter.

## Layout

| file | contents |
|---|---|
| `hemoridge/reproduce.py` | the harness: QMAP filter chain, the four models, the control check |
| `hemoridge/esm2.py` | mean-pooled ESM2-650M embeddings, as HemoLinear builds them |
| `hemoridge/features.py` | descriptor set, closed-form ridge, standardisation, penalty selection |

## Citing the benchmark

```bibtex
@misc{lavertu_qmap_2026,
    title = {{QMAP}: {A} {Benchmark} for {Standardized} {Evaluation} of {Antimicrobial}
             {Peptide} {MIC} and {Hemolytic} {Activity} {Regression}},
    url = {https://www.biorxiv.org/content/10.64898/2026.02.03.703041v1},
    doi = {10.64898/2026.02.03.703041},
    publisher = {bioRxiv},
    author = {Lavertu, Anthony and Corbeil, Jacques and Germain, Pascal},
    month = feb,
    year = {2026}
}
```

This repository contains only the HC50 reproduction and the correction submitted to the
QMAP leaderboard. It uses QMAP data exclusively; no other dataset is redistributed here.
