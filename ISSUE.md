# The text to post

Open at https://github.com/anthol42/QMAP/issues/new ; the leaderboard section of QMAP's
README asks for an issue carrying the code to reproduce and the reference paper.

**Title:** `HC50 leaderboard: the ESM2 baseline is unregularized OLS; ridge on the same embeddings gives mean PCC 0.27`

---

Thanks for QMAP: the homology-aware splits and the single `compute_metrics` entry point
made this reproducible in an afternoon.

This is both a leaderboard entry and a correction to the existing baseline, so I've
combined them into one issue.

## Issue

`eval_prev_works/HemoLinear/main.py` fits `sklearn.linear_model.LinearRegression`,
unregularized OLS, on 1280-dimensional mean-pooled ESM2-650M embeddings. After the
homology train mask that leaves between 650 and 801 training rows depending on the split,
against 1280 predictors. ESM2 embedding dimensions are strongly collinear, so OLS in that
regime is close to ill-posed, and the resulting instability is what the leaderboard's
`min = -0.18` records. The HC50 row therefore measures the estimator rather than the
representation.

## Reproduction, then the one-line change

Same filter chain (`with_hc50().with_canonical_only().with_l_aa_only().with_terminal_modification(False, False)`,
benchmark `with_length_range(None, 100)`), same five predefined splits, your
`compute_metrics(..., log=True)`, target `log10(hc50)`.

Pearson *r* per split:

| model | s0 | s1 | s2 | s3 | s4 | min | mean | max |
|---|---|---|---|---|---|---|---|---|
| `esm2_ols` (HemoLinear, reproduced) | -0.179 | 0.111 | 0.293 | 0.038 | 0.032 | -0.179 | **0.059** | 0.293 |
| *published README row* | *not published per split* | | | | | *-0.180* | *0.070* | *0.290* |
| `esm2_ridge` (identical embeddings) | 0.232 | 0.367 | 0.300 | 0.250 | 0.200 | **0.200** | **0.270** | 0.367 |
| `desc_ridge` (11 biophysical descriptors) | 0.198 | 0.079 | 0.014 | 0.092 | 0.058 | 0.014 | 0.088 | 0.198 |
| `desc_esm_ridge` (concatenated) | 0.230 | 0.363 | 0.298 | 0.249 | 0.200 | 0.200 | 0.268 | 0.363 |

Spearman rho, same runs: `esm2_ols` 0.059 mean, `esm2_ridge` 0.299 mean. Full per-split
Spearman, RMSE and R² are in `results/hc50_reproduction.csv` in the linked repository.

The OLS reproduction lands on the published row (0.059 vs 0.070; -0.179 vs -0.180), which
is the control: the pipeline is faithful, so the difference below it is attributable to the
estimator alone rather than to some unnoticed difference in filtering or pooling.

Replacing `LinearRegression` with ridge (**nothing else changed**) moves the mean 4.6x
and turns the worst split from -0.179 to +0.200. The embeddings also subsume the hand-built
descriptors entirely (0.268 concatenated vs 0.270 alone), which is the fair representation
comparison the OLS row could not support.

### Setting, for exactness

| split | train rows (after `get_train_mask`) | test rows | lambda chosen for `esm2_ridge` |
|---|---|---|---|
| 0 | 752 | 173 | 300 |
| 1 | 650 | 278 | 100 |
| 2 | 801 | 129 | 300 |
| 3 | 736 | 192 | 300 |
| 4 | 773 | 141 | 300 |

Training pool before the mask: 934 sequences; 930 unique sequences needed embedding across
the pool and all five benchmark splits.

Lambda is selected by 5-fold CV **within the training pool only**, over a fixed grid
(`0.1` to `1e5`, decade-ish steps), and features are standardized on training statistics.
The benchmark's test split is never read during selection or standardization, so ridge has
no information advantage over the OLS baseline: the comparison is the estimator and only
the estimator.

Environment: Python 3.12.11, `qmap-benchmark` 0.1.1, numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.9.1, torch 2.14.1, transformers 5.20.2, `facebook/esm2_t33_650M_UR50D`,
Apple silicon (MPS). Embeddings are mean-pooled over real tokens with CLS and EOS dropped,
matching `generate_esm_embeddings.py`.

### What this does not claim

0.270 is still modest, and this is not a claim that HC50 regression is solved. DBAASP
replicate assays of the same peptide agree to roughly ±0.34 log₁₀, about a two-fold
concentration step (ordinary dilution-series granularity), which puts the attainable
ceiling near *r* ≈ 0.88. Most of the headroom is still there. The claim is narrower: the
published number reflects a choice of estimator, not a limit of the representation, and the
leaderboard currently reads as though it reflected the latter.

## Suggested leaderboard rows

| Method | Year | min PCC | mean | max | Source Code |
|---|---|---|---|---|---|
| Ridge on ESM2 embeddings | 2026 | 0.20 | 0.27 | 0.37 | [`claushorn/HC50-leaderboard`](https://github.com/claushorn/HC50-leaderboard) |
| Linear (OLS) on ESM2 embeddings | N/A | -0.18 | 0.07 | 0.29 | `eval_prev_works/HemoLinear` |

I'd suggest keeping the OLS row: the gap between the two is the informative part. If you'd
rather the entry point lived in-tree, I'm happy to open a PR adding a `HemoRidge/` directory
beside `HemoLinear/` instead of pointing at an external repository.

Minor, while that table is open: the three column headers in the **Full - hc50** section
read `e. coli min PCC` / `e. coli mean` / `e. coli max PCC`, carried over from the MIC
tables above it.

## Code

https://github.com/claushorn/HC50-leaderboard

```bash
git clone https://github.com/claushorn/HC50-leaderboard
cd HC50-leaderboard
uv run python -m hemoridge.reproduce
```

All four models in one harness against your package, writing
`results/hc50_reproduction.csv`. The `esm2_ols` control is checked against the published
row automatically and the run reports the verdict, so the claim can be verified without
anyone having to trust my embedding code. First run pays for the ESM2 pass (~930 sequences,
cached thereafter); CUDA, MPS and CPU are all handled.

No reference paper: this is a measurement correction rather than a new method, which is
why I've leaned on making the control reproducible instead. Happy to share the embedding
cache if it saves anyone a GPU pass, and happy to re-run under any filter or metric
variation you'd prefer to see.

— Claus Horn (@claushorn)
