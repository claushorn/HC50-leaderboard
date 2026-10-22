# The text to post

Open at https://github.com/anthol42/QMAP/issues/new — the leaderboard section of QMAP's
README asks for an issue carrying the code to reproduce and the reference paper.

**Title:** `HC50 leaderboard: the ESM2 baseline is unregularized OLS; ridge on the same embeddings gives mean PCC 0.27`

---

Thanks for QMAP — the homology-aware splits and the single `compute_metrics` entry point
made this reproducible in an afternoon.

This is both a leaderboard entry and a correction to the existing baseline, so I've
combined them into one issue.

## Issue

`eval_prev_works/HemoLinear/main.py` fits `sklearn.linear_model.LinearRegression` —
unregularized OLS — on 1280-dimensional mean-pooled ESM2-650M embeddings with 752 training
rows after the homology train mask. ESM2 dimensions are strongly collinear, so OLS there is
close to ill-posed, and the resulting instability is what the leaderboard's `min = -0.18`
records. The HC50 row therefore measures the estimator rather than the representation.

## Reproduction, then the one-line change

Same filter chain (`with_hc50().with_canonical_only().with_l_aa_only().with_terminal_modification(False, False)`,
benchmark `with_length_range(None, 100)`), same five splits, your `compute_metrics(..., log=True)`.

| model | s0 | s1 | s2 | s3 | s4 | min | mean | max |
|---|---|---|---|---|---|---|---|---|
| `esm2_ols` (HemoLinear, reproduced) | -0.179 | 0.111 | 0.293 | 0.038 | 0.032 | -0.179 | **0.059** | 0.293 |
| *published README row* | | | | | | *-0.180* | *0.070* | *0.290* |
| `esm2_ridge` (identical embeddings) | 0.232 | 0.367 | 0.300 | 0.250 | 0.200 | **0.200** | **0.270** | 0.367 |
| `desc_ridge` (11 biophysical descriptors) | 0.198 | 0.079 | 0.014 | 0.092 | 0.058 | 0.014 | 0.088 | 0.198 |
| `desc_esm_ridge` (concatenated) | 0.230 | 0.363 | 0.298 | 0.249 | 0.200 | 0.200 | 0.268 | 0.363 |

The OLS reproduction lands on the published row (0.059 vs 0.070; -0.179 vs -0.180), which
is the control: the pipeline is faithful, so the difference below it is attributable to the
estimator alone.

Replacing `LinearRegression` with ridge — lambda chosen by 5-fold CV on the training pool
only, features standardized on train statistics, nothing else changed — moves the mean 4.6x
and turns the worst split from -0.179 to +0.200. The test split is never consulted during
penalty selection, so ridge has no information advantage over the baseline.

The embeddings also subsume the hand-built descriptors entirely (0.268 concatenated vs
0.270 alone), which is the fair representation comparison the OLS row could not support.

This is not a claim that HC50 regression is solved: 0.270 is still modest, and DBAASP
replicate assays agree to about +/-0.34 log10, putting the attainable ceiling near r = 0.88.

## Suggested leaderboard rows

| Method | Year | min PCC | mean | max | Source Code |
|---|---|---|---|---|---|
| Ridge on ESM2 embeddings | 2026 | 0.20 | 0.27 | 0.37 | [`claushorn/HC50-leaderboard`](https://github.com/claushorn/HC50-leaderboard) |
| Linear (OLS) on ESM2 embeddings | N/A | -0.18 | 0.07 | 0.29 | `eval_prev_works/HemoLinear` |

I'd suggest keeping the OLS row — the gap between the two is the informative part. If you'd
rather the entry point lived in-tree, I'm happy to open a PR adding a `HemoRidge/` directory
beside `HemoLinear/` instead of pointing at an external repository.

## Code

https://github.com/claushorn/HC50-leaderboard

```bash
git clone https://github.com/claushorn/HC50-leaderboard
cd HC50-leaderboard
uv run python -m hemoridge.reproduce
```

All four models in one harness against your package; the `esm2_ols` control is checked
against the published row automatically, so the claim can be verified without trusting my
embedding code.

No reference paper — this is a measurement correction rather than a new method. Happy to
supply the embedding cache if that saves anyone a GPU pass.
