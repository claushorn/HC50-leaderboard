"""Reproduce the QMAP HC50 baseline, then isolate the estimator.

The QMAP HC50 leaderboard carries a single entry: a linear probe on ESM2
embeddings at min/mean/max Pearson = -0.18 / 0.07 / 0.29. Reading
`eval_prev_works/HemoLinear/main.py`, that probe is `LinearRegression` --
UNREGULARISED ordinary least squares on 1280-dimensional ESM2 embeddings
against 650-801 training rows, depending on the split, after the homology train
mask. ESM2 embedding dimensions are
strongly collinear, so OLS there is close to ill-posed, and will produce
exactly the unstable, sometimes-negative cross-fold correlations reported.

So the question this script answers is not "which representation is better".
It is: how much of the published ceiling is a property of the ESM2
REPRESENTATION, and how much is an artifact of fitting it with OLS?

Four models, one harness, QMAP's own filters, splits and `compute_metrics`:

  esm2_ols        their method, reproduced verbatim (sklearn LinearRegression)
  esm2_ridge      the same embeddings, ridge, lambda by inner CV on train only
  desc_ridge      11 interpretable biophysical descriptors, ridge
  desc_esm_ridge  both concatenated, ridge

`esm2_ols` vs `esm2_ridge` isolates the estimator, with everything else held
fixed. `esm2_ridge` vs `desc_ridge` is then the representation comparison that
the OLS row could not support.

The `esm2_ols` row is the control: if it does not land on the published
-0.18 / 0.07 / 0.29, the harness is not faithful and no other row here means
anything. Check that row first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .esm2 import esm2_embeddings
from .features import desc_matrix, pick_lambda, ridge_fit, ridge_pred, standardise

# The published row this harness must reproduce before anything else is read.
PUBLISHED_OLS = {"min": -0.180, "mean": 0.070, "max": 0.290}
MODELS = ["esm2_ols", "esm2_ridge", "desc_ridge", "desc_esm_ridge"]


def filtered_pool():
    """The training pool under HemoLinear's filter chain."""
    from qmap import DBAASPDataset
    return (DBAASPDataset()
            .with_hc50()
            .with_canonical_only()
            .with_l_aa_only()
            .with_terminal_modification(False, False))


def filtered_benchmark(split: int):
    """The benchmark split under HemoLinear's filter chain.

    `with_length_range(None, 100)` is applied to the benchmark only, matching
    HemoLinear. Dropping it changes the evaluation subset and makes the numbers
    incomparable to the published row.
    """
    from qmap import QMAPBenchmark
    return (QMAPBenchmark(split)
            .with_hc50()
            .with_canonical_only()
            .with_l_aa_only()
            .with_terminal_modification(False, False)
            .with_length_range(None, 100))


def run(cache_dir: Path | str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fit all four models on all five splits. Returns one row per model/split."""
    from sklearn.linear_model import LinearRegression

    pool = filtered_pool()
    tab = pool.tabular(["sequence", "hc50"])
    seqs = tab["sequence"].tolist()
    y_all = np.log10(tab["hc50"].to_numpy())
    if verbose:
        print(f"filtered training pool: {len(seqs)} sequences")

    all_seqs = set(seqs)
    bench, bseq = {}, {}
    for s in range(5):
        b = filtered_benchmark(s)
        bench[s] = b
        bseq[s] = b.tabular(["sequence"])["sequence"].tolist()
        all_seqs |= set(bseq[s])
    if verbose:
        print(f"unique sequences needing embeddings: {len(all_seqs)}")

    esm2_embeddings(sorted(all_seqs), cache_dir)   # populate the cache once
    E_pool = esm2_embeddings(seqs, cache_dir)
    D_pool, dcols = desc_matrix(seqs)
    if verbose:
        print(f"ESM2 dim={E_pool.shape[1]}  descriptors={len(dcols)}")

    rows = []
    for s in range(5):
        if verbose:
            print(f"\n--- split {s} " + "-" * 56, flush=True)
        b = bench[s]
        te_seq = bseq[s]
        mask = b.get_train_mask(seqs, show_progress=False)
        ntr = int(np.sum(mask))
        if verbose:
            print(f"  train={ntr}  test={len(te_seq)}")

        ytr = y_all[mask]
        E_tr, E_te = E_pool[mask], esm2_embeddings(te_seq, cache_dir)
        D_tr, D_te = D_pool[mask], desc_matrix(te_seq)[0]

        preds, lams = {}, {}

        # Their exact method: unregularised OLS on the raw embeddings.
        preds["esm2_ols"] = LinearRegression().fit(E_tr, ytr).predict(E_te)

        # The same embeddings, regularised. This is the whole change.
        Etr_s, Ete_s = standardise(E_tr, E_te)
        lams["esm2_ridge"] = pick_lambda(Etr_s, ytr, seed=s)
        preds["esm2_ridge"] = ridge_pred(
            ridge_fit(Etr_s, ytr, lams["esm2_ridge"]), Ete_s)

        Dtr_s, Dte_s = standardise(D_tr, D_te)
        lams["desc_ridge"] = pick_lambda(Dtr_s, ytr, seed=s)
        preds["desc_ridge"] = ridge_pred(
            ridge_fit(Dtr_s, ytr, lams["desc_ridge"]), Dte_s)

        Ctr = np.hstack([Dtr_s, Etr_s])
        Cte = np.hstack([Dte_s, Ete_s])
        lams["desc_esm_ridge"] = pick_lambda(Ctr, ytr, seed=s)
        preds["desc_esm_ridge"] = ridge_pred(
            ridge_fit(Ctr, ytr, lams["desc_esm_ridge"]), Cte)

        if verbose:
            print("  lambda: " + "  ".join(f"{k}={v:g}" for k, v in lams.items()))
        for name in MODELS:
            p = preds[name]
            # Scored by the benchmark's own metric implementation, not ours.
            m = b.compute_metrics([{"hc50": float(v)} for v in p], log=True)["hc50"]
            d = m.dict() if hasattr(m, "dict") else {}
            rows.append(dict(split=s, model=name, n_train=ntr, n_test=len(te_seq),
                             pearson=d.get("pearson", np.nan),
                             spearman=d.get("spearman", np.nan),
                             rmse=d.get("rmse", np.nan), r2=d.get("r2", np.nan)))
            if verbose:
                print(f"    {name:16s} PCC={d.get('pearson', float('nan')):+.3f}"
                      f"  rho={d.get('spearman', float('nan')):+.3f}"
                      f"  RMSE={d.get('rmse', float('nan')):.3f}"
                      f"  R2={d.get('r2', float('nan')):+.3f}")

    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Per-model min/mean/max Pearson across the five splits."""
    piv = res.pivot(index="model", columns="split", values="pearson")
    piv["min"] = piv.iloc[:, :5].min(axis=1)
    piv["mean"] = piv.iloc[:, :5].mean(axis=1)
    piv["max"] = piv.iloc[:, :5].max(axis=1)
    return piv.loc[[m for m in MODELS if m in piv.index]]


def check_reproduction(piv: pd.DataFrame, tol: float = 0.05) -> bool:
    """Does `esm2_ols` land on the published row? Printed verdict, returns bool."""
    r = piv.loc["esm2_ols"]
    print("\nControl: reproduction of the published ESM2 linear probe")
    print(f"  published   min={PUBLISHED_OLS['min']:+.3f}  "
          f"mean={PUBLISHED_OLS['mean']:+.3f}  max={PUBLISHED_OLS['max']:+.3f}")
    print(f"  reproduced  min={r['min']:+.3f}  mean={r['mean']:+.3f}  max={r['max']:+.3f}")
    ok = all(abs(r[k] - v) <= tol for k, v in PUBLISHED_OLS.items())
    print(f"  within {tol:g} on all three: {'YES' if ok else 'NO'}")
    if not ok:
        print("  -> the harness is NOT faithful; do not read the other rows.")
    return ok


def main() -> None:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    res = run()
    res.to_csv(out / "hc50_reproduction.csv", index=False)

    print("\n" + "=" * 78)
    print("Pearson r -- identical filters, identical splits, official compute_metrics")
    print("=" * 78)
    piv = summarise(res)
    print(piv.round(3).to_string())
    check_reproduction(piv)

    print("\n" + "=" * 78)
    print("Spearman rho (same table)")
    print("=" * 78)
    pv2 = res.pivot(index="model", columns="split", values="spearman")
    pv2["mean"] = pv2.mean(axis=1)
    print(pv2.loc[[m for m in MODELS if m in pv2.index]].round(3).to_string())
    print(f"\nwrote {out / 'hc50_reproduction.csv'}")


if __name__ == "__main__":
    main()
