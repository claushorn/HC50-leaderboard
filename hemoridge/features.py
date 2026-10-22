"""Features and estimators used by the HC50 reproduction.

Nothing here is specific to the correction being submitted: these are the
biophysical descriptor set, a closed-form ridge, and the standardisation and
penalty-selection helpers that the reproduction harness shares between its
four models.

This module is the single definition of each of these quantities. The
hemolysis investigation that produced the correction imports them from here
rather than keeping its own copy, so the numbers published in the leaderboard
issue and the numbers in that investigation cannot drift apart.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Eisenberg consensus hydrophobicity scale (normalised).
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
AAS = "ACDEFGHIKLMNPQRSTVWY"
POS, NEG = set("KR"), set("DE")


def descriptors(seq: str) -> dict:
    """A priori, mechanism-motivated descriptors of membrane activity.

    Chosen from the amphipathicity-driven insertion mechanism rather than by
    feature search, so the set is fixed before any evaluation.
    """
    s = [c for c in seq.upper() if c in EISENBERG]
    n = len(s)
    if n == 0:
        return {}
    h = np.array([EISENBERG[c] for c in s])

    # Eisenberg hydrophobic moment: helical periodicity of 100 deg/residue.
    ang = np.deg2rad(100.0) * np.arange(n)
    mu = np.hypot((h * np.cos(ang)).sum(), (h * np.sin(ang)).sum()) / n

    charge = sum(c in POS for c in s) - sum(c in NEG for c in s)
    return {
        "length": n,
        "charge": charge,
        "charge_density": charge / n,
        "H_mean": h.mean(),
        "mu_H": mu,
        "f_hydrophobic": np.mean([c in "AILMFWVC" for c in s]),
        "f_aromatic": np.mean([c in "FWY" for c in s]),
        "f_K": np.mean([c == "K" for c in s]),
        "f_R": np.mean([c == "R" for c in s]),
        "f_G": np.mean([c == "G" for c in s]),
        "f_P": np.mean([c == "P" for c in s]),
    }


def desc_matrix(seqs: list[str]) -> tuple[np.ndarray, list[str]]:
    """Descriptor design matrix; empty descriptor dicts become column means."""
    rows = [descriptors(s) for s in seqs]
    cols = sorted({k for r in rows for k in r})
    X = np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    return X, cols


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge on standardised X with intercept, no sklearn."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    pen = lam * np.eye(Xb.shape[1])
    pen[0, 0] = 0.0  # never penalise the intercept
    return np.linalg.solve(Xb.T @ Xb + pen, Xb.T @ y)


def ridge_pred(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X]) @ w


def standardise(Xtr: np.ndarray, Xte: np.ndarray):
    """Centre and scale on TRAINING statistics only."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


LAMBDA_GRID = [0.1, 1, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]


def pick_lambda(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Select the ridge penalty by 5-fold CV on the training pool only.

    The benchmark's test split is never consulted, so this adds no information
    beyond what the published OLS baseline already had access to.
    """
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 5, len(X))
    best, bl = -np.inf, 1.0
    for lam in LAMBDA_GRID:
        rs = []
        for k in range(5):
            a, b = fold != k, fold == k
            if b.sum() < 5:
                continue
            p = ridge_pred(ridge_fit(X[a], y[a], lam), X[b])
            if np.std(p) == 0:
                continue
            rs.append(stats.pearsonr(y[b], p)[0])
        if rs and np.mean(rs) > best:
            best, bl = float(np.mean(rs)), lam
    return bl
