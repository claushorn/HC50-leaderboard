"""Ridge on ESM2 embeddings for the QMAP HC50 benchmark.

A correction to the published HC50 baseline: the leaderboard's linear probe is
unregularised OLS on 1280 collinear dimensions against 650-801 training rows
(depending on the split), and
regularising it -- changing nothing else -- moves mean Pearson r from 0.059 to
0.270 and the worst split from -0.179 to +0.200.
"""

from .esm2 import MODEL, esm2_embeddings
from .features import (
    AAS,
    EISENBERG,
    desc_matrix,
    descriptors,
    pick_lambda,
    ridge_fit,
    ridge_pred,
    standardise,
)
from .reproduce import (
    MODELS,
    PUBLISHED_OLS,
    check_reproduction,
    filtered_benchmark,
    filtered_pool,
    run,
    summarise,
)

__all__ = [
    "AAS", "EISENBERG", "MODEL", "MODELS", "PUBLISHED_OLS",
    "check_reproduction", "desc_matrix", "descriptors", "esm2_embeddings",
    "filtered_benchmark", "filtered_pool", "pick_lambda", "ridge_fit",
    "ridge_pred", "run", "standardise", "summarise",
]
