"""Mean-pooled ESM2-650M embeddings, as the QMAP HemoLinear baseline builds them.

Reproduces `eval_prev_works/HemoLinear/generate_esm_embeddings.py`:
last hidden state, CLS and EOS dropped, mean over the remaining real tokens.
Deliberately identical to the published baseline, because the point of the
submission is that the embeddings are NOT what limits the reported score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_CACHE = Path("data") / "cache"
BATCH = 16


def esm2_embeddings(seqs: list[str], cache_dir: Path | str | None = None) -> np.ndarray:
    """Embed `seqs`, caching per sequence so repeated splits cost nothing.

    Returns one row per element of `seqs`, in order (duplicates included).
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    store = cache / "esm2_650M.npz"

    have: dict[str, np.ndarray] = {}
    if store.exists():
        z = np.load(store, allow_pickle=True)
        have = {k: z[k] for k in z.files}

    todo = sorted({s for s in seqs if s not in have})
    if todo:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dev = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  embedding {len(todo)} new sequences with {MODEL} on {dev}", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL)
        mod = AutoModel.from_pretrained(MODEL).to(dev).eval()
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            inp = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024)
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.no_grad():
                hs = mod(**inp).last_hidden_state
            for j, s in enumerate(batch):
                m = inp["attention_mask"][j].bool()
                e = hs[j][m][1:-1].mean(dim=0)  # drop CLS and EOS, then mean pool
                have[s] = e.float().cpu().numpy()
            if (i // BATCH) % 10 == 0:
                print(f"    {i + len(batch)}/{len(todo)}", flush=True)
        np.savez_compressed(store, **have)
        print(f"  cached {len(have)} embeddings", flush=True)

    return np.stack([have[s] for s in seqs])
