"""Hybrid retrieval: BM25 + MiniLM embeddings fused with reciprocal rank fusion.
`python -m src.retrieval` builds and caches the embedding index."""
import re

import numpy as np
import pandas as pd

from . import data
from .config import ART, EMB_MODEL


def _tok(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion. Each ranking lists ids best-first."""
    s: dict[int, float] = {}
    for r in rankings:
        for i, d in enumerate(r):
            s[d] = s.get(d, 0.0) + 1.0 / (k + i + 1)
    return sorted(s, key=s.get, reverse=True)


class Retriever:
    def __init__(self, corpus: pd.DataFrame | None = None, cache=None):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMB_MODEL)
        self.docs = (data.corpus() if corpus is None else corpus).reset_index(drop=True)
        text = (self.docs.title + ". " + self.docs.sentences.map(" ".join)).tolist()
        self.bm25 = BM25Okapi([_tok(t) for t in text])
        name = f"emb_{EMB_MODEL.split('/')[-1]}.npy"
        self.emb = self._embeddings(text, cache or ART / "index" / name)

    def _embeddings(self, text, path):
        if path.exists():
            e = np.load(path)
            if len(e) == len(text):
                return e
        e = self.model.encode(text, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, e)
        return e

    def search(self, claim: str, k: int = 3, pool: int = 50, mode: str = "hybrid") -> pd.DataFrame:
        b = np.argsort(-self.bm25.get_scores(_tok(claim)))[:pool]
        d = np.argsort(-(self.emb @ self.model.encode(claim, normalize_embeddings=True)))[:pool]
        ids = {"bm25": b, "dense": d}.get(mode)
        ids = rrf([b.tolist(), d.tolist()]) if ids is None else ids.tolist()
        return self.docs.iloc[ids[:k]]


if __name__ == "__main__":
    Retriever()
