"""Cross-encoder reranker for picking evidence sentences.
Swap models with RERANK_MODEL (e.g. ncbi/MedCPT-Cross-Encoder for biomedical text)."""
import numpy as np

from .config import RERANK_MODEL, device


class Reranker:
    def __init__(self, name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(name, max_length=256, device=device())

    def scores(self, claim: str, texts: list[str]) -> np.ndarray:
        return np.asarray(self.m.predict([(claim, t) for t in texts]))
