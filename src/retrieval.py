import os
import re
 
import numpy as np
import pandas as pd
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
 
CORPUS_URL = "hf://datasets/allenai/scifact@refs/convert/parquet/corpus/train/*.parquet"
METHODS = ("bm25", "embedding", "hybrid")
 
 
def tok(t):
    return re.findall(r"\w+", t.lower())
 
 
def ranks(s):
    r = np.empty(len(s))
    r[np.argsort(-s)] = np.arange(len(s))
    return r
 
 
def load_corpus():
    df = pd.DataFrame(load_dataset("parquet", data_files={"c": CORPUS_URL})["c"])
    df["sentences"] = df["abstract"].apply(list)
    df["text"] = df["title"] + " " + df["sentences"].apply(" ".join)
    return df
 
 
class Retriever:
    """BM25, dense (MiniLM) and hybrid (reciprocal rank fusion) retrieval over SciFact."""
 
    def __init__(self, cache="data/corpus_emb.npy"):
        self.df = load_corpus()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if os.path.exists(cache) and np.load(cache).shape[0] == len(self.df):
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(self.df["text"].tolist(), batch_size=64,
                                         normalize_embeddings=True, show_progress_bar=True)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, self.emb)
        self.bm25 = BM25Okapi([tok(t) for t in self.df["text"]])
 
    def scores(self, claim, method="hybrid"):
        if method == "bm25":
            return self.bm25.get_scores(tok(claim))
        if method == "embedding":
            return self.emb @ self.model.encode(claim, normalize_embeddings=True)
        # reciprocal rank fusion of the two rankings (k=60 is the usual default)
        return sum(1 / (60 + ranks(self.scores(claim, m))) for m in ("bm25", "embedding"))
 
    def search(self, claim, k=3, method="hybrid"):
        s = self.scores(claim, method)
        idx = np.argsort(-s)[:k]
        return self.df.iloc[idx].assign(score=s[idx])
 
 
if __name__ == "__main__":
    r = Retriever()
    print(r.search("0-dimensional biomaterials lack inductive properties.", k=3)[["doc_id", "title", "score"]])
 