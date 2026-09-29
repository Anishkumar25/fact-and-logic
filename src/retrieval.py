import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def load_corpus():
    ds = load_dataset(
        "parquet",
        data_files={"corpus": "hf://datasets/allenai/scifact@refs/convert/parquet/corpus/train/*.parquet"},
    )
    corpus_df = pd.DataFrame(ds["corpus"])
    corpus_df["text"] = corpus_df["title"] + " " + corpus_df["abstract"].apply(lambda x: " ".join(x))
    return corpus_df


# ---------- BM25 baseline (kept for comparison in the report) ----------

def build_bm25_index(corpus_df):
    tokenized = [word_tokenize(t.lower()) for t in corpus_df["text"]]
    return BM25Okapi(tokenized)


def retrieve_bm25(claim, corpus_df, bm25, k=3):
    tokens = word_tokenize(claim.lower())
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[::-1][:k]
    return corpus_df.iloc[top_idx][["doc_id", "title", "text"]]


# ---------- Embedding retrieval (the one we actually use) ----------

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_embedding_index(corpus_df):
    model = get_embedder()
    embeddings = model.encode(corpus_df["text"].tolist(), show_progress_bar=True, batch_size=64)
    return embeddings


def retrieve_embedding(claim, corpus_df, corpus_embeddings, k=3):
    model = get_embedder()
    claim_emb = model.encode([claim])[0]
    sims = corpus_embeddings @ claim_emb / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(claim_emb)
    )
    top_idx = sims.argsort()[::-1][:k]
    return corpus_df.iloc[top_idx][["doc_id", "title", "text"]], sims


if __name__ == "__main__":
    corpus_df = load_corpus()
    test_claim = "0-dimensional biomaterials lack inductive properties."
    gold_doc_id = 31715818

    print("Building embedding index (one-time, ~1-2 min on CPU)...")
    corpus_embeddings = build_embedding_index(corpus_df)

    results, sims = retrieve_embedding(test_claim, corpus_df, corpus_embeddings, k=5)
    print("\nTop 5 (embedding retrieval):")
    print(results)

    gold_idx = corpus_df[corpus_df["doc_id"] == gold_doc_id].index[0]
    gold_rank = (sims.argsort()[::-1] == gold_idx).nonzero()[0][0]
    print(f"\nGold doc rank: {gold_rank} out of {len(sims)}")