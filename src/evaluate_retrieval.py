import pandas as pd
from datasets import load_dataset
from retrieval import load_corpus, build_embedding_index, get_embedder
import numpy as np

# load claims (only ones with a gold evidence doc)
ds = load_dataset(
    "parquet",
    data_files={
        "train": "hf://datasets/allenai/scifact@refs/convert/parquet/claims/train/*.parquet",
    },
)
claims_df = pd.DataFrame(ds["train"])
claims_df = claims_df[claims_df["evidence_doc_id"] != ""].copy()
claims_df["evidence_doc_id"] = claims_df["evidence_doc_id"].astype(int)

corpus_df = load_corpus()
print("Building embedding index...")
corpus_embeddings = build_embedding_index(corpus_df)
model = get_embedder()

doc_ids = corpus_df["doc_id"].tolist()
doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

sample = claims_df.sample(min(100, len(claims_df)), random_state=42)
claim_embeddings = model.encode(sample["claim"].tolist())

hits_at_1, hits_at_3, hits_at_5, ranks = 0, 0, 0, []

for i, (_, row) in enumerate(sample.iterrows()):
    gold_id = row["evidence_doc_id"]
    if gold_id not in doc_id_to_idx:
        continue
    sims = corpus_embeddings @ claim_embeddings[i] / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(claim_embeddings[i])
    )
    ranked = sims.argsort()[::-1]
    gold_idx = doc_id_to_idx[gold_id]
    rank = int((ranked == gold_idx).nonzero()[0][0])
    ranks.append(rank)
    if rank < 1:
        hits_at_1 += 1
    if rank < 3:
        hits_at_3 += 1
    if rank < 5:
        hits_at_5 += 1

n = len(ranks)
print(f"\nEvaluated on {n} claims")
print(f"Hit@1: {hits_at_1/n:.2%}")
print(f"Hit@3: {hits_at_3/n:.2%}")
print(f"Hit@5: {hits_at_5/n:.2%}")
print(f"Median rank: {np.median(ranks):.0f}")