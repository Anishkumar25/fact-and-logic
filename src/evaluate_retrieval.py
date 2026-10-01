import os
 
import numpy as np
import pandas as pd
 
from src.data import load_claims
from src.retrieval import METHODS, Retriever
 
KS = (1, 3, 5, 10, 20)
 
if __name__ == "__main__":
    r = Retriever()
    claims = load_claims("validation")
    claims = claims[claims["gold"].apply(len) > 0]  # NEI claims have no gold doc
    ids = r.df["doc_id"].values
    rows = []
    for m in METHODS:
        rk = []
        for c in claims.itertuples():
            order = np.argsort(-r.scores(c.claim, m))
            hit = np.flatnonzero(np.isin(ids[order], list(c.gold)))
            rk.append(hit.min() if len(hit) else len(order))
        rk = np.array(rk)
        rows.append({"method": m, "n_claims": len(rk), "median_rank": int(np.median(rk)),
                     **{f"hit@{k}": round(float((rk < k).mean()), 3) for k in KS}})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/retrieval.csv", index=False)
 