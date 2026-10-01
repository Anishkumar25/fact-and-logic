import pandas as pd
from datasets import load_dataset


def load_claims(split):
    """One row per claim: id, claim, label, gold (set of evidence doc ids)."""
    url = f"hf://datasets/allenai/scifact@refs/convert/parquet/claims/{split}/*.parquet"
    df = pd.DataFrame(load_dataset("parquet", data_files={split: url})[split])
    rows = {}
    for r in df.itertuples():
        d = rows.setdefault(r.id, {"id": r.id, "claim": r.claim, "gold": set(),
                                   "label": r.evidence_label or "NOT_ENOUGH_INFO"})
        if r.evidence_doc_id != "":
            d["gold"].add(int(r.evidence_doc_id))
    return pd.DataFrame(rows.values())
