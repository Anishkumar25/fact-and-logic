"""SciFact loaders. Parsing is kept apart from I/O so it can be unit-tested."""
import pandas as pd

from .config import LABELS, SCIFACT


def parse_claims(raw: pd.DataFrame) -> pd.DataFrame:
    lab = raw["evidence_label"].fillna("").astype(str).str.upper()
    lab = lab.where(lab.isin(LABELS), "NOT_ENOUGH_INFO")
    return pd.DataFrame({
        "id": raw["id"],
        "claim": raw["claim"],
        "label": lab,
        "gold_doc": pd.to_numeric(raw["evidence_doc_id"], errors="coerce"),
        "gold_sents": raw["evidence_sentences"].map(
            lambda x: [int(i) for i in x] if x is not None else []),
    })


def corpus() -> pd.DataFrame:
    raw = pd.read_parquet(SCIFACT.format(cfg="corpus", split="train"))
    return pd.DataFrame({"doc_id": raw["doc_id"].astype(int), "title": raw["title"],
                         "sentences": raw["abstract"].map(list)})


def claims(split: str = "validation") -> pd.DataFrame:
    return parse_claims(pd.read_parquet(SCIFACT.format(cfg="claims", split=split)))
