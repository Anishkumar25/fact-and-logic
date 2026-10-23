"""SciFact loaders. Parsing is kept apart from I/O so it can be unit-tested."""
import pandas as pd

from .config import LABELS, SCIFACT


def parse_claims(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per claim. The raw table can hold one row per (claim, evidence doc), so group by id.
    gold_docs: every document that holds evidence; gold_sents: {doc_id: [sentence indices]}."""
    lab = raw["evidence_label"].fillna("").astype(str).str.upper()
    rows = pd.DataFrame({
        "id": raw["id"], "claim": raw["claim"],
        "label": lab.where(lab.isin(LABELS), "NOT_ENOUGH_INFO"),
        "doc": pd.to_numeric(raw["evidence_doc_id"], errors="coerce"),
        "sents": raw["evidence_sentences"].map(lambda x: [int(i) for i in x] if x is not None else []),
    })
    out = []
    for cid, g in rows.groupby("id", sort=False):
        ev, gold = g[g.doc.notna()], {}
        for d, s in zip(ev.doc, ev.sents):
            gold.setdefault(int(d), set()).update(s)
        out.append({"id": cid, "claim": g.claim.iloc[0],
                    "label": ev.label.iloc[0] if len(ev) else "NOT_ENOUGH_INFO",
                    "gold_docs": sorted(gold), "gold_sents": {d: sorted(s) for d, s in gold.items()}})
    return pd.DataFrame(out)


def corpus() -> pd.DataFrame:
    raw = pd.read_parquet(SCIFACT.format(cfg="corpus", split="train"))
    return pd.DataFrame({"doc_id": raw["doc_id"].astype(int), "title": raw["title"],
                         "sentences": raw["abstract"].map(list)})


def claims(split: str = "validation") -> pd.DataFrame:
    return parse_claims(pd.read_parquet(SCIFACT.format(cfg="claims", split=split)))
