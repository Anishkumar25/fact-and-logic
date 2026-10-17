"""Evaluation helpers and CLI.  python -m src.evaluate {retrieval,sentences,verdict} [--limit N]

Thresholds are tuned on the SciFact train split (the NLI model never trains on it) and
reported on validation, so the headline number isn't tuned on the data it's scored on."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from . import data
from .config import ART, LABELS, REPORTS, THRESHOLDS
from .verdict import decide


def retrieval(r, df, ks=(1, 3, 5, 20), mode="hybrid") -> dict:
    df = df[df.gold_doc.notna()]
    ranks = []
    for c, g in zip(df.claim, df.gold_doc):
        ids = r.search(c, max(ks), mode=mode).doc_id.tolist()
        ranks.append(ids.index(int(g)) if int(g) in ids else 10**9)
    ranks = np.array(ranks)
    return {f"hit@{k}": float((ranks < k).mean()) for k in ks}


def sentence_recall(v, df) -> float:
    """Share of claims where a gold evidence sentence is among the verifier's top-n sentences."""
    df = df[df.gold_doc.notna()]
    hits = []
    for c, g, gs in zip(df.claim, df.gold_doc, df.gold_sents):
        top = v.top_sentences(c, v.r.search(c, v.k))
        hits.append(any(s[0] == int(g) and s[2] in gs for s in top))
    return float(np.mean(hits))


def score_all(v, df, cache=None) -> pd.DataFrame:
    if cache and cache.exists():
        return pd.read_parquet(cache)
    out = pd.DataFrame([v.score(c) for c in df.claim])[["sup", "con"]]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache)
    return out


def _pred(s, ts, tc):
    return [decide(a, b, ts, tc) for a, b in zip(s.sup, s.con)]


def tune(s, y, grid=np.arange(0.3, 0.96, 0.05)) -> dict:
    y, best = list(y), (-1.0, 0.5, 0.5)
    for ts in grid:
        for tc in grid:
            f = f1_score(y, _pred(s, ts, tc), average="macro", labels=list(LABELS), zero_division=0)
            if f > best[0]:
                best = (f, float(ts), float(tc))
    return {"macro_f1": best[0], "ts": round(best[1], 2), "tc": round(best[2], 2)}


def report(s, y, ts, tc) -> dict:
    y, p = list(y), _pred(s, ts, tc)
    return {"ts": ts, "tc": tc, "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", labels=list(LABELS), zero_division=0),
            "per_class": classification_report(y, p, labels=list(LABELS), output_dict=True,
                                               zero_division=0)}


def main():
    from .retrieval import Retriever
    from .verdict import Verifier

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["retrieval", "sentences", "verdict"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, help="use only the first N claims (smoke test)")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()

    def cut(df):
        return df.head(a.limit) if a.limit else df

    r = Retriever()
    if a.task == "retrieval":
        df = cut(data.claims(a.split))
        out = {m: retrieval(r, df, mode=m) for m in ("bm25", "dense", "hybrid")}
    elif a.task == "sentences":
        df = cut(data.claims(a.split))
        out = {name: sentence_recall(Verifier(r, reranker=rr), df)
               for name, rr in (("bi-encoder", False), ("cross-encoder", True))}
    else:
        v = Verifier(r, reranker=not a.no_rerank)
        tag = f"{'bi' if a.no_rerank else 'rr'}_{a.limit or 'all'}"
        tr, va = cut(data.claims("train")), cut(data.claims(a.split))
        st = score_all(v, tr, ART / f"scores_train_{tag}.parquet")
        sv = score_all(v, va, ART / f"scores_{a.split}_{tag}.parquet")
        best = tune(st, tr.label)
        out = {"tuned_on": "train", "reported_on": a.split, "thresholds": best,
               "default_0.5": report(sv, va.label, 0.5, 0.5),
               "tuned": report(sv, va.label, best["ts"], best["tc"])}
        if not a.limit:
            THRESHOLDS.write_text(json.dumps({"ts": best["ts"], "tc": best["tc"]}, indent=2))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{a.task}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
