"""Evaluation helpers and CLI.  python -m src.evaluate {retrieval,sentences,verdict} [--limit N]

Thresholds are tuned on the SciFact train split (the NLI model never trains on it) and
reported on validation, so the headline number isn't tuned on the data it's scored on."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from . import data
from .config import ART, LABELS, REPORTS
from .verdict import decide, save_thresholds


def retrieval(r, df, ks=(1, 3, 5, 20), mode="hybrid") -> dict:
    df = df[df.gold_docs.map(len) > 0]
    ranks = []
    for c, g in zip(df.claim, df.gold_docs):
        ids = r.search(c, max(ks), mode=mode).doc_id.tolist()
        ranks.append(min((ids.index(d) for d in g if d in ids), default=10**9))
    ranks = np.array(ranks)
    return {f"hit@{k}": float((ranks < k).mean()) for k in ks}


def sentence_recall(v, df) -> float:
    """Share of claims where a gold evidence sentence is among the verifier's top-n sentences."""
    df = df[df.gold_docs.map(len) > 0]
    hits = []
    for c, gs in zip(df.claim, df.gold_sents):
        top = v.top_sentences(c, v.r.search(c, v.k))
        hits.append(any(int(s[0]) in gs and s[2] in gs[int(s[0])] for s in top))
    return float(np.mean(hits))


def score_all(v, df, cache=None) -> pd.DataFrame:
    if cache and cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) == len(df):  # a cache from an older run with different data is ignored
            return cached
    out = pd.DataFrame([v.score(c) for c in df.claim])[["sup", "con"]]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache)
    return out


def _pred(s, ts, tc):
    return [decide(a, b, ts, tc) for a, b in zip(s.sup, s.con)]


def tune(s, y, grid=np.r_[np.arange(0.1, 0.96, 0.05), 0.99]) -> dict:
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
    ap.add_argument("--nli", help="fine-tuned NLI model (folder or Hub id) instead of the zero-shot one")
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
        v = Verifier(r, reranker=not a.no_rerank, nli_model=a.nli)
        tag = f"{'bi' if a.no_rerank else 'rr'}_{'ft' if a.nli else 'zs'}_{a.limit or 'all'}"
        tr, va = cut(data.claims("train")), cut(data.claims(a.split))
        if a.nli:  # tune on claims the fine-tuned model never trained on
            from .train_nli import split_claims
            tr = split_claims(tr)[1]
        st = score_all(v, tr, ART / f"scores_train_{tag}.parquet")
        sv = score_all(v, va, ART / f"scores_{a.split}_{tag}.parquet")
        best = tune(st, tr.label)
        tuned_on = "held-out train claims" if a.nli else "train"
        out = {"tuned_on": tuned_on, "reported_on": a.split, "thresholds": best,
               "default_0.5": report(sv, va.label, 0.5, 0.5),
               "tuned": report(sv, va.label, best["ts"], best["tc"])}
        if not a.limit:
            save_thresholds(best["ts"], best["tc"], a.nli)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{a.task}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
