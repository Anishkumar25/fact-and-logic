import pandas as pd

from src import verdict
from src.train_nli import build_pairs, split_claims
from src.verdict import load_thresholds, save_thresholds


class R:
    docs = pd.DataFrame({"doc_id": [1, 2], "title": ["a", "b"],
                         "sentences": [["s0", "s1", "s2"], ["t0", "t1"]]})

    def search(self, claim, k):
        return self.docs


class V:
    r, k = R(), 3

    def top_sentences(self, claim, docs):
        return [(int(d.doc_id), d.title, 0, d.sentences[0]) for d in docs.itertuples()]


def test_split_is_disjoint_and_reproducible():
    df = pd.DataFrame({"claim": [str(i) for i in range(100)]})
    tr, dev = split_claims(df)
    assert len(dev) == 15 and not set(tr.index) & set(dev.index)
    assert split_claims(df)[1].index.tolist() == dev.index.tolist()


def test_build_pairs_uses_gold_rationale_and_non_evidence_negatives():
    claims = pd.DataFrame({"claim": ["c1", "c2"], "label": ["SUPPORT", "NOT_ENOUGH_INFO"],
                           "gold_sents": [{1: [0, 2]}, {}]})
    rows = build_pairs(claims, V(), neg=1).values.tolist()
    assert ["s0 s2", "c1", "entailment"] in rows
    assert ["t0", "c1", "neutral"] in rows and ["s0", "c1", "neutral"] not in rows
    assert sum(r[1] == "c2" and r[2] == "neutral" for r in rows) == 1


def test_thresholds_are_stored_per_model(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict, "THRESHOLDS", tmp_path / "t.json")
    save_thresholds(0.3, 0.9)
    save_thresholds(0.5, 0.6, "me/nli")
    assert load_thresholds() == {"ts": 0.3, "tc": 0.9}
    assert load_thresholds("me/nli") == {"ts": 0.5, "tc": 0.6}
    assert load_thresholds("other") == {"ts": 0.3, "tc": 0.9}
