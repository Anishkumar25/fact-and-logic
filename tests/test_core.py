import numpy as np
import pandas as pd

from src.data import parse_claims
from src.evaluate import tune
from src.retrieval import rrf
from src.verdict import decide


def test_rrf_prefers_docs_ranked_well_by_both():
    assert rrf([[1, 2, 3], [3, 1, 4]])[0] == 1


def test_rrf_keeps_docs_seen_by_one_ranker():
    assert set(rrf([[1, 2], [3]])) == {1, 2, 3}


def test_decide_thresholds():
    assert decide(0.9, 0.05) == "SUPPORT"
    assert decide(0.05, 0.9) == "CONTRADICT"
    assert decide(0.4, 0.3) == "NOT_ENOUGH_INFO"
    assert decide(0.6, 0.1, ts=0.7) == "NOT_ENOUGH_INFO"


def test_parse_claims_handles_missing_evidence():
    raw = pd.DataFrame({
        "id": [1, 2], "claim": ["a", "b"],
        "evidence_doc_id": ["123", ""],
        "evidence_label": ["support", ""],
        "evidence_sentences": [np.array([0, 2]), np.array([])],
    })
    out = parse_claims(raw)
    assert out.label.tolist() == ["SUPPORT", "NOT_ENOUGH_INFO"]
    assert out.gold_doc.iloc[0] == 123 and np.isnan(out.gold_doc.iloc[1])
    assert out.gold_sents.iloc[0] == [0, 2]


def test_tune_finds_separating_thresholds():
    s = pd.DataFrame({"sup": [0.9, 0.1, 0.1, 0.2], "con": [0.05, 0.85, 0.1, 0.2]})
    y = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"]
    assert tune(s, y)["macro_f1"] == 1.0
