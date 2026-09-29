from transformers import pipeline
from retrieval import load_corpus, build_index, retrieve

_nli = None


def get_nli():
    global _nli
    if _nli is None:
        _nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _nli


def get_verdict(claim, evidence_text):
    nli = get_nli()
    result = nli(
        evidence_text[:1000],
        candidate_labels=["supports", "contradicts", "not enough info"],
        hypothesis_template=f"This passage {{}} the claim: '{claim}'",
    )
    return result["labels"][0]


if __name__ == "__main__":
    corpus_df = load_corpus()
    bm25 = build_index(corpus_df)
    claim = "0-dimensional biomaterials lack inductive properties."
    top_doc = retrieve(claim, corpus_df, bm25, k=1).iloc[0]["text"]
    print("Claim:", claim)
    print("Evidence:", top_doc[:200], "...")
    print("Verdict:", get_verdict(claim, top_doc))