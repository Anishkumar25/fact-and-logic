import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.retrieval import Retriever

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class Verifier:
    """claim -> retrieve docs -> pick best sentences -> NLI -> SUPPORT / CONTRADICT / NOT_ENOUGH_INFO."""

    def __init__(self, retriever=None, tau=0.5):
        self.r = retriever or Retriever()
        self.tau = tau
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.nli = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
        names = [self.nli.config.id2label[i].lower() for i in range(self.nli.config.num_labels)]
        self.i_ent, self.i_con = names.index("entailment"), names.index("contradiction")

    def top_sentences(self, claim, docs, n=5):
        sents = [(d.doc_id, d.title, s) for d in docs.itertuples() for s in d.sentences]
        e = self.r.model.encode([s[2] for s in sents], normalize_embeddings=True)
        q = self.r.model.encode(claim, normalize_embeddings=True)
        return [sents[i] for i in np.argsort(-(e @ q))[:n]]

    @torch.no_grad()
    def nli_probs(self, claim, sents):
        enc = self.tok([s[2] for s in sents], [claim] * len(sents), return_tensors="pt",
                       padding=True, truncation=True, max_length=256)
        return self.nli(**enc).logits.softmax(-1).numpy()

    def score(self, claim, k=3, n=5):
        sents = self.top_sentences(claim, self.r.search(claim, k), n)
        p = self.nli_probs(claim, sents)
        sup, con = p[:, self.i_ent], p[:, self.i_con]
        j = int(np.argmax(sup if sup.max() >= con.max() else con))
        return {"sup": float(sup.max()), "con": float(con.max()),
                "doc_id": sents[j][0], "source": sents[j][1], "evidence": sents[j][2]}

    def label(self, s, tau=None):
        tau = self.tau if tau is None else tau
        if max(s["sup"], s["con"]) < tau:
            return "NOT_ENOUGH_INFO"
        return "SUPPORT" if s["sup"] >= s["con"] else "CONTRADICT"

    def verify(self, claim):
        s = self.score(claim)
        return {**s, "label": self.label(s)}


if __name__ == "__main__":
    v = Verifier()
    for c in ["0-dimensional biomaterials lack inductive properties.",
              "1 in 5 million in UK have abnormal PrP positivity."]:
        print(c, "->", v.verify(c))
