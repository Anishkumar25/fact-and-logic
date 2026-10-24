"""claim -> retrieve k docs -> rerank sentences -> NLI -> SUPPORT / CONTRADICT / NOT_ENOUGH_INFO."""
import json

import numpy as np

from .config import NLI_MODEL, THRESHOLDS, device


def decide(sup: float, con: float, ts: float = 0.5, tc: float = 0.5) -> str:
    if con > sup and con >= tc:
        return "CONTRADICT"
    if sup >= con and sup >= ts:
        return "SUPPORT"
    return "NOT_ENOUGH_INFO"


class Verifier:
    """reranker: True = build the default cross-encoder, False/None = bi-encoder fallback,
    or pass a Reranker instance. Thresholds default to thresholds.json if present."""

    def __init__(self, retriever=None, reranker=True, ts=None, tc=None, k=3, n=5, pair=True):
        from .rerank import Reranker
        from .retrieval import Retriever

        saved = json.loads(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {}
        self.ts = saved.get("ts", 0.5) if ts is None else ts
        self.tc = saved.get("tc", 0.5) if tc is None else tc
        self.k, self.n, self.pair = k, n, pair
        self.r = retriever or Retriever()
        self.rr = Reranker() if reranker is True else (reranker or None)
        self._nli = None

    def load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self._nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval().to(self.dev)
        names = [self._nli.config.id2label[i].lower() for i in range(self._nli.config.num_labels)]
        self.i_ent, self.i_con = names.index("entailment"), names.index("contradiction")

    def top_sentences(self, claim, docs):
        """Returns (doc_id, title, sentence_idx, text) for the n best sentences."""
        sents = [(d.doc_id, d.title, i, s) for d in docs.itertuples() for i, s in enumerate(d.sentences)]
        texts = [s[3] for s in sents]
        if self.rr:
            sc = self.rr.scores(claim, texts)
        else:
            m = self.r.model
            sc = m.encode(texts, normalize_embeddings=True) @ m.encode(claim, normalize_embeddings=True)
        return [sents[i] for i in np.argsort(-sc)[:self.n]]

    def _premises(self, sents):
        prem = [(s[0], s[1], [s[2]], s[3]) for s in sents]
        top = [s for s in sents if s[0] == sents[0][0]][:2]  # pair only within one document
        if self.pair and len(top) == 2:
            top = sorted(top, key=lambda s: s[2])
            prem.append((top[0][0], top[0][1], [s[2] for s in top], " ".join(s[3] for s in top)))
        return prem

    def nli_probs(self, claim, premises):
        import torch

        if self._nli is None:
            self.load()
        enc = self.tok(premises, [claim] * len(premises), return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(self.dev)
        with torch.no_grad():
            return self._nli(**enc).logits.softmax(-1).cpu().numpy()

    def score(self, claim):
        prem = self._premises(self.top_sentences(claim, self.r.search(claim, self.k)))
        p = self.nli_probs(claim, [x[3] for x in prem])
        sup, con = p[:, self.i_ent], p[:, self.i_con]
        j = int(np.argmax(sup if sup.max() >= con.max() else con))
        doc_id, title, ids, text = prem[j]
        return {"sup": float(sup.max()), "con": float(con.max()), "doc_id": int(doc_id),
                "source": title, "sentence_ids": ids, "evidence": text}

    def verify(self, claim):
        s = self.score(claim)
        lab = decide(s["sup"], s["con"], self.ts, self.tc)
        conf = {"SUPPORT": s["sup"], "CONTRADICT": s["con"]}.get(lab, 1 - max(s["sup"], s["con"]))
        return {**s, "label": lab, "confidence": float(conf)}
