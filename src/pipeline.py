"""Ties the fact-check and fallacy modules together."""
import logging
from functools import lru_cache

from .config import FALLACY_MIN_PROB

log = logging.getLogger("fact-and-logic")


class Pipeline:
    def __init__(self, verifier, fallacy=None, min_prob=FALLACY_MIN_PROB):
        self.verifier, self.fallacy, self.min_prob = verifier, fallacy, min_prob

    def run(self, text: str) -> dict:
        guesses = self.fallacy.predict(text) if self.fallacy else []
        return {"claim": text, "verdict": self.verifier.verify(text),
                "fallacies": [f for f in guesses if f["prob"] >= self.min_prob],
                "fallacy_model": self.fallacy is not None}


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    from .fallacy import FallacyDetector
    from .verdict import Verifier

    try:
        fallacy = FallacyDetector()
    except Exception as e:  # the fact-checker still works without the fallacy model
        log.warning("fallacy model unavailable (%s); running without it", e)
        fallacy = None
    return Pipeline(Verifier(), fallacy)
