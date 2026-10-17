"""Ties the fact-check and fallacy modules together."""
import logging
from functools import lru_cache

log = logging.getLogger("fact-and-logic")


class Pipeline:
    def __init__(self, verifier, fallacy=None):
        self.verifier, self.fallacy = verifier, fallacy

    def run(self, text: str) -> dict:
        return {"claim": text, "verdict": self.verifier.verify(text),
                "fallacies": self.fallacy.predict(text) if self.fallacy else []}


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
