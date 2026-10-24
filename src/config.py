"""Central config. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
THRESHOLDS = ROOT / "thresholds.json"
REPORTS = ROOT / "reports"

NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FALLACY_MODEL = os.getenv("FALLACY_MODEL", str(ART / "fallacy"))  # local dir or Hub id
FALLACY_BASE = os.getenv("FALLACY_BASE", "distilroberta-base")
FALLACY_DS = "tasksource/logical-fallacy"
# The fallacy model has no "no fallacy" class, so only show guesses above this confidence.
FALLACY_MIN_PROB = float(os.getenv("FALLACY_MIN_PROB", "0.5"))

SCIFACT = "hf://datasets/allenai/scifact@refs/convert/parquet/{cfg}/{split}/0000.parquet"
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")


def device() -> str:
    """DEVICE env var wins (set to cpu on Hugging Face Spaces); otherwise cuda if present."""
    if os.getenv("DEVICE"):
        return os.environ["DEVICE"]
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
