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

SCIFACT = "hf://datasets/allenai/scifact@refs/convert/parquet/{cfg}/{split}/0000.parquet"
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
