---
title: Fact and Logic
sdk: gradio
app_file: app.py
---
 
# Fact and Logic
 
[![CI](https://github.com/Anishkumar25/fact-and-logic/actions/workflows/ci.yml/badge.svg)](https://github.com/Anishkumar25/fact-and-logic/actions/workflows/ci.yml)
 
Checks a claim against scientific abstracts and flags logical fallacies in the way it is argued.
 
[Live demo](https://huggingface.co/spaces/Anish-25/fact-and-logic) · [Fallacy model](https://huggingface.co/Anish-25/fact-and-logic)
 
## What it does
 
Give it a sentence. It returns:
 
- a verdict (supported, refuted, or not enough information), the abstract sentence the verdict rests on, and the paper it came from
- the logical fallacy the text most resembles, shown only when the model is confident enough
Evidence comes from SciFact, a corpus of about 5,000 biomedical abstracts. Claims outside that field will mostly come back as "not enough information", which is the correct answer for this corpus.
 
## How it works
 
```
claim
  |-- BM25 + MiniLM embeddings, rank-fused ------> top 3 abstracts
  |-- cross-encoder sentence reranker -----------> 5 best sentences (plus the top pair)
  |-- DeBERTa NLI + tuned thresholds ------------> SUPPORT / CONTRADICT / NOT_ENOUGH_INFO
  '-- DistilRoBERTa fallacy classifier ----------> fallacy type, if above the confidence cutoff
```
 
| Stage | Model |
|---|---|
| Retrieval | `BM25` and `sentence-transformers/all-MiniLM-L6-v2`, combined with reciprocal rank fusion |
| Sentence selection | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Verdict | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, used zero-shot |
| Fallacy | `distilroberta-base` fine-tuned on `tasksource/logical-fallacy` (13 classes) |
 
## Results
 
Evaluated on the SciFact validation split (300 claims, 188 of them with evidence).
 
| Retrieval | hit@1 | hit@3 | hit@5 | hit@20 |
|---|---|---|---|---|
| BM25 | 70.2% | 84.0% | 90.4% | 95.7% |
| Dense (MiniLM) | 61.7% | 82.4% | 87.8% | 96.3% |
| Hybrid | 72.3% | 88.8% | 93.1% | 96.8% |
 
| Evidence sentence selection | Gold sentence in top 5 |
|---|---|
| Bi-encoder | 71.3% |
| Cross-encoder reranker | 77.1% |
 
| Verdict | Macro-F1 | Accuracy |
|---|---|---|
| Thresholds at 0.5 | 0.551 | 0.553 |
| Thresholds tuned on the train split | 0.585 | 0.590 |
 
Thresholds are tuned on the train split and reported on validation, so the headline number isn't tuned on the data it's scored on. SUPPORT claims are the weak spot: 59 of 124 are recovered, and most of the rest are called "not enough information".
 
| Fallacy classifier | Accuracy | Macro-F1 |
|---|---|---|
| Test split, 511 examples, 13 classes | 0.44 | 0.40 |
 
Random guessing would score about 0.08 and always predicting the most common class about 0.17. It does best on ad hominem (F1 0.71) and worst on equivocation and fallacy of relevance.
 
## Quick start
 
Python 3.10 or newer.
 
```
git clone https://github.com/Anishkumar25/fact-and-logic.git
cd fact-and-logic
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-train.txt
python -m src.retrieval
python app.py
```
 
On macOS or Linux, activate with `source .venv/bin/activate` instead. `src.retrieval` builds the search index once, which takes a few minutes on a CPU. Without a trained fallacy model the app still runs and skips that panel.
 
## Reproduce the results
 
```
python -m src.evaluate retrieval
python -m src.evaluate sentences
python -m src.evaluate verdict
python -m src.fallacy --epochs 4
```
 
Each command writes its numbers to `reports/`, and `evaluate verdict` also writes `thresholds.json`, which the app reads at startup. Add `--limit 50` to any evaluate command for a quick check. The same steps are in `notebooks/` and run on Google Colab's free tier.
 
## API
 
```
uvicorn src.api:app --port 8000
```
 
```
curl -X POST http://localhost:8000/verify -H "Content-Type: application/json" -d "{\"text\": \"Vitamin D deficiency is associated with an increased risk of tuberculosis.\"}"
```
 
The response has this shape (values are illustrative):
 
```json
{
  "claim": "...",
  "verdict": {"label": "SUPPORT", "confidence": 0.97, "evidence": "...", "source": "...", "doc_id": 123, "sup": 0.97, "con": 0.01},
  "fallacies": [{"label": "ad hominem", "prob": 0.81}]
}
```
 
`GET /health` is the liveness check and `/docs` has the interactive reference.
 
## Deployment
 
The demo runs on a free Hugging Face Space (Gradio SDK, ZeroGPU hardware). `app.py` keeps inference on the CPU when it detects a Space, loads all models at startup, and registers a no-op `@spaces.GPU` function because ZeroGPU won't start a Space without one. Set the Space variable `FALLACY_MODEL` to the Hub id of the fallacy model.
 
For any host that runs containers, the `Dockerfile` serves the API on port 8000 with the model weights and search index baked into the image.
 
## Configuration
 
| Variable | Default | Purpose |
|---|---|---|
| `NLI_MODEL` | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | verdict model |
| `RERANK_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | sentence reranker |
| `EMB_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | retrieval embeddings |
| `FALLACY_MODEL` | artifacts/fallacy | local folder or Hub id |
| `FALLACY_MIN_PROB` | 0.5 | fallacy guesses below this confidence are hidden |
| `DEVICE` | auto | `cpu` or `cuda` |
| `ARTIFACTS_DIR` | artifacts | search index, cached scores, trained models |
 
## Repository layout
 
```
app.py             Gradio demo and Space entry point
src/api.py         FastAPI service
src/pipeline.py    verdict and fallacy detection together
src/retrieval.py   BM25, dense retrieval and rank fusion
src/rerank.py      cross-encoder sentence reranker
src/verdict.py     NLI scoring and thresholds
src/fallacy.py     fallacy classifier: training and inference
src/evaluate.py    metrics, threshold tuning and CLI
src/data.py        SciFact loaders
notebooks/         data, retrieval, verdict, fallacy and end-to-end notebooks
tests/             unit and API tests, none of which download models
reports/           metrics written by the evaluate commands
thresholds.json    verdict thresholds tuned on the train split
```
 
CI runs `ruff check .` and `pytest` on every push.
 
## Limitations
 
- The corpus is small and biomedical, so this is a research prototype and not a general fact-checker. It shouldn't be used to settle real disputes.
- The verdict model is used zero-shot and reaches about 59% accuracy on three classes.
- The fallacy dataset has no "no fallacy" class, so the model always names some fallacy. The confidence cutoff hides low-confidence guesses, but it can't tell a sound argument from a fallacious one.
- Fallacy labels are partly subjective, and several of the broader categories overlap.
- The API has no authentication or rate limiting. Add both before exposing it publicly.
## Roadmap
 
- Fine-tune the NLI model on SciFact's training claims to lift SUPPORT recall.
- Calibrate the fallacy confidence cutoff against ordinary non-fallacious text.
- Try a biomedical reranker such as `ncbi/MedCPT-Cross-Encoder`.
## Data and models
 
- [SciFact](https://huggingface.co/datasets/allenai/scifact) for claims and evidence abstracts
- [tasksource/logical-fallacy](https://huggingface.co/datasets/tasksource/logical-fallacy) for fallacy labels
- Pretrained models listed under How it works, all from the Hugging Face Hub
 
