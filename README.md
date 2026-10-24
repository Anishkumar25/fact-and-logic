---
title: Fact and Logic
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Fact and Logic

Paste a claim and get two things back: whether published scientific evidence supports or contradicts it, and whether the way it's argued commits a logical fallacy.

It's a college ML mini project, built to be run, tested and deployed like a real service rather than left as a notebook.

## How it works

```
claim
  |-- hybrid retrieval (BM25 + MiniLM, rank-fused) -> top 3 abstracts
  |-- cross-encoder picks the 5 best sentences
  |-- DeBERTa NLI scores each sentence (plus the top pair) against the claim
  |-- tuned thresholds -> SUPPORT / CONTRADICT / NOT_ENOUGH_INFO
  '-- fine-tuned DistilRoBERTa -> likely fallacy type
```

Evidence comes from the SciFact corpus (about 5k biomedical abstracts). A claim outside that area will mostly come back as NOT_ENOUGH_INFO, and that's the correct behaviour, not a bug.

## Results

Measured so far on the SciFact validation split:

| Stage | Result |
|---|---|
| Hybrid retrieval | hit@5 92.0%, hit@20 97.9% |
| Zero-shot NLI, thresholds 0.5 | macro-F1 0.546 |
| Zero-shot NLI, tuned thresholds | macro-F1 0.581 (tuned on the same split, so slightly optimistic) |

Still to measure with this repo's code. Run the commands below and paste the numbers from `reports/` here:

| Stage | Command |
|---|---|
| Reranker vs bi-encoder sentence recall | `python -m src.evaluate sentences` |
| Verdict with reranker, thresholds tuned on train | `python -m src.evaluate verdict` |
| Fallacy classifier | `python -m src.fallacy --epochs 4` |

## Run it locally

Windows, from the repo folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-train.txt
python -m src.retrieval
python -m src.evaluate verdict
python app.py
```

`src.retrieval` builds the search index once (a few minutes on CPU). `src.evaluate verdict` scores the train and validation claims, tunes the thresholds, and writes `thresholds.json`, which the app picks up automatically. Add `--limit 50` to any evaluate command for a quick smoke test.

Train the fallacy model on a free Colab GPU with `notebooks/04_fallacy.ipynb`, or locally with `python -m src.fallacy --epochs 4`. Without it the app still works and just skips the fallacy panel.

## Deploy for free

**Hugging Face Spaces (recommended).** The front matter at the top of this file is what Spaces reads.

1. Create a Space, pick the Gradio SDK and the free CPU hardware.
2. Push the repo to it: `git remote add space https://huggingface.co/spaces/<you>/fact-and-logic`, then `git push space main`.
3. Push your trained fallacy model to the Hub (`python -m src.fallacy --push <you>/fact-and-logic-fallacy`) and set a Space variable `FALLACY_MODEL` to that id.
4. The first boot builds the search index, which takes a few minutes. Free Spaces sleep when idle, so expect a slow first request after a quiet spell.

**Docker / API.** The same pipeline is available as a REST service:

```
docker build -t fact-and-logic .
docker run -p 8000:8000 -e FALLACY_MODEL=<you>/fact-and-logic-fallacy fact-and-logic
```

Then `POST /verify` with `{"text": "..."}`. Interactive docs are at `/docs` and `/health` is the liveness check. Any container host with a free tier works (Render, Fly.io, Cloud Run).

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `NLI_MODEL` | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | verdict model |
| `RERANK_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | sentence reranker; try `ncbi/MedCPT-Cross-Encoder` for biomedical text |
| `EMB_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | retrieval embeddings |
| `FALLACY_MODEL` | artifacts/fallacy | local folder or Hub id |
| `FALLACY_MIN_PROB` | 0.5 | fallacy guesses below this confidence are hidden |
| `ARTIFACTS_DIR` | artifacts | index, scores and trained models |

## Project layout

```
app.py             Gradio demo (Spaces entry point)
src/api.py         FastAPI service
src/pipeline.py    verdict + fallacy together
src/retrieval.py   BM25 + dense + rank fusion
src/rerank.py      cross-encoder sentence reranker
src/verdict.py     NLI verdict and thresholds
src/fallacy.py     fallacy classifier (train + inference)
src/evaluate.py    metrics, threshold tuning, CLI
src/data.py        SciFact loaders
notebooks/         01 data, 02 retrieval, 03 verdict, 04 fallacy, 05 end to end
tests/             unit and API tests (no model downloads needed)
thresholds.json    written by `evaluate verdict`
```

Tests and lint run on every push through GitHub Actions: `pip install -r requirements-dev.txt`, then `ruff check .` and `pytest`.

## Limits worth knowing

- The corpus is small and biomedical. This is a research prototype, not a general fact-checker, and it shouldn't be used to settle real disputes.
- The NLI model is used zero-shot. Fine-tuning it on SciFact's training claims is the most obvious next improvement.
- The fallacy dataset is small, and fallacy labels are subjective, so treat the fallacy output as a hint.
- There's no authentication or rate limiting on the API. Add both before exposing it publicly.
