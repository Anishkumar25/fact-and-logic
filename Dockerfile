FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models ARTIFACTS_DIR=/models/artifacts
WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY src ./src
COPY thresholds.js[o]n ./

# Bake weights and the search index into the image so cold starts are fast.
RUN python -m src.retrieval \
 && python -c "from src.rerank import Reranker; Reranker()" \
 && python -c "from src.verdict import Verifier; v = Verifier(reranker=False); v._load()"

RUN useradd -m app && chown -R app /app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
