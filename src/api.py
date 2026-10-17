"""FastAPI service.  uvicorn src.api:app --host 0.0.0.0 --port 8000"""
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, StringConstraints

from .pipeline import get_pipeline

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]


class In(BaseModel):
    text: Text


class Verdict(BaseModel):
    label: str
    confidence: float
    evidence: str
    source: str
    doc_id: int
    sup: float
    con: float


class Fallacy(BaseModel):
    label: str
    prob: float


class Out(BaseModel):
    claim: str
    verdict: Verdict
    fallacies: list[Fallacy]


@asynccontextmanager
async def lifespan(_):
    get_pipeline()  # load models at startup so the first request isn't slow
    yield


app = FastAPI(title="Fact and Logic", version="1.0.0", lifespan=lifespan)


def pipe():
    return get_pipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify", response_model=Out)
def verify(body: In, p=Depends(pipe)):
    return p.run(body.text)
