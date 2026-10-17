from fastapi.testclient import TestClient

from src.api import app, pipe


class Stub:
    def run(self, text):
        return {"claim": text,
                "verdict": {"label": "SUPPORT", "confidence": 0.9, "evidence": "e", "source": "s",
                            "doc_id": 1, "sup": 0.9, "con": 0.05, "sentence_ids": [0]},
                "fallacies": [{"label": "ad hominem", "prob": 0.7}]}


app.dependency_overrides[pipe] = Stub
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_verify_returns_verdict_and_fallacies():
    r = client.post("/verify", json={"text": "  Aspirin thins blood.  "})
    assert r.status_code == 200
    body = r.json()
    assert body["claim"] == "Aspirin thins blood."
    assert body["verdict"]["label"] == "SUPPORT"
    assert body["fallacies"][0]["label"] == "ad hominem"


def test_verify_rejects_empty_and_oversized_input():
    assert client.post("/verify", json={"text": "  "}).status_code == 422
    assert client.post("/verify", json={"text": "x" * 1001}).status_code == 422
