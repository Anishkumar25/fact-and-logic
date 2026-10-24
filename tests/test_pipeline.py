from src.pipeline import Pipeline


class V:
    def verify(self, text):
        return {"label": "SUPPORT"}


class F:
    def predict(self, text):
        return [{"label": "ad hominem", "prob": 0.8}, {"label": "false dilemma", "prob": 0.2}]


def test_low_confidence_fallacies_are_hidden():
    out = Pipeline(V(), F(), min_prob=0.5).run("x")
    assert [f["label"] for f in out["fallacies"]] == ["ad hominem"]
    assert out["fallacy_model"] is True


def test_works_without_fallacy_model():
    out = Pipeline(V(), None).run("x")
    assert out["fallacies"] == [] and out["fallacy_model"] is False
