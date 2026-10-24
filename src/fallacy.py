"""Logical-fallacy classifier. FallacyDetector for inference, train() for fine-tuning.
python -m src.fallacy --epochs 4 [--push user/repo]"""
import argparse
import json

import numpy as np

from .config import ART, FALLACY_BASE, FALLACY_DS, FALLACY_MODEL, REPORTS, device

TEXT, LABEL = "source_article", "logical_fallacies"


class FallacyDetector:
    def __init__(self, path: str = FALLACY_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path).eval().to(self.dev)

    def predict(self, text: str, top: int = 3) -> list[dict]:
        import torch

        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=256).to(self.dev)
        with torch.no_grad():
            p = self.m(**enc).logits.softmax(-1)[0].cpu().numpy()
        i2l = self.m.config.id2label
        return [{"label": i2l[int(i)], "prob": float(p[i])} for i in np.argsort(-p)[:top]]


def _labels(split):
    f = split.features[LABEL]
    return [f.int2str(x) for x in split[LABEL]] if hasattr(f, "int2str") else [str(x) for x in split[LABEL]]


def train(out=None, base=FALLACY_BASE, epochs=4, lr=3e-5, bs=16, push=None):
    import torch
    from datasets import Dataset, load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    out = str(out or ART / "fallacy")
    ds = load_dataset(FALLACY_DS)
    if TEXT not in ds["train"].column_names:
        raise KeyError(f"expected column {TEXT!r}, found {ds['train'].column_names}")
    names = sorted(set(_labels(ds["train"])))
    l2i = {n: i for i, n in enumerate(names)}
    tok = AutoTokenizer.from_pretrained(base)

    def prep(split):
        d = Dataset.from_dict({"text": list(split[TEXT]), "label": [l2i[n] for n in _labels(split)]})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=256),
                     batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = prep(ds["train"]), prep(ds.get("validation", ds.get("dev"))), prep(ds["test"])
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(names), id2label=dict(enumerate(names)), label2id=l2i)

    def metrics(ev):
        y = ev.predictions.argmax(-1)
        return {"accuracy": accuracy_score(ev.label_ids, y),
                "macro_f1": f1_score(ev.label_ids, y, average="macro")}

    args = TrainingArguments(
        output_dir=out, num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01,
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs * 2,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, report_to="none",
        fp16=torch.cuda.is_available(), seed=42)
    tr = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                 data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    tr.train()
    res = tr.evaluate(test_ds, metric_key_prefix="test")
    tr.save_model(out)
    tok.save_pretrained(out)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "fallacy.json").write_text(json.dumps(res, indent=2))
    if push:
        tr.model.push_to_hub(push)
        tok.push_to_hub(push)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--push", help="Hub repo id, e.g. user/fact-and-logic-fallacy")
    a = ap.parse_args()
    print(train(epochs=a.epochs, lr=a.lr, bs=a.bs, push=a.push))
