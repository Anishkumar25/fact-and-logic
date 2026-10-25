"""Fine-tune the NLI verdict model on SciFact's training claims.  python -m src.train_nli --epochs 3

Training pairs: the gold rationale plus its claim is entailment (SUPPORT) or contradiction
(CONTRADICT); the top-ranked sentences from retrieved abstracts that are not evidence are neutral.
15% of the train claims are held out: they are used for early stopping and for tuning thresholds,
so the official validation split stays untouched for the final report."""
import argparse
import json

import pandas as pd

from .config import ART, NLI_MODEL, REPORTS

NAMES = {"SUPPORT": "entailment", "CONTRADICT": "contradiction"}


def split_claims(train: pd.DataFrame, dev_frac: float = 0.15, seed: int = 42):
    dev = train.sample(frac=dev_frac, random_state=seed)
    return train.drop(dev.index), dev


def build_pairs(claims: pd.DataFrame, v, neg: int = 1) -> pd.DataFrame:
    """v is a Verifier; only its retriever and sentence reranker are used."""
    sents = dict(zip(v.r.docs.doc_id, v.r.docs.sentences))
    rows = []
    for c, lab, gold in zip(claims.claim, claims.label, claims.gold_sents):
        if lab in NAMES:
            rows += [(" ".join(sents[d][i] for i in idx), c, NAMES[lab])
                     for d, idx in gold.items() if d in sents]
        docs = v.r.search(c, v.k)
        docs = docs[~docs.doc_id.isin(list(gold))]
        if len(docs):
            rows += [(s[3], c, "neutral") for s in v.top_sentences(c, docs)[:neg]]
    return pd.DataFrame(rows, columns=["premise", "hypothesis", "label"])


def train(out=None, base=NLI_MODEL, epochs=3, lr=1e-5, bs=16, neg=1, seed=42):
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from . import data
    from .verdict import Verifier

    out = str(out or ART / "nli")
    v = Verifier()  # supplies retrieval and sentence reranking; its NLI model is never loaded
    tr_claims, dev_claims = split_claims(data.claims("train"))
    tr, dev = build_pairs(tr_claims, v, neg), build_pairs(dev_claims, v, neg)
    print("train pairs", tr.label.value_counts().to_dict(), "| dev pairs", dev.label.value_counts().to_dict())

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base)
    l2i = {n.lower(): i for i, n in model.config.id2label.items()}

    def prep(df):
        d = Dataset.from_dict({"premise": df.premise.tolist(), "hypothesis": df.hypothesis.tolist(),
                               "label": [l2i[x] for x in df.label]})
        return d.map(lambda b: tok(b["premise"], b["hypothesis"], truncation=True, max_length=256),
                     batched=True, remove_columns=["premise", "hypothesis"])

    def metrics(ev):
        y = ev.predictions.argmax(-1)
        return {"accuracy": accuracy_score(ev.label_ids, y),
                "macro_f1": f1_score(ev.label_ids, y, average="macro")}

    args = TrainingArguments(
        output_dir=str(ART / "nli_runs"), num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01,
        warmup_steps=int(0.1 * epochs * -(-len(tr) // bs)),
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs * 2,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, report_to="none",
        fp16=torch.cuda.is_available(), seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=prep(tr), eval_dataset=prep(dev),
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    res = trainer.evaluate()
    trainer.save_model(out)
    tok.save_pretrained(out)
    res.update(train_pairs=len(tr), dev_pairs=len(dev))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "nli_train.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--neg", type=int, default=1, help="neutral pairs per claim")
    a = ap.parse_args()
    print(train(epochs=a.epochs, lr=a.lr, bs=a.bs, neg=a.neg))
