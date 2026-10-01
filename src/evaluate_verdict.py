import numpy as np
from sklearn.metrics import classification_report, f1_score

from src.data import load_claims
from src.verdict import Verifier

if __name__ == "__main__":
    v = Verifier()
    dev = load_claims("train").sample(300, random_state=0)  # only used to pick tau
    test = load_claims("validation")                         # reported numbers
    dev_s = [v.score(c) for c in dev["claim"]]
    test_s = [v.score(c) for c in test["claim"]]

    taus = np.arange(0.3, 0.96, 0.05)
    f1s = [f1_score(dev["label"], [v.label(s, t) for s in dev_s], average="macro") for t in taus]
    tau = float(taus[int(np.argmax(f1s))])
    print(f"best tau = {tau:.2f} (dev macro-F1 {max(f1s):.3f})\n")
    print(classification_report(test["label"], [v.label(s, tau) for s in test_s], digits=3))
