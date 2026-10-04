import itertools
import os
import pickle
import sys
 
import numpy as np
from sklearn.metrics import classification_report, f1_score
 
from src.data import load_claims
from src.verdict import Verifier
 
K, N = (int(a) for a in sys.argv[1:3]) if len(sys.argv) >= 3 else (3, 5)  # usage: python -m src.evaluate_verdict K N
CACHE = f"data/verdict_scores_k{K}_n{N}.pkl"
OUT = f"results/verdict_k{K}_n{N}.txt"
 
 
def cached(fn):
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    out = fn()
    pickle.dump(out, open(CACHE, "wb"))
    return out
 
 
if __name__ == "__main__":
    v = Verifier(k=K, n=N)
    dev = load_claims("train").sample(300, random_state=0)  # only used to pick thresholds
    test = load_claims("validation")                         # reported numbers
    dev_s, test_s = cached(lambda: ([v.score(c) for c in dev["claim"]],
                                    [v.score(c) for c in test["claim"]]))
 
    grid = np.arange(0.3, 0.96, 0.1)
    f1 = {(ts, tc): f1_score(dev["label"], [v.label(s, ts, tc) for s in dev_s], average="macro")
          for ts, tc in itertools.product(grid, grid)}
    ts, tc = max(f1, key=f1.get)
    report = classification_report(test["label"], [v.label(s, ts, tc) for s in test_s], digits=3)
    head = f"thresholds: SUPPORT>={ts:.2f}, CONTRADICT>={tc:.2f} (dev macro-F1 {f1[(ts, tc)]:.3f}) k={K} n={N}"
    print(head + "\n\n" + report)
    os.makedirs("results", exist_ok=True)
    open(OUT, "w").write(head + "\n\n" + report)
 