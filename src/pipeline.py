import re
 
from src.verdict import Verifier
 
K, N = 3, 5  # deployed config: change after comparing results/verdict_k*_n*.txt
 
 
def thresholds(path=f"results/verdict_k{K}_n{N}.txt"):
    """Read the tuned SUPPORT / CONTRADICT thresholds saved by evaluate_verdict."""
    try:
        return tuple(map(float, re.findall(r"[\d.]+", open(path).readline())[:2]))
    except (OSError, ValueError):
        return 0.5, 0.5
 
 
class FactChecker:
    def __init__(self):
        ts, tc = thresholds()
        self.v = Verifier(ts=ts, tc=tc, k=K, n=N)
 
    def check(self, claim):
        return self.v.verify(claim)
 