"""Does the fuzzy context rule help? Build 'words' of EMNIST test characters (all digits or all letters)
and compare per-character accuracy with and without the rule."""
import numpy as np
from data import load
from classifier import CharClassifier
from segment import apply_context

x, y = load("test")
clf = CharClassifier()
P = clf.proba(x)
rng = np.random.default_rng(1)
digits, letters = np.where(y < 10)[0], np.where(y >= 10)[0]
for name, pool in (("digit words", digits), ("letter words", letters)):
    base = ctx = tot = 0
    for _ in range(3000):
        ids = rng.choice(pool, rng.integers(2, 7), replace=False)
        Pw = P[ids]
        base += (Pw.argmax(1) == y[ids]).sum(); ctx += (apply_context(Pw).argmax(1) == y[ids]).sum(); tot += len(ids)
    print(f"{name}: without context {base / tot:.1%}   with context {ctx / tot:.1%}")
