"""Train the fuzzy classifier on EMNIST-balanced and report test accuracy."""
import sys, time, os
import numpy as np
from data import load
from features import extract
from fuzzy_model import FuzzyClassifier

rules = int(sys.argv[1]) if len(sys.argv) > 1 else 4
t = time.time()
xtr, ytr = load("train"); xte, yte = load("test")
Ftr = np.array([extract(i) for i in xtr]); Fte = np.array([extract(i) for i in xte])
model = FuzzyClassifier(rules).fit(Ftr, ytr)
pred = model.predict(Fte)
acc = (pred == yte).mean()
print(f"rules/class={rules}  rules total={len(model.centres)}  test accuracy={acc:.4f}  time={time.time()-t:.0f}s")
os.makedirs("models", exist_ok=True)
model.save("models/fuzzy_model.npz")
np.save("models/test_pred.npy", pred)
