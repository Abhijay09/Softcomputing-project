"""Evaluate the trained model on the EMNIST test set and save figures for the report."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load, label_chars
from features import extract
from fuzzy_model import FuzzyClassifier
from fuzzy_logic import reliability, triangle, IN_SETS, GAP_SETS, OUT_SETS

os.makedirs("docs/figs", exist_ok=True)
xte, yte = load("test")
F = np.array([extract(i) for i in xte])
model = FuzzyClassifier.load("models/fuzzy_model.npz")
M = model.memberships(F)
order = np.argsort(-M, axis=1)
top1 = model.classes[order[:, 0]]
acc1 = (top1 == yte).mean()
acc3 = np.mean([yte[i] in model.classes[order[i, :3]] for i in range(len(yte))])
strength = M[np.arange(len(M)), order[:, 0]]
gap = strength - M[np.arange(len(M)), order[:, 1]]
rel = np.array([reliability(s, g) for s, g in zip(strength, gap)])
correct = top1 == yte
buckets = {}
for name, lo, hi in [("LOW (<35)", 0, 35), ("MEDIUM (35-65)", 35, 65), ("HIGH (>=65)", 65, 101)]:
    m = (rel >= lo) & (rel < hi)
    buckets[name] = {"samples": int(m.sum()), "accuracy": float(correct[m].mean()) if m.any() else None}
res = {"top1": float(acc1), "top3": float(acc3), "rules": int(len(model.centres)), "reliability_buckets": buckets,
       "strength_pctiles": np.percentile(strength, [5, 25, 50, 75, 95]).round(3).tolist(),
       "gap_pctiles": np.percentile(gap, [5, 25, 50, 75, 95]).round(3).tolist()}
json.dump(res, open("docs/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))

names = label_chars()
n = len(model.classes)
cm = np.zeros((n, n), int)
for t, p in zip(yte, top1): cm[t, p] += 1
fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels([names[c] for c in model.classes], fontsize=6); ax.set_yticklabels([names[c] for c in model.classes], fontsize=6)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix (EMNIST-balanced test, accuracy {acc1:.1%})")
fig.tight_layout(); fig.savefig("docs/figs/confusion.png", dpi=150); plt.close(fig)

per = [(cm[i, i] / cm[i].sum()) for i in range(n)]
fig, ax = plt.subplots(figsize=(9, 3))
ax.bar([names[c] for c in model.classes], per, color="#4c78a8"); ax.set_ylabel("accuracy"); ax.set_title("Per-class accuracy")
fig.tight_layout(); fig.savefig("docs/figs/per_class.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(15, 3))
for ax, (title, sets, rng) in zip(axs[:3], [("Input 1: winning strength", IN_SETS, (0, 1)), ("Input 2: gap to runner-up", GAP_SETS, (0, .5)), ("Output: reliability %", OUT_SETS, (0, 100))]):
    x = np.linspace(*rng, 300)
    for k, v in sets.items(): ax.plot(x, triangle(x, *v), label=k)
    ax.set_title(title); ax.legend(fontsize=7)
axs[3].hist(rel, bins=30, color="#59a14f"); axs[3].set_title("Reliability on test set")
fig.tight_layout(); fig.savefig("docs/figs/fuzzy_sets.png", dpi=150); plt.close(fig)
