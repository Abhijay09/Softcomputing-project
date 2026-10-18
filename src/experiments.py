"""Small experiments for the report: sample images, effect of rules per class, effect of features."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load, label_chars
from features import extract, zone_features
from fuzzy_model import FuzzyClassifier

xtr, ytr = load("train"); xte, yte = load("test"); names = label_chars()

# 1. one row of samples for a few classes
fig, axs = plt.subplots(6, 12, figsize=(10, 5.2))
for r, cls in enumerate([3, 10, 17, 24, 36, 41]):
    idx = np.where(ytr == cls)[0][:12]
    for c, i in enumerate(idx):
        axs[r, c].imshow(xtr[i], cmap="gray"); axs[r, c].axis("off")
    axs[r, 0].set_title(names[cls], loc="left", fontsize=10)
fig.suptitle("EMNIST samples: the same character written by different people")
fig.tight_layout(); fig.savefig("docs/figs/samples.png", dpi=130); plt.close(fig)

Ftr = np.array([extract(i) for i in xtr]); Fte = np.array([extract(i) for i in xte])

# 2. rules per class
res = {"rules": {}, "features": {}}
for k in (1, 2, 4, 8):
    m = FuzzyClassifier(k).fit(Ftr, ytr)
    res["rules"][k] = float((m.predict(Fte) == yte).mean()); print("rules", k, res["rules"][k])

# 3. feature sets (4 rules/class): zones only (first 49), directions only (last 128), both
for name, sl in {"ink zones only (49)": slice(0, 49), "stroke directions only (128)": slice(49, 177), "both (177)": slice(0, 177)}.items():
    m = FuzzyClassifier(4).fit(Ftr[:, sl], ytr)
    res["features"][name] = float((m.predict(Fte[:, sl]) == yte).mean()); print(name, res["features"][name])
json.dump(res, open("docs/experiments.json", "w"), indent=2)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
axs[0].plot(list(res["rules"].keys()), [v * 100 for v in res["rules"].values()], "o-"); axs[0].set_xlabel("fuzzy rules per class"); axs[0].set_ylabel("test accuracy %"); axs[0].set_title("Effect of number of rules")
axs[1].barh(list(res["features"].keys()), [v * 100 for v in res["features"].values()], color="#7c5cff"); axs[1].set_xlabel("test accuracy %"); axs[1].set_title("Effect of feature type")
fig.tight_layout(); fig.savefig("docs/figs/experiments.png", dpi=130)

# 4. what the rules look like: ink-zone part of the 4 rule centres for the letter 'A' (class 10)
m = FuzzyClassifier.load("models/fuzzy_model.npz")
cls = 10; ci = list(m.classes).index(cls)
fig, axs = plt.subplots(1, 4, figsize=(8, 2.4))
for a, r in zip(axs, np.where(m.rule_class == ci)[0]):
    a.imshow(m.centres[r][:49].reshape(7, 7), cmap="magma"); a.axis("off"); a.set_title(f"rule {list(np.where(m.rule_class == ci)[0]).index(r) + 1}")
fig.suptitle("The four fuzzy rules for 'A' (ink-zone part of the rule centres)")
fig.tight_layout(); fig.savefig("docs/figs/rules_A.png", dpi=130)
