"""Final numbers for the neural-network classifier and the full camera pipeline."""
import json
import numpy as np
from data import load, label_chars
from classifier import CharClassifier
from segment import recognise, to_text
from benchmark import run

x, y = load("test")
clf = CharClassifier(); names = label_chars()
P = clf.proba(x)
order = np.argsort(-P, axis=1)
res = {"top1": float((order[:, 0] == y).mean()),
       "top3": float(np.mean([y[i] in order[i, :3] for i in range(len(y))]))}
acc, cnt = run(lambda f: to_text(recognise(f, clf, names, context=False)) or "")
res["camera_benchmark_char_accuracy"], res["camera_benchmark_count_correct"] = float(acc), float(cnt)
json.dump(res, open("docs/results_mlp.json", "w"), indent=2)
print(json.dumps(res, indent=2))
