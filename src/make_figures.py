"""Figures for the report: pipeline illustration on a sample image."""
import sys, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import label_chars
from fuzzy_model import FuzzyClassifier
from features import normalize_char
from segment import binarize, recognise, to_text

names = label_chars()
model = FuzzyClassifier.load("models/fuzzy_model.npz")
frame = cv2.imread("docs/figs/sample_input.png")
ink = binarize(frame)
lines = recognise(frame, model, names)
chars = [c for l in lines for c in l]
col = {"HIGH": "#3ddc97", "MEDIUM": "#ffb454", "LOW": "#ff5c7a"}

fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(3, 8, height_ratios=[3, 3, 1.4])
a = fig.add_subplot(gs[0, 0:4]); a.imshow(frame[..., ::-1]); a.set_title("1. Camera frame"); a.axis("off")
b = fig.add_subplot(gs[0, 4:8]); b.imshow(ink, cmap="gray"); b.set_title("2. Adaptive threshold (ink = white)"); b.axis("off")
c = fig.add_subplot(gs[1, 0:4]); c.imshow(frame[..., ::-1]); c.set_title("3. Segmentation + fuzzy decision (colour = reliability)"); c.axis("off")
for ch in chars:
    x, y, w, h = ch["box"]
    c.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=col[ch["level"]], lw=2))
    c.text(x, y - 5, ch["char"], color=col[ch["level"]], fontsize=11, weight="bold")
c.set_xlim(0, 640); c.set_ylim(300, 150)
for i, ch in enumerate(chars[:8]):
    ax = fig.add_subplot(gs[1, 4 + i // 2] if False else gs[1, 4 + min(i // 2, 3)]) if False else None
sub = fig.add_subplot(gs[1, 4:8]); sub.axis("off"); sub.set_title("4. Normalised 28x28 characters")
strip = np.hstack([normalize_char(ink[y:y + h, x:x + w]) for (x, y, w, h) in [c_["box"] for c_ in chars]])
sub.imshow(strip, cmap="gray"); 
t = fig.add_subplot(gs[2, :]); t.axis("off")
t.text(0, 0.5, "5. Output text:  " + to_text(lines) + "      reliability per char (%): " + " ".join(f"{c_['reliability']:.0f}" for c_ in chars),
       fontsize=13, family="monospace", va="center")
fig.tight_layout(); fig.savefig("docs/figs/pipeline.png", dpi=140); print("ok")

# block diagram
fig, ax = plt.subplots(figsize=(13, 2)); ax.axis("off")
steps = ["Camera\nframe", "Pre-process\n(gray, blur,\nadaptive\nthreshold)", "Segment\n(connected\ncomponents)", "Normalise\n28x28", "Features\n(177 values)", "Fuzzy rules\n188 rules\nmembership", "Defuzzify\n(max)", "Mamdani\nreliability", "Text +\nconfidence"]
for i, s in enumerate(steps):
    ax.add_patch(plt.Rectangle((i * 1.4, 0), 1.2, 1, fc="#e8e4ff", ec="#7c5cff", lw=1.5))
    ax.text(i * 1.4 + .6, .5, s, ha="center", va="center", fontsize=9)
    if i: ax.annotate("", (i * 1.4, .5), (i * 1.4 - .2, .5), arrowprops=dict(arrowstyle="->", color="#7c5cff"))
ax.set_xlim(-.1, len(steps) * 1.4); ax.set_ylim(-.1, 1.1)
fig.savefig("docs/figs/blocks.png", dpi=140, bbox_inches="tight")
