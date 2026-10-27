"""Train the MLP on EMNIST-balanced (handwriting) + printed letters from system fonts, both augmented."""
import sys, time
import numpy as np
import cv2
from data import load
from mlp import MLP, Adam
from augment import augment_batch
from features import normalize_char
from fonts import font_files, glyphs, render

HELD_OUT = ("Liberation", "NotoSans-Bold")          # fonts kept out of training so the printed benchmark is fair
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PRINTED_PER_EPOCH = 25000


def glyph_bank():
    """Pre-render every glyph in every training font, in a normal, a bold and a light version."""
    bank = []
    for f in font_files():
        if f.split("/")[-1].startswith(HELD_OUT):
            continue
        for k, ch in glyphs():
            g = render(ch, f)
            if g is None: continue
            light = cv2.erode(g, np.ones((3, 3), np.uint8))
            if (light > 0).sum() < 0.3 * (g > 0).sum(): light = g                       # very thin fonts would vanish
            bank.append((k, [g, cv2.dilate(g, np.ones((5, 5), np.uint8)), light]))
    return bank


xtr, ytr = load("train"); xte, yte = load("test")
bank = glyph_bank()
print(f"{len(bank)} printed glyphs", flush=True)
rng = np.random.default_rng(0)
net, opt = MLP(), Adam()
t0 = time.time()
for ep in range(epochs):
    lr = 1.5e-3 * 0.5 * (1 + np.cos(np.pi * ep / epochs)) + 5e-5                 # cosine decay
    pick = rng.integers(len(bank), size=PRINTED_PER_EPOCH)
    pimgs = np.stack([normalize_char(bank[i][1][rng.integers(3)]) for i in pick])
    plabels = np.array([bank[i][0] for i in pick])
    X = np.concatenate([xtr, pimgs]); Y = np.concatenate([ytr, plabels]).astype(np.int64)
    xa = augment_batch(X, rng).reshape(len(X), -1)
    net.fit_epoch(xa, Y, lr, opt, rng=rng)
    if ep % 3 == 2 or ep == epochs - 1:
        acc = (net.predict_proba(xte.reshape(len(xte), -1)).argmax(1) == yte).mean()
        print(f"epoch {ep + 1}/{epochs}  handwriting test acc {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
net.save("models/mlp.npz")
