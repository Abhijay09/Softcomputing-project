"""Train the MLP on EMNIST-balanced with augmentation; report clean test accuracy."""
import sys, time
import numpy as np
from data import load
from mlp import MLP, Adam
from augment import augment_batch

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
xtr, ytr = load("train"); xte, yte = load("test")
rng = np.random.default_rng(0)
net, opt = MLP(), Adam()
t0 = time.time()
for ep in range(epochs):
    lr = 1.5e-3 * 0.5 * (1 + np.cos(np.pi * ep / epochs)) + 5e-5                 # cosine decay
    xa = augment_batch(xtr, rng).reshape(len(xtr), -1)
    net.fit_epoch(xa, ytr.astype(np.int64), lr, opt, rng=rng)
    if ep % 3 == 2 or ep == epochs - 1:
        acc = (net.predict_proba(xte.reshape(len(xte), -1)).argmax(1) == yte).mean()
        print(f"epoch {ep + 1}/{epochs}  test acc {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
net.save("models/mlp.npz")
