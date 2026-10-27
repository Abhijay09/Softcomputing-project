"""A small multilayer perceptron (784 -> 512 -> 256 -> 47) trained with backpropagation + Adam. NumPy only."""
import numpy as np


class MLP:
    def __init__(self, sizes=(784, 512, 256, 47), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]

    def forward(self, x, drop=0.0, rng=None):
        acts, masks = [x], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            if i < len(self.W) - 1:
                z = np.maximum(z, 0)                                   # ReLU
                if drop:
                    m = (rng.random(z.shape) > drop).astype(np.float32) / (1 - drop)
                    z *= m; masks.append(m)
            acts.append(z)
        return acts, masks

    def predict_proba(self, x):
        z = self.forward(np.atleast_2d(x).astype(np.float32))[0][-1]
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)                             # softmax

    def fit_epoch(self, X, y, lr, opt, batch=256, drop=0.2, rng=None):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            ids = order[s:s + batch]; xb, yb = X[ids], y[ids]
            acts, masks = self.forward(xb, drop, rng)
            z = acts[-1]; e = np.exp(z - z.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
            g = p; g[np.arange(len(yb)), yb] -= 1; g /= len(yb)         # d loss / d logits (cross-entropy)
            grads = []
            for i in reversed(range(len(self.W))):
                grads.append((acts[i].T @ g, g.sum(0)))
                if i:
                    g = (g @ self.W[i].T) * (acts[i] > 0)
                    if masks: g *= masks[i - 1]
            grads.reverse()
            opt.step(self, grads, lr)

    def save(self, path):
        np.savez(path, **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path); m = cls.__new__(cls)
        n = len([k for k in d.files if k.startswith("W")])
        m.W = [d[f"W{i}"] for i in range(n)]; m.b = [d[f"b{i}"] for i in range(n)]
        return m


class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.b1, self.b2, self.t, self.m, self.v = b1, b2, 0, None, None

    def step(self, net, grads, lr):
        params = [p for W, b in zip(net.W, net.b) for p in (W, b)]
        gs = [g for gw, gb in grads for g in (gw, gb)]
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, gs, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + 1e-8)
