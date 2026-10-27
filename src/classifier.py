"""Character classifier used by the app: a trained MLP gives class scores, read as fuzzy memberships."""
import os
import numpy as np
from mlp import MLP

ROOT = os.path.join(os.path.dirname(__file__), "..")
DIGIT_CLASSES = np.arange(10)                 # EMNIST-balanced label 0-9 are the digits


class CharClassifier:
    def __init__(self, path=None):
        self.net = MLP.load(path or os.path.join(ROOT, "models", "mlp.npz"))

    def proba(self, imgs28):
        """imgs28: (n,28,28) in [0,1]. Returns (n,47) memberships that add up to 1 (average of 5 tiny shifts)."""
        X = np.asarray(imgs28, np.float32)
        out = 0
        for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            out = out + self.net.predict_proba(np.roll(X, (dy, dx), axis=(1, 2)).reshape(len(X), -1))
        return out / 5
