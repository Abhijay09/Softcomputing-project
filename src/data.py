import gzip, os
import numpy as np

D = os.path.join(os.path.dirname(__file__), "..", "data", "gzip")


def load(split):
    """EMNIST-balanced: 47 classes (0-9, A-Z, and a few lowercase). Returns images (n,28,28) in [0,1], labels."""
    with gzip.open(f"{D}/emnist-balanced-{split}-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(f"{D}/emnist-balanced-{split}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return np.transpose(x, (0, 2, 1)).astype(np.float32) / 255.0, y   # EMNIST is stored transposed


def label_chars():
    out = {}
    for line in open(f"{D}/emnist-balanced-mapping.txt"):
        k, v = line.split()
        out[int(k)] = chr(int(v))
    return out
