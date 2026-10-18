"""Turn a 28x28 character image into 177 numbers: 49 ink-zone values + 128 stroke-direction values."""
import cv2
import numpy as np

GRID = 7          # 7x7 zones over a 28x28 image -> 49 features


def normalize_char(binary):
    """binary: white ink on black, any size (tight crop). Returns 28x28 float image in [0,1].
    Same recipe EMNIST/MNIST use: fit into 20x20 keeping aspect, centre in 28x28."""
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)           # soft strokes like the training data
    canvas = np.zeros((28, 28), np.float32)
    h, w = crop.shape
    canvas[(28 - h) // 2:(28 - h) // 2 + h, (28 - w) // 2:(28 - w) // 2 + w] = crop / 255.0
    return canvas


def zone_features(img28):
    """Ink density: average of each 4x4 block of the 28x28 image -> 49 values in [0,1]."""
    b = 28 // GRID
    return img28.reshape(GRID, b, GRID, b).mean(axis=(1, 3)).ravel().astype(np.float32)


def direction_features(img28):
    """Stroke direction (HOG-style): split image into 4x4 cells; in each cell sum the gradient
    strength falling in 8 angle bins (0..180 deg) -> 128 values, scaled to unit length."""
    im = img28.astype(np.float32)
    gx, gy = cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    bins = np.minimum(((np.arctan2(gy, gx) % np.pi) / np.pi * 8).astype(int), 7)
    out = np.zeros((4, 4, 8), np.float32)
    for r in range(4):
        for c in range(4):
            sl = (slice(r * 7, r * 7 + 7), slice(c * 7, c * 7 + 7))
            out[r, c] = np.bincount(bins[sl].ravel(), mag[sl].ravel(), 8)
    return (out / (np.linalg.norm(out) + 1e-6)).ravel()


def extract(img28):
    """Full feature vector (177 values) for one normalised character."""
    return np.concatenate([zone_features(img28), direction_features(img28)])
