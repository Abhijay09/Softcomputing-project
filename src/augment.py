"""Random changes that imitate camera photos of handwriting: tilt, size, slant, shift, pen thickness, blur, noise."""
import cv2
import numpy as np


def augment_one(img, rng):
    c = 13.5
    ang, sc, shear = rng.uniform(-14, 14), rng.uniform(0.82, 1.15), rng.uniform(-0.25, 0.25)
    M = cv2.getRotationMatrix2D((c, c), ang, sc)
    M[0, 1] += shear; M[:, 2] += rng.uniform(-2.5, 2.5, 2)
    out = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_LINEAR)
    r = rng.random()
    if r < 0.25: out = cv2.dilate(out, np.ones((2, 2), np.uint8))          # thicker pen
    elif r < 0.55: out = cv2.erode(out, np.ones((2, 2), np.uint8))         # thinner pen
    if rng.random() < 0.3: out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 0.9))
    if rng.random() < 0.3: out = out + rng.normal(0, 0.05, out.shape).astype(np.float32)
    return np.clip(out * rng.uniform(0.7, 1.0) if rng.random() < 0.3 else out, 0, 1).astype(np.float32)


def augment_batch(X, rng):
    return np.stack([augment_one(x, rng) for x in X])
