"""Makes a handwritten-looking sample page out of real handwritten characters (EMNIST) laid out like one person wrote it
(capitals/digits full height, small letters x-height, descenders hanging below the line), and remembers where every
character is. Used by the demo mode."""
import cv2
import numpy as np
from data import load, label_chars

LINES = ["Soft computing Fuzzy logic", "Abhijay Shrivastava - 2430030333", "Aarush Dayal - 2430030303"]
OWN_LOWER = set("abdefghnqrt")          # lowercase letters that have their own EMNIST class
ASCENDERS, DESCENDERS = set("bdfhklt"), set("gjpqy")
_cache = {}


def _pick(xs, ys, cls, rng):
    """A typical sample of the class: stroke amount near the middle of the class."""
    ids = np.where(ys == cls)[0][:500]
    ink = xs[ids].sum((1, 2))
    mid = ids[(ink > np.percentile(ink, 30)) & (ink < np.percentile(ink, 70))]
    return xs[rng.choice(mid)]


def make_sample(path=None, size=(1280, 720), seed=23):
    """Returns (bgr image, lines) where lines[i] = [(char, (x, y, w, h)), ...] (hyphens and spaces left out)."""
    if "data" not in _cache:
        _cache["data"] = load("test")
    xs, ys = _cache["data"]
    idx = {v: k for k, v in label_chars().items()}
    rng = np.random.default_rng(seed)
    W, H = size
    paper = np.full((H, W), 238, np.float32) + np.linspace(-6, 6, W)[None, :] + np.linspace(4, -4, H)[:, None]
    ink = np.zeros((H, W), np.float32)
    Hc = 42                                                # capital / digit height
    Hx = int(Hc * 0.62)                                    # small-letter height
    out, base = [], 190
    for text in LINES:
        x, wob, row = 40.0, 0.0, []
        for ch in text:
            if ch == " ":
                x += Hc * 0.5; continue
            wob = 0.75 * wob + rng.normal(0, 1.4)          # slowly drifting baseline, like a hand
            y_base = base + wob
            if ch == "-":
                cv2.line(ink, (int(x), int(y_base - Hc * 0.35)), (int(x + Hc * 0.4), int(y_base - Hc * 0.35 + rng.normal(0, 1.5))), 0.9, 3, cv2.LINE_AA)
                x += Hc * 0.65; continue
            cls = idx[ch] if (ch.isdigit() or ch.isupper() or ch in OWN_LOWER) else idx[ch.upper()]
            g = _pick(xs, ys, cls, rng)
            yy, xx = np.nonzero(g > 0.25)
            g = g[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
            if ch.isdigit() or ch.isupper() or ch in ASCENDERS: h, top = Hc, y_base - Hc
            elif ch in DESCENDERS: h, top = int(Hx + 0.38 * Hc), y_base - Hx
            else: h, top = Hx, y_base - Hx
            w = int(np.clip(g.shape[1] * h / g.shape[0], 0.18 * Hc, 0.9 * Hc))
            if ch in "il1I": w = max(w, 5)
            tile = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.normal(0, 2.5), 1.0)
            tile = cv2.warpAffine(tile, M, (w, h)) * rng.uniform(0.9, 1.0)
            x0, y0 = int(x), int(top)
            if x0 + w >= W - 10:
                break
            ink[y0:y0 + h, x0:x0 + w] = np.maximum(ink[y0:y0 + h, x0:x0 + w], tile)
            row.append((ch, (x0, y0, w, h)))
            x += w + 5 + rng.uniform(-1, 2.5)
        out.append(row)
        base += 170
    ink = cv2.GaussianBlur(np.clip(ink * 1.15, 0, 1), (0, 0), 0.7)
    col = np.array([85, 50, 30], np.float32)               # dark blue pen (BGR)
    img = np.stack([paper] * 3, -1)
    for c in range(3):
        img[..., c] = paper * (1 - ink) + col[c] * ink
    img += rng.normal(0, 2.0, img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), 0.5).astype(np.uint8)
    if path: cv2.imwrite(path, img)
    return img, out


if __name__ == "__main__":
    import sys
    make_sample(sys.argv[1])
