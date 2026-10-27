"""DEMO MODE: a mock-up of how the finished reader should look. The image is built from real EMNIST characters and the
boxes come from the real segmentation, but the letters/confidences shown are SCRIPTED, not recognised.
Run with:  python src/app.py --demo"""
import cv2
import numpy as np
from data import load, label_chars
from segment import recognise

SCRIPT = ["FUZZY LOGIC", "2430030333"]


def make_demo_image(path):
    xs, ys = load("test"); names = label_chars(); idx = {v: k for k, v in names.items()}
    rng = np.random.default_rng(5)
    canvas = np.full((480, 640), 232, np.float32)
    for row, text in enumerate(SCRIPT):
        x, y0, h = 30, 120 + row * 150, 52
        for ch in text:
            if ch == " ":
                x += 40; continue
            i = rng.choice(np.where(ys == idx[ch])[0])
            im = (xs[i] * 255).astype(np.uint8)
            yy, xx = np.nonzero(im > 60); im = im[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
            tw = max(14, min(int(im.shape[1] * h / im.shape[0]), int(0.85 * h)))
            im = cv2.resize(im, (tw, h), interpolation=cv2.INTER_CUBIC)
            if x + tw > 625: break
            roi = canvas[y0:y0 + h, x:x + im.shape[1]]
            canvas[y0:y0 + h, x:x + im.shape[1]] = np.minimum(roi, 232 - (232 - 35) * im.astype(np.float32) / 255.0)
            x += tw + 12
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8) + rng.normal(0, 2.5, canvas.shape)
    cv2.imwrite(path, cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))


def demo_recognise(frame, clf, names, context=True, tracker=None, roi=None, mode="auto"):
    """Real boxes, scripted text."""
    real = recognise(frame, clf, names, context, tracker, roi, mode)
    rng = np.random.default_rng(11)
    for li, line in enumerate(real):
        script = [c for c in SCRIPT[li] if c != " "] if li < len(SCRIPT) else []
        for k, c in enumerate(line):
            if k >= len(script): break
            ch = script[k]
            rel = float(rng.uniform(72, 97)); p = rel / 100
            alts = [a for a in c["top3"] if a[0] != ch][:2]
            c.update(char=ch, reliability=rel, level="HIGH" if rel >= 65 else "MEDIUM", strength=p,
                     top3=[(ch, p)] + [(n, (1 - p) * (0.6 if i == 0 else 0.3)) for i, (n, _) in enumerate(alts)])
    return real
