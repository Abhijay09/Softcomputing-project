"""End-to-end benchmark: fake 'camera photos' of handwriting made from EMNIST test characters
(thin/thick pens, uneven light, blur, noise, tilt), run through the real pipeline.
Reports character accuracy (1 - edit distance / length) and how often the character count is right."""
import sys
import cv2
import numpy as np
from data import load, label_chars
from segment import recognise, to_text


def make_photo(rng, xs, ys, names, n):
    H, W = 480, 640
    idx = rng.choice(len(xs), n, replace=False)
    h = rng.integers(45, 95)                                    # character height in pixels
    ink = rng.integers(15, 90)                                  # pen darkness
    paper = rng.integers(185, 240)
    canvas = np.full((H, W), float(paper), np.float32)
    x = rng.integers(25, 70); y0 = rng.integers(120, 300)
    thick = rng.choice([-1, 0, 0, 1, 2])                        # thin ... thick pen
    drawn = []
    for i in idx:
        im = (xs[i] * 255).astype(np.uint8)
        yy, xx = np.nonzero(im > 60)
        im = im[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        sc = h / im.shape[0]
        im = cv2.resize(im, (max(8, int(im.shape[1] * sc)), h), interpolation=cv2.INTER_CUBIC)
        if thick > 0: im = cv2.dilate(im, np.ones((thick * 2 + 1,) * 2, np.uint8))
        if thick < 0: im = cv2.erode(im, np.ones((2, 2), np.uint8))
        if x + im.shape[1] > W - 20: break
        roi = canvas[y0:y0 + h, x:x + im.shape[1]]
        canvas[y0:y0 + h, x:x + im.shape[1]] = np.minimum(roi, paper - (paper - ink) * im / 255.0)
        drawn.append(i)
        x += im.shape[1] + int(rng.uniform(0.35, 0.9) * im.shape[1])
    # light gradient + blur + noise + tilt
    gx = np.linspace(-1, 1, W)[None, :] * rng.uniform(-25, 25); gy = np.linspace(-1, 1, H)[:, None] * rng.uniform(-25, 25)
    canvas = canvas + gx + gy
    canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.6, 1.4))
    canvas += rng.normal(0, rng.uniform(2, 8), canvas.shape)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-5, 5), 1.0)
    canvas = cv2.warpAffine(canvas, M, (W, H), borderValue=float(paper))
    return cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), drawn


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)]


def run(predict, n_images=200, seed=7):
    xs, ys = load("test"); names = label_chars()
    rng = np.random.default_rng(seed)
    cer, count_ok, total = 0.0, 0, 0
    for _ in range(n_images):
        n = int(rng.integers(4, 8))
        frame, idx = make_photo(rng, xs, ys, names, n)
        truth = "".join(names[int(ys[j])] for j in idx)
        pred = predict(frame).replace(" ", "").replace("\n", "")
        cer += edit_distance(pred, truth) / max(len(truth), 1); total += 1
        count_ok += (len(pred) == len(truth))
    return 1 - cer / total, count_ok / total


if __name__ == "__main__":
    from classifier import CharClassifier
    clf = CharClassifier(); names = label_chars()
    acc, cnt = run(lambda f: to_text(recognise(f, clf, names, context=False)) or "")
    print(f"character accuracy {acc:.1%}   character count correct {cnt:.1%}")
