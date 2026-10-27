"""DEMO MODE: a mock-up of how the finished reader should look. The page is a generated handwriting-style image and the
letters, boxes and confidences shown are SCRIPTED from where the generator put each character. Nothing is recognised.
Run with:  python src/app.py --demo"""
import cv2
import numpy as np
from handwriting_sample import make_sample
from segment import recognise

LOOKALIKE = {"o": "0", "0": "O", "l": "1", "1": "l", "i": "j", "j": "i", "S": "5", "5": "S", "g": "q", "q": "g",
             "t": "f", "f": "t", "u": "v", "v": "u", "a": "o", "c": "e", "e": "c", "n": "h", "h": "n", "3": "B", "8": "B"}


LAST_SCRIPTED = False        # was the last result the scripted one (True) or a real reading of some other picture?


def make_demo_image(path):
    make_sample(path)


def demo_recognise(frame, clf, names, context=True, tracker=None, roi=None, mode="auto"):
    """Scripted result for the generated page (scaled if the frame size differs)."""
    ref, lines = make_sample()
    ref = cv2.resize(ref, (frame.shape[1], frame.shape[0]))
    global LAST_SCRIPTED
    LAST_SCRIPTED = np.abs(ref.astype(np.int16) - frame.astype(np.int16)).mean() <= 12
    if not LAST_SCRIPTED:                                                       # some other picture: read it for real
        return recognise(frame, clf, names, context, tracker, roi, mode)
    sx, sy = frame.shape[1] / 1280, frame.shape[0] / 720
    rng = np.random.default_rng(11)
    out = []
    for row in lines:
        res = []
        for ch, (x, y, w, h) in row:
            rel = float(rng.uniform(68, 97)); p = rel / 100
            alt = LOOKALIKE.get(ch, "e")
            res.append({"box": (int(x * sx), int(y * sy), int(w * sx), int(h * sy)), "char": ch, "strength": p,
                        "reliability": rel, "level": "HIGH" if rel >= 65 else "MEDIUM",
                        "top3": [(ch, p), (alt, (1 - p) * 0.6), ("r" if alt != "r" else "n", (1 - p) * 0.3)]})
        out.append(res)
    return out
