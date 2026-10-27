"""DEMO MODE: a mock-up of how the finished reader should look. The page is a generated handwriting-style image and the
letters, boxes and confidences shown are SCRIPTED from where the generator put each character. Nothing is recognised.
Run with:  python src/app.py --demo"""
import numpy as np
from handwriting_sample import make_sample

LOOKALIKE = {"o": "0", "0": "O", "l": "1", "1": "l", "i": "j", "j": "i", "S": "5", "5": "S", "g": "q", "q": "g",
             "t": "f", "f": "t", "u": "v", "v": "u", "a": "o", "c": "e", "e": "c", "n": "h", "h": "n", "3": "B", "8": "B"}


def make_demo_image(path):
    make_sample(path)


def demo_recognise(frame, clf, names, context=True, tracker=None, roi=None, mode="auto"):
    """Scripted result for the generated page (scaled if the frame size differs)."""
    _, lines = make_sample()
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
