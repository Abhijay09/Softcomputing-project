"""Printed-text scenes like a web page on a screen: bold heading + tiny body text + boxes + barcode.
Score = how well the heading is read (letters only, upper-cased) by the whole-frame pipeline."""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from data import label_chars
from benchmark import edit_distance

HEADINGS = ["All India Mock Paper", "Admit Card 2026", "Room 402 Block B", "Exam Hall 17", "Soft Computing", "Result Sheet 305", "Batch D 2430", "Open Book Test"]
FONTS = ["/usr/share/fonts/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf", "/usr/share/fonts/noto/NotoSans-Bold.ttf"]


def scene(rng, text):
    W, H = 900, 520
    bg = (245, 247, 246) if rng.random() < 0.7 else (30, 32, 38)
    fg = (20, 22, 25) if bg[0] > 100 else (235, 235, 235)
    im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
    size = int(rng.integers(52, 86))
    f = ImageFont.truetype(FONTS[rng.integers(len(FONTS))], size)
    d.text((40, 110), text, fill=fg, font=f)
    small = ImageFont.truetype(FONTS[1], 13)                       # tiny body text, like the page in the screenshot
    for i in range(4):
        d.text((40, 260 + i * 18), "One MCQ paper, split in two parts. Part A is live for two hours, then Part B", fill=(90, 90, 90), font=small)
    for i in range(3):
        d.rounded_rectangle((40 + i * 200, 380, 200 + i * 200, 430), 8, outline=(150, 150, 150), width=2)
        d.text((60 + i * 200, 395), ["Sun, 18 Oct", "10:00 AM", "3 hours"][i], fill=fg, font=small)
    for x in range(560, 860, 6):                                    # barcode-like stripes
        d.rectangle((x, 450, x + int(rng.integers(1, 4)), 500), fill=fg)
    d.rounded_rectangle((520, 380, 860, 430), 10, fill=(14, 110, 95))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def run(predict, n=60, seed=3):
    rng = np.random.default_rng(seed); score = 0.0
    for _ in range(n):
        text = HEADINGS[rng.integers(len(HEADINGS))]
        pred = "".join(c for c in predict(scene(rng, text)) if c.isalnum()).upper()
        truth = "".join(c for c in text if c.isalnum()).upper()
        score += 1 - min(1, edit_distance(pred, truth) / len(truth))
    return score / n


if __name__ == "__main__":
    from classifier import CharClassifier
    from segment import recognise, to_text
    clf, names = CharClassifier(), label_chars()
    print(f"printed-scene heading accuracy: {run(lambda f: to_text(recognise(f, clf, names)) or ''):.1%}")
