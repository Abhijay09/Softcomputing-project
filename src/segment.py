"""Find handwritten characters in a camera frame and recognise them."""
import cv2
import numpy as np
from features import normalize_char, extract
from fuzzy_logic import reliability, reliability_label


def binarize(frame):
    """Ink = white, paper = black. Adaptive threshold copes with uneven room light."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 12)
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def find_boxes(ink):
    """Connected blobs of ink -> boxes (x, y, w, h), filtered by size and sorted into reading order."""
    H, W = ink.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = [tuple(s[:4]) for s in stats[1:] if s[4] > 0.0004 * H * W and 0.05 * H < s[3] < 0.95 * H]
    if not boxes:
        return []
    med_h = np.median([b[3] for b in boxes])
    boxes.sort(key=lambda b: b[1] + b[3] / 2)                 # group into text lines by vertical centre
    lines, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if abs((b[1] + b[3] / 2) - np.mean([c[1] + c[3] / 2 for c in cur])) > 0.7 * med_h:
            lines.append(cur); cur = []
        cur.append(b)
    lines.append(cur)
    return [sorted(l, key=lambda b: b[0]) for l in lines]


def recognise(frame, model, names):
    """Returns a list of text lines; each is a list of dicts {box, char, strength, reliability, level, top3}."""
    ink = binarize(frame)
    result = []
    for line in find_boxes(ink):
        out = []
        for (x, y, w, h) in line:
            f = extract(normalize_char(ink[y:y + h, x:x + w]))
            m = model.memberships(f)[0]
            o = np.argsort(-m)
            strength, gap = float(m[o[0]]), float(m[o[0]] - m[o[1]])
            r = reliability(strength, gap)
            out.append({"box": (x, y, w, h), "char": names[int(model.classes[o[0]])], "strength": strength,
                        "reliability": r, "level": reliability_label(r),
                        "top3": [(names[int(model.classes[i])], float(m[i])) for i in o[:3]]})
        result.append(out)
    return result


def to_text(lines):
    """Join characters; a wide horizontal gap becomes a space."""
    rows = []
    for line in lines:
        s = ""
        med_w = np.median([c["box"][2] for c in line])
        for i, c in enumerate(line):
            if i:
                p = line[i - 1]["box"]
                if c["box"][0] - (p[0] + p[2]) > 1.2 * med_w:
                    s += " "
            s += c["char"]
        rows.append(s)
    return "\n".join(rows)
