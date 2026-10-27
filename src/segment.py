"""Find handwritten characters in a camera frame and read them."""
import cv2
import numpy as np
from features import normalize_char
from fuzzy_logic import reliability, reliability_label
from classifier import DIGIT_CLASSES


# ---------- 1. find the ink ----------
def binarize(frame):
    """Ink = white, paper = black. A local threshold copes with uneven light; the second test throws away paper texture."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(31, (min(H, W) // 10) | 1)
    local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 9)
    dark = (gray < np.percentile(gray, 60) - 12).astype(np.uint8) * 255
    return cv2.morphologyEx(cv2.bitwise_and(local, dark), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


# ---------- 2. split it into characters ----------
def _merge(boxes, med_h):
    """Glue fragments that belong together (dot of an i, the two bars of '=', a pen-lift gap)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                gap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                nh = max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1])
                if ox > 0.5 * min(a[2], b[2]) and gap < 0.35 * med_h and nh < 1.6 * med_h:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    boxes[i] = [x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, nh]
                    del boxes[j]; changed = True
                    break
            if changed: break
    return [tuple(b) for b in boxes]


def find_boxes(ink):
    """Returns text lines; each line is a left-to-right list of character boxes (x, y, w, h)."""
    H, W = ink.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps, m = [], int(0.02 * max(H, W))                                   # keep-out strip along the frame border
    for x, y, w, h, area in stats[1:]:
        if area < max(10, 0.00003 * H * W): continue                        # speckle
        if x < m or y < m or x + w > W - m or y + h > H - m: continue          # paper / table edges and shadows
        if h > 0.9 * H or w > 0.6 * W: continue
        comps.append((x, y, w, h, area))
    main = [c for c in comps if c[4] > 0.0003 * H * W and c[3] > 0.05 * H]
    if not main:
        return []
    med_h = float(np.median([c[3] for c in main]))
    boxes = _merge([c[:4] for c in comps], med_h)
    boxes = [b for b in boxes if b[3] >= 0.35 * med_h and b[3] <= 2.2 * med_h]
    if not boxes:
        return []
    med_h = float(np.median([b[3] for b in boxes]))
    boxes.sort(key=lambda b: b[1] + b[3] / 2)                               # lines by vertical centre
    lines, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if abs((b[1] + b[3] / 2) - np.mean([c[1] + c[3] / 2 for c in cur])) > 0.7 * med_h:
            lines.append(cur); cur = []
        cur.append(b)
    lines.append(cur)
    return [sorted(l, key=lambda b: b[0]) for l in lines]


def split_words(line):
    """Indices of the characters of one line, grouped into words by wide gaps."""
    med_w = np.median([b[2] for b in line])
    words, cur = [], [0]
    for i in range(1, len(line)):
        p, c = line[i - 1], line[i]
        if c[0] - (p[0] + p[2]) > 1.2 * med_w:
            words.append(cur); cur = []
        cur.append(i)
    words.append(cur)
    return words


# ---------- 3. fuzzy context rule ----------
def apply_context(P):
    """P: (k,47) memberships of the characters of one word. Rules:
         IF the word looks like digits  THEN digits get more likely (letters less)
         IF the word looks like letters THEN letters get more likely (digits less)
    'looks like digits' = average membership of the digit classes, turned into a fuzzy degree with a ramp."""
    if len(P) < 2:
        return P
    D = P[:, DIGIT_CLASSES].sum(1).mean()
    mu_digit = np.clip((D - 0.4) / 0.4, 0, 1)
    mu_letter = np.clip((0.6 - D) / 0.4, 0, 1)
    f = np.ones(P.shape[1], np.float32)
    f[DIGIT_CLASSES] = (1 + 2 * mu_digit) * (1 - 0.7 * mu_letter)
    letters = np.ones(P.shape[1], bool); letters[DIGIT_CLASSES] = False
    f[letters] = (1 + 2 * mu_letter) * (1 - 0.7 * mu_digit)
    Q = P * f
    return Q / Q.sum(1, keepdims=True)


# ---------- 4. smoothing over video frames ----------
class Tracker:
    """Averages each character's memberships over a few frames, so the text does not flicker."""
    def __init__(self, keep=0.6):
        self.prev, self.keep = [], keep

    def reset(self):
        self.prev = []

    def update(self, boxes, P):
        out, new = P.copy(), []
        used = set()
        for i, (x, y, w, h) in enumerate(boxes):
            cx, cy = x + w / 2, y + h / 2
            best, bd = None, 0.5 * h
            for j, (pcx, pcy, ph, pP) in enumerate(self.prev):
                d = np.hypot(cx - pcx, cy - pcy)
                if j not in used and d < bd:
                    best, bd = j, d
            if best is not None:
                used.add(best)
                out[i] = self.keep * self.prev[best][3] + (1 - self.keep) * P[i]
            new.append((cx, cy, h, out[i]))
        self.prev = new
        return out


# ---------- 5. put it together ----------
def recognise(frame, clf, names, context=True, tracker=None):
    """Returns a list of text lines; each is a list of dicts {box, char, strength, reliability, level, top3}."""
    ink = binarize(frame)
    lines = find_boxes(ink)
    flat = [b for l in lines for b in l]
    if not flat:
        if tracker: tracker.update([], np.zeros((0, 47), np.float32))
        return []
    P = clf.proba(np.stack([normalize_char(ink[y:y + h, x:x + w]) for (x, y, w, h) in flat]))
    if tracker:
        P = tracker.update(flat, P)
    result, k = [], 0
    for line in lines:
        Pl = P[k:k + len(line)]; k += len(line)
        if context:
            Pl = Pl.copy()
            for word in split_words(line):
                Pl[word] = apply_context(Pl[word])
        out = []
        for box, m in zip(line, Pl):
            o = np.argsort(-m)
            strength, gap = float(m[o[0]]), float(m[o[0]] - m[o[1]])
            r = reliability(strength, gap)
            out.append({"box": box, "char": names[int(o[0])], "strength": strength, "reliability": r,
                        "level": reliability_label(r), "top3": [(names[int(i)], float(m[i])) for i in o[:3]]})
        result.append(out)
    return result


def to_text(lines):
    """Join characters; a wide horizontal gap becomes a space."""
    rows = []
    for line in lines:
        words = split_words([c["box"] for c in line])
        rows.append(" ".join("".join(line[i]["char"] for i in w) for w in words))
    return "\n".join(rows)
