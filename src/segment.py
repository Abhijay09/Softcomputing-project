"""Find handwritten characters in a camera frame and read them."""
import cv2
import numpy as np
from features import normalize_char
from fuzzy_logic import reliability, reliability_label
from classifier import DIGIT_CLASSES


# ---------- 1. find the ink ----------
def binarize(frame):
    """Ink = white, paper = black. Works for dark ink on light paper AND light strokes on a dark screen:
    both are tried and the one with less 'ink' wins (ink is always the minority). A local threshold copes with uneven light."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(31, (min(H, W) // 10) | 1)
    med = float(np.median(gray))
    dark_mass = np.clip(med - gray.astype(np.float32) - 40, 0, None).sum()       # how much is much darker than the background
    bright_mass = np.clip(gray.astype(np.float32) - med - 40, 0, None).sum()     # ... and much brighter
    g = 255 - gray if bright_mass > 2 * dark_mass else gray                      # light strokes on a dark screen -> flip
    local = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 9)
    far = (g < np.percentile(g, 60) - 12).astype(np.uint8) * 255                 # clearly darker than the usual background
    return cv2.morphologyEx(cv2.bitwise_and(local, far), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


# ---------- 2. split it into characters ----------
def _attach_fragments(dom, frags, med_h):
    """Glue small pieces (dot of an i or j) onto the character box they sit above or below."""
    dom = [list(b) for b in dom]
    for fx, fy, fw, fh in frags:
        cx = fx + fw / 2
        for b in dom:
            gap = max(fy, b[1]) - min(fy + fh, b[1] + b[3])
            if b[0] - 0.15 * b[2] <= cx <= b[0] + 1.15 * b[2] and gap < 0.5 * med_h:
                x0, y0 = min(fx, b[0]), min(fy, b[1])
                x1, y1 = max(fx + fw, b[0] + b[2]), max(fy + fh, b[1] + b[3])
                if y1 - y0 < 1.5 * med_h:
                    b[:] = [x0, y0, x1 - x0, y1 - y0]
                    break
    return [tuple(b) for b in dom]


def _not_a_character(mask):
    """Hollow frames / buttons, and barcode-like stripe patterns are never letters."""
    h, w = mask.shape
    if (w > 1.8 * h or h > 1.8 * w) and mask.mean() < 0.25:
        inner = mask[int(0.25 * h):int(0.75 * h) + 1, int(0.25 * w):int(0.75 * w) + 1]
        if inner.mean() < 0.03:
            return True                                                     # outline with an empty middle
    col = mask.mean(0)
    full, empty = col > 0.85, col < 0.1
    if w >= 12 and (np.diff(full.astype(int)) != 0).sum() >= 5 and ((~full) & (~empty)).mean() < 0.2:
        return True                                                         # vertical stripes
    return False


def _dominant_size(boxes):
    """Pages hold text of many sizes (headings, tiny body text, labels). Group the boxes by height and keep the group
    with the most ink, i.e. the main thing in view."""
    hs = sorted(set(b[3] for b in boxes))
    groups, cur = [], [hs[0]]
    for h in hs[1:]:
        if h > 1.6 * cur[-1]:
            groups.append(cur); cur = []
        cur.append(h)
    groups.append(cur)
    best = max(groups, key=lambda g: sum(b[2] * b[3] for b in boxes if g[0] <= b[3] <= g[-1]))
    lo, hi = best[0] * 0.8, best[-1] * 1.25
    return [b for b in boxes if lo <= b[3] <= hi], float(np.median([b[3] for b in boxes if best[0] <= b[3] <= best[-1]]))


def _split_touching(ink, box, h_ref, depth=0):
    """Letters that touch (bold print, joined strokes) come out as one wide blob. Cut it where a single thin stroke
    joins two parts that are both nearly full height."""
    x, y, w, h = box
    if w <= 2.3 * h_ref or depth > 6:
        return [box]
    crop = ink[y:y + h, x:x + w] > 0
    col = crop.sum(0).astype(float)
    pen = max(1.0, 4 * cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, 3)[crop].mean() / 1.66)
    margin = int(0.2 * h_ref)
    best = None
    for c in range(margin, w - margin):
        if col[c] > 1.6 * pen: continue
        lh = np.nonzero(crop[:, :c].any(1))[0]; rh = np.nonzero(crop[:, c + 1:].any(1))[0]
        if len(lh) == 0 or len(rh) == 0: continue
        if lh.max() - lh.min() < 0.55 * h_ref or rh.max() - rh.min() < 0.55 * h_ref: continue
        if best is None or col[c] < col[best] - 0.5: best = c
    if best is None and w > 3.2 * h_ref:                                     # clearly several letters: cut at the weakest point
        mid = range(int(0.3 * w), int(0.7 * w) + 1)
        best = min(mid, key=lambda c: col[c])
    if best is None:
        return [box]
    out = []
    for sub in ((x, y, best, h), (x + best + 1, y, w - best - 1, h)):
        yy, xx = np.nonzero(ink[sub[1]:sub[1] + sub[3], sub[0]:sub[0] + sub[2]])
        if len(yy) < 8: continue
        tight = (sub[0] + xx.min(), sub[1] + yy.min(), xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)
        out += _split_touching(ink, tight, h_ref, depth + 1)
    return out


def find_boxes(ink, border=True):
    """Returns text lines; each is a left-to-right list of character boxes (x, y, w, h)."""
    H, W = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps, m = [], int(0.02 * max(H, W)) if border else 2                   # keep-out strip along the frame border
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if area < max(10, 0.00003 * H * W): continue                        # speckle
        if x < m or y < m or x + w > W - m or y + h > H - m: continue       # paper / table edges and shadows
        if h > 0.9 * H or w > 0.6 * W: continue
        if w > 6 * h or h > 6 * w: continue                                 # long thin lines
        if area < 0.09 * w * h: continue                                    # very hollow
        if w > 12 and _not_a_character(labels[y:y + h, x:x + w] == i): continue
        comps.append((x, y, w, h))
    if not comps:
        return []
    boxes, med_h = _dominant_size(comps)                                    # the main text size in view
    frags = [c for c in comps if c[3] < 0.5 * med_h and c not in boxes]
    boxes = _attach_fragments(boxes, frags, med_h)
    boxes = [t for b in boxes for t in _split_touching(ink, b, med_h)]
    boxes = [b for b in boxes if b[3] >= 0.5 * med_h and b[3] > 0.03 * H]
    if not boxes:
        return []
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


def _maybe_split(ink, box, h_ref, clf):
    """A wide blob is either one wide letter (m, w) or two letters that touch (00). Try the best cut and keep it only if
    the two halves are clearly read with more confidence than the whole blob."""
    x, y, w, h = box
    if w < 1.2 * h_ref or w < 1.25 * h:
        return [box]
    crop = ink[y:y + h, x:x + w] > 0
    col = crop.sum(0)
    lo, hi = int(0.3 * w), int(0.7 * w) + 1
    sm = np.convolve(col.astype(float), np.ones(5) / 5, mode="same")
    c = lo + int(np.argmin(sm[lo:hi] + 0.2 * np.abs(np.arange(lo, hi) - w / 2)))     # weakest point, leaning to the middle
    halves = []
    for sub in ((x, y, c, h), (x + c, y, w - c, h)):
        yy, xx = np.nonzero(ink[sub[1]:sub[1] + sub[3], sub[0]:sub[0] + sub[2]])
        if len(yy) < 10: return [box]
        halves.append((sub[0] + xx.min(), sub[1] + yy.min(), xx.max() - xx.min() + 1, yy.max() - yy.min() + 1))
    if min(hb[2] for hb in halves) < 0.25 * h_ref or min(hb[3] for hb in halves) < 0.4 * h_ref:
        return [box]
    whole = clf.proba(normalize_char(ink[y:y + h, x:x + w])[None])[0].max()
    parts = clf.proba(np.stack([normalize_char(ink[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]) for b in halves])).max(1)
    return halves if (whole < 0.6 and parts.min() > 0.6 and parts.mean() > whole + 0.25) else [box]


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
def recognise(frame, clf, names, context=True, tracker=None, roi=None, mode="auto"):
    """Returns a list of text lines; each is a list of dicts {box, char, strength, reliability, level, top3}.
    roi = (x0, y0, x1, y1): only read inside this rectangle (boxes are still given in full-frame coordinates)."""
    ox = oy = 0
    if roi:
        ox, oy = roi[0], roi[1]
        frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
    ink = binarize(frame)
    lines = find_boxes(ink, border=roi is None)
    for li, line in enumerate(lines):
        med_h = float(np.median([b[3] for b in line]))
        lines[li] = sorted([t for b in line for t in _maybe_split(ink, b, med_h, clf)], key=lambda b: b[0])
    flat = [b for l in lines for b in l]
    if not flat:
        if tracker: tracker.update([], np.zeros((0, 47), np.float32))
        return []
    P = clf.proba(np.stack([normalize_char(ink[y:y + h, x:x + w]) for (x, y, w, h) in flat]))
    if mode != "auto":                                  # reading only digits (e.g. a roll number) or only letters
        keep = np.zeros(P.shape[1], bool)
        keep[DIGIT_CLASSES] = True
        P = P * (keep if mode == "digits" else ~keep)
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
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
            out.append({"box": (box[0] + ox, box[1] + oy, box[2], box[3]), "char": names[int(o[0])], "strength": strength, "reliability": r,
                        "level": reliability_label(r), "top3": [(names[int(i)], float(m[i])) for i in o[:3]]})
        result.append(out)
    return result


def to_text(lines, min_reliability=8):
    """Join characters; a wide horizontal gap becomes a space. A character the controller does not trust becomes '?'."""
    rows = []
    for line in lines:
        words = split_words([c["box"] for c in line])
        rows.append(" ".join("".join(line[i]["char"] if line[i]["reliability"] >= min_reliability else "?" for i in w) for w in words))
    return "\n".join(rows)
