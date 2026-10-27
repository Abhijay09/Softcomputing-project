"""Whole-line handwriting reader using a pretrained model (TrOCR, Microsoft). Used by the 'Read with AI' button.
The model is downloaded from Hugging Face the first time (about 1.3 GB)."""
import cv2
import numpy as np

MODEL = "microsoft/trocr-base-handwritten"
_cache = {}


def _load():
    if "m" not in _cache:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _cache["p"] = TrOCRProcessor.from_pretrained(MODEL)
        _cache["m"] = VisionEncoderDecoderModel.from_pretrained(MODEL).eval()
    return _cache["p"], _cache["m"]


def split_lines(frame):
    """Cut a page photo into text-line pictures. Ruled lines of a notebook are removed before looking for text."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    ink = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, max(31, (H // 12) | 1), 12)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, W // 6), 1)))
    ink = cv2.subtract(ink, cv2.dilate(rules, np.ones((3, 3), np.uint8)))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    rows = cv2.GaussianBlur((ink > 0).sum(1).astype(np.float32)[:, None], (1, 0), 3)[:, 0]
    on = rows > max(3, 0.04 * rows.max())
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None: start = y
        if not v and start is not None:
            if y - start >= 14: bands.append((start, y))
            start = None
    if start is not None and H - start >= 14: bands.append((start, H))
    out = []
    for y0, y1 in bands:
        cols = np.nonzero((ink[y0:y1] > 0).sum(0) > 0)[0]
        if len(cols) == 0: continue
        pad = max(6, (y1 - y0) // 3)
        a, b = max(0, y0 - pad), min(H, y1 + pad)
        c0, c1 = max(0, cols.min() - pad), min(W, cols.max() + pad)
        out.append(((c0, a, c1 - c0, b - a), frame[a:b, c0:c1], (ink[a:b, c0:c1] > 0).sum(0)))
    return out


def chunks(crop, colink, max_aspect=4.5):
    """TrOCR squeezes its input to a square, so a long line must be cut into pieces of a few words, at gaps between words."""
    h, w = crop.shape[:2]
    if w <= max_aspect * h:
        return [crop]
    empty = colink == 0
    gap_min = max(4, int(0.25 * h))
    cuts, run = [], 0
    for x, e in enumerate(empty):                      # centres of the gaps between words
        run = run + 1 if e else 0
        if run == gap_min: cuts.append(x - gap_min // 2)
    out, start, last = [], 0, None
    for c in cuts + [w]:
        if c - start > max_aspect * h and last is not None:
            out.append((start, last)); start = last
        last = c
    out.append((start, w))
    return [crop[:, a:b] for a, b in out if b - a > 0.5 * h]


def read_page(frame, progress=None):
    """Returns [(box, text), ...] one entry per text line, top to bottom."""
    import torch
    proc, model = _load()
    lines = split_lines(frame)
    results = []
    for i, (box, crop, colink) in enumerate(lines):
        words = []
        for piece in chunks(crop, colink):
            if piece.shape[0] < 80:                                   # small writing: enlarge so strokes are clear
                f = 80 / piece.shape[0]; piece = cv2.resize(piece, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                ids = model.generate(proc(images=rgb, return_tensors="pt").pixel_values, max_new_tokens=32)
            words.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        results.append((box, " ".join(words)))
        if progress: progress(i + 1, len(lines))
    return results
