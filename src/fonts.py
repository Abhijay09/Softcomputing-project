"""Printed letters/digits from the system fonts, used both to train on and to build test scenes."""
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# EMNIST-balanced classes: 0-9, A-Z, then a b d e f g h n q r t. Lowercase c i j k l m o p s u v w x y z share the capital's class.
CLASS_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abdefghnqrt")
MERGED = set("cijklmopsuvwxyz")
LATIN = ["NotoSans-", "NotoSerif-", "NotoSansMono-", "Liberation", "Nimbus", "URWGothic", "URWBookman", "C059", "P052", "DejaVu"]


def font_files():
    files = []
    for pat in ("/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf"):
        for f in glob.glob(pat, recursive=True):
            name = f.split("/")[-1]
            if any(name.startswith(p) for p in LATIN) and "Emoji" not in name:
                files.append(f)
    return sorted(files)


def glyphs():
    """Yield (class index, character) for every glyph to render."""
    for k, ch in enumerate(CLASS_CHARS):
        yield k, ch
        if ch.isalpha() and ch.upper() == ch and ch.lower() in MERGED:
            yield k, ch.lower()


def render(ch, font_file, size=80):
    """White ink on black, tight crop. None if the font lacks the glyph."""
    font = ImageFont.truetype(font_file, size)
    im = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(im).text((size // 2, size // 4), ch, fill=255, font=font)
    a = np.array(im)
    ys, xs = np.nonzero(a > 100)
    if len(ys) < 20:
        return None
    return ((a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 110) * 255).astype(np.uint8)
