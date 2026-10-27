"""Fuzzy Handwriting Reader - Tkinter app. Point a camera at handwriting; it reads it live."""
import os, sys, time
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(__file__))
from data import label_chars
from classifier import CharClassifier
from segment import recognise, to_text, Tracker

ROOT = os.path.join(os.path.dirname(__file__), "..")
BG, PANEL, FG, DIM, ACCENT = "#0f1117", "#181b24", "#e8eaf0", "#8a90a2", "#7c5cff"
LEVEL_COL = {"HIGH": "#3ddc97", "MEDIUM": "#ffb454", "LOW": "#ff5c7a"}   # also used as BGR below
VIDEO_W, VIDEO_H = 640, 480        # size of the picture shown in the window


def bgr(hexcol):
    return tuple(int(hexcol[i:i + 2], 16) for i in (5, 3, 1))


class App:
    def __init__(self, root, cam_index=0, image=None):
        self.root = root
        root.title("Fuzzy Handwriting Reader")
        root.configure(bg=BG)
        self.model = CharClassifier()
        self.tracker = Tracker()
        self.context = tk.BooleanVar(value=True)
        self.names = label_chars()
        self.cam_index, self.cap, self.frozen, self.dirty, self.current, self.shown = cam_index, None, False, False, None, None
        self._build()
        if image:                                   # start on a still image instead of the camera
            self.current = cv2.imread(image); self.frozen = self.dirty = True
            self.btn_freeze.config(text="Resume")
        else:
            self.open_camera()
        self.tick()

    # ---------- layout ----------
    def _build(self):
        left = tk.Frame(self.root, bg=BG); left.grid(row=0, column=0, padx=16, pady=16, sticky="n")
        tk.Label(left, text="FUZZY HANDWRITING READER", bg=BG, fg=FG, font=("Helvetica", 16, "bold")).pack(anchor="w")
        tk.Label(left, text="alphanumeric detection using fuzzy rules", bg=BG, fg=DIM, font=("Helvetica", 10)).pack(anchor="w", pady=(0, 8))
        self.video = tk.Label(left, bg="black", width=VIDEO_W, height=VIDEO_H); self.video.pack()
        bar = tk.Frame(left, bg=BG); bar.pack(fill="x", pady=10)
        self.btn_freeze = self._btn(bar, "Freeze", self.toggle_freeze, ACCENT)
        self._btn(bar, "Save snapshot", self.save_snapshot)
        self._btn(bar, "Open image…", self.open_image)
        self._btn(bar, "Switch camera", self.next_camera)
        tk.Checkbutton(bar, text="Context rules", variable=self.context, bg=BG, fg=FG, selectcolor="#262a36", activebackground=BG, activeforeground=FG, highlightthickness=0, command=self._redo).pack(side="left", padx=(8, 0))
        self.status = tk.Label(left, text="", bg=BG, fg=DIM, font=("Helvetica", 9)); self.status.pack(anchor="w")

        right = tk.Frame(self.root, bg=PANEL, width=380); right.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="ns")
        right.grid_propagate(False)
        tk.Label(right, text="DETECTED TEXT", bg=PANEL, fg=DIM, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=16, pady=(16, 4))
        self.text = tk.Label(right, text="—", bg=PANEL, fg=FG, font=("Courier", 26, "bold"), justify="left", anchor="w", wraplength=340)
        self.text.pack(anchor="w", padx=16)
        tk.Label(right, text="FUZZY RELIABILITY (avg)", bg=PANEL, fg=DIM, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=16, pady=(16, 4))
        self.rel_canvas = tk.Canvas(right, width=340, height=18, bg="#262a36", highlightthickness=0); self.rel_canvas.pack(padx=16)
        self.rel_label = tk.Label(right, text="", bg=PANEL, fg=FG, font=("Helvetica", 10)); self.rel_label.pack(anchor="w", padx=16)
        tk.Label(right, text="PER CHARACTER  (best 3 guesses, membership 0–1)", bg=PANEL, fg=DIM, font=("Helvetica", 9, "bold")).pack(anchor="w", padx=16, pady=(16, 4))
        self.detail = tk.Label(right, text="", bg=PANEL, fg=FG, font=("Courier", 10), justify="left", anchor="nw")
        self.detail.pack(anchor="w", padx=16, fill="x")
        tk.Label(right, text="Write dark ink on plain paper, one line, big letters,\nhold it steady inside the frame.", bg=PANEL, fg=DIM, font=("Helvetica", 9), justify="left").pack(side="bottom", anchor="w", padx=16, pady=12)

    def _btn(self, parent, text, cmd, col="#262a36"):
        b = tk.Button(parent, text=text, command=cmd, bg=col, fg=FG, activebackground=ACCENT, activeforeground=FG,
                      relief="flat", bd=0, padx=12, pady=6, highlightthickness=0)
        b.pack(side="left", padx=(0, 8)); return b

    # ---------- camera ----------
    def open_camera(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # sharper strokes
        self.tracker.reset()
        self.status.config(text=f"camera {self.cam_index}" if self.cap.isOpened() else f"camera {self.cam_index} not available - use 'Open image…'")

    def next_camera(self):
        self.cam_index = (self.cam_index + 1) % 4; self.open_camera()

    def _redo(self):
        self.dirty = True

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self.btn_freeze.config(text="Resume" if self.frozen else "Freeze")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        img = cv2.imread(path) if path else None
        if img is not None:
            self.current = cv2.resize(img, (VIDEO_W, int(img.shape[0] * VIDEO_W / img.shape[1])))
            self.frozen = self.dirty = True
            self.btn_freeze.config(text="Resume")

    def save_snapshot(self):
        if self.shown is None: return
        os.makedirs(os.path.join(ROOT, "captures"), exist_ok=True)
        p = os.path.join(ROOT, "captures", time.strftime("%Y%m%d_%H%M%S.png"))
        cv2.imwrite(p, self.shown); self.status.config(text=f"saved {os.path.relpath(p, ROOT)}")

    # ---------- main loop ----------
    def tick(self):
        if not self.frozen and self.cap and self.cap.isOpened():
            ok, f = self.cap.read()
            if ok: self.current = f
        if getattr(self, "current", None) is not None and (not self.frozen or self.dirty):
            self.dirty = False
            self.last_result = recognise(self.current, self.model, self.names, self.context.get(),
                                         None if self.frozen else self.tracker)
            self.shown = self.draw(self.current, self.last_result)
            self.update_panel(self.last_result)
            h, w = self.shown.shape[:2]
            img = cv2.cvtColor(cv2.resize(self.shown, (VIDEO_W, round(h * VIDEO_W / w))), cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.video.config(image=self.photo, width=img.shape[1], height=img.shape[0])
        self.root.after(150, self.tick)

    def draw(self, frame, lines):
        out = frame.copy()
        for line in lines:
            for c in line:
                x, y, w, h = c["box"]; col = bgr(LEVEL_COL[c["level"]])
                k = max(1.0, out.shape[1] / 640)
                cv2.rectangle(out, (x, y), (x + w, y + h), col, round(2 * k))
                cv2.putText(out, c["char"], (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, col, round(2 * k), cv2.LINE_AA)
        return out

    def update_panel(self, lines):
        chars = [c for l in lines for c in l]
        self.text.config(text=to_text(lines) if chars else "—")
        avg = sum(c["reliability"] for c in chars) / len(chars) if chars else 0
        self.rel_canvas.delete("all")
        lvl = "HIGH" if avg >= 65 else "MEDIUM" if avg >= 35 else "LOW"
        self.rel_canvas.create_rectangle(0, 0, 340 * avg / 100, 18, fill=LEVEL_COL[lvl], width=0)
        self.rel_label.config(text=f"{avg:.0f}%  {lvl}" if chars else "no characters found")
        rows = [f"{c['char']}  " + "  ".join(f"{n}:{m:.2f}" for n, m in c["top3"]) + f"  [{c['level'][0]}]" for c in chars[:14]]
        self.detail.config(text="\n".join(rows))


if __name__ == "__main__":
    root = tk.Tk()
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    App(root, int(arg)) if arg.isdigit() else App(root, image=arg)
    root.mainloop()
