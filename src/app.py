"""Fuzzy Handwriting Reader - Tkinter app. Point a camera at handwriting; it reads it live."""
import os, sys, time, threading
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(__file__))
from data import label_chars
from classifier import CharClassifier
from segment import recognise, to_text, Tracker
import demo

ROOT = os.path.join(os.path.dirname(__file__), "..")
BG, PANEL, FG, DIM, ACCENT = "#0f1117", "#181b24", "#e8eaf0", "#8a90a2", "#7c5cff"
LEVEL_COL = {"HIGH": "#3ddc97", "MEDIUM": "#ffb454", "LOW": "#ff5c7a"}   # also used as BGR below
VIDEO_W, VIDEO_H = 640, 480        # size of the picture shown in the window


def bgr(hexcol):
    return tuple(int(hexcol[i:i + 2], 16) for i in (5, 3, 1))


class App:
    def __init__(self, root, cam_index=0, image=None, demo_mode=False):
        self.root = root
        root.title("Fuzzy Handwriting Reader" + ("  [DEMO MODE - simulated output]" if demo_mode else ""))
        root.configure(bg=BG)
        self.demo_mode = demo_mode
        self.model = CharClassifier()
        self.tracker = Tracker()
        self.context = tk.BooleanVar(value=True)
        self.mode = tk.StringVar(value="auto")
        self.names = label_chars()
        self.cam_index, self.cap, self.frozen, self.dirty, self.current, self.shown = cam_index, None, False, False, None, None
        self.ai_mode, self.ai_lines, self.ai_busy = False, [], False
        self.roi, self.drag, self.scale = None, None, 1.0          # area to read (frame pixels), mouse drag, frame px per shown px
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
        self.video = tk.Label(left, bg="black", width=VIDEO_W, height=VIDEO_H, cursor="crosshair"); self.video.pack()
        self.video.bind("<ButtonPress-1>", self.drag_start); self.video.bind("<B1-Motion>", self.drag_move)
        self.video.bind("<ButtonRelease-1>", self.drag_end); self.video.bind("<Button-3>", lambda e: self.clear_area())
        bar = tk.Frame(left, bg=BG); bar.pack(fill="x", pady=10)
        self._btn(bar, "Read with AI", self.read_ai, "#1f9d6b")
        self.btn_freeze = self._btn(bar, "Freeze", self.toggle_freeze, ACCENT)
        self._btn(bar, "Save snapshot", self.save_snapshot)
        self._btn(bar, "Open image…", self.open_image)
        self._btn(bar, "Clear area", self.clear_area)
        self._btn(bar, "Switch camera", self.next_camera)
        tk.OptionMenu(bar, self.mode, "auto", "digits", "letters", command=lambda _: self._redo()).pack(side="left", padx=(8, 0))
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
        tk.Label(right, text="Drag on the video to read only that area (right-click clears).\nDark ink on plain paper works best. ? = not sure.", bg=PANEL, fg=DIM, font=("Helvetica", 9), justify="left").pack(side="bottom", anchor="w", padx=16, pady=12)

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

    # ---------- choose the area to read ----------
    def drag_start(self, e):
        self.drag = [e.x, e.y, e.x, e.y]

    def drag_move(self, e):
        if self.drag:
            self.drag[2:] = [e.x, e.y]; self.dirty = True

    def drag_end(self, e):
        if not self.drag: return
        x0, y0, x1, y1 = self.drag; self.drag = None
        if abs(x1 - x0) > 25 and abs(y1 - y0) > 25 and self.current is not None:
            k, (h, w) = self.scale, self.current.shape[:2]
            self.roi = (max(0, int(min(x0, x1) * k)), max(0, int(min(y0, y1) * k)), min(w, int(max(x0, x1) * k)), min(h, int(max(y0, y1) * k)))
            self.tracker.reset()
        self.dirty = True

    def clear_area(self):
        self.roi = None; self.drag = None; self.tracker.reset(); self.dirty = True

    def _redo(self):
        self.dirty = True

    def read_ai(self):
        """Pretrained handwriting model (TrOCR) reads the whole page / selected area, line by line."""
        if self.current is None or self.ai_busy: return
        self.ai_busy, self.frozen = True, True
        self.btn_freeze.config(text="Resume")
        frame, roi = self.current.copy(), self.roi
        ox, oy = (roi[0], roi[1]) if roi else (0, 0)
        crop = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame
        self.status.config(text="AI is reading… (first time it downloads the model, give it a minute)")
        def work():
            try:
                import htr
                res = htr.read_page(crop, lambda i, n: self.root.after(0, lambda: self.status.config(text=f"AI reading line {i}/{n}…")))
                self.root.after(0, lambda: self.show_ai([((b[0] + ox, b[1] + oy, b[2], b[3]), t) for b, t in res]))
            except Exception as e:
                self.root.after(0, lambda: self.status.config(text=f"AI reader failed: {e}"))
            self.ai_busy = False
        threading.Thread(target=work, daemon=True).start()

    def show_ai(self, lines):
        self.ai_lines, self.ai_mode, self.dirty = lines, True, True
        self.text.config(text="\n".join(t for _, t in lines) or "(no text found)", font=("Courier", 11), wraplength=345)
        self.detail.config(text="")
        self.rel_label.config(text="read by a pretrained handwriting model (TrOCR)")
        self.rel_canvas.delete("all")
        self.status.config(text=f"AI read {len(lines)} lines")

    def toggle_freeze(self):
        self.ai_mode = False; self.text.config(font=("Courier", 26, "bold"), wraplength=340)
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
            self.last_result = (demo.demo_recognise if self.demo_mode else recognise)(self.current, self.model, self.names, self.context.get(),
                                         None if self.frozen else self.tracker, self.roi, self.mode.get())
            self.shown = self.draw(self.current, self.last_result)
            if self.demo_mode:
                kk = max(1.0, self.shown.shape[1] / 640)
                cv2.rectangle(self.shown, (0, self.shown.shape[0] - round(30 * kk)), (round(330 * kk), self.shown.shape[0]), (0, 0, 0), -1)
                cv2.putText(self.shown, "DEMO - simulated output", (round(8 * kk), self.shown.shape[0] - round(9 * kk)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * kk, (60, 160, 255), round(2 * kk), cv2.LINE_AA)
            self.update_panel(self.last_result)
            h, w = self.shown.shape[:2]
            self.scale = w / VIDEO_W
            img = cv2.cvtColor(cv2.resize(self.shown, (VIDEO_W, round(h * VIDEO_W / w))), cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.video.config(image=self.photo, width=img.shape[1], height=img.shape[0])
        self.root.after(150, self.tick)

    def draw(self, frame, lines):
        out = frame.copy()
        k0 = max(1.0, out.shape[1] / 640)
        if self.roi:
            cv2.rectangle(out, self.roi[:2], self.roi[2:], bgr(ACCENT), round(2 * k0))
        if self.drag:
            d = [int(v * self.scale) for v in self.drag]
            cv2.rectangle(out, (d[0], d[1]), (d[2], d[3]), bgr(ACCENT), round(1 * k0))
        for i, (b, _) in enumerate(self.ai_lines if self.ai_mode else []):
            cv2.rectangle(out, b[:2], (b[0] + b[2], b[1] + b[3]), (60, 160, 255), round(2 * k0))
            cv2.putText(out, str(i + 1), (b[0] + 4, b[1] + 22 * round(k0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k0, (60, 160, 255), round(2 * k0), cv2.LINE_AA)
        for line in ([] if self.ai_mode else lines):
            for c in line:
                x, y, w, h = c["box"]; col = bgr(LEVEL_COL[c["level"]])
                k = max(1.0, out.shape[1] / 640)
                cv2.rectangle(out, (x, y), (x + w, y + h), col, round(2 * k))
                cv2.putText(out, c["char"], (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, col, round(2 * k), cv2.LINE_AA)
        return out

    def update_panel(self, lines):
        if self.ai_mode: return
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
    if arg == "--demo":
        path = os.path.join(ROOT, "docs", "figs", "demo_input.png"); demo.make_demo_image(path)
        App(root, image=path, demo_mode=True)
    else:
        App(root, int(arg)) if arg.isdigit() else App(root, image=arg)
    root.mainloop()
