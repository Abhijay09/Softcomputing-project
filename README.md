# Softcomputing-project

Handwritten letter and digit detection with fuzzy logic and a small neural network. Made for INT3141 Soft Computing Techniques.

You point a webcam at some handwriting and the program reads it out live. It finds each character in the frame,
decides which character it is, and gives a confidence (low / medium / high) for every one.

## Running it

You need Python 3 with tkinter.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python src/app.py

`python src/app.py 1` uses camera 1 instead of 0, and `python src/app.py some_photo.png` reads a picture instead of the camera.
Works best with dark letters on white paper and decent light. The "Context rules" box turns the digit/letter rule on and off.

The trained models are already in `models/`. To train again, download EMNIST from
https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip, put the `emnist-balanced-*` files in `data/gzip/` and run

    python src/train_mlp.py      # neural network, about 6 minutes
    python src/evaluate_mlp.py   # final numbers

## How it works

1. Camera frame -> local threshold -> connected components. Broken pieces (dot of an i, two bars of =) are merged,
   and anything touching the frame edge is ignored.
2. Each character is cropped, the pen thickness is matched to the training data, and it is resized to 28x28.
3. A multilayer perceptron (784-512-256-47, backpropagation + Adam, written in NumPy) gives a membership for each of the 47 classes.
   It was trained on augmented EMNIST (tilt, slant, size, thin/thick pen, blur, noise) so it copes with camera photos.
4. Fuzzy context rule: if the other characters in a word look like digits, digits become more likely (and the same for letters).
   This fixes things like O/0 and I/1 inside numbers.
5. Memberships are averaged over a few video frames so the text does not flicker.
6. A small Mamdani controller looks at how strong the winning membership is and how far ahead it is of the runner-up,
   and outputs a reliability percentage.

## Results

| | top-1 | top-3 |
|---|---|---|
| fuzzy rule classifier (first version, `src/fuzzy_model.py`) | 66.3% | 86.8% |
| neural network (current) | 87.2% | 98.7% |

`src/benchmark.py` makes fake camera photos (uneven light, blur, noise, tilt, thin and thick pens) from EMNIST test characters
and runs the whole pipeline: the first version reads 60.0% of the characters right, the current one 83.2%.
Reliability groups LOW / MEDIUM / HIGH are right 46% / 70% / 98% of the time.
The context rule adds about 5 points on words that are all digits or all letters (`src/benchmark_context.py`).

## Known problems

- No cursive, and characters that touch each other are not split.
- Some upper/lower case pairs (C/c, O/o, S/s ...) are one class in EMNIST so the app can't tell them apart.
- The benchmark is fake photos. Real handwriting in your room will be a bit worse.

## Newer bits

- Drag a rectangle on the video to read only that area (right-click or "Clear area" resets).
- "Read as: digits / letters / auto" - pick digits for numbers like roll numbers, it removes most 0/O, 1/I, 2/Z mix-ups.
- Works on light strokes on a dark screen too (the polarity is detected automatically).
- "Read with AI" runs a pretrained handwriting model (TrOCR) on the page line by line. Optional and slow, it needs
  `pip install torch transformers tiktoken sentencepiece protobuf` and downloads about 1.3 GB the first time. Small or blurry
  writing comes out badly, it works best on big clear lines.
- `src/benchmark_print.py` makes printed-text scenes. `src/train_mlp.py` can also train on printed letters from system fonts
  (that run was not finished, the model in `models/` is the handwriting-only one).
