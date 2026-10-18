# Softcomputing-project

Handwritten letter and digit detection using fuzzy logic. Made for INT3141 Soft Computing Techniques.

You point a webcam at some handwriting and the program reads it out live. It finds each character in the frame,
turns it into a list of numbers, and checks it against a set of fuzzy rules to decide which character it is.
It also gives a confidence (low / medium / high) for every character.

## Running it

You need Python 3 with tkinter.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python src/app.py

`python src/app.py 1` uses camera 1 instead of 0, and `python src/app.py some_photo.png` reads a picture instead of the camera.
Works best with big dark letters on white paper and decent light.

The trained model is already in `models/`. To train again, download EMNIST from
https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip, put the `emnist-balanced-*` files in `data/gzip/` and run

    python src/train.py
    python src/evaluate.py

## How it works

1. Camera frame -> adaptive threshold -> connected components, one per character.
2. Each character is resized to 28x28 and described with 177 numbers (ink in each zone + stroke directions).
3. 188 fuzzy rules (4 per class). Each rule says "feature 1 is about c1 AND feature 2 is about c2 ...", using gaussian
   membership functions. The rule centres come from fuzzy c-means on the training data.
4. The class with the highest membership wins.
5. A small Mamdani controller looks at how strong the winning membership is and how far ahead it is of the runner-up,
   and outputs a reliability percentage.

Trained on EMNIST balanced (47 classes). On its test set it gets about 66% top-1 and 87% top-3. Most mistakes are look-alikes
like O/0, 1/I/L and 9/q/g. Details and the maths are in `report/`.

## Known problems

- No cursive, and characters that touch each other are not split.
- Some upper/lower case pairs (C/c, O/o, S/s ...) are one class in EMNIST so the app can't tell them apart.
- Real camera images are messier than EMNIST so expect lower accuracy than the numbers above.
