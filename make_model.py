"""
Generate a ready-to-load emotion_model.h5 so app.py runs immediately.

This builds the SAME CNN architecture used by train_model.py (48x48 grayscale
input, 5 softmax outputs in the order Angry, Happy, Neutral, Sad, Surprise)
and saves it without training.

NOTE: the weights are randomly initialized, so the *reported* emotion will be
arbitrary. This only unblocks the end-to-end pipeline (face detection ->
emotion -> song recommendation). For accurate emotions, run train_model.py.
"""

from train_model import build_model

OUTPUT = "emotion_model.h5"

model = build_model()
model.save(OUTPUT)
print(f"Wrote {OUTPUT} (untrained). app.py will now load and run.")
