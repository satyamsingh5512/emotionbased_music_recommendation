"""
Train an emotion-recognition CNN that matches what app.py expects.

app.py requires a Keras model with:
  * input  : 48x48 grayscale images, shape (N, 48, 48, 1), pixels scaled 0-1
  * output : 5 softmax classes in EXACTLY this order:
             ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

This script trains on the FER-2013 dataset and saves the result as
"emotion_model.h5" in the project root.

It supports the two common ways FER-2013 is distributed:

  1. CSV format  -> a single file `fer2013.csv` with columns
                    emotion,pixels,Usage
                    (the classic Kaggle "fer2013" download).

  2. Folder format -> image folders, one sub-folder per emotion, e.g.
                    dataset/train/<emotion>/*.jpg
                    dataset/test/<emotion>/*.jpg
                    (the Kaggle "msambare/fer2013" download).

Usage:
    python train_model.py --data fer2013.csv
    python train_model.py --data path/to/dataset_folder
    python train_model.py            # auto-detect in current directory

Run `python train_model.py --help` for all options.
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
# The order here MUST match emotion_labels in app.py.
APP_LABELS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
APP_INDEX = {name: i for i, name in enumerate(APP_LABELS)}

# FER-2013 CSV uses integer codes for 7 emotions:
#   0=Angry 1=Disgust 2=Fear 3=Happy 4=Sad 5=Surprise 6=Neutral
# We keep only the 5 the app uses and map them to the app's order.
FER_CSV_TO_APP = {
    0: APP_INDEX["Angry"],
    3: APP_INDEX["Happy"],
    6: APP_INDEX["Neutral"],
    4: APP_INDEX["Sad"],
    5: APP_INDEX["Surprise"],
    # 1 (Disgust) and 2 (Fear) are intentionally dropped.
}

# FER-2013 folder format uses lowercase emotion names as directory names.
FER_DIR_TO_APP = {
    "angry": APP_INDEX["Angry"],
    "happy": APP_INDEX["Happy"],
    "neutral": APP_INDEX["Neutral"],
    "sad": APP_INDEX["Sad"],
    "surprise": APP_INDEX["Surprise"],
    # "disgust" and "fear" directories are ignored if present.
}

IMG_SIZE = 48
NUM_CLASSES = len(APP_LABELS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_from_csv(csv_path):
    """Load FER-2013 from the classic single-CSV format."""
    import pandas as pd

    print(f"Loading CSV dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"emotion", "pixels"}
    if not required.issubset(df.columns):
        sys.exit(
            f"CSV is missing required columns {required}. "
            f"Found columns: {list(df.columns)}"
        )

    x_train, y_train, x_test, y_test = [], [], [], []
    has_usage = "Usage" in df.columns

    for _, row in df.iterrows():
        code = int(row["emotion"])
        if code not in FER_CSV_TO_APP:
            continue  # skip Disgust / Fear

        label = FER_CSV_TO_APP[code]
        pixels = np.fromstring(row["pixels"], sep=" ", dtype="float32")
        if pixels.size != IMG_SIZE * IMG_SIZE:
            continue
        img = pixels.reshape(IMG_SIZE, IMG_SIZE) / 255.0

        # Use the dataset's own train/test split if available.
        usage = str(row["Usage"]).strip() if has_usage else "Training"
        if usage == "Training":
            x_train.append(img)
            y_train.append(label)
        else:  # PublicTest / PrivateTest
            x_test.append(img)
            y_test.append(label)

    return (
        _finalize(x_train, y_train),
        _finalize(x_test, y_test),
    )


def load_from_folder(root):
    """Load FER-2013 from the image-folder format.

    Expects something like:
        root/train/<emotion>/*.png
        root/test/<emotion>/*.png   (or 'validation' instead of 'test')
    If there is no train/test split, all images are loaded and split later.
    """
    import cv2

    def find_split_dir(*names):
        for n in names:
            p = os.path.join(root, n)
            if os.path.isdir(p):
                return p
        return None

    train_dir = find_split_dir("train", "Training")
    test_dir = find_split_dir("test", "validation", "Validation", "PublicTest")

    def load_dir(directory):
        xs, ys = [], []
        if directory is None:
            return xs, ys
        for emotion_name, label in FER_DIR_TO_APP.items():
            class_dir = os.path.join(directory, emotion_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if img.shape != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                xs.append(img.astype("float32") / 255.0)
                ys.append(label)
        return xs, ys

    if train_dir is None and test_dir is None:
        # No split folders: treat `root` itself as a set of emotion folders.
        print(f"Loading folder dataset (no train/test split): {root}")
        xs, ys = load_dir(root)
        if not xs:
            sys.exit(
                f"No images found under {root}. Expected emotion sub-folders "
                f"like {list(FER_DIR_TO_APP)}."
            )
        return _split(xs, ys)

    print(f"Loading folder dataset: {root}")
    x_train, y_train = load_dir(train_dir)
    x_test, y_test = load_dir(test_dir)

    if not x_train:
        sys.exit(f"No training images found under {train_dir!r}.")
    if not x_test:
        # No usable test split -> carve one out of training data.
        return _split(x_train, y_train)

    return _finalize(x_train, y_train), _finalize(x_test, y_test)


def _finalize(xs, ys):
    """Convert lists to model-ready arrays: (N, 48, 48, 1) and one-hot labels."""
    from tensorflow.keras.utils import to_categorical

    if not xs:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 1), "float32"), np.empty(
            (0, NUM_CLASSES), "float32"
        )
    x = np.array(xs, dtype="float32").reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(np.array(ys), num_classes=NUM_CLASSES)
    return x, y


def _split(xs, ys, test_frac=0.15, seed=42):
    """Random train/test split when the dataset doesn't provide one."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xs))
    cut = int(len(xs) * (1 - test_frac))
    train_idx, test_idx = idx[:cut], idx[cut:]
    xs = np.array(xs, dtype="float32")
    ys = np.array(ys)
    return (
        _finalize(list(xs[train_idx]), list(ys[train_idx])),
        _finalize(list(xs[test_idx]), list(ys[test_idx])),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        BatchNormalization,
        Dropout,
        Flatten,
        Dense,
    )

    model = Sequential(
        [
            Conv2D(64, (3, 3), padding="same", activation="relu",
                   input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Dataset auto-detection
# ---------------------------------------------------------------------------
def autodetect_data():
    # 1. a CSV in the current directory
    for name in ("fer2013.csv", "icml_face_data.csv"):
        if os.path.isfile(name):
            return name
    # 2. a common dataset folder name
    for name in ("dataset", "data", "fer2013", "images", "archive"):
        if os.path.isdir(name):
            return name
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train emotion_model.h5 for the music recommender."
    )
    parser.add_argument(
        "--data",
        help="Path to fer2013.csv or to the dataset folder. "
        "If omitted, the script tries to auto-detect it.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="emotion_model.h5")
    args = parser.parse_args()

    data_path = args.data or autodetect_data()
    if not data_path:
        sys.exit(
            "Could not find a dataset. Download FER-2013 and pass it with\n"
            "    python train_model.py --data fer2013.csv\n"
            "or\n"
            "    python train_model.py --data path/to/dataset_folder"
        )
    if not os.path.exists(data_path):
        sys.exit(f"Data path does not exist: {data_path}")

    # Load data (CSV vs folder).
    if os.path.isfile(data_path) and data_path.lower().endswith(".csv"):
        (x_train, y_train), (x_test, y_test) = load_from_csv(data_path)
    elif os.path.isdir(data_path):
        (x_train, y_train), (x_test, y_test) = load_from_folder(data_path)
    else:
        sys.exit(f"Unsupported data path (need a .csv file or a folder): {data_path}")

    print(f"Train samples: {len(x_train)}   Test samples: {len(x_test)}")
    print(f"Class order  : {APP_LABELS}")
    if len(x_train) == 0:
        sys.exit("No training samples were loaded; check the dataset format.")

    # Data augmentation helps a lot on FER-2013.
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint,
    )

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-6),
        ModelCheckpoint(args.output, monitor="val_accuracy",
                        save_best_only=True),
    ]

    has_val = len(x_test) > 0
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        validation_data=(x_test, y_test) if has_val else None,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ModelCheckpoint already saved the best model, but save again to be safe
    # in case there was no validation split.
    model.save(args.output)
    print(f"\nSaved trained model to: {os.path.abspath(args.output)}")
    print("You can now run:  python app.py")


if __name__ == "__main__":
    main()
