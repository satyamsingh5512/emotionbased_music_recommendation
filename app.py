import cv2
import os
import sys
import subprocess
import numpy as np
from tensorflow.keras.models import load_model
import random


def play_song(path):
    """Open a local audio file with the OS default media player."""
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to load songs
def load_songs(filename):
    with open(filename, 'r') as file:
        songs = file.readlines()
    return [song.strip() for song in songs]

# Song dictionary
songs = {
    'Happy': load_songs('songs/happy.txt'),
    'Sad': load_songs('songs/sad.txt'),
    'Angry': load_songs('songs/angry.txt'),
    'Neutral': load_songs('songs/neutral.txt'),
    'Surprise': load_songs('songs/surprise.txt')
}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

# Keep the capture buffer small so we always grab the latest frame
# (prevents lag building up over time).
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

song_opened = False

# Only run the (expensive) emotion model every N frames instead of every frame.
PREDICT_EVERY = 5
frame_count = 0
last_emotion = None


def predict_emotion(roi):
    """Fast single-image inference.

    Using model(x, training=False) instead of model.predict() avoids the
    per-call overhead and graph retracing that make predict() slow down
    progressively when called in a loop.
    """
    roi = cv2.resize(roi, (48, 48)) / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1)).astype("float32")
    prediction = model(roi, training=False)
    return emotion_labels[int(np.argmax(prediction))]


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        # Run the model only on selected frames; reuse the last result otherwise.
        if frame_count % PREDICT_EVERY == 0 or last_emotion is None:
            last_emotion = predict_emotion(roi_gray)

        emotion = last_emotion

        # Display emotion
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Recommend song
        if not song_opened:
            song = random.choice(songs[emotion])

            print(f"Detected Emotion: {emotion}")
            print(f"Recommended Song: {song}")

            play_song(song)

            song_opened = True

    cv2.imshow(
        "Emotion Music Recommendation System",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()