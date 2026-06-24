import cv2 #error fix kare 
import numpy as np
from tensorflow.keras.models import load_model
import random
import webbrowser

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

song_opened = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi_gray = roi_gray / 255.0

        roi_gray = np.reshape(
            roi_gray,
            (1, 48, 48, 1)
        )

        prediction = model.predict(roi_gray)

        max_index = int(np.argmax(prediction))

        emotion = emotion_labels[max_index]

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

            webbrowser.open(song)

            song_opened = True

    cv2.imshow(
        "Emotion Music Recommendation System",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows