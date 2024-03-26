import os
import pickle
import mediapipe as mp
import cv2

mphands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir in os.listdir(DATADIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        img = cv2.imread(os.path.join(DATADIR, dir, imgpath))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = min(l.x for l in hand_landmarks.landmark)
                y_min = min(l.y for l in hand_landmarks.landmark)
                landmarks_normalized = [(l.x - x_min, l.y - y_min) for l in hand_landmarks.landmark]
                data.append(landmarks_normalized)
                labels.append(dir)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
