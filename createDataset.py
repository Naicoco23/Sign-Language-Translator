import os
import pickle
import mediapipe as mp
import cv2

mpHands = mp.solutions.hands

hands = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)

datasetDirectories = './Dataset'

data = []
labels = []

for dir_ in os.listdir(datasetDirectories):
    for img_path in os.listdir(os.path.join(datasetDirectories, dir_)):
        img = cv2.imread(os.path.join(datasetDirectories, dir_, img_path))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        dataAux = []
        x1 = []
        y1 = []
        results = hands.process(imgRGB)
       
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for i in range(len(hand.landmark)):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    x1.append(x)
                    y1.append(y)
                for i in range(len(hand.landmark)):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    dataAux.append(x - min(x1))
                    dataAux.append(y - min(y1))
            
            data.append(dataAux)
            labels.append(dir_)
        
with open('data.pickle', 'wb') as f:
    pickle.dump({'data' : data, 'labels': labels}, f)
