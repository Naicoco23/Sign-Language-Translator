import cv2 as cv
import time
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf
import pyautogui
import face_recognition
import copy

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Initialize video capture
cap = cv.VideoCapture('C:\\Users\\Nico\\OneDrive - Bina Nusantara\\Documents\\BINUS\\SMT 4\\Research Methodology in Computer Science\\CodeNew\\RM.mov')

# Set video window size
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', screen_width, screen_height)

# Initialize eye detection model
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model for hand gesture recognition
model = pickle.load(open('model.p', 'rb'))

# Label dictionary for hand gestures
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

# Initialize variables for tracking eye gaze duration and person detection
gaze_start = None
last_eye_seen_time = None
gaze_thresh = 5
person_detected = False
countItem=1

countOutsideBox=0
countInsideBox1=0
countInsideBox0=0
countInsideBox2=0

# Function to detect hands and perform gestures
def detect_hands(frame, model, labels, person_boxs, mouse_coordinate_cur):
    global countItem
    global countOutsideBox
    global countInsideBox1
    global countInsideBox0
    global countInsideBox2
    width, height = pyautogui.size()
    
    mouse_x, mouse_y = mouse_coordinate_cur
    totalbox = len(person_boxs)
    for index, box in enumerate(person_boxs):
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * width, xmax * width,
                                        ymin * height, ymax * height)
        if(left <= mouse_x <= right and top<= mouse_y <= bottom):
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4))
                    data = []
                    x_ = []
                    y_ = []
                    for landmark in hand.landmark:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)
                        data.append(x - min(x_))
                        data.append(y - min(y_))
                    if data:
                        x1 = int(min(x_) * frame.shape[1]) - 10
                        y1 = int(min(y_) * frame.shape[0]) - 10

                        x2 = int(max(x_) * frame.shape[1]) + 10
                        y2 = int(max(y_) * frame.shape[0]) + 1
                        prediction = model.predict([np.asarray(data)])
                        predicted_char = labels[int(prediction[0])]
                        print('Test number : '+ str(countItem))
                        countItem=countItem+1
                        if(x1>=left and y1>=top and x2<=right and y2<=bottom):
                            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                            cv.putText(frame, predicted_char, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3) 
                            if(index==1):
                                countInsideBox1=countInsideBox1+1
                            elif(index==0):
                                countInsideBox0=countInsideBox0+1
                            elif(index==2):
                                countInsideBox2=countInsideBox2+1
                            print('totalBox : '+ str(totalbox))
                        else:
                            print('Outside box!')
                            countOutsideBox=countOutsideBox+1
    print('Box Number 1 : '+ str(countInsideBox0))
    print('Box Number 2 : '+ str(countInsideBox1))
    print('Box Number 3 : '+ str(countInsideBox2))
    print('Outside box : ' +  str(countOutsideBox))
# Detect  Person Box ------------------------------------------------------------------------------------------

# Load the TensorFlow model
model_dir = "C:\\Users\\Nico\\OneDrive - Bina Nusantara\\Documents\\BINUS\\SMT 4\\Research Methodology in Computer Science\\CodeNew\\ssdlite_mobilenet_v2_coco_2018_05_09\\saved_model"
detection_model = tf.saved_model.load(model_dir)
detect_fn = detection_model.signatures['serving_default']

def run_inference(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    return output_dict

def draw_boxes(image, output_dict):
    width, height = pyautogui.size()
    person_box = []
    for i in range(output_dict['num_detections']):
        if output_dict['detection_scores'][i] > 0.5:  # Only consider detections with a confidence > 50%
            class_id = output_dict['detection_classes'][i]
            if class_id == 1:  # Class ID 1 for 'person' in COCO dataset
                box = output_dict['detection_boxes'][i]
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * width, xmax * width,
                                              ymin * height, ymax * height)
                cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                cv.putText(image, 'Person', (int(left), int(top-10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                person_box.append(box)
    return image, person_box

# Eye detection --------------------------------------------------------------------------------------------------------

pyautogui.FAILSAFE=False

def maxAndMin(featCoords, mult=1):
    adj = 10 / mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX) - adj, min(listY) - adj, max(listX) + adj, max(listY) + adj])
    return (maxminList * mult).astype(int), (np.array([sum(listX) / len(listX) - maxminList[0],
                                                        sum(listY) / len(listY) - maxminList[1]]) * mult).astype(int)

def process(im):
    left_eye = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    left_eye = cv.resize(left_eye, dsize=(100, 50))
    return left_eye

def capture_positions(direction, samples=20):
    positions = []
    webcam = cv.VideoCapture(0)
    delay = 0.1  # Adjust this delay as needed
    eye_width = 60  # Define the width of the eye region
    eye_height = 40  # Define the height of the eye region
    margin = 10  # Movement threshold to trigger window update
    last_center = None  # Last position of the eye center

    if not webcam.isOpened():
        print("Cannot open webcam")
        return None

    print(f"Please look {direction}. Capturing positions...")
    input()

    try:
        for _ in range(samples):
            ret, frame = webcam.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv.flip(frame, 1)  # Flip horizontally
            smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=0.15, fx=0.15)
            smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

            feats = face_recognition.face_landmarks(smallframe)
            if len(feats) > 0:
                leBds, _ = maxAndMin(feats[0]['right_eye'], mult=1 / 0.15)
                center_x, center_y = (leBds[0] + leBds[2]) // 2, (leBds[1] + leBds[3]) // 2  # Calculate center

                if last_center is None:
                    last_center = (center_x, center_y)  # Initialize last_center

                # Check if movement is less than the margin
                if abs(center_x - last_center[0]) < margin and abs(center_y - last_center[1]) < margin:
                    center_x, center_y = last_center  # Use last known center
                else:
                    last_center = (center_x, center_y)  # Update last known center

                # Ensure the ROI does not go out of frame bounds
                x1 = max(center_x - (eye_width // 2) + 2, 0)
                y1 = max(center_y - eye_height // 2, 0)
                x2 = min(center_x + (eye_width // 2) + 10, frame.shape[1])
                y2 = min(center_y + eye_height // 2, frame.shape[0])

                left_eye = frame[y1:y2, x1:x2]

                left_eye_gray = process(left_eye)
                _, threshold = cv.threshold(left_eye_gray, 48, 255, cv.THRESH_BINARY_INV)
                contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
                if contours:
                    (x, y, w, h) = cv.boundingRect(contours[0])
                    positions.append((x + w//2, y + h//2))

            time.sleep(delay)  # Small delay to allow for eye movement between captures
    finally:
        webcam.release()

    if positions:
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        return (avg_x, avg_y)
    else:
        return None

def capture_gaze_positions():
    directions = ["top left", "top right", "bottom left", "bottom right"]
    gaze_coordinates = {}
    for direction in directions:
        position = capture_positions(direction)
        if position:
            print(f"Average position for {direction}: {position}")
            gaze_coordinates[direction] = position
        else:
            print(f"Failed to capture positions for {direction}")
    return gaze_coordinates

def create_virtual_screen(gaze_coordinates):
    top_left_x, top_left_y = gaze_coordinates['top left']
    top_right_x, top_right_y = gaze_coordinates['top right']
    bottom_left_x, bottom_left_y = gaze_coordinates['bottom left']
    bottom_right_x, bottom_right_y = gaze_coordinates['bottom right']
    #top, bottom, left, right
    screen_coordinate = [0, 0, 0, 0]
    if(top_left_y > top_right_y):
        screen_coordinate[0] = top_right_y
    else:
        screen_coordinate[0] = top_left_y

    if(bottom_left_y > bottom_right_y):
        screen_coordinate[1] = bottom_left_y
    else:
        screen_coordinate[1] = bottom_right_y

    if(top_left_x > bottom_left_x):
        screen_coordinate[2] = bottom_left_x
    else:
        screen_coordinate[2] = top_left_x

    if(top_right_x>bottom_right_x):
        screen_coordinate[3] = top_right_x
    else:
        screen_coordinate[3] = bottom_right_x

    return screen_coordinate
#-------------------------------------------------------------------------------------------------------

def maxAndMin(featCoords, mult=1):
    adj = 10 / mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX) - adj, min(listY) - adj, max(listX) + adj, max(listY) + adj])
    return (maxminList * mult).astype(int), (np.array([sum(listX) / len(listX) - maxminList[0],
                                                        sum(listY) / len(listY) - maxminList[1]]) * mult).astype(int)

def process(im):
    left_eye = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    left_eye = cv.resize(left_eye, dsize=(100, 50))
    return left_eye

def move_mouse(current_eye_center, screen_coordinates):
    screen_width, screen_height = pyautogui.size()
    top, bottom, left, right = screen_coordinates  # Unpack top, bottom, left, and right coordinates
    
    mouse_move_x = screen_width * ((current_eye_center[0] - left) / abs(right - left))
    mouse_move_y = screen_height * ((current_eye_center[1] - top) / abs(bottom - top))
    
    pyautogui.moveTo(mouse_move_x, mouse_move_y)

    return (mouse_move_x, mouse_move_y)

# Main loop -------------------------------------------------------------------------------------------------------------

# Eye Gaze Train
gaze_coordinates = capture_gaze_positions()
screen_coordinate = create_virtual_screen(gaze_coordinates)

webcam = cv.VideoCapture(0)
delay = 0.1  # Adjust this delay as needed
eye_width = 60  # Define the width of the eye region
eye_height = 40  # Define the height of the eye region
margin = 10  # Movement threshold to trigger window update
last_center = None  # Last position of the eye center
mouse_coordinate = None

while True:
    # Eye Detect
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv.flip(frame, 1)  # Flip horizontally
    smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=0.15, fx=0.15)
    smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

    feats = face_recognition.face_landmarks(smallframe)
    if len(feats) > 0:
        leBds, _ = maxAndMin(feats[0]['right_eye'], mult=1 / 0.15)
        center_x, center_y = (leBds[0] + leBds[2]) // 2, (leBds[1] + leBds[3]) // 2  # Calculate center

        if last_center is None:
            last_center = (center_x, center_y)  # Initialize last_center

        # Check if movement is less than the margin
        if abs(center_x - last_center[0]) < margin and abs(center_y - last_center[1]) < margin:
            center_x, center_y = last_center  # Use last known center
        else:
            last_center = (center_x, center_y)  # Update last known center

        # Ensure the ROI does not go out of frame bounds
        x1 = max(center_x - (eye_width // 2) + 2, 0)
        y1 = max(center_y - eye_height // 2, 0)
        x2 = min(center_x + (eye_width // 2) + 10, frame.shape[1])
        y2 = min(center_y + eye_height // 2, frame.shape[0])

        left_eye = frame[y1:y2, x1:x2]

        left_eye_gray = process(left_eye)
        _, threshold = cv.threshold(left_eye_gray, 48, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        rows, cols = threshold.shape

        for cnt in contours:
            (x, y, w, h) = cv.boundingRect(cnt)
            cv.drawContours(threshold, [cnt], -1, (0, 0, 255), 3)
            cv.rectangle(threshold, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.line(threshold, (x+int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv.line(threshold, (0, y+int(h/2)), (cols, y+int(h/2)), (0, 255, 0), 2)
            current_eye_center = (x+w//2, y+h//2)
            mouse_coordinate=move_mouse(current_eye_center, screen_coordinate)
            break

        # Display the thresholded left eye
        cv.imshow('Left Eye', threshold)

    ret, frame = cap.read()
    eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    current_time = time.time()

    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    # Run detection
    output_dict = run_inference(detect_fn, frame)
    
    # Draw bounding boxes
    frame, person_box = draw_boxes(frame, output_dict)

    if len(eyes) > 0:
        if gaze_start is None:
            gaze_start = current_time
        last_eye_seen_time = current_time
        person_detected = (current_time - gaze_start) >= gaze_thresh
    else:
        if last_eye_seen_time is not None and (current_time - last_eye_seen_time) >= 3:
            gaze_start = None
            person_detected = False

    if person_detected:
        cv.putText(frame, "Person detected!!!", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        detect_hands(frame, model, labels, person_box, mouse_coordinate)
    elif gaze_start is not None:
        elapsed_time = int(current_time - gaze_start)
        cv.putText(frame, f"Eye is detected for {elapsed_time} seconds...", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()