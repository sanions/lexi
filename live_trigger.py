import cv2
import time
import json
from datetime import datetime
import numpy as np
import threading
import speech_recognition as sr
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from ocr_full import analyze_and_save
from process_target import process_target

# Constants
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
TRIGGER = "lex"

# Shared state
latest_result = None
triggered = False  # Flag set by speech thread

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Hand landmarker setup
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_hand_detection_confidence=0.05,
    min_hand_presence_confidence=0.1, 
    min_tracking_confidence=0.5
)

# Speech recognition thread
def speech_listener():
    global triggered
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Lexi is listening...say \"Hey, Lexi!\" to trigger a save.")
        while True:
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                print("[Speech] You said:", text)
                if TRIGGER in text.lower():
                    print("[Speech] Scene captured!")
                    triggered = True
            except Exception as e:
                print("[Speech Error]", e)

# Start speech thread
speech_thread = threading.Thread(target=speech_listener, daemon=True)
speech_thread.start()

# Start camera and hand tracking
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

landmarker = HandLandmarker.create_from_options(options)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    display_frame = frame

    if latest_result and latest_result.handedness:
        display_frame = draw_landmarks_on_image(frame, latest_result)

    # Save frame if triggered
    if triggered:
        triggered = False
        # timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_filename = f"saved_frames/triggered/frame.png"
        landmarks_filename = f"saved_frames/triggered/landmarks.json"

        print(f"[Main] Saving current frame to {frame_filename}")
        cv2.imwrite(frame_filename, frame)

        height, width, num_channels = frame.shape 

        if latest_result and latest_result.hand_landmarks:
            landmarks_data = []
            for hand_landmarks, handedness in zip(latest_result.hand_landmarks, latest_result.handedness):
                hand_data = {
                    # "handedness": handedness[0].category_name,
                    "landmarks": [
                        {"x": int(lm.x * width), "y": int(lm.y * height)}
                        for ix, lm in enumerate(hand_landmarks) if ix in [6, 8]
                    ]
                }
                landmarks_data.append(hand_data)

            with open(landmarks_filename, "w") as f:
                json.dump(landmarks_data, f, indent=2)

            print(f"[Main] Saved landmarks to {landmarks_filename}")
            print("Analyzing...")
            analyze_and_save(
                frame_path=frame_filename,
                landmark_path=landmarks_filename,
                save_dir="saved_frames/triggered/ocr"
            )
            print("OCR completed. Processing targets.")
            definition, img = process_target("saved_frames/triggered/ocr/targets.json")


        else:
            print("[Main] No landmarks to save.")

    cv2.imshow('Live Feed with Landmarks', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
