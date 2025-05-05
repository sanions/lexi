
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os
import math
import json
from datetime import datetime

# ===================== OCR + Pointing Logic =====================

def preprocess_image_for_ocr(image):
    print("preprocessing...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, binary_img = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img

def extract_bounding_boxes(image):
    preprocessed = preprocess_image_for_ocr(image)
    ocr_data = pytesseract.image_to_data(preprocessed, output_type=Output.DICT)

    print("extracting bounding boxes...")
    words = []
    n = len(ocr_data['text'])
    for i in range(n):
        text = ocr_data['text'][i].strip()
        if text == "":
            continue
        x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
        words.append({
            "text": text,
            "bbox": (x, y, x + w, y + h),
            "index": i
        })
    return words

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def do_lines_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def ray_intersects_box(ray_start, ray_end, box):
    (x1, y1, x2, y2) = box
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    return any(do_lines_intersect(ray_start, ray_end, a, b) for (a, b) in edges)

def draw_bounding_boxes(image, words, target_idx, context_indices):
    for word in words:
        x1, y1, x2, y2 = word["bbox"]
        if word["index"] == target_idx:
            color = (255, 0, 0)  # Red
        elif word["index"] in context_indices:
            color = (0, 0, 255)  # Blue
        else:
            color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

def get_target_word_from_pointing(image, index_tip, index_pip, context_window=10):
    words = extract_bounding_boxes(image)
    dx, dy = index_tip[0] - index_pip[0], index_tip[1] - index_pip[1]
    ray_end = (index_tip[0] + 1000 * dx, index_tip[1] + 1000 * dy)

    closest_word = None
    min_dist = float("inf")
    target_idx = None

    for word in words:
        box = word["bbox"]
        if ray_intersects_box(index_tip, ray_end, box):
            box_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            d = distance(index_tip, box_center)
            if d < min_dist:
                min_dist = d
                closest_word = word["text"]
                target_idx = word["index"]

    if closest_word and target_idx is not None:
        start = max(0, target_idx - context_window)
        end = min(len(words), target_idx + context_window + 1)
        context_indices = list(range(start, end))
        context_words = [w["text"] for w in words if w["index"] in context_indices]
        context_string = ' '.join(context_words)
        draw_bounding_boxes(image, words, target_idx, context_indices)
        return closest_word, context_string, image
    else:
        return None, None, image

# ===================== Entry Point =====================

def load_landmark_coordinates(json_path: str) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Get first hand, first and second landmarks (PIP and TIP)
    landmarks = data[0]["landmarks"]
    pip = (landmarks[0]["x"], landmarks[0]["y"])
    tip = (landmarks[1]["x"], landmarks[1]["y"])
    
    return {"pip": pip, "tip": tip}

# def analyze_pointing_frame_from_files(frame_path, landmark_path, context_window=5):
#     frame = cv2.imread(frame_path)
#     if frame is None:
#         raise FileNotFoundError(f"Could not load image at {frame_path}")
    
#     finger_coords = load_landmark_coordinates(landmark_path)
#     tip = finger_coords["tip"]
#     pip = finger_coords["pip"]

#     print("retrieved frame and pointer coordinates")
    
#     return get_target_word_from_pointing(frame, tip, pip, context_window)

# target_word, context = analyze_pointing_frame_from_files(
#     "saved_frames/triggered/frame.png",
#     "saved_frames/triggered/landmarks.json"
# )
# print(target_word, context)

def analyze_and_save(frame_path, landmark_path, save_dir, context_window=5):
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at {frame_path}")

    finger_coords = load_landmark_coordinates(landmark_path)
    tip = finger_coords["tip"]
    pip = finger_coords["pip"]

    target_word, context_string, annotated_image = get_target_word_from_pointing(frame, tip, pip, context_window)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "frame.png"), annotated_image)
    with open(os.path.join(save_dir, "targets.json"), "w") as f:
        json.dump({"target_word": target_word, "context": context_string}, f, indent=2)


analyze_and_save(
    "saved_frames/triggered/frame.png",
    "saved_frames/triggered/landmarks.json",
    "saved_frames/triggered/ocr"
)