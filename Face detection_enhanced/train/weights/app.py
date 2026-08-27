import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# 1. Download official MediaPipe Face Landmarker Task model if missing
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task (~1.4MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# 2. Initialize MediaPipe Face Landmarker with Blendshapes enabled
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 3. Load YOLO model
model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 4. Extract facial blendshape scores
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    state = "Neutral"
    color = (255, 255, 0) # Cyan

    blink_score = 0.0
    smile_score = 0.0
    angry_score = 0.0
    mouth_open = 0.0

    if detection_result.face_blendshapes:
        blendshapes = {
            b.category_name: b.score for b in detection_result.face_blendshapes[0]
        }

        # Individual feature values
        blink_l = blendshapes.get('eyeBlinkLeft', 0)
        blink_r = blendshapes.get('eyeBlinkRight', 0)
        blink_score = (blink_l + blink_r) / 2.0

        smile_l = blendshapes.get('mouthSmileLeft', 0)
        smile_r = blendshapes.get('mouthSmileRight', 0)
        smile_score = (smile_l + smile_r) / 2.0

        mouth_open = blendshapes.get('jawOpen', 0)

        # Multi-factor Anger Calculation
        brow_down = (blendshapes.get('browDownLeft', 0) + blendshapes.get('browDownRight', 0)) / 2.0
        squint = (blendshapes.get('eyeSquintLeft', 0) + blendshapes.get('eyeSquintRight', 0)) / 2.0
        frown = (blendshapes.get('mouthFrownLeft', 0) + blendshapes.get('mouthFrownRight', 0)) / 2.0
        nose_sneer = (blendshapes.get('noseSneerLeft', 0) + blendshapes.get('noseSneerRight', 0)) / 2.0

        # Weighted composite score
        angry_score = (brow_down * 0.55) + (squint * 0.25) + (frown * 0.10) + (nose_sneer * 0.10)

        # Expression State Hierarchy (Tested thresholds)
        if blink_score > 0.45:
            state = "Blinking / Eyes Closed"
            color = (255, 120, 0)  # Blue
        elif angry_score > 0.18:
            state = "Angry >:("
            color = (0, 0, 255)    # Red
        elif smile_score > 0.35:
            state = "Smiling :)"
            color = (0, 255, 0)    # Green
        elif mouth_open > 0.35:
            state = "Mouth Open :O"
            color = (0, 165, 255)  # Orange
        else:
            state = "Neutral"
            color = (255, 255, 0)  # Cyan

    # 5. Detect face bounding box using YOLO
    results = model.predict(frame, conf=0.35, imgsz=640, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw state badge
            label = f"State: {state}"
            cv2.rectangle(frame, (x1, max(0, y1 - 32)), (x1 + len(label) * 14, max(30, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 6, max(22, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 6. Live Debug HUD (Top-Left Corner)
    hud_bg = (30, 30, 30)
    cv2.rectangle(frame, (10, 10), (280, 130), hud_bg, -1)
    cv2.putText(frame, f"Angry Score : {angry_score:.2f} (Target > 0.18)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if angry_score > 0.18 else (200, 200, 200), 1)
    cv2.putText(frame, f"Blink Score : {blink_score:.2f} (Target > 0.45)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0) if blink_score > 0.45 else (200, 200, 200), 1)
    cv2.putText(frame, f"Smile Score : {smile_score:.2f} (Target > 0.35)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if smile_score > 0.35 else (200, 200, 200), 1)
    cv2.putText(frame, f"Mouth Open  : {mouth_open:.2f} (Target > 0.35)", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255) if mouth_open > 0.35 else (200, 200, 200), 1)

    cv2.imshow("YOLOv8 + Real-Time Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()