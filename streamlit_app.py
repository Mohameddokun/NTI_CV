import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

st.set_page_config(page_title="Real-Time Face & State Detector", layout="wide")

# Find weights file automatically
POSSIBLE_PATHS = [
    "best.pt",
    os.path.join(os.path.dirname(__file__), "best.pt"),
    r"E:\mohammad\Programming\Pytthon_4_AI\NTI_CV\face_det\Face detection_enhanced\train\weights\best.pt"
]

MODEL_WEIGHTS_PATH = next((p for p in POSSIBLE_PATHS if os.path.exists(p)), "best.pt")

# Cache models without spinner conflict
@st.cache_resource(show_spinner=False)
def load_models():
    # Download MediaPipe face landmarker model if missing
    task_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(task_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(url, task_path)

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    yolo = YOLO(MODEL_WEIGHTS_PATH)
    return landmarker, yolo

detector, yolo_model = load_models()

# Sidebar Controls
st.sidebar.title("⚙️ Detection Controls")
start_cam = st.sidebar.toggle("Start Camera", value=False)

conf_thresh = st.sidebar.slider("YOLO Confidence", 0.1, 1.0, 0.35, 0.05)
angry_thresh = st.sidebar.slider("Angry Sensitivity", 0.10, 0.40, 0.18, 0.01)
smile_thresh = st.sidebar.slider("Smile Sensitivity", 0.15, 0.60, 0.35, 0.02)
blink_thresh = st.sidebar.slider("Blink Sensitivity", 0.20, 0.70, 0.45, 0.02)
show_hud = st.sidebar.checkbox("Show Score HUD", value=True)

st.title("Real-Time Face & State Detection")
frame_window = st.image([])

if start_cam:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while start_cam:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access webcam.")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Facial Expression Analysis via MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        state = "Neutral"
        color = (0, 255, 255) # Cyan (RGB)

        blink_score, smile_score, angry_score, mouth_open = 0.0, 0.0, 0.0, 0.0

        if detection_result.face_blendshapes:
            blendshapes = {b.category_name: b.score for b in detection_result.face_blendshapes[0]}

            blink_score = (blendshapes.get('eyeBlinkLeft', 0) + blendshapes.get('eyeBlinkRight', 0)) / 2.0
            smile_score = (blendshapes.get('mouthSmileLeft', 0) + blendshapes.get('mouthSmileRight', 0)) / 2.0
            mouth_open = blendshapes.get('jawOpen', 0)

            brow_down = (blendshapes.get('browDownLeft', 0) + blendshapes.get('browDownRight', 0)) / 2.0
            squint = (blendshapes.get('eyeSquintLeft', 0) + blendshapes.get('eyeSquintRight', 0)) / 2.0
            frown = (blendshapes.get('mouthFrownLeft', 0) + blendshapes.get('mouthFrownRight', 0)) / 2.0
            nose_sneer = (blendshapes.get('noseSneerLeft', 0) + blendshapes.get('noseSneerRight', 0)) / 2.0

            angry_score = (brow_down * 0.55) + (squint * 0.25) + (frown * 0.10) + (nose_sneer * 0.10)

            if blink_score > blink_thresh:
                state = "Blinking / Eyes Closed"
                color = (0, 120, 255)
            elif angry_score > angry_thresh:
                state = "Angry >:("
                color = (255, 0, 0)
            elif smile_score > smile_thresh:
                state = "Smiling :)"
                color = (0, 255, 0)
            elif mouth_open > 0.35:
                state = "Mouth Open :O"
                color = (255, 165, 0)

        # YOLO Face Bounding Box
        results = yolo_model.predict(rgb_frame, conf=conf_thresh, imgsz=640, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                label = f"State: {state}"
                cv2.rectangle(rgb_frame, (x1, max(0, y1 - 32)), (x1 + len(label) * 14, max(30, y1)), color, -1)
                cv2.putText(rgb_frame, label, (x1 + 6, max(22, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Score HUD
        if show_hud:
            cv2.rectangle(rgb_frame, (10, 10), (320, 130), (30, 30, 30), -1)
            cv2.putText(rgb_frame, f"Angry Score : {angry_score:.2f} (Cutoff: {angry_thresh:.2f})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0) if angry_score > angry_thresh else (220, 220, 220), 1)
            cv2.putText(rgb_frame, f"Blink Score : {blink_score:.2f} (Cutoff: {blink_thresh:.2f})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255) if blink_score > blink_thresh else (220, 220, 220), 1)
            cv2.putText(rgb_frame, f"Smile Score : {smile_score:.2f} (Cutoff: {smile_thresh:.2f})", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if smile_score > smile_thresh else (220, 220, 220), 1)
            cv2.putText(rgb_frame, f"Mouth Open  : {mouth_open:.2f} (Cutoff: 0.35)", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0) if mouth_open > 0.35 else (220, 220, 220), 1)

        frame_window.image(rgb_frame)

    cap.release()
else:
    st.info("👈 Toggle 'Start Camera' in the sidebar to begin.")