# 🎭 Real-Time Face & Emotion Recognition System
An end-to-end Computer Vision pipeline featuring **YOLOv8** for real-time face localization and a custom **Deep CNN / MobileNetV2** model for 7-class facial emotion classification, wrapped in an interactive **Streamlit Dashboard** with automated CSV logging.

## 📌 Members:
- Ahmed Badawy Mohamed Hamdy
- Mohamed Ahmed Ibrahim Hassan
- Mohamed Ahmed Kamal El din
- Hassan Mohamed Ali
- Mohamed Mahmoud Shahin
- Amr Ayman Abd El-Raouf 

## 📌 Features:
- **Decoupled Two-Stage Architecture**: High-precision face bounding box extraction followed by dedicated emotion classification.
- **7 Discrete Facial Expressions**: Recognizes `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, and `Surprise`.
- **High-Throughput CPU Inference**: Direct tensor execution (`training=False`) bypassing TensorFlow `.predict()` pipeline overhead for steady real-time frame rates (~20–30 FPS).
- **Interactive Streamlit Web UI**: Real-time sensitivity threshold sliders, live preview, dynamic confidence bounding boxes, and tabular logging.
- **Automated Logging & Export**: Throttled 1-second CSV logging to record timestamps, detected individuals, emotion states, and confidence scores.


## 🧠 System Pipeline

[ Live Webcam Feed ]
         │
         ▼
[ Stage 1: YOLOv8n Face Detection ] ────► (Extracts Bounding Boxes)
         │
         ▼
[ Crop & Preprocess Face (48x48x3) ]
         │
         ▼
[ Stage 2: Keras Emotion Classifier ] ──► (Predicts Softmax Probabilities)
         │
         ▼
[ Streamlit Web App / CSV Logger ]
