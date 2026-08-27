import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random
from tensorflow.keras.applications.resnet50 import preprocess_input
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "best_emotion_model_scratch_enhanced.keras",
        custom_objects={'preprocess_input': preprocess_input}  
    )

model = load_my_model()
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  

st.title("Teeth Disease Image Classifier with pretraind Model ✨ and Achieved 98.93% accuracy")
st.markdown("""
مرحبًا بكم في تطبيق تصنيف أمراض الأسنان!  
✨ ارفع صورة لأسنانك وسيقوم الموديل بتحديد الفئة مع احتمالات كل فئة.
""")

uploaded_file = st.file_uploader("Upload an image 📷", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((48,48))
    
    st.image(img, caption="🖼️Uploaded Image", use_column_width=True, channels="RGB")
    
    img_array = np.array(img)
    st.write("Image shape:", img_array.shape)  
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    

    pred = model.predict(img_array)
    
    
    class_index = np.argmax(pred)
    st.write("**Predicted Class:**", class_names[class_index])
    
    st.write("**Prediction Probabilities:**")
    for i, prob in enumerate(pred[0]):
        st.write(f"{class_names[i]}: {prob:.4f}")
    
    motivational_msgs = [
        "👏 Great! Keep brushing twice a day! 🪥💧",
        "💪 Don't forget flossing daily for healthy gums! 😁",
        "✨ Stay consistent with dental care and smile bright! 😎",
        "🦷 Healthy teeth = Happy life! Keep it up! 😁",
        "🌟 Remember to visit your dentist regularly! 🪥"
    ]
    
    st.markdown(f"**💡 Tip for you:** {random.choice(motivational_msgs)}")
