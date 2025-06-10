import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('kidney_classifier_model.h5')

# Class names (adjust if needed)
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# --- Streamlit Layout ---
st.set_page_config(page_title="Kidney CT Classifier", layout="centered")

# Sidebar controls
st.sidebar.title("App Settings")
st.sidebar.markdown("Upload a kidney CT scan to classify it.")

st.title("🧠 Kidney CT Image Classifier")
st.caption("Detects: Cyst, Normal, Stone, Tumor")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a CT image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🔍 Predicted: **{predicted_class}**")
    st.markdown(f"🧪 Confidence: `{confidence:.2%}`")

    # Show full probability breakdown
    st.markdown("#### 🔬 All Class Probabilities:")
    prob_table = {cls: f"{prob:.2%}" for cls, prob in zip(class_names, prediction)}
    st.table(prob_table)
