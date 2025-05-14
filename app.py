import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(page_title="🫁 Skin Cancer Detector", layout="centered")
st.title("🩺 Skin lesions Image Classifier")
st.markdown("Upload a Skin lesions image to classify it as benign or malignant.")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("a1_model2.h5")
    return model

model = load_model()

# Use the correct class order from your training
classes = ['benign', 'malignant']

# File upload UI
uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    IMG_SIZE = 256
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Only if you used rescale=1./255 during training
    img = np.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions)
    predicted_class = classes[predicted_index]
    confidence = float(np.max(predictions))

    # Display results
    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    st.success(f"**Class:** `{predicted_class}`")
    st.info(f"**Confidence:** `{confidence:.2f}`")

    # Optional: Show confidence for all classes
    st.markdown("### 📊 Confidence Scores")
    for i, label in enumerate(classes):
        st.write(f"{label}: `{predictions[0][i]:.2f}`")
else:
    st.info("Please upload a histopathology image to classify.")
