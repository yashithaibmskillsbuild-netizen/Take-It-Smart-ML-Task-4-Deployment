import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# -------------------------------
# App Title
# -------------------------------
st.title("🥔 Potato Disease Detection")
st.write("Upload a potato leaf image to detect disease")

# -------------------------------
# Download Model from Google Drive
# -------------------------------

MODEL_PATH = "potato_model.h5"

# Google Drive file ID
FILE_ID = "1Q8Eb24NQN53reQxOlONZwp3sFXoGlp0F"

# Direct download link
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... Please wait ⏳")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------------
# Class Labels
# -------------------------------
class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Preprocess Image
    # -------------------------------
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # -------------------------------
    # Display Result
    # -------------------------------
    st.success(f"✅ Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
