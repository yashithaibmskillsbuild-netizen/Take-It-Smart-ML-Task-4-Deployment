import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

url = "https://drive.google.com/uc?id=1qU7yjbftkmyh6gvolJ20UTXyfJTqYAiY"

model_path = "model.keras"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

class_names = ["Early Blight", "Late Blight", "Healthy"]

st.title("Potato Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    st.image(image, caption="Uploaded Image")

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
