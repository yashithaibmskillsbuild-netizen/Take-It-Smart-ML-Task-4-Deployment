import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

url = "https://drive.google.com/file/d/1Q8Eb24NQN53reQxOlONZwp3sFXoGlp0F/view?usp=drive_link"
model_path = "model.keras"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

class_names = ["Early Blight", "Late Blight", "Healthy"]

st.title("Potato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    st.write(result)
