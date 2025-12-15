import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Handwritten Digit Recognition")

model = tf.keras.models.load_model("digit_model.h5")

uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28,28))
    st.image(img, caption="Uploaded Image")

    img = np.array(img) / 255.0
    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    st.write("Predicted Digit:", np.argmax(prediction))
