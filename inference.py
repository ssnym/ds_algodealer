import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("digit_model.h5")

def predict_digit(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28,28))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    return np.argmax(prediction)

digit = predict_digit("digit.png")
print("Predicted Digit:", digit)
