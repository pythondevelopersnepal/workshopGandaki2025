import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((28, 28)).convert('L')  # Resize to 28x28 and convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
    return img_array

# Make prediction
def predict_digit(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_class, confidence

def main():
    st.title("MNIST Handwritten Digit Recognition")
    st.write("Upload an image of a handwritten digit (0-9)")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        processed_image = preprocess_image(image)
        prediction, confidence = predict_digit(model, processed_image)

        st.write("## Prediction:")
        st.write(f"The predicted digit is: **{prediction}**")
        st.write(f"Confidence: **{confidence:.4f}**")

if __name__ == "__main__":
    main()