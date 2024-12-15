import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from footer_utils import load_footer

# Load the model
import os
file_path = os.path.abspath("image_classifier.keras")
print("Loading model from:", file_path)  # Debugging the path
model = tf.keras.models.load_model(file_path)

# Define function to predict
def predict_image(img):
    # Preprocess the image
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get prediction
    prediction = model.predict(img_array)
    return 'Dog' if prediction[0] > 0.5 else 'Cat'

# Streamlit App
st.title('🐶 vs 😺 Image Classifier')
st.markdown('Upload an image of a dog or a cat to predict.')

# Inputs
uploaded_file = st.file_uploader("🔗 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)

    # Display the image
    st.image(img, caption='Uploaded Image', width=150)
    st.write("")
    
    # Prediction
    prediction = predict_image(img)
    st.success(f'☝️ This is a {prediction}.')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Display the footer 
footer_html = load_footer()
st.markdown(footer_html, unsafe_allow_html=True)