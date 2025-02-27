import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="RecycAI - Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# Title and description
st.title("RecycAI")
st.subheader("Smart Waste Classification System")
st.write("Upload an image to classify waste as Organic or Recyclable")

# Define the prediction function
def getPrediction(img):
    # Create a temporary directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    
    # Save the uploaded image temporarily
    img_path = os.path.join('temp', 'uploaded_image.jpg')
    img.save(img_path)
    
    # Load the model
    model = tf.keras.models.load_model("final_model_weights.hdf5")
    
    # Preprocess the image
    img = load_img(img_path, target_size=(180, 180))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction
    probability = model.predict(img_array)
    prediction = np.argmax(probability, axis=1)[0]
    
    # Process result
    category = "Recycle" if prediction == 1 else "Organic"
    probability_result = probability[0][prediction]
    
    return category, probability_result, img_path

# File uploader
st.markdown("### Upload an image of waste")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Add a predict button
    if st.button('Classify Waste'):
        with st.spinner('Classifying...'):
            # Make prediction
            category, confidence, _ = getPrediction(image)
            confidence_percentage = float(confidence) * 100
            
            # Display result
            st.markdown("## Classification Result:")
            if category == "Organic":
                st.success(f"🌱 **ORGANIC WASTE**\nConfidence: {confidence_percentage:.2f}%")
                st.markdown("#### Disposal Tips for Organic Was
