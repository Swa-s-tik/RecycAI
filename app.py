import numpy as np
import keras
import streamlit as st
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import gdown

# Define file ID and destination path
file_id = "1QdRP8z77dO2MhK2IKOrotjmrplGPge94"
output_path = "slowfast_model.hdf5"

# Download the file from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

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
    # Create a temporary directory to save the uploaded image if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    # Save the uploaded image temporarily
    img_path = os.path.join('temp', 'uploaded_image.jpg')
    img.save(img_path)
    
    # Load the model
    model = tf.keras.models.load_model(r"https://drive.google.com/uc?id=1QdRP8z77dO2MhK2IKOrotjmrplGPge94")
    
    # Preprocess the image
    img = load_img(img_path, target_size=(180, 180))
    img_array = img_to_array(img)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction
    probability = model.predict(img_array)
    
    # Handle the case for newer TF versions where predict_classes is deprecated
    try:
        category = model.predict_classes(img_array)
        answer = category[0]
    except AttributeError:
        # For newer TensorFlow versions
        prediction = np.argmax(probability, axis=1)
        answer = prediction[0]
    
    # Process result
    if answer == 1:
        category = "Recycle"
        probability_result = probability[0][1]
    else:
        category = "Organic"
        probability_result = probability[0][0]
    
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
            
            # Display result with a colored box based on the prediction
            st.markdown("## Classification Result:")
            
            result_container = st.container()
            with result_container:
                if category == "Organic":
                    st.success(f"🌱 **ORGANIC WASTE**\nConfidence: {confidence_percentage:.2f}%")
                    
                    st.markdown("#### Disposal Tips for Organic Waste:")
                    st.markdown("- Compost if possible")
                    st.markdown("- Use in garden as fertilizer")
                    st.markdown("- Dispose in the organic waste bin")
                    
                else:
                    st.info(f"♻️ **RECYCLABLE WASTE**\nConfidence: {confidence_percentage:.2f}%")
                    
                    st.markdown("#### Disposal Tips for Recyclable Waste:")
                    st.markdown("- Clean before recycling")
                    st.markdown("- Check local recycling guidelines")
                    st.markdown("- Separate different materials if required")
            
            # Display confidence visualization
            st.markdown("### Confidence Level")
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Confidence'], [confidence_percentage], 
                    color='green' if category == "Organic" else 'blue')
            ax.set_xlim(0, 100)
            for i, v in enumerate([confidence_percentage]):
                ax.text(v + 3, i, f"{v:.2f}%", va='center')
            plt.tight_layout()
            st.pyplot(fig)

# Add information section
with st.expander("About RecycAI"):
    st.write("""
    RecycAI is an application that uses machine learning to classify waste as either Organic or Recyclable.
    
    - **Organic waste** includes food scraps, yard trimmings, and other biodegradable materials.
    - **Recyclable waste** includes paper, plastic, glass, and metal items that can be processed and reused.
    
    The model was trained on a dataset of waste images and can help you determine the proper disposal method for your waste.
    """)

# Add sidebar with additional information
st.sidebar.image("logo.png", width=100)
st.sidebar.title("RecycAI")
st.sidebar.markdown("---")
st.sidebar.markdown("### How to use:")
st.sidebar.markdown("1. Upload an image of waste")
st.sidebar.markdown("2. Click 'Classify Waste'")
st.sidebar.markdown("3. View the classification result")
st.sidebar.markdown("---")
st.sidebar.markdown("### Accuracy")
st.sidebar.write("The model provides a confidence score with each prediction. Higher confidence scores indicate more reliable predictions.")
st.sidebar.markdown("---")
st.sidebar.markdown("Project developed as part of UCF Project 03")

# Footer
st.markdown("---")
st.caption("© 2025 RecycAI | UCF Project 03")