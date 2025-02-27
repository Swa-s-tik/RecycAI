import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import matplotlib.pyplot as plt

def set_background(image_file):
    """
    Sets a background image for the Streamlit web app.
    """
    import base64
    from pathlib import Path

    # Read the image file
    img_path = Path(image_file)
    if not img_path.exists():
        st.error(f"Background image '{image_file}' not found.")
        return

    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    # Custom CSS to set background
    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    
    # Inject the CSS into Streamlit
    st.markdown(bg_style, unsafe_allow_html=True)

# Call the function to set background
set_background("bg.jpg")


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
