import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import matplotlib.pyplot as plt

# ✅ Set page configuration FIRST
st.set_page_config(
    page_title="RecycAI - Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# ✅ Simple background image using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("bg.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Title and description
st.title("RecycAI")
st.subheader("Smart Waste Classification System")
st.write("Upload an image to classify waste as Organic or Recyclable")

# ✅ Define the prediction function
def getPrediction(img):
    os.makedirs('temp', exist_ok=True)  # Ensure temp folder exists
    img_path = os.path.join('temp', 'uploaded_image.jpg')
    img.save(img_path)

    model = tf.keras.models.load_model("final_model_weights.hdf5")

    img = load_img(img_path, target_size=(180, 180))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probability = model.predict(img_array)
    prediction = np.argmax(probability, axis=1)[0]

    category = "Recycle" if prediction == 1 else "Organic"
    probability_result = probability[0][prediction]

    return category, probability_result

# ✅ File uploader
st.markdown("### Upload an image of waste")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify Waste'):
        with st.spinner('Classifying...'):
            category, confidence = getPrediction(image)
            confidence_percentage = float(confidence) * 100

            st.markdown("## Classification Result:")
            if category == "Organic":
                st.success(f"🌱 **ORGANIC WASTE**\nConfidence: {confidence_percentage:.2f}%")
            else:
                st.info(f"♻️ **RECYCLABLE WASTE**\nConfidence: {confidence_percentage:.2f}%")

            # ✅ Confidence Level Bar Chart
            st.markdown("### Confidence Level")
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Confidence'], [confidence_percentage], color='green' if category == "Organic" else 'blue')
            ax.set_xlim(0, 100)
            for i, v in enumerate([confidence_percentage]):
                ax.text(v + 3, i, f"{v:.2f}%", va='center')
            plt.tight_layout()
            st.pyplot(fig)

# ✅ Sidebar Info
st.sidebar.image("logo.png", width=100)
st.sidebar.title("RecycAI")
st.sidebar.markdown("---")
st.sidebar.markdown("### How to use:")
st.sidebar.markdown("1. Upload an image of waste")
st.sidebar.markdown("2. Click 'Classify Waste'")
st.sidebar.markdown("3. View the classification result")
st.sidebar.markdown("---")
st.sidebar.markdown("### Accuracy")
st.sidebar.write("The model provides a confidence score with each prediction.")
st.sidebar.markdown("---")
st.sidebar.markdown("Project developed as part of UCF Project 03")

# ✅ Footer
st.markdown("---")
st.caption("© 2025 RecycAI | UCF Project 03")
