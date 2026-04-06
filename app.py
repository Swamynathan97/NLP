import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model safely
@st.cache_resource
def load_my_model():
    return load_model("potatoes.h5", compile=False)

model = load_my_model()

# ✅ IMPORTANT: Adjusted class order (most likely correct)
# Change this if needed after checking predictions
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Image preprocessing
def preprocess_image(image):
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize((256, 256))  # match training size
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# UI
st.title("🥔 Potato Disease Classification App")
st.write("Upload a potato leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)

    # Debug info (VERY IMPORTANT)
    st.subheader("🔍 Debug Info")
    st.write("Raw Prediction:", prediction)

    for i, prob in enumerate(prediction[0]):
        st.write(f"Class {i} ({class_names[i]}): {prob:.4f}")

    pred_index = np.argmax(prediction)

    predicted_class = class_names[pred_index]
    confidence = float(np.max(prediction))

    # Final Output
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")
