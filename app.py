#==================================================================================================#
# Import Libraries
#==================================================================================================#
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import cv2
import gdown
import os

#==================================================================================================#
# Page Config
#==================================================================================================#
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)

#==================================================================================================#
# PreprocessLayer
#==================================================================================================#
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return preprocess_input(inputs)

#==================================================================================================#
# Load Model
#==================================================================================================#
@st.cache_resource
def load_face_mask_model():
    model_path = "face_mask_model.keras"
    if not os.path.exists(model_path):
        with st.spinner("Downloading Model... Please Wait ⏳"):
            gdown.download(
                "https://drive.google.com/uc?id=19i_A_Zzyu9uaECCCUeD1as2nV8eWbrF8",
                model_path,
                quiet=False
            )
    model = load_model(
        model_path,
        custom_objects={"PreprocessLayer": PreprocessLayer}
    )
    return model

model = load_face_mask_model()
class_names = ['WithMask', 'WithoutMask']

#==================================================================================================#
# Face Detection Function
#==================================================================================================#
def detect_and_crop_face(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)
        face_crop = img_array[y:y+h, x:x+w]
        return Image.fromarray(face_crop), True, (x, y, w, h)
    return image, False, None

#==================================================================================================#
# Prediction Function
#==================================================================================================#
def predict(image):
    cropped, face_found, coords = detect_and_crop_face(image)
    if not face_found:
        return None, None, None, None, False, None

    img = cropped.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)
    confidence = float(pred[0][0])

    if confidence >= 0.5:
        label = "WithoutMask"
        emoji = "❌"
        color = "red"
        conf_score = confidence * 100
    else:
        label = "WithMask"
        emoji = "✅"
        color = "green"
        conf_score = (1 - confidence) * 100

    return label, emoji, color, conf_score, True, coords

#==================================================================================================#
# Draw Bounding Box Function
#==================================================================================================#
def draw_bounding_box(image, coords, label, color):
    img_array = np.array(image).copy()
    x, y, w, h = coords
    bgr_color = (0, 255, 0) if color == "green" else (255, 0, 0)
    cv2.rectangle(img_array, (x, y), (x+w, y+h), bgr_color, 2)
    cv2.putText(
        img_array, label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, bgr_color, 2
    )
    return Image.fromarray(img_array)

#==================================================================================================#
# UI Header
#==================================================================================================#
st.title("😷 Face Mask Detector")
st.markdown("ResNet50 Transfer Learning Model")
st.markdown("---")

#==================================================================================================#
# Input Method
#==================================================================================================#
option = st.radio("Choose Input Method:", ["📁 Image Upload", "📷 Camera"])

#==================================================================================================#
# Image Upload
#==================================================================================================#
if option == "📁 Image Upload":
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Predicting..."):
            label, emoji, color, conf_score, face_found, coords = predict(image)

        st.markdown("---")

        if not face_found:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.warning("⚠️ No face detected! Please face the camera directly.")
        else:
            boxed_image = draw_bounding_box(image, coords, label, color)
            st.image(boxed_image, caption="Detected Face", use_container_width=True)
            if color == "green":
                st.success(f"Prediction: {label} {emoji}")
            else:
                st.error(f"Prediction: {label} {emoji}")
            st.metric(label="Confidence Score", value=f"{conf_score:.2f}%")

#==================================================================================================#
# Camera Input
#==================================================================================================#
elif option == "📷 Camera":
    st.info("💡 Tip: Face the camera directly for accurate results!")
    camera_image = st.camera_input("Take a Photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

        with st.spinner("Predicting..."):
            label, emoji, color, conf_score, face_found, coords = predict(image)

        st.markdown("---")

        if not face_found:
            st.warning("⚠️ No face detected! Please face the camera directly.")
        else:
            boxed_image = draw_bounding_box(image, coords, label, color)
            st.image(boxed_image, caption="Detected Face", use_container_width=True)
            if color == "green":
                st.success(f"Prediction: {label} {emoji}")
            else:
                st.error(f"Prediction: {label} {emoji}")
            st.metric(label="Confidence Score", value=f"{conf_score:.2f}%")

#==================================================================================================#
# Footer
#==================================================================================================#
st.markdown("---")
st.caption("Powered by ResNet50 + Streamlit 🚀")