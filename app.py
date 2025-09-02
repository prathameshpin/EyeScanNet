# Importing required libaries
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore

st.title("Detect eye disease from retina image")

if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

if not st.session_state.uploaded_img:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:  
        img = load_img(uploaded_file, target_size=(224, 224))
        st.session_state.uploaded_img = img
        st.session_state.prediction_made = False
        st.image(img, caption="Uploaded Image", width = 100)

if st.session_state.uploaded_img:
    if not st.session_state.prediction_made:
        if st.button("Get Prediction"):

            # Converting uploaded image to required format
            img_array = img_to_array(st.session_state.uploaded_img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Loading the pre-trained model
            model = load_model("model.keras")
            categories = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

            # Predicting class of image from the loaded model
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_label = categories[predicted_index]
            predicted_prob = predictions[0][predicted_index]

            st.session_state.prediction_made = predicted_label
            st.rerun()


    if st.session_state.prediction_made:
        st.write(f"Prediction: {st.session_state.prediction_made}") 
        if st.button("Clear Image"):
            st.session_state.uploaded_img = None
            st.session_state.prediction_made = False
            st.rerun()


