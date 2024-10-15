import streamlit as st
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("HAM10000_100epochs.h5")
    return model

# Function to make predictions
def getPrediction(image, model):
    classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 
               'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
    le = LabelEncoder()
    le.fit(classes)
    
    SIZE = 32  # Resize to the same size as training images
    img = np.asarray(image.resize((SIZE, SIZE)))
    img = img / 255.  # Normalize image pixels
    
    img = np.expand_dims(img, axis=0)  # Prepare for model input
    pred = model.predict(img)  # Make prediction
    
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    return pred_class

# Streamlit app interface
st.title('Skin Cancer Classification')

# Instructions
st.write("""
The 7 classes of skin cancer lesions included in this dataset are:
- Melanocytic nevi (nv)
- Melanoma (mel)
- Benign keratosis-like lesions (bkl)
- Basal cell carcinoma (bcc)
- Actinic keratoses (akiec)
- Vascular lesions (vas)
- Dermatofibroma (df)
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Load model
    model = load_trained_model()
    
    # Make prediction
    label = getPrediction(image, model)
    
    # Display the result
    st.write(f"Diagnosis is: {label}")
