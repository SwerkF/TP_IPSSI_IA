import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle
import joblib
import torch
import torch.nn as nn

# Define your PyTorch model architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Passe à vide pour calculer la taille d'entrée pour la couche dense
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            output_size = self.features(dummy_input).view(-1).shape[0]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Function to load a TensorFlow or MLP model (.keras or .pkl file)
@st.cache_resource
def load_selected_model(model_path):
    try:
        if model_path.endswith('.keras'):
            # Load a TensorFlow model
            model = models.load_model(model_path)
            return model, "tensorflow"
        elif model_path.endswith('.pkl'):
            # Load a scikit-learn model (pkl)
            with open(model_path, 'rb') as file:
                try:
                    model = pickle.load(file)
                except Exception:
                    model = joblib.load(model_path)  # Load with joblib if pickle fails
            return model, "mlp"
        elif model_path.endswith('.pth'):
            # Load a PyTorch model
            model = CNNModel()
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            return model, "pytorch"
        else:
            st.error("Unsupported model format.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Function to predict with the selected model
def predict_image(file, model, model_type):
    if model_type == "tensorflow":
        # Load and resize for CNNs
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
        prediction = model.predict(img_array)
        return "Malignant" if prediction[0][0] > 0.5 else "Benign"
    elif model_type == "mlp":
        # Load and resize for MLPs
        img = image.load_img(file, target_size=(50, 50))  # Size suitable for MLP
        img_array = image.img_to_array(img)
        img_flat = img_array.flatten().reshape(1, -1)  # Flatten for scikit-learn
        prediction = model.predict(img_flat)
        return "Malignant" if prediction[0] == 1 else "Benign"
    elif model_type == "pytorch":
        # Load and resize for PyTorch CNNs
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_tensor = torch.Tensor(img_array).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            prediction = model(img_tensor)
        return "Malignant" if prediction.item() > 0.5 else "Benign"

# Streamlit interface
st.title("Image Analysis: Benign or Malignant")
st.write("Upload an image and select a model to analyze the image.")

# List of available models in the `data/saved_models` folder
model_dir = './saved_models'
models_available = [f for f in os.listdir(model_dir) if f.endswith('.keras') or f.endswith('.pkl') or f.endswith('.pth')]

if not models_available:
    st.error("No models available. Please save a model first.")
    st.stop()

# Model selection
selected_model = st.selectbox("Select a model:", models_available)
model_path = os.path.join(model_dir, selected_model)

# Load the selected model
model, model_type = load_selected_model(model_path)
st.success(f"Model loaded: {selected_model} (Type: {model_type})")

# Image upload
uploaded_files = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Analyzing image: {file.name}")
        st.image(file, caption="Image to analyze", use_column_width=True)

        # Prediction with the selected model
        prediction = predict_image(file, model, model_type)
        st.success(f"Prediction: {prediction}")