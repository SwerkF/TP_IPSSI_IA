import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle
import joblib

# Fonction pour charger un modèle TensorFlow ou un modèle MLP (fichier .h5 ou .pkl)
@st.cache_resource
def load_selected_model(model_path):
    try:
        if model_path.endswith('.h5'):
            # Charger un modèle TensorFlow
            model = models.load_model(model_path)
            return model, "tensorflow"
        elif model_path.endswith('.pkl'):
            # Charger un modèle scikit-learn (pkl)
            with open(model_path, 'rb') as file:
                try:
                    model = pickle.load(file)
                except Exception:
                    model = joblib.load(model_path)  # Charger avec joblib si pickle échoue
            return model, "mlp"
        else:
            st.error("Format de modèle non pris en charge.")
            st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

# Fonction pour prédire avec le modèle sélectionné
def predict_image(file, model, model_type):
    if model_type == "tensorflow":
        # Chargement et redimensionnement pour les CNN
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
        prediction = model.predict(img_array)
        return "Malignant" if prediction[0][0] > 0.5 else "Benign"
    elif model_type == "mlp":
        # Chargement et redimensionnement pour les MLP
        img = image.load_img(file, target_size=(50, 50))  # Taille adaptée au MLP
        img_array = image.img_to_array(img)
        img_flat = img_array.flatten().reshape(1, -1)  # Aplatir pour scikit-learn
        prediction = model.predict(img_flat)
        return "Malignant" if prediction[0] == 1 else "Benign"


def model_page():
    # Interface Streamlit Models
    st.title("Analyse d'images : Bénain ou Malin")
    st.write("Téléchargez une image et sélectionnez un modèle pour analyser l'image.")

    # Liste des modèles disponibles dans le dossier `data/saved_models`
    model_dir = './data/saved_models'
    models_available = [f for f in os.listdir(model_dir) if f.endswith('.h5') or f.endswith('.pkl')]

    if not models_available:
        st.error("Aucun modèle disponible dans le dossier 'data/saved_models'. Veuillez ajouter des fichiers .h5 ou .pkl.")
        st.stop()

    # Sélection du modèle
    selected_model = st.selectbox("Sélectionnez un modèle :", models_available)
    model_path = os.path.join(model_dir, selected_model)

    # Charger le modèle sélectionné
    model, model_type = load_selected_model(model_path)
    st.success(f"Modèle chargé : {selected_model} (Type : {model_type})")

    # Upload d'une image
    uploaded_files = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Analyse de l'image : {file.name}")
            st.image(file, caption="Image à analyser", use_column_width=True)

            # Prédiction avec le modèle sélectionné
            prediction = predict_image(file, model, model_type)
            st.success(f"Prédiction : {prediction}")

def about_page():
    st.title("À propos")
    st.write("Cette application a été développée dans le cadre du TP IPSSI IA.")
    st.write("Elle permet d'analyser des images médicales pour détecter des tumeurs bénignes ou malignes.")
    st.write("Elle utilise des modèles de Machine Learning (MLP) et de Deep Learning (CNN) pour effectuer les prédictions.")
    st.write("Développée par : [Prénom Nom](https://www.linkedin.com/in/username/)")

curr_page = "Models"

def change_page(page):
    if page == "Models":
        model_page()
    elif page == "About":
        about_page()

st.sidebar.title("Navigation")
st.sidebar.button("Modèles", on_click=lambda: change_page("Models"))
st.sidebar.button("À propos", on_click=lambda: change_page("About"))






