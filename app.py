import streamlit as st
from PIL import Image
import os
import tensorflow as tf  # Pour charger les modèles TensorFlow
from tensorflow.keras import models  # Pour charger les modèles .keras
import pickle  # Pour charger les modèles scikit-learn sauvegardés en .pkl
import joblib  # Pour charger les modèles scikit-learn en utilisant joblib
import torch  # Pour charger les modèles PyTorch
import torch.nn as nn  # Pour définir et utiliser le modèle CNN PyTorch
import numpy as np  # Pour travailler avec des matrices d'image

# Définir le modèle CNN pour PyTorch
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
        
        # Calculer la taille d'entrée pour la couche dense
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

# Configuration de la page Streamlit (unique, doit être la première commande)
st.set_page_config(
    page_title="Détection du Cancer de la Peau",
    page_icon="🏥",
    layout="wide"
)

# Barre latérale pour la navigation entre les pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse Exploratoire des Données", "Faire une Prédiction"])

# Liste des modèles disponibles pour la prédiction (pour la page "Faire une Prédiction")
model_dir = './saved_models'
models_available = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.pkl', '.pth'))]

# Page d'accueil
if page == "Accueil":
    st.title("🏥 Prédire un cancer de la peau avec une image")

    # Introduction du projet
    st.markdown("""
    ### Introduction du projet 🏥

    Ce projet a été développé dans le cadre du **cours d'analyse de données pour une IA**, afin de mettre en pratique les concepts de **deep learning** et de **machine learning**. L'objectif est de créer un **outil accessible** qui permet de détecter les signes précoces de **cancer de la peau** à partir d'une simple image de la lésion cutanée. En utilisant des modèles avancés, nous souhaitons non seulement aider à la sensibilisation sur les risques potentiels des lésions cutanées, mais aussi acquérir des compétences en développement de modèles IA appliqués.

    ### Modèles Utilisés
    Pour ce projet, nous avons mis en place et testé plusieurs modèles de réseaux de neurones avancés, parmi lesquels :

    - **VGG16** : Modèle pré-entraîné pour la classification d'images.
    - **ResNet50** : Réseau résidu profond permettant de travailler sur des architectures complexes.
    - **EfficientNetB0** : Modèle moderne, équilibrant précision et efficacité des calculs.
    - **MLP (Perceptron multicouche)** : Pour servir de baseline et comparer avec des modèles plus avancés.
    - **CNN PyTorch** et un **modèle TensorFlow personnalisé** pour plus de flexibilité.

    ### Comment Utiliser l'Application
    Pour utiliser cette application, c'est simple :
    1. **Naviguez vers la page "Faire une Prédiction"** en utilisant la barre latérale.
    2. **Choisissez un modèle** parmi ceux disponibles (par exemple, VGG16, ResNet50).
    3. **Téléchargez une image** de la lésion cutanée que vous souhaitez analyser.
    4. Obtenez une prédiction indiquant si la lésion est probablement **bénigne** ou **maligne**.
    """)

# Page d'analyse exploratoire des données
elif page == "Analyse Exploratoire des Données":
    st.title("📊 Analyse Exploratoire des Données")

    # Introduction à l'analyse exploratoire
    st.markdown("""
    ### Analyse Exploratoire avec un Arbre de Décision

    Ce projet démontre l'utilisation d'un modèle **Arbre de Décision** pour analyser un dataset lié aux maladies de la peau. 
    L'objectif est de trouver la profondeur optimale de l'arbre et d'évaluer ses performances à l'aide de plusieurs métriques et visualisations.
    """)

    # Étapes dans le code
    st.markdown("""
    #### Étapes dans l'Analyse Exploratoire
    1. **Chargement des données** :
        - Le dataset est chargé à partir d'un fichier Excel.
        - La colonne cible est `HadSkinCancer`, qui indique si une personne a eu un cancer de la peau.
    2. **Prétraitement des données** :
        - Les fonctions `preprocess_data` et `clean_data` sont utilisées pour nettoyer et encoder les variables catégoriques.
    3. **Séparation des données** :
        - Les données sont divisées en un ensemble d'entraînement (80%) et un ensemble de test (20%).
    4. **Entraînement et évaluation du modèle** :
        - Une boucle teste des modèles d'Arbre de Décision avec des profondeurs variant de 1 à 20.
        - Pour chaque profondeur, les métriques suivantes sont calculées :
          - Précision sur les données d'entraînement.
          - Précision sur les données de test.
          - Écart de précision (*Accuracy Gap*).
          - Taille de l'arbre.
    """)

    # Affichage des graphiques
    st.markdown("#### Résultats et Visualisations")

    # Graphique 1 : Accuracy Gap Analysis
    st.subheader("Train vs Test Accuracy and Accuracy Gap")
    st.image("data/images/accuracy_gap_analysis.png", caption="Train vs Test Accuracy and Accuracy Gap")

    # Graphique 2 : Arbre de Décision
    st.subheader("Visualisation du Meilleur Arbre")
    st.image("data/images/best_decision_tree_visualization.png",
             caption="Arbre de Décision avec la Meilleure Profondeur")

    # Graphique 3 : Importances des Caractéristiques (Profondeur Optimale)
    st.subheader("Importances des Caractéristiques (Profondeur Optimale)")
    st.image("data/images/feature_importances_best_depth.png",
             caption="Importances des Caractéristiques pour la Meilleure Profondeur")

    # Graphique 4 : Importances des Caractéristiques (Profondeur = 9)
    st.subheader("Importances des Caractéristiques (Profondeur = 9)")
    st.image("data/images/feature_importances_depth_9.png",
             caption="Importances des Caractéristiques pour Profondeur = 9")

    # Graphique 5 : Courbe ROC
    st.subheader("Courbe ROC")
    st.image("data/images/roc_curve.png", caption="Courbe ROC avec AUC")

    # Tableau récapitulatif
    st.subheader("Tableau Récapitulatif des Résultats")
    st.image("data/images/decision_tree_summary_table.png",
             caption="Tableau des Profondeurs, Précisions et Taille des Arbres")

# Page pour faire une prédiction
elif page == "Faire une Prédiction":
    st.title("🔍 Faire une Prédiction")

    # Vérifier s'il y a des modèles disponibles
    if not models_available:
        st.error("⚠️ Aucun modèle disponible. Veuillez ajouter des modèles dans le dossier 'saved_models'.")
        st.stop()

    # Étape 1 : Sélection du modèle
    st.markdown("## 📌 Étape 1 : Sélectionner un modèle")
    st.info("Sélectionnez le modèle que vous souhaitez utiliser pour l'analyse des lésions cutanées.")
    selected_model = st.selectbox(
        "Modèles disponibles :", 
        models_available,
        help="Choisissez un modèle pré-entraîné pour analyser l'image.",
    )

    # Charger le modèle sélectionné
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
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load to CPU
                model.eval()  # Set the model to evaluation mode
                return model, "pytorch"
            else:
                st.error("Unsupported model format.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    model_path = os.path.join(model_dir, selected_model)
    with st.spinner(f"Chargement du modèle {selected_model}..."):
        model, model_type = load_selected_model(model_path)
    st.success(f"Modèle chargé : {selected_model} (Type : {model_type})")

    # Étape 2 : Téléchargement de l'image
    st.markdown("## 📌 Étape 2 : Télécharger une image")
    st.info("Veuillez télécharger une image de la lésion cutanée à analyser. L'image doit être au format **JPEG** ou **PNG**.")
    uploaded_file = st.file_uploader(
        "Choisissez une image de lésion cutanée :", 
        type=["jpg", "jpeg", "png"],
        help="Téléchargez une image de la peau au format JPG, JPEG ou PNG pour effectuer une prédiction.",
    )

    # Fonction de prédiction
    def predict_image(file, model, model_type):
        if model_type == "tensorflow":
            # Préparer l'image pour le modèle TensorFlow
            img = Image.open(file).resize((224, 224))
            img_array = np.array(img) / 255.0  # Normalisation de l'image
            img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension de batch
            prediction = model.predict(img_array)
            return "Malignant" if prediction[0][0] > 0.5 else "Benign"
        elif model_type == "mlp":
            # Préparer l'image pour le modèle MLP (scikit-learn)
            img = Image.open(file).resize((50, 50))  # Redimensionner pour MLP
            img_array = np.array(img).flatten().reshape(1, -1)  # Aplatir pour MLP
            prediction = model.predict(img_array)
            return "Malignant" if prediction[0] == 1 else "Benign"
        elif model_type == "pytorch":
            # Préparer l'image pour le modèle PyTorch
            img = Image.open(file).resize((224, 224))
            img_array = np.array(img).transpose((2, 0, 1)) / 255.0  # Convertir à un format CxHxW
            img_tensor = torch.Tensor(img_array).unsqueeze(0)  # Ajouter une dimension de batch
            with torch.no_grad():
                prediction = model(img_tensor)
            return "Malignant" if prediction.item() > 0.5 else "Benign"

    if uploaded_file:
        # Afficher l'image téléchargée
        st.markdown("### 🖼️ Image téléchargée")
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée à analyser", use_column_width=True)

        # Étape 3 : Résultat de la prédiction
        st.markdown("## 📌 Étape 3 : Résultat de la prédiction")
        with st.spinner("🧪 Analyse en cours..."):
            # Faire la prédiction
            prediction = predict_image(uploaded_file, model, model_type)

        # Afficher le résultat de la prédiction
        if prediction == "Benign":
            st.success("✅ Résultat : La lésion semble **BÉNIGNE**.")
            st.markdown("### Recommandation :")
            st.write("Cela semble être une lésion bénigne. Toutefois, si vous avez des préoccupations, veuillez consulter un professionnel de santé.")
        else:
            st.error("⚠️ Résultat : La lésion pourrait être **MALIGNE**.")
            st.markdown("### Recommandation :")
            st.write("Veuillez **consulter un professionnel de santé** pour un diagnostic plus approfondi.")
