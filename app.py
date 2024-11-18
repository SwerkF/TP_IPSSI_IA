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
import pandas as pd

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
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse Exploratoire des Données", "Faire une Prédiction", "Benchmark des Modèles"])

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
    st.title("📊 Analyse et Visualisation avec un Arbre de Décision")

    st.markdown("""
        Ce projet démontre l'utilisation d'un modèle **Arbre de Décision** et **Forêt Aléatoire** pour analyser un dataset lié aux maladies de la peau. L'objectif est d'identifier les variables les plus importantes et de trouver des modèles efficaces pour prédire si une personne a eu un cancer de la peau (`HadSkinCancer`).

        ## **1. Explication du Code**

        Le projet utilise Python avec des bibliothèques comme `scikit-learn` et `matplotlib`. Voici les étapes principales :

        ### **Étapes dans le Code**

        1. **Chargement des données :**
        - Le dataset est chargé à partir d'un fichier Excel.
        - La colonne cible est `HadSkinCancer`, qui indique si une personne a eu un cancer de la peau.

        2. **Prétraitement des données :**
        - Les fonctions `preprocess_data` et `clean_data` sont utilisées pour nettoyer et encoder les variables catégoriques.

        3. **Matrice de corrélation :**
        - Une matrice de corrélation est calculée pour visualiser les relations entre toutes les variables, y compris la cible (`HadSkinCancer`).

        4. **Séparation des données :**
        - Les données sont divisées en un ensemble d'entraînement (80%) et un ensemble de test (20%).

        5. **Entraînement et évaluation des modèles :**
        - **Arbre de Décision :** Une boucle teste des modèles d'Arbre de Décision avec des profondeurs variant de 1 à 20.
            - Les métriques suivantes sont calculées : précision (train et test), écart de précision (*Accuracy Gap*), taille de l'arbre.
        - **Forêt Aléatoire :** Un modèle avec plusieurs arbres est entraîné pour comparer ses performances à celles de l'Arbre de Décision.

        6. **Visualisations générées :**
        - **Matrice de corrélation** : Relations entre les variables.
        - **Meilleur arbre** : Visualisation de l'arbre pour la profondeur optimale.
        - **Importances des caractéristiques** : Variables ayant le plus d'influence dans les prédictions (pour les deux modèles).
        - **Courbes ROC** : Capacité des modèles à distinguer les classes.
        - **Analyse des précisions** : Comparaison des précisions d'entraînement et de test pour l'Arbre de Décision.

        ## **2. Interprétation des Résultats**

        ### **Graphique 1 : Train vs Test Accuracy and Accuracy Gap**
    """)
    image1 = Image.open("data/images/accuracy_gap_analysis.png")
    st.image(image1, caption="Image téléchargée à analyser", use_container_width=True)
    st.markdown("""
        - **Observation** : 
            - La précision d'entraînement augmente avec la profondeur, tandis que la précision de test se stabilise pour des profondeurs entre 5 et 10.
            - L'écart de précision (*Accuracy Gap*) reste faible pour des profondeurs modérées mais augmente pour des arbres plus profonds.
        - **Interprétation** :
            - Les arbres profonds tendent à sur-apprendre (*overfitting*), ce qui réduit leur capacité à généraliser sur de nouvelles données.

        ### **Graphique 2 : Visualisation du Meilleur Arbre**
    """)
    image2 = Image.open("data/images/best_decision_tree_visualization.png")
    st.image(image2, caption="Image téléchargée à analyser", use_container_width=True)
    st.markdown("""
        - **Observation** :
            - L’arbre montre que les premières décisions sont principalement basées sur `AgeCategory`, suivi de `EthnicityCategory`.
            - Les nœuds supérieurs divisent les données en groupes significatifs, maximisant la séparation entre les classes.
            - **Interprétation détaillée** :
            - **Critères de décision (nœuds supérieurs)** :
                - L'utilisation de `AgeCategory` en haut de l’arbre reflète son rôle dominant dans la prédiction. Par exemple :
                - Si une personne est dans une catégorie d'âge avancé, le modèle peut prédire avec une grande probabilité la présence d'un risque accru.
                - `EthnicityCategory`, souvent utilisé dans les premiers nœuds, divise les individus en fonction de leur sensibilité aux dommages UV.
            - **Décisions fines (nœuds inférieurs)** :
                - Les caractéristiques comme `AlcoholDrinkers` ou `BMI`, bien que moins importantes globalement, sont utilisées pour affiner les prédictions dans des sous-groupes spécifiques.
            - **Feuilles de l'arbre** :
                - Les probabilités présentes aux feuilles permettent d'identifier la classe prédite (cancer de la peau ou non) ainsi que le degré de certitude de la prédiction.
            - **Explication contextuelle** :
            - Cet arbre de décision met en lumière les relations clés dans les données et peut être utilisé pour :
                - Identifier rapidement les groupes à haut risque.
                - Élaborer des stratégies ciblées, comme des campagnes de dépistage adaptées aux profils les plus vulnérables.


            ### **Graphique 3 : Importances des Caractéristiques (Profondeur Optimale)**
    """)
    image3 = Image.open("data/images/feature_importances_best_depth.png")
    st.image(image3, caption="Image téléchargée à analyser", use_container_width=True)
    st.markdown("""
        - **Observation** :
            - `AgeCategory` est la variable la plus influente, suivie de `EthnicityCategory`. Les autres caractéristiques, comme `BMI` et `AlcoholDrinkers`, ont un impact limité.
            - **Interprétation détaillée** :
            - **AgeCategory (importance élevée)** :
                - L’âge est un facteur critique dans les maladies de la peau, notamment le cancer de la peau, car :
                - L’exposition cumulative au soleil au fil des ans augmente les risques.
                - Les processus biologiques liés au vieillissement réduisent les capacités de réparation de l’ADN après des dommages causés par les UV.
                - Les groupes d'âge plus avancés sont plus fréquemment diagnostiqués avec des cancers de la peau dans les études épidémiologiques.
            - **EthnicityCategory (importance élevée)** :
                - L’ethnicité joue un rôle clé, car :
                - Les personnes ayant une peau plus claire ont généralement une concentration plus faible de mélanine, ce qui les rend plus vulnérables aux dommages UV.
                - Les différences dans les comportements culturels, comme la protection solaire ou les habitudes d'exposition, peuvent également influencer ce facteur.
            - **AlcoholDrinkers et BMI (importance modérée)** :
                - Bien que ces variables aient une importance moindre, elles peuvent refléter des comportements liés à la santé globale :
                - Une consommation excessive d'alcool peut affaiblir le système immunitaire et réduire la capacité à réparer les cellules endommagées.
                - L’indice de masse corporelle (BMI) est parfois lié à des comportements de santé globaux (par exemple, l'activité physique et l'exposition au soleil).
            - **Explication contextuelle** :
            - Ces résultats confirment des tendances bien documentées dans les recherches médicales.

        ### **Graphique 4 : Importances des Caractéristiques (Profondeur = 9)**
    """)
    image4 = Image.open("data/images/feature_importances_depth_9.png")
    st.image(image4, caption="Image téléchargée à analyser", use_container_width=True)
    st.markdown("""
        - **Observation** :
            - Les tendances sont similaires à celles de la profondeur optimale.
        - **Interprétation** :
            - Une profondeur de 9 est cohérente avec le meilleur modèle et offre un bon équilibre entre performance et simplicité.


        ### **Graphique 5 : Courbes ROC**
    """)
    image5 = Image.open("data/images/roc_curve.png")
    image5_1 = Image.open("data/images/rf_roc_curve.png")
    st.image(image5, caption="Image téléchargée à analyser", use_container_width=True)
    st.image(image5_1, caption="Image téléchargée à analyser", use_container_width=True)
    st.markdown("""
        - **Observation** :
            - L'AUC pour la Forêt Aléatoire est légèrement meilleure que celle de l'Arbre de Décision.
        - **Interprétation** :
            - La Forêt Aléatoire offre une meilleure capacité de discrimination entre les classes.

        ### **Graphique 6 : Matrice de Corrélation**
    """)
    image6 = Image.open("data/images/correlation_matrix_with_target.png")
    st.image(image6, caption="Image téléchargée à analyser", use_container_width=True)
    st.markdown("""
        - **Observation** :
            - `AgeCategory` et `EthnicityCategory` montrent des corrélations significatives avec la cible `HadSkinCancer`.
        - **Interprétation** :
            - Ces relations confirment les résultats des modèles, mettant en avant leur pertinence pour prédire le cancer de la peau.

        ### **Tableau récapitulatif des résultats**
    """)

    image7 = Image.open("data/images/decision_tree_summary_table.png")
    st.image(image7, caption="Image téléchargée à analyser", use_container_width=True)
    st.markdown("""
        - **Observation** :
            - Une profondeur de 5 offre une précision optimale avec un écart de précision minimal.
        - **Interprétation** :
            - Les profondeurs supérieures à 10 n'apportent pas de gains significatifs et augmentent le risque de sur-apprentissage.
    """)


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
        st.image(image, caption="Image téléchargée à analyser", use_container_width=True)

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

# Page pour le benchmark des modèles
elif page == "Benchmark des Modèles":
    st.title("📊 Benchmark des Modèles")

    st.markdown("""
    Cette page compare les performances des différents modèles utilisés pour la détection du cancer de la peau.
    
    ## Comparaison des Résultats des Modèles
    Les tableaux et graphiques ci-dessous montrent les performances (accuracy, AUC, temps d'entraînement, etc.) de chaque modèle.
    """)

    # Ajouter une visualisation de type tableau pour comparer les performances des modèles
    benchmark_results = {
        "Modèle": ["VGG", "ResNet", "EfficientNet", "MLP", "Sequential"],
        "Accuracy": [0.85, 0.87, 0.86, 0.78, 0.79],
        "Training Time (seconds)": [300, 350, 280, 100, 150],
        "AUC": [0.87, 0.88, 0.85, 0.76, 0.80]
    }

    benchmark_df = pd.DataFrame(benchmark_results)
    st.dataframe(benchmark_df)

    # Graphique pour comparer les accuracy des modèles
    st.markdown("### Comparaison de l'Accuracy des Modèles")
    st.bar_chart(benchmark_df.set_index("Modèle")["Accuracy"])

    # Graphique pour comparer le temps d'entraînement
    st.markdown("### Temps d'Entraînement des Modèles")
    st.bar_chart(benchmark_df.set_index("Modèle")["Training Time (seconds)"])

    # Afficher des barres de progression pour les performances
    st.markdown("### Visualisation des Performances des Modèles")
    for index, row in benchmark_df.iterrows():
        st.markdown(f"**{row['Modèle']}**")
        percentage = int(row['Accuracy'] * 100)
        st.progress(percentage)
        st.write(f"Accuracy: {percentage}%")


    curve_dir = './training_curves'
    model_names = ["VGG", "ResNet", "EfficientNet", "Sequential"]

    for model_name in model_names:
        accuracy_curve_path = os.path.join(curve_dir, f"{model_name}_accuracy_curve.png")
        loss_curve_path = os.path.join(curve_dir, f"{model_name}_loss_curve.png")

        if os.path.exists(accuracy_curve_path) and os.path.exists(loss_curve_path):
            st.markdown(f"#### Courbes d'Apprentissage pour le Modèle {model_name}")
            st.image(accuracy_curve_path, caption=f"Courbe d'Accuracy - {model_name}", use_container_width=True)
            st.image(loss_curve_path, caption=f"Courbe de Loss - {model_name}", use_container_width=True)

    # Courbes d'apprentissage pour chaque modèle
    st.markdown("### Courbes d'Apprentissage des Modèles")
    st.markdown("""
        Les courbes d'apprentissage ci-dessous montrent l'évolution des métriques de **Loss** et **Accuracy** pendant l'entraînement de chaque modèle. Ces courbes permettent de visualiser le comportement des modèles au fur et à mesure de l'apprentissage, tant sur l'ensemble d'entraînement que sur l'ensemble de validation.

        - **Courbe de Loss** : Représente la mesure de l'erreur de prédiction du modèle au fil des epochs. Une baisse régulière de la loss indique que le modèle apprend correctement.
        - **Courbe d'Accuracy** : Montre l'évolution de la précision du modèle. Plus la courbe monte, plus le modèle devient performant.

        Les courbes permettent de déterminer si le modèle est en train de **sous-apprendre** (les deux courbes sont faibles) ou de **sur-apprendre** (forte différence entre les courbes d'entraînement et de validation).
    """)

    st.markdown("#### Modèles MLP")
    mlp_image = Image.open("data/images/mlp_results.png")
    st.image(mlp_image, caption="Courbes d'Apprentissage - Modèles MLP", use_container_width=True)

    # Conclusion sur le benchmark
    st.markdown("## Conclusion")
    st.markdown("""
    Après avoir comparé les performances des différents modèles, il semble que **ResNet** soit le modèle le plus performant, avec une **accuracy** de 0.88 et un **AUC** de 0.88. Toutefois, cela a un coût en termes de temps d'entraînement, qui est relativement élevé.

    Pour des applications où la précision est cruciale, **ResNet** semble être le meilleur choix. Si le temps d'entraînement est une contrainte importante, alors **EfficientNet** offre un bon compromis entre performance et temps d'entraînement.
    """)