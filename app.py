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
import matplotlib.pyplot as plt

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
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse Exploratoire des Données", "Faire une Prédiction par Image", "Faire une Prédiction par Profil" ,"Benchmark des Modèles"])

# Liste des modèles disponibles pour la prédiction (pour la page "Faire une Prédiction")
model_dir = './saved_models'
models_available = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.pkl', '.pth')) and not f.startswith(('encoders', 'mlp_model', 'scaler'))]

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
    st.image(image1, caption="Image téléchargée à analyser")
    st.markdown("""
        - **Observation** : 
            - La précision d'entraînement augmente avec la profondeur, tandis que la précision de test se stabilise pour des profondeurs entre 5 et 10.
            - L'écart de précision (*Accuracy Gap*) reste faible pour des profondeurs modérées mais augmente pour des arbres plus profonds.
        - **Interprétation** :
            - Les arbres profonds tendent à sur-apprendre (*overfitting*), ce qui réduit leur capacité à généraliser sur de nouvelles données.

        ### **Graphique 2 : Visualisation du Meilleur Arbre**
    """)
    image2 = Image.open("data/images/best_decision_tree_visualization.png")
    st.image(image2, caption="Image téléchargée à analyser")
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
    st.image(image3, caption="Image téléchargée à analyser")
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
    st.image(image4, caption="Image téléchargée à analyser")
    st.markdown("""
        - **Observation** :
            - Les tendances sont similaires à celles de la profondeur optimale.
        - **Interprétation** :
            - Une profondeur de 9 est cohérente avec le meilleur modèle et offre un bon équilibre entre performance et simplicité.


        ### **Graphique 5 : Courbes ROC**
    """)
    image5 = Image.open("data/images/roc_curve.png")
    image5_1 = Image.open("data/images/rf_roc_curve.png")
    st.image(image5, caption="Image téléchargée à analyser")
    st.image(image5_1, caption="Image téléchargée à analyser")
    st.markdown("""
        - **Observation** :
            - L'AUC pour la Forêt Aléatoire est légèrement meilleure que celle de l'Arbre de Décision.
        - **Interprétation** :
            - La Forêt Aléatoire offre une meilleure capacité de discrimination entre les classes.

        ### **Graphique 6 : Matrice de Corrélation**
    """)
    image6 = Image.open("data/images/correlation_matrix_with_target.png")
    st.image(image6, caption="Image téléchargée à analyser")
    st.markdown("""
        - **Observation** :
            - `AgeCategory` et `EthnicityCategory` montrent des corrélations significatives avec la cible `HadSkinCancer`.
        - **Interprétation** :
            - Ces relations confirment les résultats des modèles, mettant en avant leur pertinence pour prédire le cancer de la peau.

        ### **Tableau récapitulatif des résultats**
    """)

    image7 = Image.open("data/images/decision_tree_summary_table.png")
    st.image(image7, caption="Image téléchargée à analyser")
    st.markdown("""
        - **Observation** :
            - Une profondeur de 5 offre une précision optimale avec un écart de précision minimal.
        - **Interprétation** :
            - Les profondeurs supérieures à 10 n'apportent pas de gains significatifs et augmentent le risque de sur-apprentissage.
    """)

    # Titre principal
    st.title("📊 Analyse de Modèles : KNN, K-Means et Régression Logistique")

    # Section 1 : Modèle KNN
    st.header("Modèle K-Nearest Neighbors (KNN)")

    # Résultats du modèle
    st.subheader("Résultats du Modèle KNN")
    st.image("data/images/results_knn.png", caption="Résultats du modèle KNN")

    # Méthode du coude
    st.subheader("Méthode du Coude")
    st.image("data/images/elbow_knn.png", caption="Méthode du coude")
    st.markdown("""
    La méthode du coude a été utilisée pour choisir une valeur optimale pour k.  
    Le taux d'erreur diminue rapidement jusqu'à une stabilisation autour de **k=10**, suggérant un équilibre entre complexité et performance.  
    **Observation :**  
        - Une valeur de k=10 est recommandée pour obtenir un compromis entre un faible taux d'erreur et une complexité acceptable.
    """)

    # Matrice de confusion
    st.subheader("Matrice de Confusion")
    st.markdown("""
    |                | Prédit : Non | Prédit : Oui |
    |----------------|--------------|--------------|
    | **Réel : Non** | 43,406 (TN)  | 155 (FP)     |
    | **Réel : Oui** | 3,910 (FN)   | 55 (TP)      |
    """)
    st.markdown("""
    **Observations :**  
        - Classe 0 (Non) : Très efficace avec 43,406 prédictions correctes et 155 erreurs.  
        - Classe 1 (Oui) : Faible performance avec seulement 55 cas correctement détectés sur 3,965.
    """)

    # Rapport de classification
    st.subheader("Rapport de Classification")
    st.markdown("""
    | Classe | Précision | Rappel | F1-Score | Support |
    |--------|-----------|--------|----------|---------|
    | 0.0    | 0.92      | 1.00   | 0.96     | 43,561  |
    | 1.0    | 0.26      | 0.01   | 0.03     | 3,965   |
    | **Macro Avg** | 0.59 | 0.51 | 0.49 | 47,526 |
    | **Weighted Avg** | 0.86 | 0.91 | 0.88 | 47,526 |
    """)
    st.markdown("""
    - Classe 0 (Non) :
        Excellente performance globale avec un F1-Score de 0.96.
    - Classe 1 (Oui) :
        Précision de 26% et rappel de 1%, reflétant une faible capacité à détecter les cas positifs.
    - Macro Avg : Le déséquilibre des performances entre les deux classes est notable.
    - Weighted Avg : Pondéré par le déséquilibre des classes, il masque les lacunes sur la classe minoritaire.
    """)

    # Scores de performance
    st.subheader("Scores de Performance")
    st.markdown("""
    **Accuracy :** 91%, trompeuse en raison de la domination de la classe majoritaire (92% de "Non").  
    **ROC-AUC :** 0.73, indique une capacité modérée à séparer les classes, dépassant légèrement le seuil de hasard (0.5).
    """)
    
    # Impact du Nombre de Voisins (k)
    st.subheader("Impact du Nombre de Voisins (k)")
    st.markdown("""
    | k      | Accuracy | 
    |--------|-----------|
    | 1      | 0.86      | 
    | 2-6    | 0.91      |    
    | 7-14   | 0.91-0.92 |   
    | 15-20  | 0.92      |           
    
    L'accuracy reste stable autour de 91-92 %, soulignant que le déséquilibre des classes est un problème inhérent.
    """)

    # Recommandations pour amélioration
    st.subheader("Améliorations Recommandées")
    st.markdown("""
    1. **Gestion du déséquilibre des classes :**
        - Sur-échantillonnage : Techniques pour augmenter artificiellement les données de la classe minoritaire.
        - Sous-échantillonnage : Réduire les échantillons de la classe majoritaire.
        - Pondération des Classes : Ajuster les poids pour accorder plus d'importance à la classe minoritaire.
    2. **Explorer d'autres modèles :**
        - Modèles robustes au déséquilibre comme Random Forest, Gradient Boosting ou SVM.
    3. **Métriques adaptées :**
        - Se concentrer sur le F1-Score, le Rappel, et le ROC-AUC pour mieux évaluer les performances sur la classe minoritaire.
    """)
    
    # Conclusion
    st.header("Conclusion KNN")
    st.markdown("""
    Le modèle KNN affiche une accuracy élevée (91 %), mais échoue à détecter efficacement la classe minoritaire.  
    Pour améliorer la détection des cas critiques (classe "Oui"), il est essentiel de combiner :  
        - Rééquilibrage des données  
        - Optimisation des métriques pertinentes  
        - Exploration de modèles alternatifs  
    Ces ajustements amélioreront significativement la capacité du modèle à traiter les cas rares mais critiques, comme dans les contextes de santé ou de détection d'anomalies.
    """)

    # Section 2 : Clustering K-Means
    st.header("Clustering avec K-Means")

    # Méthode du coude
    st.subheader("Méthode du Coude")
    st.image("data/images/elbow_kmeans.png", caption="Méthode du coude")
    st.markdown("""
    La méthode du coude montre une diminution rapide de l'inertie jusqu'à **k=10**, suggérant que 10 clusters pourraient être optimaux.  
    Pour cette analyse, nous avons utilisé k=2 pour explorer une séparation binaire simple.
    """)

    # Résultats du clustering
    st.subheader("Résultats du Clustering avec k=2")
    st.image("data/images/k_means.png", caption="Visualisation des clusters (PCA)")
    st.markdown("""
    **Matrice de Dispersion des Clusters (Réduction PCA) :**  
        - Le clustering a été visualisé à l’aide de la réduction de dimensions PCA sur 2 composantes principales.  
        - Les clusters générés sont bien distincts dans l’espace PCA, mais ils ne correspondent pas directement aux classes définies par HadSkinCancer.  
    """)
    
    # Dispersion des Cancer de la Peau dans le Dataset 
    st.subheader("Dispersion des Cancer de la Peau dans le Dataset ")
    st.image("data/images/dispersion.png", caption="Dispersion dans le Dataset")
    st.markdown("""
    **Lors de la visualisation des clusters en fonction de la cible HadSkinCancer :**  
    - Les individus sont répartis différemment par le clustering K-Means et la cible réelle (HadSkinCancer).
        
    **Observation :**  
        - Les clusters K-Means ne parviennent pas à capturer la distinction entre les individus ayant ou non un cancer de la peau.  
        - Cela indique que le modèle utilise des patterns différents ou des caractéristiques non liées à HadSkinCancer.  
    """)
    
    # Résumé des Résultats
    st.subheader("Résumé des Résultats")
    st.markdown("""
    1. **Nombre Optimal de Clusters : 10 (méthode du coude)**

    2. **Clustering avec k=2 :**
        - Les clusters sont distincts mais ne reflètent pas la présence ou l'absence de cancer de la peau.
        - Cela pourrait être dû à un bruit dans les données ou à des caractéristiques non discriminantes.

    3. **Pertinence des Caractéristiques :**
        - Les variables fournies semblent insuffisantes pour capturer la relation avec HadSkinCancer.
        - Le clustering révèle potentiellement d'autres structures dans les données, mais celles-ci ne correspondent pas à l'objectif défini.
    """)

    # Recommandations pour amélioration
    st.subheader("Améliorations Recommandées")
    st.markdown("""
    1. **Explorer d'autres modèles de clustering :**
        - DBSCAN pour détecter des clusters de formes variées.
        - GMM pour capturer des relations probabilistes.
    2. **Sélection de caractéristiques :**
        - Identifier et intégrer des caractéristiques plus corrélées à la cible HadSkinCancer.
        - Réduire le bruit dans les données par un nettoyage plus rigoureux.
    3. **Ajustement du Nombre de Clusters :**
        - Tester avec k=10 (nombre suggéré par la méthode du coude) pour une segmentation plus fine et voir si elle révèle des groupes plus significatifs.
    4. **Ajout de Métriques d'Évaluation :**
        - Calculer le silhouette score pour évaluer la cohérence des clusters.
        - Utiliser des métriques supervisées pour mesurer la pertinence des clusters par rapport à HadSkinCancer.
    """)
    
    # Conclusion
    st.header("Conclusion K-Means")
    st.markdown("""
    Le clustering K-Means a permis de révéler des structures dans les données, mais ces structures ne reflètent pas la présence ou l'absence de cancer de la peau.
    Pour améliorer la pertinence du clustering dans ce contexte, il est essentiel :
    - d'Intégrer des caractéristiques plus significatives,
    - d'Explorer des algorithmes alternatifs,
    - d'Ajuster le nombre de clusters.  
    
    Ces améliorations permettront d’obtenir une segmentation plus adaptée, particulièrement utile dans des applications critiques comme la détection de maladies.
    """)

    # Section 3 : Modèle de Régression Logistique
    st.header("Modèle de Régression Logistique")

    # Résultats du modèle
    st.subheader("Résultats du Modèle de Régression Logistique")
    st.image("data/images/results_reglog.png", caption="Résultats de la régression logistique")

    # Matrice de confusion
    st.subheader("Matrice de Confusion")
    st.markdown("""
    |                | Prédit : Non | Prédit : Oui |
    |----------------|--------------|--------------|
    | **Réel : Non** | 43,561 (TN)  | 0 (FP)       |
    | **Réel : Oui** | 3,965 (FN)   | 0 (TP)       |
    """)
    st.markdown("""
    **Observations :**  
        - Classe 0 (Non) : Le modèle est très performant pour détecter les cas "Non", avec toutes les prédictions "Non" correctement classées (43,561).  
        - Classe 1 (Oui) : Aucune prédiction correcte pour la classe "Oui", avec toutes les instances "Oui" classées comme "Non" (0 TP).  
    Le modèle souffre d'un biais important vers la classe majoritaire et ne détecte pas du tout les cas "Oui".  
    """)
    
    # Rapport de classification
    st.subheader("Rapport de Classification")
    st.markdown("""
    | Classe | Précision | Rappel | F1-Score | Support |
    |--------|-----------|--------|----------|---------|
    | 0.0    | 0.92      | 1.00   | 0.96     | 43,561  |
    | 1.0    | 0.00      | 0.00   | 0.00     | 3,965   |
    | **Macro Avg**    | 0.46 | 0.50 | 0.48 | 47,526 |
    | **Weighted Avg** | 0.84 | 0.92 | 0.88 | 47,526 |
    """)
    st.markdown("""
    **Observations :**  
        - Classe 0 (Non) : Excellente performance pour cette classe avec un F1-Score de 0.96.  
        - Classe 1 (Oui) : Le modèle échoue complètement à détecter les cas "Oui", avec un F1-Score de 0.0.  
        - Macro Avg : La moyenne non pondérée indique de faibles performances globales, avec un F1-Score de 0.48.  
        - Weighted Avg : Le F1-Score élevé de 0.88 est dû à la prédominance de la classe 0, masquant ainsi la mauvaise performance pour la classe 1.
    """)
    
    # Scores de performance
    st.subheader("Scores de Performance")
    st.markdown("""
    **Accuracy :** 92 %, le modèle est principalement performant pour la classe majoritaire, mais cette métrique est trompeuse en raison du déséquilibre des classes.
    """)

    # Importance des caractéristiques
    st.subheader("Importance des Caractéristiques")
    st.image("data/images/features_importance.png", caption="Importance des caractéristiques")
    st.markdown("""
    Pour mieux comprendre le fonctionnement du modèle, l'importance des caractéristiques a été calculée en fonction des coefficients du modèle. 
    Les caractéristiques les plus influentes sont :
    - **Âge :** La variable la plus importante pour prédire les résultats.
    - **Ethnie :** Joue un rôle significatif dans les prédictions.
    - **Facteurs secondaires :**
        - Consommation d'alcool
        - Tabagisme
        - Statut VIH (positif/négatif)
    Ces résultats soulignent l'importance de variables démographiques (âge, ethnie) et comportementales dans la prédiction, même si la performance globale est limitée par le déséquilibre des classes.
    """)
    
    # Impact du Déséquilibre des Classes
    st.subheader("Impact du Déséquilibre des Classes")
    st.markdown("""
    Le modèle de régression logistique affiche une accuracy élevée de 92 %, mais cette performance est totalement biaisée par la classe majoritaire (0.0).  
    Le modèle n'a pas réussi à identifier un seul cas de la classe minoritaire (1.0), ce qui montre son incapacité à gérer un déséquilibre marqué entre les classes.
    """)
    
    # Recommandations pour amélioration
    st.subheader("Améliorations Recommandées")
    st.markdown("""
    1. **Gestion du Déséquilibre des Classes :**
        - Sur-échantillonnage : Augmenter artificiellement les données de la classe minoritaire (par exemple avec SMOTE).
        - Sous-échantillonnage : Réduire les données de la classe majoritaire pour équilibrer les proportions.
        - Pondération des Classes : Ajuster les poids des classes dans le modèle pour donner plus d'importance à la classe minoritaire.
    2. **Exploration d'Autres Algorithmes :** 
        - Tester des modèles plus robustes face au déséquilibre des classes, tels que Random Forest, Gradient Boosting, ou SVM.
    3. **Utilisation de Métriques Adaptées :**
        - Se concentrer sur des métriques comme le F1-Score, le Rappel et **l'AUC-ROC pour évaluer les performances, surtout sur la classe minoritaire.
    """)
    
    # Conclusion
    st.header("Conclusion Régression Logistique")
    st.markdown("""
    Bien que le modèle de régression logistique atteigne une accuracy élevée de 92 %, il est complètement inefficace pour détecter la classe minoritaire (1.0). 
    Ce modèle est fortement biaisé par le déséquilibre des classes, et il est essentiel de :
    - Appliquer des techniques pour traiter ce déséquilibre (rééchantillonnage, pondération des classes),
    - Choisir des métriques adaptées à la classe minoritaire,
    - Explorer des modèles plus robustes aux déséquilibres comme Random Forest ou Gradient Boosting.
    Cela permettra d’améliorer les performances du modèle, en particulier pour des tâches critiques comme la détection de maladies ou la classification d’anomalies.
    """)

    # Conclusion globale
    st.header("Conclusion Globale")
    st.markdown("""
    Les modèles KNN, K-Means et régression logistique montrent des limites dans la gestion des déséquilibres de classes.
    **Ces analyses mettent en évidence des pistes claires d'amélioration :**
    1. Rééquilibrage des données.
    2. Exploration de modèles plus robustes.
    3. Utilisation de métriques adaptées (F1-Score, ROC-AUC).
    Ces actions permettront d'améliorer les performances pour des tâches critiques comme la détection de maladies.
    """)

# Page pour faire une prédiction
elif page == "Faire une Prédiction par Image":
    st.title("🔍 Faire une Prédiction par Image")

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
            img = Image.open(file).resize((224, 224))  # Redimensionner pour MLP
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
        st.image(image, caption="Image téléchargée à analyser")

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

elif page == "Faire une Prédiction par Profil":
    # Interface utilisateur
    st.title("🤖 Prédiction personnalisée par profil")
    st.markdown("""
    Cette page permet d'effectuer une prédiction basée sur des informations personnelles. Le modèle utilisé pour cette prédiction est un **Perceptron Multicouche (MLP)**, avec une architecture optimisée et entraînée spécifiquement pour détecter les risques de cancer de la peau.

    ## Configuration du Modèle 🔧
    - **Type** : Perceptron Multicouche (MLP)
    - **Architecture** :
    - Première couche : **50 neurones**
    - Deuxième couche : **30 neurones**
    - **500 époques maximum**
    - Fonction d'activation : **ReLU** (Rectified Linear Unit) 

    ### Objectif 🎯
    Le modèle prédit la probabilité actuelle de risque de cancer de la peau.

    ---

    ## Formulaire Utilisateur 📝
    Pour effectuer la prédiction, l'utilisateur doit fournir les informations suivantes :
    1. **État de résidence** : Choisissez parmi une liste d'États américains (ex. Alabama, Alaska, etc.)
    2. **Sexe** : Homme ou Femme 
    3. **Catégorie d'âge** : Une des plages d'âge prédéfinies (ex. 18-24 ans, 25-29 ans, etc.)
    4. **IMC (Indice de Masse Corporelle)** : Un curseur permet de définir une valeur entre **10.0** et **70.0**
    5. **Statut de fumeur** : Indiquez si vous êtes fumeur (Oui/Non)
    6. **Utilisation de cigarettes électroniques** : Oui ou Non
    7. **Ethnicité** : Sélectionnez une catégorie (ex. White only, Non-Hispanic)

    ---

    ## Résultats de la Prédiction 📊
    Après soumission des informations :
    1. **Probabilités prédictives** :
    - **Classe Positive (cancer probable)** : Affichage de la probabilité en pourcentage.
    - **Classe Négative (cancer improbable)** : Affichage de la probabilité en pourcentage.
    2. **Représentation graphique** :
    - Un graphique en barres montre la distribution des probabilités entre les deux classes.

    ---

    ### Exemple de Résultat
    - **Classe Positive** : 65.34% (indique un risque élevé de cancer de la peau).
    - **Classe Négative** : 34.66% (indique un risque faible).

    Un graphique est généré pour permettre une analyse visuelle rapide de la probabilité.

    ---

    > **Note** : Les prédictions fournies par cet outil sont uniquement à titre informatif. Pour tout doute ou risque de santé, il est fortement recommandé de consulter un professionnel de santé.
    """)

    # Création du formulaire utilisateur
    state = st.selectbox("État", ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Guam", "Hawaii", "Idaho", "Illinois"])
    sex = st.selectbox("Sexe", ["Male", "Female"])
    age_category = st.selectbox("Catégorie d'âge", ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59", "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 to older"])
    bmi = st.slider("IMC (Indice de Masse Corporelle)", 10.0, 70.0, 25.0, step=0.1)
    smoker_status = st.selectbox("Statut de fumeur", ["Never smoked", "Former smoker", "Current smoker - now smokes every day"])
    e_cigarette_usage = st.selectbox("Utilisation de cigarettes électroniques", [
        "Never used e-cigarettes in my entire life",
        "Not at all (right now)",
        "Use them some days"
    ])
    race_ethnicity = st.selectbox("Ethnicité", ["Black only, Non-Hispanic", "Hispanic", "Multiracial, Non-Hispanic", "Other race only, Non-Hispanic", "White only, Non-Hispanic"])
    alchool_drinkers = st.selectbox("Alcool", ["Yes", "No"])
    hiv_testing = st.selectbox("Test VIH", ["Yes", "No"])

    # process data to int
    hiv_testing = 1 if hiv_testing == "Yes" else 0
    alchool_drinkers = 1 if alchool_drinkers == "Yes" else 0

    # Lorsque l'utilisateur clique sur "Prédire"
    if st.button("Faire une prédiction"):
        # Fonction pour faire une prédiction
        def predict_profil(model, features):
            return model.predict_proba([features])[0]

        model, model_type = load_selected_model("./saved_models/mlp_model.pkl")
        if not model:
            st.error("⚠️ Aucun modèle disponible. Veuillez ajouter des modèles dans le dossier 'saved_models' en lançant le fichier train_neuron.py.")
            st.stop()

        # Convertir les entrées en format utilisable pour le modèle
        user_features = [
            state, sex, age_category, bmi, smoker_status, e_cigarette_usage, race_ethnicity, alchool_drinkers, hiv_testing
        ]
        
        encoder = joblib.load("./saved_models/encoders.pkl")
        if not encoder:
            st.error("⚠️ Aucun encoder disponible. Veuillez ajouter des encoders dans le dossier 'saved_models' en lançant le fichier train_neuron.py.")
            st.stop()

        user_features_encoded = []
        important_columns = ['State', 'Sex', 'AgeCategory', 'BMI', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AlcoholDrinkers', 'HIVTesting']
        for i, col in enumerate(important_columns):
            if col in encoder:  # Vérifie si la colonne a un encodeur
                feature_value = user_features[i]
                if feature_value not in encoder[col].classes_:
                    st.error(f"⚠️ Valeur inconnue '{feature_value}' détectée pour la colonne '{col}'. Veuillez vérifier vos données.")
                    st.stop()
                transformed = encoder[col].transform([feature_value])
                user_features_encoded.append(transformed[0])
            else:
                # Ajouter directement les valeurs numériques ou non encodées
                user_features_encoded.append(user_features[i])

        print(user_features_encoded)
        user_features_encoded = np.array(user_features_encoded) 

        # Prédiction
        prediction = predict_profil(model, user_features_encoded)
        
        # Afficher les résultats
        st.markdown("## 📊 Résultats de la Prédiction")
        st.write(f"Classe Positive (cancer probable) : {prediction[1] * 100:.2f}%")
        st.write(f"Classe Négative (cancer improbable) : {prediction[0] * 100:.2f}%")

        # Visualisation des probabilités
        fig, ax = plt.subplots()
        ax.bar(["Classe Négative", "Classe Positive"], prediction, color=["green", "red"])
        ax.set_ylabel("Probabilité")
        ax.set_title("Probabilités Prédictives")
        st.pyplot(fig)
        

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
        "Modèle": ["VGG", "ResNet", "EfficientNet", "MLP (Moyenne)", "Sequential"],
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
            st.image(accuracy_curve_path, caption=f"Courbe d'Accuracy - {model_name}")
            st.image(loss_curve_path, caption=f"Courbe de Loss - {model_name}")

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
    st.image(mlp_image, caption="Courbes d'Apprentissage - Modèles MLP")

    st.markdown("""
        Ce graphique permet de représenter l'efficacité des modèles MLP en fonction du nombre de neurones dans les couches cachées et en soulignant l'accuracy, le temps d'entraînement et le temps d'évaluation.
        Grâce à ce graphique, nous pouvons voir qu'un grand nombre de neurones dans les couches cachées n'entraîne pas nécessairement une meilleure accuracy et prends plus de temps d'entraînement. Cependant, un certains nombre de neuronnes est requis pour le bon fonctionnement du modèle.
        Comme on peut le voir, le modèle contenant uniquement 1 neuronne dans la couche cachée est le moins performant avec une accuracy en dessous de 50%.      
        
        En analysant les performances des modèles, nous pouvons déterminer le modèle le plus efficace qui est celui à deux couches cachées avec 64 et 32 neurones respectivement.
        Ce modèle a le temps d'entrainement le plus court et une accuracy similaire aux autres modèles.      
    """)

    # Conclusion sur le benchmark
    st.markdown("## Conclusion")
    st.markdown("""
    Après avoir comparé les performances des différents modèles, il semble que **ResNet** soit le modèle le plus performant, avec une **accuracy** de 0.88 et un **AUC** de 0.88. Toutefois, cela a un coût en termes de temps d'entraînement, qui est relativement élevé.

    Pour des applications où la précision est cruciale, **ResNet** semble être le meilleur choix. Si le temps d'entraînement est une contrainte importante, alors **EfficientNet** offre un bon compromis entre performance et temps d'entraînement.
                
    Le pire choix reste le modèle MLP à 1 neurone.
    """)
