
# Analyse et Visualisation avec un Arbre de Décision

Ce projet démontre l'utilisation d'un modèle **Arbre de Décision** pour analyser un dataset lié aux maladies de la peau. 
L'objectif est de trouver la profondeur optimale de l'arbre et d'évaluer ses performances à l'aide de plusieurs métriques et visualisations.

## **0. Installation**

Pour lancer le projet, installez les dépendances nécessaires via le fichier `requirements.txt`

```py
pip install -r requirements.txt
```

Ensuite, téléchargez le jeu de données format `.zip` et dézippez le contenu dans à la racine du projet: https://mediafire.com...

Une fois ces deux étapes effectuées, lancez l'application via la commande suivante:

```
streamlit run app.py
```

## **1. Explication du Code**

Le projet utilise Python avec des bibliothèques comme `scikit-learn` et `matplotlib`. Voici les étapes principales :

### **Étapes dans le Code**

1. **Chargement des données :**
   - Le dataset est chargé à partir d'un fichier Excel.
   - La colonne cible est `HadSkinCancer`, qui indique si une personne a eu un cancer de la peau.

2. **Prétraitement des données :**
   - Les fonctions `preprocess_data` et `clean_data` sont utilisées pour nettoyer et encoder les variables catégoriques.

3. **Séparation des données :**
   - Les données sont divisées en un ensemble d'entraînement (80%) et un ensemble de test (20%).

4. **Entraînement et évaluation du modèle :**
   - Une boucle teste des modèles d'Arbre de Décision avec des profondeurs variant de 1 à 20.
   - Pour chaque profondeur, les métriques suivantes sont calculées :
     - Précision sur les données d'entraînement.
     - Précision sur les données de test.
     - Écart de précision (*Accuracy Gap*).
     - Taille de l'arbre.

5. **Visualisations générées :**
   - **Meilleur arbre** : Visualisation de l'arbre pour la profondeur optimale.
   - **Importances des caractéristiques** : Variables ayant le plus d'influence sur les prédictions.
   - **Courbe ROC** : Capacité du modèle à distinguer les classes.
   - **Analyse des précisions** : Comparaison des précisions d'entraînement et de test, et de l'écart entre elles.

## **2. Interprétation des Résultats**

### **Graphique 1 : Train vs Test Accuracy and Accuracy Gap**
![Train vs Test Accuracy and Accuracy Gap](data/images/accuracy_gap_analysis.png)
- **Observation** : 
  - La précision d'entraînement augmente avec la profondeur, tandis que la précision de test se stabilise pour des profondeurs entre 5 et 10.
  - L'écart de précision (*Accuracy Gap*) reste faible pour des profondeurs modérées mais augmente pour des arbres plus profonds.
- **Interprétation** :
  - Les arbres profonds tendent à sur-apprendre (*overfitting*), ce qui réduit leur capacité à généraliser sur de nouvelles données.

### **Graphique 2 : Visualisation du Meilleur Arbre**
![Visualisation du Meilleur Arbre](data/images/best_decision_tree_visualization.png)
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
![Importances des Caractéristiques (Profondeur Optimale)](data/images/feature_importances_best_depth.png)
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
![Importances des Caractéristiques (Profondeur = 9)](data/images/feature_importances_depth_9.png)
- **Observation** :
  - Les tendances sont similaires à celles de la profondeur optimale.
- **Interprétation** :
  - Une profondeur de 9 est cohérente avec le meilleur modèle et offre un bon équilibre entre performance et simplicité.

### **Graphique 5 : Courbe ROC**
![Courbe ROC](data/images/roc_curve.png)
- **Observation** :
  - La courbe ROC affiche une AUC de 0.78, indiquant une bonne capacité à distinguer les classes.
- **Interprétation** :
  - Le modèle a une performance correcte mais pourrait être amélioré.

### **Tableau récapitulatif des résultats**
![Tableau récapitulatif](data/images/decision_tree_summary_table.png)
- **Observation** :
  - Une profondeur de 5 offre une précision optimale avec un écart de précision minimal.
  - La taille de l'arbre augmente rapidement avec la profondeur.
- **Interprétation** :
  - Les profondeurs supérieures à 10 n'apportent pas de gains significatifs et augmentent le risque de sur-apprentissage.
