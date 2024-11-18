import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Charger les données
data = pd.read_excel('./data/dataset.xlsx')

# Sélection des colonnes importantes
important_columns = ['State', 'Sex', 'AgeCategory', 'BMI', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AlcoholDrinkers', 'HIVTesting']

# Filtrer les colonnes dans le DataFrame pour l'entraînement
X = data[important_columns]
y = data['HadSkinCancer'].apply(lambda x: 1 if x == 1 else 0)

# Initialiser des encodeurs pour chaque colonne et les stocker
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])

# Séparer en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner le modèle
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, activation='relu', random_state=42)
mlp.fit(X_train, y_train)

# Évaluation du modèle
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude :", accuracy)

# Enregistrer le modèle MLP
joblib.dump(mlp, './saved_models/mlp_model.pkl')

# Enregistrer le scaler
joblib.dump(scaler, './saved_models/scaler.pkl')

# Enregistrer les encodeurs
joblib.dump(encoders, './saved_models/encoders.pkl')

"""
Voici le code pour faire une simulation sans streamlit depuis le code
# Créer la fonction de prédiction
def predict_skin_cancer_proba(model, scaler, input_data):
    input_data_scaled = scaler.transform([input_data])
    proba = model.predict_proba(input_data_scaled)[:, 1][0]
    return proba * 100

# Exemple d'entrée uniquement avec les colonnes importantes
person_data = {
    'State': 'Alabama',
    'Sex': 'Male',
    'AgeCategory': 'Age 25 to 29',
    'BMI': 25.4,
    'SmokerStatus': 'Former smoker',
    'ECigaretteUsage': 'Not at all (right now)',
    'RaceEthnicityCategory': 'White only, Non-Hispanic',
    'AlcoholDrinkers': 0,
    'HIVTesting': 0
}

# Encoder l'entrée de manière cohérente
input_data_list = []
for col in important_columns:
    if col in encoders:  # Si la colonne est catégorique
        input_data_list.append(encoders[col].transform([person_data[col]])[0])
    else:  # Si la colonne est numérique
        input_data_list.append(person_data[col])

# Prédire la probabilité
proba = predict_skin_cancer_proba(mlp, scaler, input_data_list)
print(f"Probabilité de cancer de la peau : {proba:.2%}")

# Générer des âges de 20 à 80 ans et calculer la probabilité
ages = np.arange(30, 75)
probabilities = []

for age in ages:
    input_data = input_data_list.copy()
    age_category = f"Age {age // 5 * 5} to {age // 5 * 5 + 4}"
    if age_category in encoders['AgeCategory'].classes_:
        input_data[0] = encoders['AgeCategory'].transform([age_category])[0]
    probabilities.append(predict_skin_cancer_proba(mlp, scaler, input_data))

# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(ages, probabilities, label='Probabilité de cancer de la peau', color='red')
plt.xlabel("Âge")
plt.ylabel("Probabilité")
plt.title("Probabilité de cancer de la peau en fonction de l'âge")
plt.grid(True)
plt.legend()
plt.show()
"""