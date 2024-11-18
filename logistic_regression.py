import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tools import preprocess_data, clean_data

# Étape 1 : Charger le dataset
data = pd.read_excel('data/dataset.xlsx')
print("Data loaded successfully!")

# Étape 2 : Prétraitement des données
preprocessed_data, initial_data = preprocess_data(data, n_rows=None, remove_other_diseases=True)
cleaned_data, label_encoders, category_mappings = clean_data(preprocessed_data)

# Étape 3 : Séparation des caractéristiques et de la cible
X = cleaned_data.drop('HadSkinCancer', axis=1)
y = cleaned_data['HadSkinCancer']

# Étape 4 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 5 : Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===========================
# Logistic Regression
# ===========================

# Initialisation et entraînement du modèle
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Évaluation du modèle Logistic Regression
y_pred_log = log_reg.predict(X_test)

print("\n=== Logistic Regression ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, zero_division=0))

# Importance des caractéristiques 
feature_names = X.columns
coef_log = log_reg.coef_[0]
sorted_indices_log = np.argsort(np.abs(coef_log))[::-1]

# Visualisation
plt.figure(figsize=(10, 6))
plt.title("Importance des caractéristiques - Logistic Regression")
plt.bar(range(len(feature_names)), coef_log[sorted_indices_log], align="center", color="orange")
plt.xticks(range(len(feature_names)), feature_names[sorted_indices_log], rotation=45, ha='right')
plt.tight_layout()
plt.show()