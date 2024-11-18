import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
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
# Méthode du coude pour KNN
# ===========================

# Calcul des erreurs pour différents nombres de voisins
errors = []
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    errors.append(error)

# Tracer la courbe du coude
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), errors, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Neighbors')
plt.ylabel('Taux d\'Erreur')
plt.show()

# ===========================
# Modèle K-Nearest Neighbors
# ===========================

# Entraînement et évaluation du modèle KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Résultats du modèle KNN
print("\n=== Résultats KNN ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred_knn):.2f}")

# Tester différents paramètres
print("\nTest de différents nombres de voisins...")
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Pour k={k}, Accuracy: {acc:.2f}")
    
# ROC-AUC Score
print("\n=== ROC-AUC ===")
y_prob = knn.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc_score:.2f}")