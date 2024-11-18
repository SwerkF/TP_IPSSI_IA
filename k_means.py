import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
# Méthode du coude pour K-Means
# ===========================

# Initialisation de la liste pour la somme des erreurs quadratiques (SSE)
sse = []
for i in range(1, 21):
    km = KMeans(n_clusters=i, init='random', n_init=10, random_state=0)
    km.fit(X_train) 
    sse.append(km.inertia_)

# Tracer la courbe du coude
plt.plot(range(1, 21), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# ===========================
# K-Means Clustering
# ===========================

# Exécution de KMeans avec k=2
km = KMeans(n_clusters=2, init='random', n_init=10, random_state=0)
km.fit(X_scaled)

# Attribution des labels de clusters au dataset
cleaned_data['cluster'] = km.labels_

# Réduction de dimensions avec PCA pour la visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)  

# Ajout des coordonnées PCA au dataset pour la visualisation
cleaned_data['PCA1'] = X_pca[:, 0]
cleaned_data['PCA2'] = X_pca[:, 1]

# Visualisation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', palette='Set1', data=cleaned_data)
plt.title('K-Means Clustering with PCA Reduction')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# ==============================
# Visualisation de la dispersion 
# ==============================

# Visualisation de la dispersion des individus ayant ou non un cancer de la peau
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='HadSkinCancer', palette='Set2', data=cleaned_data)
plt.title('Dispersion of Skin Cancer in the Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()