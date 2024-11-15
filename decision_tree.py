import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tools import preprocess_data, clean_data, plot_feature_importances, plot_roc_curve

print("Starting Decision Tree model...")
# Load the dataset
data = pd.read_excel('data/dataset.xlsx')
target = 'HadSkinCancer'

print("Data loaded successfully!")

preprocessed_data, initial_data = preprocess_data(data, n_rows=None, remove_other_deseas=True)

cleaned_data, label_encoders, category_mappings = clean_data(preprocessed_data, target)

# Show mapping
for column, mapping in category_mappings.items():
    print(f"\n\nMappings pour {column}:")
    for original, encoded in mapping.items():
        print(f"  {original} -> {encoded}")

X = cleaned_data.drop(columns=[target])
y = cleaned_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depths = range(1, 21)
best_depth = None
best_accuracy = 0

# Tester chaque profondeur et évaluer le modèle
for depth in max_depths:
    clf = DecisionTreeClassifier(random_state=42, max_depth=depth)
    clf.fit(X_train, y_train)

    # Prédire et évaluer l'exactitude
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy of the Decision Tree model with max_depth={depth}: {accuracy:.2f}")

    # Vérifier si cette profondeur est la meilleure jusqu'à présent
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_depth = depth

print(f"\nBest max_depth: {best_depth} with Accuracy: {best_accuracy:.2f}")

# Entraîner le modèle avec la meilleure profondeur trouvée
best_clf = DecisionTreeClassifier(random_state=42, max_depth=9)
best_clf.fit(X_train, y_train)

# Visualiser l'arbre de décision avec la meilleure profondeur
plt.figure(figsize=(20, 10))
class_names = label_encoders[target].inverse_transform([0, 1])
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=class_names)
plt.title("Best Decision Tree Visualization")

plt.savefig('best_decision_tree_visualization.png', format='png', dpi=1000)
plt.show()

plot_feature_importances(best_clf, X.columns)

y_scores = best_clf.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive
plot_roc_curve(y_test, y_scores)