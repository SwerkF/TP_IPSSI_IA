import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_curve, auc
from tools import preprocess_data, clean_data, plot_feature_importances

print("Starting Decision Tree model...")

# Load the dataset
data = pd.read_excel('data/dataset.xlsx')
target = 'HadSkinCancer'

print("Data loaded successfully!")

# Preprocess and clean data
preprocessed_data, initial_data = preprocess_data(data, n_rows=None, remove_other_deseas=True)
cleaned_data, label_encoders, category_mappings = clean_data(preprocessed_data, target)

# Show mappings
for column, mapping in category_mappings.items():
    print(f"\n\nMappings pour {column}:")
    for original, encoded in mapping.items():
        print(f"  {original} -> {encoded}")

X = cleaned_data.drop(columns=[target])
y = cleaned_data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize variables for tracking results
max_depths = range(1, 21)
best_depth = None
best_accuracy = 0
results = []  # Store results for the summary table

# Output directory
output_dir = 'data/images'
os.makedirs(output_dir, exist_ok=True)

# Test each depth and evaluate the model
for depth in max_depths:
    clf = DecisionTreeClassifier(random_state=42, max_depth=depth)
    clf.fit(X_train, y_train)

    # Predict and evaluate accuracy
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    size_tree = clf.tree_.node_count

    print(f"Accuracy of the Decision Tree model with max_depth={depth}: Train={train_accuracy:.3f}, Test={test_accuracy:.3f}")

    # Append the results to the summary
    results.append({
        'Max Depth': depth,
        'Train Accuracy': round(train_accuracy, 3),
        'Test Accuracy': round(test_accuracy, 3),
        'Accuracy Gap': round(abs(train_accuracy - test_accuracy), 3),
        'Tree Size': size_tree
    })

    # Check if this depth is the best so far
    if test_accuracy >= best_accuracy:
        best_accuracy = test_accuracy
        best_depth = depth

print(f"\nBest max_depth: {best_depth} with Test Accuracy: {best_accuracy:.3f}")

# Train the model with the best depth
best_clf = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
best_clf.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
class_names = label_encoders[target].inverse_transform([0, 1])
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=class_names)
plt.title("Best Decision Tree Visualization")
decision_tree_path = os.path.join(output_dir, 'best_decision_tree_visualization.png')
plt.savefig(decision_tree_path, format='png', dpi=300)
plt.close()

# Plot feature importances for the best depth
importances = best_clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances for Best Depth')
feature_importances_path = os.path.join(output_dir, 'feature_importances_best_depth.png')
plt.savefig(feature_importances_path, format='png', dpi=300)
plt.close()

# Plot ROC curve
y_scores = best_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
plt.savefig(roc_curve_path, format='png', dpi=300)
plt.close()

# Feature Importances for depth = 9
depth_9_clf = DecisionTreeClassifier(random_state=42, max_depth=9)
depth_9_clf.fit(X_train, y_train)

importances_depth_9 = depth_9_clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances_depth_9, color='lightgreen')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances for Depth 9')
feature_importances_depth_9_path = os.path.join(output_dir, 'feature_importances_depth_9.png')
plt.savefig(feature_importances_depth_9_path, format='png', dpi=300)
plt.close()

# Create a summary table of results
results_df = pd.DataFrame(results)
results_df['Best Model'] = results_df['Max Depth'] == best_depth

# Save the summary table as a CSV file
results_csv_path = os.path.join(output_dir, 'decision_tree_summary.csv')
results_df.to_csv(results_csv_path, index=False)

# Plot Train Accuracy, Test Accuracy and Accuracy Gap
plt.figure(figsize=(10, 6))
plt.plot(results_df['Max Depth'], results_df['Train Accuracy'], label='Train Accuracy', marker='o')
plt.plot(results_df['Max Depth'], results_df['Test Accuracy'], label='Test Accuracy', marker='o')
plt.plot(results_df['Max Depth'], results_df['Accuracy Gap'], label='Accuracy Gap', marker='o')

plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy and Accuracy Gap')
plt.legend()
plt.grid()
accuracy_gap_path = os.path.join(output_dir, 'accuracy_gap_analysis.png')
plt.savefig(accuracy_gap_path, format='png', dpi=300)
plt.close()

# Visualize the summary table with matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.round(3).values,  # Round values to 3 decimals
                 colLabels=results_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_df.columns))))

summary_table_path = os.path.join(output_dir, 'decision_tree_summary_table.png')
plt.savefig(summary_table_path, format='png', dpi=300)
plt.close()

# Print file paths for all saved visualizations
print("Saved visualizations:")
print(f"1. Decision Tree Visualization: {decision_tree_path}")
print(f"2. Feature Importances (Best Depth): {feature_importances_path}")
print(f"3. ROC Curve: {roc_curve_path}")
print(f"4. Accuracy Gap Analysis: {accuracy_gap_path}")
print(f"5. Feature Importances (Depth 9): {feature_importances_depth_9_path}")
print(f"6. Decision Tree Summary Table: {summary_table_path}")
