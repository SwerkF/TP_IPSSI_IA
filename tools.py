from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc


def preprocess_data(data, n_rows=None, remove_other_deseas = False):
    """
    Preprocesses the data by keeping a certain number of rows (or all) and dropping specific columns.

    Args:
    data (pandas.DataFrame): The original DataFrame to preprocess.
    n_rows (int, optional): The number of rows to keep. If None, keeps all rows. Default is None.

    Returns:
    tuple: A tuple containing (preprocessed_data, initial_data)
    """
    if n_rows is not None:
        data = data.head(n_rows)

    initial_data = data.copy()

    columns_to_drop = ['PatientID', 'HeightInMeters', 'WeightInKilograms',
                       'ChestScan', 'FluVaxLast12', 'PneumoVaxEver',
                       'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos', 'GeneralHealth']

    colums_deases = ['HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease',
     'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyBathing', 'DifficultyErrands']

    preprocessed_data = data.drop(columns=columns_to_drop)
    if remove_other_deseas:
        preprocessed_data = preprocessed_data.drop(columns=colums_deases)

    return preprocessed_data, initial_data


def create_mapping(le):
    """
    Creates a mapping dictionary between original values and encoded values.

    Args:
    le (LabelEncoder): The label encoder used for encoding.

    Returns:
    dict: A dictionary with original values as keys and encoded values as values.
    """
    return dict(zip(le.classes_, range(len(le.classes_))))


def clean_data(data, target_column=None):
    """
    Cleans the data by handling missing values and encoding categorical variables.

    Args:
    data (pandas.DataFrame): The preprocessed data to clean.
    target_column (str): The name of the target column to encode.

    Returns:
    tuple: A tuple containing (cleaned_data, label_encoders, category_mappings)
    """

    # Handling missing values
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Imputing missing values
    data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

    # Encoding categorical variables
    label_encoders = {}
    category_mappings = {}

    for column in categorical_cols:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
        category_mappings[column] = create_mapping(le)

    # Encoding the target column only if it exists
    if target_column in data.columns and data[target_column] is not None:
        le_target = LabelEncoder()
        data[target_column] = le_target.fit_transform(data[target_column])
        label_encoders[target_column] = le_target
        category_mappings[target_column] = create_mapping(le_target)

        original_classes = le_target.classes_.astype(str)
        label_encoders[target_column].classes_ = original_classes

    return data, label_encoders, category_mappings


def plot_feature_importances(model, feature_names):
    """
    Displays the feature importances of a decision tree model.

    Args:
    model: The trained decision tree model.
    feature_names (list): The list of feature names.
    """

    # Retrieve feature importances
    importances = model.feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    """
    Calcule et affiche la courbe ROC et l'AUC.

    Args:
    y_true (array-like): Les vraies étiquettes de classe.
    y_scores (array-like): Les scores de probabilité pour la classe positive.
    """
    # Calculer la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculer l'AUC
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.2f}")

    # Tracer la courbe ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonale aléatoire
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()