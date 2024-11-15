from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data, n_rows=None):
    """
    Prétraite les données en gardant un certain nombre de lignes (ou toutes) et en supprimant des colonnes spécifiques.

    Args:
    data (pandas.DataFrame): Le DataFrame d'origine à prétraiter.
    n_rows (int, optional): Le nombre de lignes à conserver. Si None, garde toutes les lignes. Par défaut None.

    Returns:
    tuple: Un tuple contenant (données_prétraitées, données_initiales)
    """
    if n_rows is not None:
        data = data.head(n_rows)

    initial_data = data.copy()

    columns_to_drop = ['PatientID', 'HeightInMeters', 'WeightInKilograms',
                       'ChestScan', 'FluVaxLast12', 'PneumoVaxEver',
                       'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos']

    preprocessed_data = data.drop(columns=columns_to_drop)

    return preprocessed_data, initial_data


def clean_data(data):
    """
    Nettoie les données en gérant les valeurs manquantes et en encodant les variables catégorielles.

    Args:
    data (pandas.DataFrame): Les données prétraitées à nettoyer.

    Returns:
    tuple: Un tuple contenant (données_nettoyées, encodeurs_étiquettes)
    """
    # Gestion des valeurs manquantes
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

    # Encodage des variables catégorielles
    label_encoders = {}
    for column in categorical_cols:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders