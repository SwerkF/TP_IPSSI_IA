
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