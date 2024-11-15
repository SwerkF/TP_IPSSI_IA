import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tools import preprocess_data, clean_data

print("Starting Decision Tree model...")
# Load the dataset
data = pd.read_excel('data/mini-dataset.xlsx')
print("Data loaded successfully!")

# Utiliser preprocess_data
preprocessed_data, initial_data = preprocess_data(data, n_rows=5000)

cleaned_data, label_encoders = clean_data(preprocessed_data)