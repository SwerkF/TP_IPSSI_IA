import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tools import preprocess_data

print("Starting Decision Tree model...")
# Load the dataset
data = pd.read_excel('data/mini-dataset.xlsx')
print("Data loaded successfully!")

# Utiliser preprocess_data
preprocessed_data, initial_data = preprocess_data(data, n_rows=5000)

# Le reste de votre code de prétraitement
# Step 2: Handle missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Separating numerical and categorical columns
numerical_cols = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = preprocessed_data.select_dtypes(include=['object']).columns

# Imputing missing values
preprocessed_data[numerical_cols] = num_imputer.fit_transform(preprocessed_data[numerical_cols])
preprocessed_data[categorical_cols] = cat_imputer.fit_transform(preprocessed_data[categorical_cols])

# Step 3: Encode categorical variables
label_encoders = {}
for column in categorical_cols:
    le = LabelEncoder()
    preprocessed_data[column] = le.fit_transform(preprocessed_data[column])
    label_encoders[column] = le
