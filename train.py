import os
import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from models.VGG import create_vgg_model
from models.ResNet import create_resnet_model
from models.EfficientNet import create_efficientnet_model
from models.Mlp import create_mlp_model
from models.Sequential import create_sequential_model
from utils.data_preprocessing import load_images_from_directory, create_data_generator
from utils.metrics import evaluate_model
import joblib

# Chargement des données
train_data, train_labels = load_images_from_directory('data/images/train')
test_data, test_labels = load_images_from_directory('data/images/test')

# Création des modèles
vgg_model = create_vgg_model()
resnet_model = create_resnet_model()
efficientnet_model = create_efficientnet_model()
mlp_model = create_mlp_model()
sequential_model = create_sequential_model()

# Compilation des modèles
vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
efficientnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
sequential_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Utilisation de l'augmentation des données
datagen = create_data_generator()
datagen.fit(train_data)

# Entraînement des modèles
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

def train_and_evaluate_model(model, model_name):
    start_time = time.time()
    model.fit(datagen.flow(train_data, train_labels, batch_size=32),
              validation_data=(test_data, test_labels),
              epochs=10, callbacks=[early_stopping])
    training_time = time.time() - start_time

    start_time = time.time()
    accuracy, auc = evaluate_model(model, test_data, test_labels)
    evaluation_time = time.time() - start_time

    print(f'{model_name} - Accuracy: {accuracy}, AUC: {auc}, Training Time: {training_time:.2f}s, Evaluation Time: {evaluation_time:.2f}s')
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'training_time': training_time,
        'evaluation_time': evaluation_time
    }

# Liste pour stocker les résultats
results = []

# Entraîner et évaluer chaque modèle
results.append(train_and_evaluate_model(vgg_model, 'VGG'))
results.append(train_and_evaluate_model(resnet_model, 'ResNet'))
results.append(train_and_evaluate_model(efficientnet_model, 'EfficientNet'))

# Entraînement et évaluation du modèle MLP
start_time = time.time()
mlp_model.fit(train_data.reshape(train_data.shape[0], -1), train_labels)
training_time = time.time() - start_time

start_time = time.time()
mlp_accuracy, mlp_auc = evaluate_model(mlp_model, test_data.reshape(test_data.shape[0], -1), test_labels)
evaluation_time = time.time() - start_time

print(f'MLP - Accuracy: {mlp_accuracy}, AUC: {mlp_auc}, Training Time: {training_time:.2f}s, Evaluation Time: {evaluation_time:.2f}s')
results.append({
    'model_name': 'MLP',
    'accuracy': mlp_accuracy,
    'training_time': training_time,
    'evaluation_time': evaluation_time
})

# Entraînement et évaluation du modèle Sequential
results.append(train_and_evaluate_model(sequential_model, 'Sequential'))

# Créer un répertoire pour les modèles sauvegardés
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')

# Sauvegarde des modèles entraînés
vgg_model.save('./saved_models/vgg_model.keras')
resnet_model.save('./saved_models/resnet_model.keras')
efficientnet_model.save('./saved_models/efficientnet_model.keras')
joblib.dump(mlp_model, './saved_models/mlp_model.pkl')
sequential_model.save('./saved_models/sequential_model.keras')

# Afficher les résultats
for result in results:
    print(result)