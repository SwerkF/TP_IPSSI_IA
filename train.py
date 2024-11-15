import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from models.VGG import create_vgg_model
from models.ResNet import create_resnet_model
from models.EfficientNet import create_efficientnet_model
from utils.data_preprocessing import load_images_from_directory, create_data_generator
from utils.metrics import evaluate_model

# Chargement des données
train_data, train_labels = load_images_from_directory('data/images/train')
test_data, test_labels = load_images_from_directory('data/images/test')

# Création des modèles
vgg_model = create_vgg_model()
resnet_model = create_resnet_model()
efficientnet_model = create_efficientnet_model()

# Compilation des modèles
vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
efficientnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Utilisation de l'augmentation des données
datagen = create_data_generator()
datagen.fit(train_data)

# Entraînement des modèles
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

vgg_model.fit(datagen.flow(train_data, train_labels, batch_size=32),
              validation_data=(test_data, test_labels),
              epochs=10, callbacks=[early_stopping])

resnet_model.fit(datagen.flow(train_data, train_labels, batch_size=32),
                 validation_data=(test_data, test_labels),
                 epochs=10, callbacks=[early_stopping])

efficientnet_model.fit(datagen.flow(train_data, train_labels, batch_size=32),
                       validation_data=(test_data, test_labels),
                       epochs=10, callbacks=[early_stopping])

# Évaluation des modèles
vgg_accuracy, vgg_auc = evaluate_model(vgg_model, test_data, test_labels)
resnet_accuracy, resnet_auc = evaluate_model(resnet_model, test_data, test_labels)
efficientnet_accuracy, efficientnet_auc = evaluate_model(efficientnet_model, test_data, test_labels)

print(f'VGG Accuracy: {vgg_accuracy}, AUC: {vgg_auc}')
print(f'ResNet Accuracy: {resnet_accuracy}, AUC: {resnet_auc}')
print(f'EfficientNet Accuracy: {efficientnet_accuracy}, AUC: {efficientnet_auc}')

# Sauvegarde des modèles entraînés
vgg_model.save('./saved_models/vgg_model.keras')
resnet_model.save('./saved_models/resnet_model.keras')
efficientnet_model.save('./saved_models/efficientnet_model.keras')