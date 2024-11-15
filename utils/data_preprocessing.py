import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_from_directory(directory, target_size=(224, 224)):
    images = []
    labels = []
    for label in ['benign', 'malignant']:
        label_path = os.path.join(directory, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = image.load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)  # Normalisation pour VGG
            images.append(img_array)
            labels.append(0 if label == 'benign' else 1)
    return np.array(images), np.array(labels)

def create_data_generator(batch_size=32):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen