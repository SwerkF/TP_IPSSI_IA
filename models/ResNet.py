from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50

def create_resnet_model(input_shape=(224, 224, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)  # Classification binaire
    model = Model(inputs=base_model.input, outputs=x)

    # Congeler les couches de base du modèle pré-entraîné
    for layer in base_model.layers:
        layer.trainable = False

    return model
