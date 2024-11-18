import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import joblib

# TODO différentes couches, nb epochs, learning rate, etc

def create_mlp_model_one_perceptron():
    return MLPClassifier(hidden_layer_sizes=(1,), max_iter=100, verbose=10, random_state=1, learning_rate_init=.00001)

def create_mlp_one_layer_model():
    return MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, verbose=10, random_state=1, learning_rate_init=.00001)

def create_mlp_two_layers_model():
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, verbose=10, random_state=1, learning_rate_init=.00001)

def create_mlp_three_layers_model():
    return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=100, verbose=10, random_state=1, learning_rate_init=.00001)

def create_mlp_four_layers_model():
    return MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=100, verbose=10, random_state=1, learning_rate_init=.00001)