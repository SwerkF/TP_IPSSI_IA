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

def create_mlp_model():
    return MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.00001)