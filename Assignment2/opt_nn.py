import pandas as pd
import numpy as np
import random as rnd
import re
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.datasets import load_iris

import mlrose_hiive

def generate_nn_model(alg_name, hidden_nodes=[30, 20], seed=None):
    nn_model = None

    ## Four Peaks
    if (alg_name == 'rhc'):
        nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                                              algorithm='random_hill_climb',
                                              restarts=50,
                                              bias=True, is_classifier=True,
                                              early_stopping=True, clip_max=5,
                                              random_state=seed,
                                              max_iters=1000,
                                              learning_rate=0.001,
                                              max_attempts=100)
    ## Flip Flop
    elif (alg_name == 'sa'):
        nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                                              algorithm='simulated_annealing',
                                              schedule=mlrose_hiive.ExpDecay(),
                                              bias=True, is_classifier=True,
                                              early_stopping=True, clip_max=5,
                                              random_state=seed,
                                              max_iters=1000,
                                              learning_rate=0.0001,
                                              max_attempts=100)
    ## Knapsack
    elif (alg_name == 'ga'):
        nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                                              algorithm='genetic_alg',
                                              pop_size = 200,
                                              mutation_prob = 0.25,
                                              bias=True, is_classifier=True,
                                              early_stopping=True, clip_max=5,
                                              random_state=seed,
                                              max_iters=1000,
                                              learning_rate=0.0001,
                                              max_attempts=100)
    else:
       print('Algorithm Name Error') 

    return nn_model


# Load the Iris dataset
data = load_iris()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
                                                    test_size = 0.2, random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

# Initialize neural network object and fit object
nn_rhc = generate_nn_model('rhc', 1)