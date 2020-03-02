import mlrose_hiive
from mlrose_hiive.runners import NNGSRunner
from mlrose_hiive import ArithDecay, ExpDecay, GeomDecay
from mlrose_hiive import algorithms as alg

import pandas as pd
import numpy as np
import random as rnd
import re

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

SEED = 1
OUTPUT_DIRECTORY = './output'

features = 'REGION-CENTROID-COL,REGION-CENTROID-ROW,REGION-PIXEL-COUNT, SHORT-LINE-DENSITY-5,SHORT-LINE-DENSITY-2,VEDGE-MEAN,VEDGE-SD,HEDGE-MEAN,HEDGE-SD,INTENSITY-MEAN,RAWRED-MEAN,RAWBLUE-MEAN,RAWGREEN-MEAN,EXRED-MEAN,EXBLUE-MEAN,EXGREEN-MEAN,VALUE-MEAN,SATURATION-MEAN,HUE-MEAN'
col_names = features.split(',') + ['Class']

dataset_dir = '../Datasets/statlog_image_segmentation'
df = pd.read_csv(dataset_dir + '/segment.dat', names=col_names, sep=' ')

column_names = df.columns
y = df['Class'].copy().values
X = df.drop(['Class'], axis=1).values

# Normalize feature data
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.fit_transform(y_test.reshape(-1, 1)).todense()


grid_search_parameters = ({
    'max_iters': [400],
    'learning_rate': [0.01],                         # nn params
    'schedule': [ExpDecay()]  # sa params
})

hidden_layer_sizes=[30, 20]

# nnr = NNGSRunner(x_train=X_train_scaled,
#                  y_train=y_train_hot,
#                  x_test=X_test_scaled,
#                  y_test=y_test_hot,
#                  experiment_name='nn',
#                  output_directory=OUTPUT_DIRECTORY,
#                  algorithm=alg.genetic_alg,
#                  grid_search_parameters=grid_search_parameters,
#                  iteration_list=[10000],
#                  hidden_layer_sizes=[hidden_layer_sizes],
#                  activation=[mlrose_hiive.neural.activation.relu],
#                  bias=True,
#                  early_stopping=True,
#                  clip_max=5,
#                  max_attempts=500,
#                  generate_curves=True,
#                  seed=SEED)

nnr = NNGSRunner(x_train=X_train_scaled,
                 y_train=y_train_hot,
                 x_test=X_test_scaled,
                 y_test=y_test_hot,
                 experiment_name='nn',
                 output_directory=OUTPUT_DIRECTORY,
                 algorithm=alg.genetic_alg,
                 pop_size = [2000],
                 mutation_prob = [0.25],
                 grid_search_parameters=grid_search_parameters,
                 iteration_list=[300],
                 hidden_layer_sizes=[hidden_layer_sizes],
                 activation=[mlrose_hiive.neural.activation.relu],
                 bias=True,
                 early_stopping=True,
                 clip_max=5,
                 max_attempts=500,
                 generate_curves=True,
                 n_jobs=2,
                 seed=SEED)

results = nnr.run()

print(nnr.best_params)

