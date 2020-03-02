import pandas as pd
import numpy as np
import random as rnd
import pickle
import re
import os
import sys
import datetime

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import mlrose_hiive

def generate_nn_model(alg_name, hidden_nodes=[30, 20], seed=None):
    nn_model = None

    ## RHC
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
    ## SA
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
    ## GA
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


def run(alg_name, n_jobs=-1):
    SEED = 1
    
    OUTPUT_DIRECTORY = './output/nn_gridsearch'
    if OUTPUT_DIRECTORY is not None:
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)

    features = 'REGION-CENTROID-COL,REGION-CENTROID-ROW,REGION-PIXEL-COUNT, SHORT-LINE-DENSITY-5,SHORT-LINE-DENSITY-2,VEDGE-MEAN,VEDGE-SD,HEDGE-MEAN,HEDGE-SD,INTENSITY-MEAN,RAWRED-MEAN,RAWBLUE-MEAN,RAWGREEN-MEAN,EXRED-MEAN,EXBLUE-MEAN,EXGREEN-MEAN,VALUE-MEAN,SATURATION-MEAN,HUE-MEAN'
    col_names = features.split(',') + ['Class']

    dataset_dir = '../Datasets/statlog_image_segmentation'
    df = pd.read_csv(dataset_dir + '/segment.dat', names=col_names, sep=' ')

    y = df['Class'].copy().values
    X = df.drop(['Class'], axis=1).values
    # to 0 - N-1
    y = y-1

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    # y_test_hot = one_hot.fit_transform(y_test.reshape(-1, 1)).todense()

    hidden_layer_sizes=[30,20]

    nn_model = generate_nn_model(alg_name, hidden_nodes=hidden_layer_sizes, seed=SEED)

    params = {'rhc': {'max_iters':[5000, 10000], 
                      'learning_rate':[0.001, 0.01, 0.1, 0.8], 
                      'max_attempts':[50, 100]}, 
              'sa': {'max_iters':[5000, 10000], 
                     'learning_rate':[0.8], 
                     'max_attempts':[50, 100]},
              'ga': {'max_iters':[500, 1000], 
                     'learning_rate':[0.8], 
                     'max_attempts':[200],
                     'pop_size':[1000, 2000],
                     'mutation_prob': [0.1, 0.25, 0.5]}
              }

    clf = GridSearchCV(nn_model, params[alg_name], cv=3, scoring='accuracy', 
                       n_jobs=n_jobs, verbose=3, refit=True)

    clf.fit(X_train_scaled, y_train_hot)

    now = datetime.datetime.now()
    stime = now.strftime('%Y%m%d%H%M_%S%f')

    df = pd.DataFrame.from_dict([clf.best_params_])
    fname = '{0}/nngs_{1}_params_{2}.csv'.format(OUTPUT_DIRECTORY, alg_name, stime)
    df.to_csv(fname, index=False)

    fname = '{0}/nngs_{1}_esitmator_{2}.p'.format(OUTPUT_DIRECTORY, alg_name, stime)
    with open(fname, 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

    print(clf.best_params_)

    ## trainning
    print('------ Train Set ------')
    y_train_pred_hot = clf.best_estimator_.predict(X_train_scaled)
    y_train_pred = one_hot.inverse_transform(y_train_pred_hot)[:,0]

    cm_train = confusion_matrix(y_train, y_train_pred)  # Confusion Matrix
    print('Train_CM:\n', cm_train)
    acc_train = accuracy_score(y_train, y_train_pred) # Accuracy Score
    print('Train_ACC', acc_train)
    f1_train = f1_score(y_train, y_train_pred, average='macro')
    print('Train_F1', f1_train)

    print('------ Test Set ------')
    y_test_pred_hot = clf.best_estimator_.predict(X_test_scaled)
    y_test_pred = one_hot.inverse_transform(y_test_pred_hot)[:,0]
    
    cm_test = confusion_matrix(y_test, y_test_pred)  # Confusion Matrix
    print('Test_CM:\n', cm_test)
    acc_test = accuracy_score(y_test, y_test_pred) # Accuracy Score
    print('Test_ACC', acc_test)
    f1_test = f1_score(y_test, y_test_pred, average='macro')
    print('Test_F1', f1_test)

    return clf


if __name__ == '__main__':
    if (len(sys.argv) == 2):
        alg_name = str(sys.argv[1])
        run(alg_name)
    elif (len(sys.argv) == 3):
        alg_name = str(sys.argv[1])
        n_jobs = int(sys.argv[2])
        run(alg_name, n_jobs)
    else:
        print('input error')