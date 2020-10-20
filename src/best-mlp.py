import pandas as pd
import sys
import os

from utils import data
from utils import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from warnings import simplefilter

# Had to be done to avoid stdout being spammed with ConvergenceWarnings
if not sys.warnoptions:
    simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

param_grid = {
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'solver': ['adam', 'sgd']
}

# Dataset 1 (Latin letters)
# Training
trainX, trainY = data.load_data('train_1.csv')
clf = GridSearchCV(MLPClassifier(), param_grid, verbose=1, n_jobs=-1)
clf.fit(trainX, trainY)
# Validation
validX, validY = data.load_data('val_1.csv')
print(f'Score: {round(clf.score(validX, validY), 3)}')
print(f'Parameters chosen: {clf.best_params_}')
# Testing
testX, testY = data.load_data('test_with_label_1.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Best-MLP-DS1.csv')
metrics.compute(predictions, testY, 'Best-MLP-DS1.csv')
data.generate_cm(predictions, testY, 'Best-MLP-DS1.png')

# Dataset 2 (Greek letters)
# Training
trainX, trainY = data.load_data('train_2.csv')
clf = GridSearchCV(MLPClassifier(), param_grid, verbose=1, n_jobs=-1)
clf.fit(trainX, trainY)
# Validation
validX, validY = data.load_data('val_2.csv')
print(f'Score: {round(clf.score(validX, validY), 3)}')
print(f'Parameters chosen: {clf.best_params_}')
# Testing
testX, testY = data.load_data('test_with_label_2.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Best-MLP-DS2.csv')
metrics.compute(predictions, testY, 'Best-MLP-DS2.csv')
data.generate_cm(predictions, testY, 'Best-MLP-DS2.png')