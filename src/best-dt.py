import pandas as pd

from utils import data
from utils import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [10, None],
              'min_samples_split': [2, 3, 4],
              'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3],
              'class_weight': [None, 'balanced']
              }

# Dataset 1 (Latin letters)
# Training
trainX, trainY = data.load_data('train_1.csv')
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, verbose=1)
clf.fit(trainX, trainY)
# Validation
validX, validY = data.load_data('val_1.csv')
print(f'Score: {round(clf.score(validX, validY), 3)}')
print(f'Parameters chosen: {clf.best_params_}')
# Testing
testX, testY = data.load_data('test_with_label_1.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Best-DT-DS1.csv')
metrics.compute(predictions, testY, 'Best-DT-DS1.csv')
data.generate_cm(predictions, testY, 'Best-DT-DS1.png')

# Dataset 2 (Greek letters)
# Training
trainX, trainY = data.load_data('train_2.csv')
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, verbose=1)
clf.fit(trainX, trainY)
# Validation
validX, validY = data.load_data('val_2.csv')
print(f'Score: {round(clf.score(validX, validY), 3)}')
print(f'Parameters chosen: {clf.best_params_}')
# Testing
testX, testY = data.load_data('test_with_label_2.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Best-DT-DS2.csv')
metrics.compute(predictions, testY, 'Best-DT-DS2.csv')
data.generate_cm(predictions, testY, 'Best-DT-DS2.png')