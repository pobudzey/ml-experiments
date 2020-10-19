import pandas as pd

from utils import data
from utils import metrics
from sklearn.linear_model import Perceptron

# Dataset 1 (Latin letters)
# Training
trainX, trainY = data.load_data('train_1.csv')
clf = Perceptron()
clf.fit(trainX, trainY)
# Testing
testX, testY = data.load_data('test_with_label_1.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'PER-DS1.csv')
metrics.compute(predictions, testY, 'PER-DS1.csv')
data.generate_cm(predictions, testY, 'PER-DS1.png')

# Dataset 2 (Greek letters)
# Training
trainX, trainY = data.load_data('train_2.csv')
clf = Perceptron()
clf.fit(trainX, trainY)
# Testing
testX, testY = data.load_data('test_with_label_2.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'PER-DS2.csv')
metrics.compute(predictions, testY, 'PER-DS2.csv')
data.generate_cm(predictions, testY, 'PER-DS2.png')