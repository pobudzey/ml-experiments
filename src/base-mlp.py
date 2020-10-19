import pandas as pd

from utils import data
from utils import metrics
from sklearn.neural_network import MLPClassifier

# Dataset 1 (Latin letters)
# Training
trainX, trainY = data.load_data('train_1.csv')
clf = MLPClassifier(activation='logistic', solver='sgd')
clf.fit(trainX, trainY)
# Testing
testX, testY = data.load_data('test_with_label_1.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Base-MLP-DS1.csv')
metrics.compute(predictions, testY, 'Base-MLP-DS1.csv')
data.generate_cm(predictions, testY, 'Base-MLP-DS1.png')

# Dataset 2 (Greek letters)
# Training
trainX, trainY = data.load_data('train_2.csv')
clf = MLPClassifier(activation='logistic', solver='sgd')
clf.fit(trainX, trainY)
# Testing
testX, testY = data.load_data('test_with_label_2.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Base-MLP-DS2.csv')
metrics.compute(predictions, testY, 'Base-MLP-DS2.csv')
data.generate_cm(predictions, testY, 'Base-MLP-DS2.png')