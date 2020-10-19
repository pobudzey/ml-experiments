import pandas as pd

from utils import data
from utils import metrics
from sklearn.tree import DecisionTreeClassifier

# Dataset 1 (Latin letters)
# Training
trainX, trainY = data.load_data('train_1.csv')
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(trainX, trainY)
# Testing
testX, testY = data.load_data('test_with_label_1.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Base-DT-DS1.csv')
metrics.compute(predictions, testY, 'Base-DT-DS1.csv')
data.generate_cm(predictions, testY, 'Base-DT-DS1.png')

# Dataset 2 (Greek letters)
# Training
trainX, trainY = data.load_data('train_2.csv')
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(trainX, trainY)
# Testing
testX, testY = data.load_data('test_with_label_2.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'Base-DT-DS2.csv')
metrics.compute(predictions, testY, 'Base-DT-DS2.csv')
data.generate_cm(predictions, testY, 'Base-DT-DS2.png')