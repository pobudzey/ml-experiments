import pandas as pd

from utils import data
from sklearn.naive_bayes import GaussianNB

# Training
trainX, trainY = data.load_data('train_1.csv')
clf = GaussianNB()
clf.fit(trainX, trainY)

# Testing
testX, testY = data.load_data('test_with_label_1.csv')
predictions = pd.DataFrame(clf.predict(testX))
data.generate_csv(predictions, 'GNB-DS1.csv')
data.generate_cm(predictions, testY, 'GNB-DS1.png')