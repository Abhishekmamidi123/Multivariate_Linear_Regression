import numpy as np
import pandas as pd
from sklearn import linear_model

train_filename = 'data/brainhead_train.csv'
test_filename = 'data/brainhead_test.csv'

dataTrain = pd.read_csv(train_filename)
dataTest = pd.read_csv(test_filename)

x_train = dataTrain[['Gender', 'Age_range', 'Head_size']]
y_train = dataTrain['Brain_weight']

x_test = dataTest[['Gender', 'Age_range', 'Head_size']]
y_test = dataTest['Brain_weight']

ols = linear_model.LinearRegression()
model = ols.fit(x_train, y_train)

print model.predict(x_test)[0:5]
M = 5
for i in range(1, M+1):
	x_train1 = dataTrain[['Gender', 'Age_range', 'Head_size']]**i
	x_test1 = dataTest[['Gender', 'Age_range', 'Head_size']]**i
	
	x_train = np.column_stack([x_train, x_train1])
	x_test = np.column_stack([x_test, x_test1])

	ols = linear_model.LinearRegression()
	model = ols.fit(x_train, y_train)

	print model.predict(x_test)[0:5]
