import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def find_average_error(y_pred, y):
	absolute = np.absolute(y_pred - y)
	error = np.sum(absolute)
	average_error = (error*1.0)/len(absolute)
	return average_error
	
def MVLR(train_filename, test_filename, M):
	M_values = [1]
	train_errors = []
	test_errors = []

	dataTrain = pd.read_csv(train_filename)
	dataTest = pd.read_csv(test_filename)
	
	x_train = dataTrain[['Gender', 'Age_range', 'Head_size']]
	y_train = dataTrain['Brain_weight']
	
	x_test = dataTest[['Gender', 'Age_range', 'Head_size']]
	y_test = dataTest['Brain_weight']
	
	ols = linear_model.LinearRegression()
	model = ols.fit(x_train, y_train)
	
	y_train_pred = model.predict(x_train)
	train_errors.append(find_average_error(y_train_pred, y_train))
	
	y_test_pred = model.predict(x_test)
	test_errors.append(find_average_error(y_test_pred, y_test))
	
	for i in range(2, M+1):
		M_values.append(i)
		x_train1 = dataTrain[['Gender', 'Age_range', 'Head_size']]**i
		x_test1 = dataTest[['Gender', 'Age_range', 'Head_size']]**i
		
		x_train = np.column_stack([x_train, x_train1])
		x_test = np.column_stack([x_test, x_test1])
	
		model = ols.fit(x_train, y_train)
	
		y_train_pred = model.predict(x_train)
		train_errors.append(find_average_error(y_train_pred, y_train))
	
		y_test_pred = model.predict(x_test)
		test_errors.append(find_average_error(y_test_pred, y_test))
	
	return M_values, train_errors, test_errors

def plot(M_values, train_errors, test_errors, title):
	plt.plot(M_values, train_errors, label = 'train_error')
	plt.plot(M_values, test_errors, label = 'test_error')
	plt.xlabel("M")
	plt.ylabel("Average Error")
	plt.legend(loc='best')
	plt.title(title)
	plt.savefig('Average_error_vs_M.png')
	plt.show()
	plt.close()
	return 0
	
# Main function
train_filename = 'data/brainhead_train.csv'
test_filename = 'data/brainhead_test.csv'
M = 6
M_values, train_errors, test_errors = MVLR(train_filename, test_filename, M)
print M_values
print train_errors
print test_errors

title = 'Linear regression'
plot(M_values, train_errors, test_errors, title)
