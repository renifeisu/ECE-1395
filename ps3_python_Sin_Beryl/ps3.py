# ECE 1395
# Problem Set 2
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.utils import shuffle
from scipy.optimize import fmin_bfgs
from sigmoid import * # sigmoid function
from costFunction import * # cost function
from gradFunction import * # cost gradient function

#----------------------------------QUESTION-1-PART-A---------------------------------------------
# load data
data1 = pd.read_csv('input/hw3_data1.txt', sep=',', names=['Exam1', 'Exam2', 'Decision'], header=None)
# store number of rows and columns in original dataframe
row, col = data1.shape
# initialize arrays for X and y
X = np.empty((row, col), 'float')
y = np.empty((row, 1), 'float')
# features, X_0, X_1, and X_2
X[:, 2] = data1['Exam2']
X[:, 1] = data1['Exam1']
X[:, 0] = np.ones((row, ))
# labels
y[:, 0] = data1['Decision']
# results
print('Question 1A X.shape: ', X.shape) 
print('Question 1A y.shape: ', y.shape) 
print('\n') # space

#----------------------------------QUESTION-1-PART-B---------------------------------------------
# scatter plot
scatter_data1 = plt.figure(1)
admitted = plt.scatter(x=data1.loc[data1['Decision'].eq(1), 'Exam1'], y=data1.loc[data1['Decision'].eq(1), 'Exam2'], c='tab:blue', label='Admitted')
not_admitted = plt.scatter(x=data1.loc[data1['Decision'].eq(0), 'Exam1'], y=data1.loc[data1['Decision'].eq(0), 'Exam2'], c='tab:cyan', label='Not Admitted')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.savefig('output/ps3-1-b.png') # output

#----------------------------------QUESTION-1-PART-C---------------------------------------------
# a function that splits data into training and testing sets
def dataSplitter(matrix_x, vector_y, ratio):
    # rows and cols of x
    row, col = matrix_x.shape
    # create and shuffle temp_vector
    temp_vector = []
    for i in range(row):
        temp_vector.append(i)
    temp_vector = shuffle(temp_vector)
    # number of training/testing samples
    train = math.ceil(row * ratio)
    test = row - train
    # define X_train and X_test
    X_train = np.empty((train, col), 'float')
    X_test = np.empty((test, col), 'float')
    # define y_train and y_test
    y_train = np.empty((train, 1), 'float')
    y_test = np.empty((test, 1), 'float')
    # used shuffled vector to determine which rows from matrix_x become X_train and X_test
    # and which rows from vector_y become y_train and y_test
    for i in range(len(temp_vector)):
        if i < train:
            X_train[i, :] = matrix_x[temp_vector[i], :]
            y_train[i, :] = vector_y[temp_vector[i], :]
        else:
            X_test[i - train, :] = matrix_x[temp_vector[i], :]
            y_test[i - train, :] = vector_y[temp_vector[i], :]
    ratio = 0
    return [X_train, X_test, y_train, y_test]

# run function dataSplitter
[X_train, X_test, y_train, y_test] = dataSplitter(X, y, 0.9)
# print out results
print('Question 1C X_train (first 5 rows): \n', X_train[0:5, :])
print('Question 1C X_test (first 5 rows): \n', X_test[0:5, :])
print('Question 1C y_train (first 5 rows): \n', y_train[0:5, :])
print('Question 1C y_test (first 5 rows): \n', y_test[0:5, :])
print('\n') # space

#----------------------------------QUESTION-1-PART-D---------------------------------------------
# test vector, z
z = np.arange(-15, 15, 0.01)
# run function sigmoid
gz = sigmoid(z)
# plot of gz vs z
plot_sigmoid = plt.figure(2)
plt.plot(z, gz)
plt.xlabel('z')
plt.ylabel('gz')
plt.savefig('output/ps3-1-c.png') # output
# finding value of z that makes gz closest to 0.1
print('Question 1D gz=0.1, z=: ', -1 * math.log(9))
print('\n') # space

#----------------------------------QUESTION-1-PART-E---------------------------------------------
# toy dataset
toyset =np.array([[1, 0, 0], [1, 3, 1], [3, 1, 0], [3, 4, 1]])
# rows and cols of toyset df
row, col = toyset.shape
# initialize feature matrix and label vector
toy_X = np.empty((row, col), 'float')
toy_X[:, 2] = toyset[:, 1]
toy_X[:, 1] = toyset[:, 0]
toy_X[:, 0] = np.ones((row, ), 'float')
toy_y = np.empty((row, 1), 'float')
toy_y[:, 0] = toyset[:, 2]
# theta
theta = np.array([[2, 0, 0]])
# run both functions, results
print('Question 1E Cost: ', costFunction(theta, toy_X, toy_y))
print('Question 1E Gradient: \n', gradFunction(theta, toy_X, toy_y))
print('\n') # space

#----------------------------------QUESTION-1-PART-F---------------------------------------------
# initialize theta
theta = np.array([[0, 0, 0]])
# optimize cost function using fmin_bfgs
xopt = fmin_bfgs(costFunction, theta, gradFunction, args=(X_train, y_train))
# results
print('Question 1F Optimal Theta: ', xopt)
print('Question 1F Cost: \n', costFunction(xopt, X_train, y_train))
print('\n') # space

#----------------------------------QUESTION-1-PART-G---------------------------------------------
# row, col of dataset
row, col = data1.shape
# min value of Exam 1 Score (X1) using infinity-norm
temp_array = 1. / X_train[:, 1] # inverse
min1 = 1. / np.linalg.norm(np.transpose(temp_array), np.inf)
# max value of Exam 1 Score (X1) using infinity-norm
max1 = np.linalg.norm(np.transpose(X_train[:, 1]), np.inf)
# learned line equation
learned_x = np.linspace(min1, max1, 100)
learned_y = -xopt[0]/xopt[2] - xopt[1]/xopt[2] * learned_x
# scatter plot of exam scores + learned equation
scatter2_data1 = plt.figure(3)
admitted2 = plt.scatter(x=data1.loc[data1['Decision'].eq(1), 'Exam1'], y=data1.loc[data1['Decision'].eq(1), 'Exam2'], c='tab:blue', label='Admitted')
not_admitted2 = plt.scatter(x=data1.loc[data1['Decision'].eq(0), 'Exam1'], y=data1.loc[data1['Decision'].eq(0), 'Exam2'], c='tab:cyan', label='Not Admitted')
plt.plot(learned_x, learned_y, c='r')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.savefig('output/ps3-1-f.png') # output

#----------------------------------QUESTION-1-PART-H---------------------------------------------
# row, col of y_test
row, col = y_test.shape
# use learned equation on testing dataset
test_results = xopt[0] * X_test[:, 0] + xopt[1] * X_test[:, 1] + xopt[2] * X_test[:, 2]
test_results = test_results.reshape((10, 1)) # make sure shape is 10, 1 not 10, 
# make test_results 0 or 1 (if +, 1 and if -, 0)
test_results = np.where(test_results < 0, 0, 1)
accuracy = 1 - np.sum(np.abs(y_test - test_results)) / row
# results
print('Question 1H Test Results:\n ', test_results)
print('Question 1H y_test:\n ', y_test)
print('Question 1H Accuracy: ', accuracy)
print('\n') # space

#----------------------------------QUESTION-1-PART-I---------------------------------------------
# student's scores
test1 = 60
test2 = 65
# z
test_results = xopt[0] + xopt[1] * test1 + xopt[2] * test2
# Probability
p = sigmoid(test_results)
# result
test_results = np.where(test_results < 0, 0, 1)
# print statements
print('Question 1I Probability: ', p)
print('Question 1I Decision: ', test_results)
print('\n') # space

#----------------------------------QUESTION-2-PART-A---------------------------------------------
# load data
data2 = pd.read_csv('input/hw3_data2.csv', sep=',', names=['Population', 'Profit'], header=None)
# store number of rows and columns in original dataframe
row, col = data2.shape
# initialize arrays for X and y
X = np.empty((row, col + 1), 'float')
y = np.empty((row, 1), 'float')
# features, X_0 and X_1
X[:, 2] = np.square(data2['Population'])
X[:, 1] = data2['Population']
X[:, 0] = np.ones((row, ))
# labels
y[:, 0] = data2['Profit']
# function that computes the closed-form solution to linear regression
def normalEqn(X_train, y_train):
    # theta = (X'* X)^-1 * (X' * y)
    # astype was to ensure that the datatypes of the elements stayed the same when transposed..
    left = np.linalg.pinv(np.dot(np.transpose(X_train), X_train)) # n x n
    right = np.dot(np.transpose(X_train), y_train) # n x 1
    theta = np.dot(left, right) # n x 1
    return theta
# use normalEqn function
theta = normalEqn(X, y)
# results
print('Question 2A theta:\n', theta) 
print('\n') # space

#----------------------------------QUESTION-2-PART-B---------------------------------------------
# row, col of X
row, col = X.shape
# min value of Population (X1) using infinity-norm
temp_array = 1. / X[:, 1] # inverse
min1 = 1. / np.linalg.norm(np.transpose(temp_array), np.inf)
# max value of Population (X1) using infinity-norm
max1 = np.linalg.norm(np.transpose(X[:, 1]), np.inf)
# learned line equation
learned_x = np.linspace(min1, max1, 100)
learned_y = theta[0] + theta[1] * learned_x + theta[2] * np.square(learned_x)
# scatter plot of exam scores + learned equation
scatter_data2 = plt.figure(4)
plt.scatter(X[:, 1], y, c='tab:blue', label='training data')
plt.plot(learned_x, learned_y, c='r', label='fitted model')
plt.xlabel('Population in Thousands')
plt.ylabel('Profit')
plt.legend()
plt.savefig('output/ps3-2-b.png') # output

