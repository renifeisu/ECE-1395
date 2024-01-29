# ECE 1395
# Problem Set 2
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.utils import shuffle
from computeCost import * # computeCost function
from gradientDescent import * # gradientDescent function
from normalEqn import * # normalEqn function

#----------------------------------QUESTION-1----------------------------------------------------
# toy data set
# matrix for variables X, each row corresponds to a sample
X = np.array([[0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4]])
# vector for results y, each row corresponds to a sample
y = np.array([[8], [6], [4], [2]])
# estimates of theta
theta_1 = np.array([[0, 1, 0.5]])
theta_2 = np.array([[10, -1, -1]])
theta_3 = np.array([[3.5, 0, 0]])
# results
print('Question 1 (i) Cost : ', computeCost(X, y, theta_1))
print('Question 1 (ii) Cost : ', computeCost(X, y, theta_2))
print('Question 1 (iii) Cost : ', computeCost(X, y, theta_3))
print('\n') # space

#----------------------------------QUESTION-2----------------------------------------------------
np.random.seed(0)

t, c = gradientDescent(X, y, 0.001, 15)
# results
print('Question 2 results : \nCost after each iteration: \n', c)
print('\nEstimated Theta: \n', t) 
print('\n') # space

#----------------------------------QUESTION-3----------------------------------------------------
t = normalEqn(X, y)
# results
print('Question 3 theta: \n', t) 
print('\n') # space

#----------------------------------QUESTION-4-PART-A---------------------------------------------
# load data
data1 = pd.read_csv('input/hw2_data1.csv', sep=',', names=['HorsePower', 'Price'], header=None)
# results
print('Question 4A Horse Power: \n', data1['HorsePower']) 
print('Question 4A Price: \n', data1['Price']) 
print('\n') # space

#----------------------------------QUESTION-4-PART-B---------------------------------------------
# scatter plot of horse power vs price
scatter_data1 = plt.figure(1)
plt.scatter(data1['HorsePower'], data1['Price'])
plt.xlabel('Horse Power')
plt.ylabel('Price')
plt.savefig('output/ps2-4-b.png') # output

#----------------------------------QUESTION-4-PART-C---------------------------------------------
# store number of rows and columns in original dataframe
row, col = data1.shape
# initialize arrays for X and y
X = np.empty((row, 2), 'object')
y = np.empty((row, 1), 'object')
# features, X_0 and X_1
X[:, 1] = data1['HorsePower']
X[:, 0] = np.ones((row, ))
# labels
y[:, 0] = data1['Price']
# results
print('Question 4C X.shape: ', X.shape) 
print('Question 4C y.shape: ', y.shape) 
print('\n') # space

#----------------------------------QUESTION-4-PART-D---------------------------------------------
# a function that splits data into training and testing sets
def dataSplitter(matrix_x, vector_y, ratio):
    # rows and cols of x
    row, col = matrix_x.shape
    # create and  shuffle temp_vector
    temp_vector = []
    for i in range(row):
        temp_vector.append(i)
    temp_vector = shuffle(temp_vector)
    # number of training/testing samples
    train = math.ceil(row * ratio)
    test = row - train
    # define X_train and X_test
    X_train = np.empty((train, col), 'object')
    X_test = np.empty((test, col), 'object')
    # define y_train and y_test
    y_train = np.empty((train, 1), 'object')
    y_test = np.empty((test, 1), 'object')
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
print('Question 4D X_train (first 5 rows) : \n', X_train[0:5, :])
print('Question 4D X_test : \n', X_test)
print('Question 4D y_train (first 5 rows) : \n', y_train[0:5, :])
print('Question 4D y_test : \n', y_test)
print('\n') # space

#----------------------------------QUESTION-4-PART-E---------------------------------------------
# run function gradientDescent
[theta, cost] = gradientDescent(X_train, y_train, 0.3, 500)
# plot of cost vs iteration
line_q4 = plt.figure(2)
plt.plot(range(500), cost)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.savefig('output/ps2-4-e.png') # output
# results
print('Question 4E theta : \n', theta)
print('\n') # space

#----------------------------------QUESTION-4-PART-F---------------------------------------------
# min value of Horse Power (X_0) using infinity-norm
temp_array = 1. / X_train[:, 1] # inverse
min = 1. / np.linalg.norm(np.transpose(temp_array), np.inf)
# max value of Horse Power (X_0) using infinity-norm
max = np.linalg.norm(np.transpose(X_train[:, 1]), np.inf)
# learned line equation
learned_x = np.linspace(min, max, math.floor(row / 8))
learned_y = theta[0] + theta[1] * learned_x
# scatter plot of horse power vs price
scatter_data1 = plt.figure(3)
plt.scatter(data1['HorsePower'], data1['Price'], c='b', label='training data')
plt.scatter(learned_x, learned_y, c='r', label='learned model')
plt.xlabel('Horse Power')
plt.ylabel('Price')
plt.legend()
plt.savefig('output/ps2-4-f.png') # output

#----------------------------------QUESTION-4-PART-G---------------------------------------------
# cost (average mean squared error) from gradient descent
cost = computeCost(X_test, y_test, np.transpose(theta))
print('Question 4G Cost : ', cost)
print('\n') # space

#----------------------------------QUESTION-4-PART-H---------------------------------------------
# run function normalEqn
theta = normalEqn(X_train, y_train)
cost = computeCost(X_test, y_test, np.transpose(theta))
print('Question 4H Cost : ', cost)
print('\n') # space

#----------------------------------QUESTION-4-PART-I---------------------------------------------
alpha = [0.001, 0.003, 0.03, 3]
# run function gradientDescent (i)
[theta, cost] = gradientDescent(X_train, y_train, alpha[0], 300)
# plot of cost vs iteration 
line_q4 = plt.figure(4)
plt.plot(range(300), cost, label='alpha = 0.001')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.savefig('output/ps2-4-i-1.png') # output
# results
print('Question 4I (i) theta : \n', theta)

# run function gradientDescent (ii)
[theta, cost] = gradientDescent(X_train, y_train, alpha[1], 300)
# plot of cost vs iteration 
line_q4 = plt.figure(5)
plt.plot(range(300), cost, label='alpha = 0.003')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.savefig('output/ps2-4-i-2.png') # output
# results
print('Question 4I (ii) theta : \n', theta)

# run function gradientDescent (iii)
[theta, cost] = gradientDescent(X_train, y_train, alpha[2], 300)
# plot of cost vs iteration 
line_q4 = plt.figure(6)
plt.plot(range(300), cost, label='alpha = 0.03')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.savefig('output/ps2-4-i-3.png') # output
# results
print('Question 4I (iii) theta : \n', theta)

# run function gradientDescent (iv)
[theta, cost] = gradientDescent(X_train, y_train, alpha[3], 300)
# plot of cost vs iteration 
line_q4 = plt.figure(7)
plt.plot(range(300), cost, label='alpha = 3')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.savefig('output/ps2-4-i-4.png') # output
# results
print('Question 4I (iv) theta : \n', theta)
print('\n') # space

#----------------------------------QUESTION-5-PART-A---------------------------------------------
# load data
data3 = pd.read_csv('input/hw2_data3.csv', sep=',', names=['EngineSize', 'Weight', 'CO2Emission'], header=None)
# store number of rows and columns in original dataframe
row, col = data3.shape
# initialize arrays for X and y
X = np.empty((row, col), 'object')
y = np.empty((row, 1), 'object')
# features, X_0, X_1 and X_2
X[:, 2] = data3['Weight']
X[:, 1] = data3['EngineSize']
X[:, 0] = np.ones((row, ))
# statistics
engine_size_mean = np.mean(X[:, 1])
engine_size_std = np.std(X[:, 1])
weight_mean = np.mean(X[:, 2])
weight_std = np.std(X[:, 2])
# labels
y[:, 0] = data3['CO2Emission']
# results
print('Question 5A Engine Size Statistics : ') 
print('Question 5A Mean : ', engine_size_mean)
print('Question 5A Standard Deviation : ', engine_size_std) 
print('Question 5A Weight Statistics: ') 
print('Question 5A Mean : ', weight_mean)
print('Question 5A Standard Deviation : ', weight_std) 
print('Question 5A X.shape : ', X.shape) 
print('Question 5A y.shape : ', y.shape) 
print('\n') # space
# standardize the data
X[:, 2] = (X[:, 2] - np.mean(X[:, 2])) / np.std(X[:, 2])
X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])

#----------------------------------QUESTION-5-PART-B---------------------------------------------
# run function gradientDescent
[theta, cost] = gradientDescent(X, y, 0.01, 750)
# plot of cost vs iteration 
line_q5 = plt.figure(8)
plt.plot(range(750), cost)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.savefig('output/ps2-5-b.png') # output
# results
print('Question 5B theta : \n', theta)
print('\n') # space

#----------------------------------QUESTION-5-PART-C---------------------------------------------
# test sample
engine_size = (2300 - engine_size_mean) / engine_size_std
weight = (1300 - weight_mean) / weight_std
CO2Emission_pred = theta[0] + engine_size * theta[1] + weight * theta[2]
# results
print('Question 5C Prediction : ', CO2Emission_pred)