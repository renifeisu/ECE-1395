# ECE 1395
# Problem Set 4
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from Reg_normalEqn import * # regularized normal equation
from dataSplitter import * # split data into test and training sets
from computeCost import * # compute cost function
from logReg_multi import * # one vs all approach to multiclassfication

#----------------------------------QUESTION-1-PART-B---------------------------------------------
# load hw4_data1.mat
data1 = sio.loadmat('input/hw4_data1.mat')
# m samples, n features (excluding bias)
m, n = data1['X_data'].shape
# label vector
y = data1['y']
# empty feature matrix
X = np.empty((m, n + 1), dtype=float)
# append bias (X0) to X_data
X[:, 0] = np.ones((m, ))
X[:, 1:n+1] = data1['X_data']
# result
print('Question 1B X.shape: ', X.shape) 
print('Question 1B y.shape: ', y.shape) 

#----------------------------------QUESTION-1-PART-C---------------------------------------------
# keep track of iterations
training_error = np.empty((20, 8), dtype=float)
testing_error = np.empty((20, 8), dtype=float)
for i in range(20):
    # 1C-i: Split data, 85% training
    [X_train, X_test, y_train, y_test] = dataSplitter(X, y, 0.85)
    # 1C-ii: Train 8 linear regression models using lambda values
    l = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017]
    for j in range(len(l)):
        theta = Reg_normalEqn(X_train, y_train, l[j])
        # 1C-iii: compute cost using both sets and theta
        training_error[i, j] = computeCost(X_train, y_train, theta)
        testing_error[i, j] = computeCost(X_test, y_test, theta)       
# average error of each lambda value using 1-norm
avg_training_error = np.mean(training_error, 0)
avg_testing_error = np.mean(testing_error, 0)
# plot of error vs lambda
plot_error = plt.figure(1)
plt.plot(l, avg_training_error, c='tab:blue', marker='o', label='training error')
plt.plot(l, avg_testing_error, c='tab:cyan', marker='o', label='testing error')
plt.xlabel('lambda')
plt.ylabel('Average Error')
plt.legend()
plt.savefig('output/ps4-1-a.png') # output

#----------------------------------QUESTION-2-PART-A---------------------------------------------
# load hw4_data2.mat
data2 = sio.loadmat('input/hw4_data2.mat')
# initialize feaure and label matrices
X_sets = np.array([data2['X1'], data2['X2'], data2['X3'], data2['X4'], data2['X5']])
# dimensions of features df; x: # of sets, y: samples per set, z: features per set
x, y, z = X_sets.shape
features = np.ones((x, y, z+1), dtype=float)
# add bias X0
features[..., 1:] = X_sets
labels = np.array([data2['y1'], data2['y2'], data2['y3'], data2['y4'], data2['y5']])
x, y, z = features.shape
# neighbor vector
K = np.arange(1, 16, 2)
# initialize an accuracy vector to store accuracy for each K value
accuracy = np.zeros(K.shape)
for i in range(5):
    for j in range(len(K)):
        X = np.vstack(np.vstack((features[0:i], features[i+1:])))
        y = np.vstack(np.vstack((labels[0:i], labels[i+1:])))
        neigh = KNeighborsClassifier(n_neighbors=K[j])
        neigh.fit(X=X,y=y)
        accuracy[j]+=accuracy_score(labels[i], neigh.predict(features[i]))
accuracy /= 5
# plot of accuracy vs K
plot_accuracy = plt.figure(2)
plt.plot(K, accuracy, c='tab:blue', marker='o')
plt.xlabel('K')
plt.ylabel('Average Accuracy')
plt.legend()
plt.savefig('output/ps4-2-a.png') # output

#----------------------------------QUESTION-3-PART-B---------------------------------------------
# load hw4_data3.mat
data3 = sio.loadmat('input/hw4_data3.mat')
# m samples, n features (excluding bias)
m, n = data3['X_train'].shape
m2, n2 = data3['X_test'].shape
# label vector
y_train = data3['y_train']
y_test = data3['y_test']
# empty feature matrix
X_train = np.empty((m, n + 1), dtype=float)
X_test = np.empty((m2, n2 + 1), dtype=float)
# append bias (X0) to X_data
X_train[:, 0] = np.ones((m, ))
X_train[:, 1:n+1] = data3['X_train']
X_test[:, 0] = np.ones((m2, ))
X_test[:, 1:n2+1] = data3['X_test']
# compute accuracy on a one vs all approach for multiclass classification
training_acc = accuracy_score(y_train, logReg_multi(X_train, y_train, X_train))
testing_acc = accuracy_score(y_test, logReg_multi(X_train, y_train, X_test))
# tabulated result
results = [[0, training_acc, testing_acc]]
col_names = ['Iteration', 'Training Accuracy', 'Testing Accuracy']
print('Question 3A : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))