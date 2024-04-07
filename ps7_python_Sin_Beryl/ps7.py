# ECE 1395
# Problem Set 7
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.io as sio
from scipy.spatial.distance import cdist
from tabulate import tabulate
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels

from dataSplitter import * # split data into test and training sets
from randomizer import * # generates random samples from data
from predict import * # returns nn predictions
from nnCost import * # returns nn cost
from sigmoidGradient import * # sigmoid gradient function
from sGD import * # stochastic gradient descent function

#----------------------------------QUESTION-0-PART-A---------------------------------------------
# load .mat file
data = sio.loadmat('input/HW7_Data2_full.mat')
# separate into feature matrix and label matrix
X = data['X']
y = data['y_labels']
# random samples
[X_random, y_random] = randomizer(X, y, 25)
# plot of random samples and corresponding labels
fig, axs = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        axs[i][j].imshow(X_random[i * 5 + j].reshape(32, 32), cmap='gray')
        axs[i][j].set_title(y_random[i * 5 + j])
        axs[i][j].axis('off')
plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.savefig('output/ps7-0-a-1.png')
#plt.show()

#----------------------------------QUESTION-0-PART-B---------------------------------------------
# split data into training and testing sets
[X_train, X_test, y_train, y_test] = dataSplitter(X, y, 13/15)

#----------------------------------QUESTION-1-PART-A---------------------------------------------
# load .mat file
data2 = sio.loadmat('input/HW7_weights_3_full.mat')
theta1 = data2['Theta1']
theta2 = data2['Theta2']

#----------------------------------QUESTION-1-PART-B---------------------------------------------
[p, h_x] = predict(theta1, theta2, X_train)
print('Question 1-B Accuracy:', accuracy_score(y_train, p))

#----------------------------------QUESTION-2-PART-B---------------------------------------------
lmbda = [0.1, 1, 2]
cost = []
for i in range(len(lmbda)):
    cost.append(nnCost(theta1, theta2, X, y, 3, lmbda[i]))

results = []
for i in range(len(lmbda)):
    results.append([lmbda[i], cost[i]])
col_names = ['Lambda', 'Cost']
print('Question 2-B : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-3----------------------------------------------------
z = [-10, 0, 10]
g_prime = []
for i in range(len(lmbda)):
    g_prime.append(sigmoidGradient(z[i]))

results = []
for i in range(len(z)):
    results.append([z[i], g_prime[i]])
col_names = ['z', 'g_prime']
print('Question 3 : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-4----------------------------------------------------
# sample data to figure out learning rate (partitioned from learning rate)
[Xtemp_train, Xtemp_test1, ytemp_train, ytemp_test1] = dataSplitter(X_train, y_train, 1/52)
#[t1, t2] = sGD(theta1.shape[1]-1, theta1.shape[0], len(unique_labels(y)), Xtemp_train, ytemp_train, 0.1, 0.01, 100)
# some other partitioning for testing out stuff
[Xtemp_train, Xtemp_test1, ytemp_train, ytemp_test1] = dataSplitter(X_train, y_train, 1/104)
[Xtemp_test, Xtemp_test2, ytemp_test, ytemp_test2] = dataSplitter(Xtemp_test1, ytemp_test1, 1/103)

#----------------------------------QUESTION-5----------------------------------------------------
lambdas = [0.1, 1, 2]
epochs = [50, 300]
#epochs = [20, 30]
training_acc = [0, 0, 0, 0, 0, 0]
training_cost = [0, 0, 0, 0, 0, 0]
testing_acc = [0, 0, 0, 0, 0, 0]
testing_cost = [0, 0, 0, 0, 0, 0]

results = []
for i in range(2):
    for j in range(3):
        # test out compilation of results
        # [t1, t2] = sGD(theta1.shape[1]-1, theta1.shape[0], len(unique_labels(y)), Xtemp_train, ytemp_train, lambdas[j], 0.01, epochs[i])
        # [p, h_x] = predict(np.transpose(t1), np.transpose(t2), Xtemp_train)
        # training_acc[i+j] = accuracy_score(ytemp_train, p)
        # training_cost[i+j] = nnCost(np.transpose(t1), np.transpose(t2), Xtemp_train, ytemp_train, len(unique_labels(y)), lambdas[j])
        # [t1, t2] = sGD(theta1.shape[1]-1, theta1.shape[0], len(unique_labels(y)), Xtemp_test, ytemp_test, lambdas[j], 0.01, epochs[i])
        # [p, h_x] = predict(np.transpose(t1), np.transpose(t2), Xtemp_test)
        # testing_acc[i+j] = accuracy_score(ytemp_test, p)
        # testing_cost[i+j] = nnCost(np.transpose(t1), np.transpose(t2), Xtemp_test, ytemp_test, len(unique_labels(y)), lambdas[j])
        
        [t1, t2] = sGD(theta1.shape[1]-1, theta1.shape[0], len(unique_labels(y)), X_train, y_train, lambdas[j], 0.01, epochs[i])
        [p, h_x] = predict(np.transpose(t1), np.transpose(t2), X_train)
        training_acc[i+j] = accuracy_score(y_train, p)
        training_cost[i+j] = nnCost(np.transpose(t1), np.transpose(t2), X_train, y_train, len(unique_labels(y)), lambdas[j])
        [t1, t2] = sGD(theta1.shape[1]-1, theta1.shape[0], len(unique_labels(y)), X_test, y_test, lambdas[j], 0.01, epochs[i])
        [p, h_x] = predict(np.transpose(t1), np.transpose(t2), X_test)
        testing_acc[i+j] = accuracy_score(y_test, p)
        testing_cost[i+j] = nnCost(np.transpose(t1), np.transpose(t2), X_test, y_test, len(unique_labels(y)), lambdas[j])

        results.append([lambdas[j], epochs[i], training_acc[i+j], training_cost[i+j], testing_acc[i+j], testing_cost[i+j]])
col_names = ['Lambda', 'Epochs', 'Training Data Accuracy', 'Cost (Training)', 'Testing Data Accuracy', 'Cost (Testing)']
print('Question 5 : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
