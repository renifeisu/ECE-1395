# ECE 1395
# Problem Set 2
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from sklearn.utils import shuffle
import computeCost # computeCost function
import gradientDescent # gradient Descent function

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
def normalEqn(X_train, y_train):
    return

