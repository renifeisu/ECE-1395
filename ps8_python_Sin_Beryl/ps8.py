# ECE 1395
# Problem Set 8
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as sst
import scipy.io as sio
from scipy.spatial.distance import cdist
from tabulate import tabulate
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from dataSplitter import * # split data into test and training sets
from randomizer import * # generates random samples from data
# from predict import * # returns nn predictions
# from nnCost import * # returns nn cost
# from sigmoidGradient import * # sigmoid gradient function
# from sGD import * # stochastic gradient descent function

#----------------------------------QUESTION-1-PART-A---------------------------------------------
# load .mat file
data = sio.loadmat('input/HW8_data1.mat')

# separate into feature matrix and label matrix
X = data['X']
y = data['y']

# random samples
[X_random, y_random] = randomizer(X, y, 25)
# plot of random samples and corresponding labels
fig, axs = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        axs[i][j].imshow(X_random[i * 5 + j].reshape(20, 20), cmap='gray')
        axs[i][j].set_title(y_random[i * 5 + j])
        axs[i][j].axis('off')
plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.savefig('output/ps8-1-a.png')
#plt.show()

#----------------------------------QUESTION-1-PART-B---------------------------------------------
# split data into training and testing sets
[X_train, X_test, y_train, y_test] = dataSplitter(X, y, 45/50)

#----------------------------------QUESTION-1-PART-C---------------------------------------------
# bagging
[X_train1, X_test1, y_train1, y_test1] = dataSplitter(X_train, y_train, 1000/4500)
[X_train2, X_test2, y_train2, y_test2] = dataSplitter(X_train, y_train, 1000/4500)
[X_train3, X_test3, y_train3, y_test3] = dataSplitter(X_train, y_train, 1000/4500)
[X_train4, X_test4, y_train4, y_test4] = dataSplitter(X_train, y_train, 1000/4500)
[X_train5, X_test5, y_train5, y_test5] = dataSplitter(X_train, y_train, 1000/4500)
sio.savemat('input/ps8-1-c.mat', mdict={'X': (X_train1, X_train2, X_train3, X_train4, X_train5), 'y': (y_train1, y_train2, y_train3, y_train4, y_train5)})
# Check the keys of the loaded data
data = sio.loadmat('input/ps8-1-c.mat')
keys = data.keys()
# print("Keys in the loaded .mat file:", keys)
# for i in range(5):
#     print(data['X'][i].shape)
#     print(data['y'][i].shape)

#----------------------------------QUESTION-1-PART-D---------------------------------------------
# One-vs-All SVM 10-class classifier, with RBF kernel
model1 = SVC(kernel='rbf', decision_function_shape='ovr')
# train using subset X1
model1.fit(X_train1, y_train1.ravel())

results = []
for i in range(5):
    # resulting error from training sets
    y_pred = model1.predict(data['X'][i])
    acc = accuracy_score(data['y'][i], y_pred)
    classification_error = 1 - acc
    results.append(['X_'+str(i+1), acc, classification_error])

# resulting error from testing set
y_pred = model1.predict(X_test)
acc = accuracy_score(y_test, y_pred)
classification_error = 1 - acc
results.append(['X_test', acc, classification_error])

# print out results
col_names = ['Set', 'Accuracy', 'Classification Error']
print('Question 1-D : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-1-PART-E---------------------------------------------
# KNN Classifier
model2 = KNeighborsClassifier(n_neighbors=5)
# train using subset X2
model2.fit(X_train2, y_train2.ravel())

results = []
for i in range(5):
    # resulting error from training sets
    y_pred = model2.predict(data['X'][i])
    acc = accuracy_score(data['y'][i], y_pred)
    classification_error = 1 - acc
    results.append(['X_'+str(i+1), acc, classification_error])

# resulting error from testing set
y_pred = model2.predict(X_test)
acc = accuracy_score(y_test, y_pred)
classification_error = 1 - acc
results.append(['X_test', acc, classification_error])

# print out results
col_names = ['Set', 'Accuracy', 'Classification Error']
print('Question 1-E : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-1-PART-F---------------------------------------------
# Logistic Regression Classifier
model3 = LogisticRegression(max_iter=1000)
# regularize data
scaler = preprocessing.StandardScaler().fit(X_train3)
X_scaled = scaler.transform(X_train3)

# train using subset X3
model3.fit(X_train3, y_train3.ravel())

results = []
for i in range(5):
    # resulting error from training sets
    X_scaled = scaler.transform(data['X'][i])
    y_pred = model3.predict(X_scaled)
    acc = accuracy_score(data['y'][i], y_pred)
    classification_error = 1 - acc
    results.append(['X_'+str(i+1), acc, classification_error])

# resulting error from testing set
y_pred = model3.predict(X_test)
acc = accuracy_score(y_test, y_pred)
classification_error = 1 - acc
results.append(['X_test', acc, classification_error])

# print out results
col_names = ['Set', 'Accuracy', 'Classification Error']
print('Question 1-F : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-1-PART-G---------------------------------------------
# Decision Tree Classifier
model4 = DecisionTreeClassifier()
# train using subset X4
model4.fit(X_train4, y_train4.ravel())

results = []
for i in range(5):
    # resulting error from training sets
    y_pred = model4.predict(data['X'][i])
    acc = accuracy_score(data['y'][i], y_pred)
    classification_error = 1 - acc
    results.append(['X_'+str(i+1), acc, classification_error])

# resulting error from testing set
y_pred = model4.predict(X_test)
acc = accuracy_score(y_test, y_pred)
classification_error = 1 - acc
results.append(['X_test', acc, classification_error])

# print out results
col_names = ['Set', 'Accuracy', 'Classification Error']
print('Question 1-G : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-1-PART-H---------------------------------------------
# Random Forest Classifier (85 trees)
model5 = RandomForestClassifier(n_estimators=85)
# train using subset X5
model5.fit(X_train5, y_train5.ravel())

results = []
for i in range(5):
    # resulting error from training sets
    y_pred = model5.predict(data['X'][i])
    acc = accuracy_score(data['y'][i], y_pred)
    classification_error = 1 - acc
    results.append(['X_'+str(i+1), acc, classification_error])

# resulting error from testing set
y_pred = model5.predict(X_test)
acc = accuracy_score(y_test, y_pred)
classification_error = 1 - acc
results.append(['X_test', acc, classification_error])

# print out results
col_names = ['Set', 'Accuracy', 'Classification Error']
print('Question 1-H : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-1-PART-I---------------------------------------------
# Majority Voting Rule
results = []
for i in range(5):
    # resulting error from training sets
    y_preds = np.empty((data['y'][i].shape[0], 5))
    y_preds[:, 0] = model1.predict(data['X'][i])
    y_preds[:, 1] = model1.predict(data['X'][i])
    y_preds[:, 2] = model1.predict(data['X'][i])
    y_preds[:, 3] = model1.predict(data['X'][i])
    y_preds[:, 4] = model1.predict(data['X'][i])
    [y_pred, counts] = sst.mode(a=y_preds, axis=1)
    acc = accuracy_score(data['y'][i], y_pred)
    classification_error = 1 - acc
    results.append(['X_'+str(i+1), acc, classification_error])

# resulting error from testing set
y_preds = np.empty((y_test.shape[0], 5))
y_preds[:, 0] = model1.predict(X_test)
y_preds[:, 1] = model1.predict(X_test)
y_preds[:, 2] = model1.predict(X_test)
y_preds[:, 3] = model1.predict(X_test)
y_preds[:, 4] = model1.predict(X_test)
[y_pred, counts] = sst.mode(a=y_preds, axis=1)
acc = accuracy_score(y_test, y_pred)
classification_error = 1 - acc
results.append(['X_test', acc, classification_error])

# print out results
col_names = ['Set', 'Accuracy', 'Classification Error']
print('Question 1-I : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))