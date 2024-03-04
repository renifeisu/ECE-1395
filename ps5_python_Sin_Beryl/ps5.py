# ECE 1395
# Problem Set 5
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.io as sio
from scipy.spatial.distance import cdist
from tabulate import tabulate
import os
import shutil
from random import randrange
from time import sleep
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from time import process_time

from weightedKNN import * # weightedKNN
from imgSplitter import * # imgSplitter

#----------------------------------QUESTION-1-PART-B---------------------------------------------
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
# sigma vector
sigma = [0.01, 0.07, 0.15, 1.5, 3, 4.5]
# results
results = []
# results2 = []
# compute accuracy 
training_acc = np.empty((len(sigma), ), dtype=float)
testing_acc = np.empty((len(sigma), ), dtype=float)
for i in range(len(sigma)):
    training_acc[i] = accuracy_score(y_train, weightedKNN(X_train, y_train, X_train, sigma[i]))
    testing_acc[i] = accuracy_score(y_test, weightedKNN(X_train, y_train, X_test, sigma[i]))
    results.append([sigma[i], training_acc[i], testing_acc[i]])
    temp_pred = weightedKNN(X_train, y_train, X_test, sigma[i])
    # results2.append(temp_pred)
# results2.append(y_test.reshape(25, ))
# results2 = np.transpose(np.array(results2))
# tabulated result
col_names = ['Sigma', 'Training Accuracy', 'Testing Accuracy']
# col_names2 = ['0.01', '0.07', '0.15', '1.5', '3', '4.5', 'Real']
print('Question 1B : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
# print('Testing Results : ') 
# print(tabulate(results2, headers=col_names2, tablefmt="fancy_grid"))

#----------------------------------QUESTION-2-PART-0---------------------------------------------
# ensure folders are empty for reruns
shutil.rmtree('input/train')
shutil.rmtree('input/test')
os.mkdir('input/train')
os.mkdir('input/test')
# some paths
path_all = 'input/all/s'
path_train = 'input/train/'
path_test = 'input/test/'
# random training/testing data
for i in range(len(next(os.walk('input/all'))[1])):
    imgSplitter(i+1, 0.8) # file splitter
# size of training/testing data
print('Question 2-0 Number of training samples:', len(os.listdir('input/train')))
print('Question 2-0 Number of testing samples:', len(os.listdir('input/test')))

# check random training samples
folder = []
number = []
while len(folder) < 3:
    # check if img is a training sample
        try:
            temp_fol = randrange(1, 40)
            temp_num = randrange(1, 10)
            plt.imread(path_train+'s'+str(temp_fol)+'-'+str(temp_num)+'.pgm') 
        except FileNotFoundError:
            sleep(1)
            print('Retrying..')
        else:
            folder.append(temp_fol)
            number.append(temp_num)
            print('Sample generated')

f1 = 's'+str(folder[0])+'-'+str(number[0])+'.pgm'
f2 = 's'+str(folder[1])+'-'+str(number[1])+'.pgm'
f3 = 's'+str(folder[2])+'-'+str(number[2])+'.pgm'

# subplot
fig, axs = plt.subplots(1, 3)
f = plt.imread(path_train+f1)
axs[0].imshow(f, cmap='gray')
axs[0].set_title(f1)
f = plt.imread(path_train+f2)
axs[1].imshow(f, cmap='gray')
axs[1].set_title(f2)
f = plt.imread(path_train+f3)
axs[2].imshow(f, cmap='gray')
axs[2].set_title(f3)
fig.tight_layout()
plt.savefig('output/ps5-2-0.png')

#----------------------------------QUESTION-2-PART-1A--------------------------------------------
print('Question 2-1:')
# Matrix T (for training)
T = np.empty((10304, 320), dtype=float)
# keep track of y_train
y_train = np.empty((320, ), dtype=int)
# Matrix T_test (for testing)
T_test = np.empty((10304, 80), dtype=float)
# keep track of y_test
y_test = np.empty((80, ), dtype=int)
# statistics
retries = 0
count = 0
count2 = 0
count3 = 0
temp_fol = 1
temp_num = 1
while temp_fol < 41:
    try:
        temp_f = plt.imread(path_train+'s'+str(temp_fol)+'-'+str(temp_num)+'.pgm') 
    except FileNotFoundError:
        retries+=1
        print('Retrying.. ('+str(retries)+')')
        # for test
        count2+=1
        count3+=1
        temp_f = plt.imread(path_test+'s'+str(temp_fol)+'-'+str(temp_num)+'.pgm') 
        temp_f = temp_f.reshape(-1, )
        T_test[:, count2-1] = temp_f
        y_test[count2-1] = temp_fol
        # continue
        if ((count3 % 10) == 0):
            temp_fol+=1
            temp_num = 1
            print('Folder '+str(temp_fol-1)+' finished')
        else:
            temp_num+=1
    else:
        count+=1
        count3+=1
        temp_f = temp_f.reshape(-1, )
        T[:, count-1] = temp_f
        y_train[count-1] = temp_fol
        print('Temp_num:', temp_num, 'Count:', count)
        if ((count3 % 10) == 0):
            temp_fol+=1
            temp_num = 1
            print('Folder '+str(temp_fol-1)+' finished')
        else:
            temp_num+=1
print('Matrix T updated')
# grayscale plot of T
fig2 = plt.figure(2) 
plt.imshow(T, cmap='gray')
plt.xticks(np.arange(0, 400, 80)) 
plt.gca().set_aspect('auto', adjustable='box')
plt.savefig('output/ps5-2-1-a.png')

#----------------------------------QUESTION-2-PART-1B--------------------------------------------
# mean across row (average face vector)
m = np.mean(T, axis=1)
# display (reshaped)
fig3 = plt.figure(3)
plt.imshow(m.reshape(112, 92), cmap='gray')
plt.savefig('output/ps5-2-1-b.png')

#----------------------------------QUESTION-2-PART-1C--------------------------------------------
# centered data matrix, A
A = np.subtract(T, m.reshape(T.shape[0], 1))
# covariance matrix, c
C = np.matmul(A, np.transpose(A))
# grayscale plot of covariance matrix
fig3 = plt.figure(3) 
plt.imshow(C, cmap='gray')
plt.gca().set_aspect('auto', adjustable='box')
plt.savefig('output/ps5-2-1-c.png')

#----------------------------------QUESTION-2-PART-1D--------------------------------------------
# eigenvalues/vectors
eigval, eigvec = np.linalg.eig(np.matmul(np.transpose(A), A))
# eigenvalues (descending order)
eigval = -np.sort(-eigval)
idx = np.argsort(eigval)
# percentage of variance
k = range(len(eigval))
v_k = []
for i in k:
    numer = np.sum(eigval[0:i+1])
    denom = np.sum(eigval)
    v_k.append(numer/denom)
# number of eigenvectors that capture 95% of the variance
K = np.array(v_k) >= 0.95
K_num = np.where(K)[0][0]+1
print('Question 2-1D, Number of eigenvectors:', K_num)
# plot
fig4 = plt.figure(4)
plt.plot(k, v_k, c='tab:blue')
plt.plot(np.where(K)[0], np.array(v_k)[K], c='tab:cyan', label='v_k >= 0.95')
plt.xlabel('k')
plt.ylabel('v_k')
plt.title('k vs v_k')
plt.legend()
plt.savefig('output/ps5-2-1-d.png')

#----------------------------------QUESTION-2-PART-1E--------------------------------------------
print('Working...')
# retrieve K dominant eigenvectors
eigval, eigvec = np.linalg.eig(C)
eigval = -np.sort(-eigval)
idx = np.argsort(eigval)
# basis matrix
U = np.real(eigvec[:, idx[0:K_num]])
# plot of first 9 eigenfaces
eigfaces = np.reshape(np.real(eigvec), (112, 92, -1))
fig5, axs = plt.subplots(1, 9, figsize=(15, 15))
for i in range(9):
    axs[i].imshow(eigfaces[:, :, i], cmap='gray')
    axs[i].set_title(str(i+1))
fig5.tight_layout()
# plt.imshow(U[:, range(9)].reshape(112, 92 * 9), cmap='gray')
plt.savefig('output/ps5-2-1-e.png')
print('Question 2-1E, U.shape:', U.shape) 

#----------------------------------QUESTION-2-PART-2A--------------------------------------------
print('Question 2-2:')
# Training
W_training = np.empty((320, K_num), dtype=float)
for i in range(320):
    try:
        W_training[i, :] = np.matmul(np.transpose(U), T[:, i] - m)
    except:
        W_training[i, :] = np.matmul(np.transpose(U), T[:, i] - m.reshape(T.shape[0], 1))
#W_training = np.matmul(np.transpose(U), np.subtract(T, m.reshape(T.shape[0], 1)))
print('Question 2-2A, W_training.shape:', W_training.shape) 
print(W_training)

#----------------------------------QUESTION-2-PART-2B--------------------------------------------
# Testing
W_testing = np.empty((80, K_num), dtype=float)
for i in range(80):
    try:
        W_testing[i, :] = np.matmul(np.transpose(U), T_test[:, i] - m)
    except:
        W_testing[i, :] = np.matmul(np.transpose(U), T_test[:, i] - m.reshape(T.shape[0], 1))
#W_testing = np.matmul(np.transpose(U), np.subtract(T_test, m.reshape(T.shape[0], 1)))
print('Question 2-2B, W_testing.shape:', W_testing.shape) 
print(W_testing)

#----------------------------------QUESTION-2-PART-3A--------------------------------------------
print('Question 2-3:')
# neighbor vector
K = np.arange(1, 12, 2)
# initialize an accuracy vector to store accuracy for each K value
accuracy = np.zeros(K.shape)
for i in range(len(K)):
    neigh = KNeighborsClassifier(n_neighbors=K[i])
    neigh.fit(X=W_training,y=y_train.ravel())
    y_pred = neigh.predict(W_testing)
    accuracy[i] = accuracy_score(y_test, y_pred)

# checking result..
results = []
for i in range(len(y_test)):
    results.append([y_test[i], y_pred[i]])
col_names = ['Actual', 'Predicted']
print('Question 2-3A Testing : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
check = ''
while check != 'y': 
    check = input("Continue?")

# tabulated result
results = []
for i in range(len(K)):
    results.append([K[i], accuracy[i]])
col_names = ['K', 'Accuracy']
print('Question 2-3A : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))

#----------------------------------QUESTION-2-PART-3B--------------------------------------------
models = ['Linear (One-vs-One)', 'Linear (One-vs-All)', 'Polynomial (One-vs-One)', 'Polynomial (One-vs-All)', 'RBF (One-vs-One)', 'RBF (One-vs-All)']
training_time = []
testing_acc = []

# linear one-vs-one SVM classifier
start = process_time()
model1 = SVC(kernel='linear', decision_function_shape='ovo')
model1.fit(W_training, y_train.ravel())
training_time.append(process_time() - start)
testing_acc.append(accuracy_score(y_test, model1.predict(W_testing)))

# Testing
y_pred = model1.predict(W_testing)
results = []
for i in range(len(y_test)):
    results.append([y_test[i], y_pred[i]])
col_names = ['Actual', 'Predicted']
print('Model 1 Testing : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
check = ''
while check != 'y': 
    check = input("Continue?")

# linear one-vs-all SVM classifier
start = process_time()
model2 = SVC(kernel='linear', decision_function_shape='ovr')
model2.fit(W_training, y_train.ravel())
training_time.append(process_time() - start)
testing_acc.append(accuracy_score(y_test, model2.predict(W_testing)))

# Testing
y_pred = model2.predict(W_testing)
results = []
for i in range(len(y_test)):
    results.append([y_test[i], y_pred[i]])
col_names = ['Actual', 'Predicted']
print('Model 2 Testing : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
check = ''
while check != 'y': 
    check = input("Continue?")

# 3rd-order one-vs-one SVM classifier
start = process_time()
model3 = SVC(kernel='poly', degree=3, decision_function_shape='ovo')
model3.fit(W_training, y_train.ravel())
training_time.append(process_time() - start)
testing_acc.append(accuracy_score(y_test, model3.predict(W_testing)))

# Testing
y_pred = model3.predict(W_testing)
results = []
for i in range(len(y_test)):
    results.append([y_test[i], y_pred[i]])
col_names = ['Actual', 'Predicted']
print('Model 3 Testing : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
check = ''
while check != 'y': 
    check = input("Continue?")

# 3rd-order one-vs-all SVM classifier
start = process_time()
model4 = SVC(kernel='poly', degree=3, decision_function_shape='ovr')
model4.fit(W_training, y_train.ravel())
training_time.append(process_time() - start)
testing_acc.append(accuracy_score(y_test, model4.predict(W_testing)))

# Testing
y_pred = model4.predict(W_testing)
results = []
for i in range(len(y_test)):
    results.append([y_test[i], y_pred[i]])
col_names = ['Actual', 'Predicted']
print('Model 4 Testing : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
check = ''
while check != 'y': 
    check = input("Continue?")

# Gaussian rbf one-vs-one SVM classifier
start = process_time()
model5 = SVC(kernel='rbf', decision_function_shape='ovo')
model5.fit(W_training, y_train.ravel())
training_time.append(process_time() - start)
testing_acc.append(accuracy_score(y_test, model5.predict(W_testing)))

# Testing
y_pred = model5.predict(W_testing)
results = []
for i in range(len(y_test)):
    results.append([y_test[i], y_pred[i]])
col_names = ['Actual', 'Predicted']
print('Model 5 Testing : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
check = ''
while check != 'y': 
    check = input("Continue?")

# Gaussian rbf one-vs-all SVM classifier
start = process_time()
model6 = SVC(kernel='rbf', decision_function_shape='ovr')
model6.fit(W_training, y_train.ravel())
training_time.append(process_time() - start)
testing_acc.append(accuracy_score(y_test, model6.predict(W_testing)))

# Testing
y_pred = model6.predict(W_testing)
results = []
for i in range(len(y_test)):
    results.append([y_test[i], y_pred[i]])
col_names = ['Actual', 'Predicted']
print('Model 6 Testing : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
check = ''
while check != 'y': 
    check = input("Continue?")

# tabulated result
results = []
for i in range(len(models)):
    results.append([models[i], training_time[i], testing_acc[i]])
col_names = ['SVM Classifier', 'Training Time', 'Testing Accuracy']
print('Question 2-3B : ') 
print(tabulate(results, headers=col_names, tablefmt="fancy_grid"))
