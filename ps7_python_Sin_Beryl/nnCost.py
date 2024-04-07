import numpy as np
import math
from predict import * 

# a function that computes the cost of a nn
def nnCost(Theta1, Theta2, X, y, K, lmbda):
    [p, h_x] = predict(Theta1, Theta2, X)
    y_k = np.zeros((y.shape[0], 3))
    s1 = 0
    s2 = 0
    s3 = 0
    for i in range(y.shape[0]):
        y_k[i, y[i]-1] = 1
    for i in range(X.shape[0]):
        for k in range(K):
            s1 += y_k[i][k] * np.log(h_x[i][k]) + (1 - y_k[i][k]) * np.log(1 - h_x[i][k])

    for i in range(Theta1.shape[0]):
        for j in range(Theta1.shape[1]):
            s2 += Theta1[i][j] ** 2

    for i in range(Theta2.shape[0]):
        for j in range(Theta2.shape[1]):
            s3 += Theta2[i][j] ** 2

    cost = -1/X.shape[0] * s1 + lmbda/(2*X.shape[0]) * (s2 + s3)
    return cost