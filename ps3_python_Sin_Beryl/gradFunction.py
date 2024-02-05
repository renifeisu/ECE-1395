import numpy as np
from costFunction import *

# function that computes the gradient of the cost function
# dJ = 1 / m * sum((h(x)-y) * x_j)
def gradFunction(theta, X_train, y_train):
    # m: samples, n: features
    m, n = X_train.shape
    # make sure theta's shape is unchanged
    theta = np.reshape(theta, (1, n))
    # empty vector for gradient, same size as theta
    grad = np.empty((n, ), 'float')
    # update gradient for each theta
    for i in range(0, n):
        # h(x) = g(theta' * X)
        h = sigmoid(np.dot(X_train, np.transpose(theta))) # m x 1
        xj = np.empty((m, 1), 'float')
        xj[:, 0] = X_train[:, i]
        sum = (h - y_train) *  xj  # m x 1
        # calculate gradient
        grad[i] = np.sum(sum) / m # 1 x 1
    return grad