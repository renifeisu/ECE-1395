import numpy as np
from sigmoid import *

# function to compute the cost given theta
# J = 1/2m * sum((h(x)-y)^2)
# J = 1/m * sum(-y*log(h(x)) - (1-y)log(1-h(x)))
def costFunction(theta, X_train, y_train):
    # m: samples, n: features
    m, n = X_train.shape
    # make sure theta's shape is unchanged
    theta = np.reshape(theta, (1, n))
    # h(x) = g(theta' * X)
    h = sigmoid(np.dot(X_train, np.transpose(theta))) # m x 1
    # -y*log(h(x))
    leftsum = np.dot(np.transpose(-y_train), np.log(h), ) #  1 x 1
    # (1-y)log(1-h(x)))
    rightsum = np.dot(np.transpose(1-y_train), np.log(1-h)) # 1 x 1
    # cost
    J = (leftsum - rightsum) / m # 1 x 1
    return np.sum(J)