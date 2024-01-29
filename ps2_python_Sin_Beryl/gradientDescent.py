import numpy as np
from computeCost import *

# function that computes the gradient descent solution to linear regression
def gradientDescent(X_train, y_train, alpha, iters):
    # m: samples, n: features
    m, n = X_train.shape
    # random initialization of theta
    theta = np.random.randn(n, 1)

    # empty vector for cost
    cost = np.empty((iters, 1), "object")

    # update cost and theta every iteration
    for i in range(0, iters):
        # theta := theta - alpha * derivative(cost)
        error = y_train - np.dot(X_train, theta)
        derivative =  1/(2*m) * np.dot(-2 * np.transpose(X_train), error)
        theta = theta - alpha * derivative
        cost[i] = computeCost(X_train, y_train, np.transpose(theta))

    return [theta, cost]