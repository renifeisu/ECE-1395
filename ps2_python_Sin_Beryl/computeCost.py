import numpy as np

# function to compute the cost given theta
# J = 1/2m * sum((h(x)-y)^2)
def computeCost(X, y, theta):
    # error: h(x) - y = theta^T * X - y
    error = y - np.dot(X, np.transpose(theta))
    # sum = error^2 = error^T * error
    sum = np.dot(np.transpose(error), error)
    # m, number of samples
    m = len(y)
    
    # cost
    J = sum / (2 * m)
    return J