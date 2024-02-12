import numpy as np

# function that computes the closed-form solution to linear regression with regularization
def Reg_normalEqn(X_train, y_train, l):
    # theta = (X'*X + lambda*D)^-1 * (X' * y)
    # X_train: m x n, y_train: m x 1, theta: n x 1
    # dimensions of X_train, m samples x n features
    m, n = X_train.shape
    D = np.eye(n)
    D[0, 0] = 0
    prod1 = np.dot(np.transpose(X_train), X_train) # n x n
    prod2 = l * D # n x n
    prod3 = np.dot(np.transpose(X_train), y_train) # n x 1
    inv = np.linalg.pinv(prod1 + prod2) # n x n
    theta = np.dot(inv, prod3) # n x 1
    return theta