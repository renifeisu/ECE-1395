import numpy as np

# function that computes the closed-form solution to linear regression
def normalEqn(X_train, y_train):
    # theta = (X'* X)^-1 * (X' * y)
    # astype was to ensure that the datatypes of the elements stayed the same when transposed..
    left = np.linalg.pinv(np.dot(np.transpose(X_train.astype(np.float64)), X_train.astype(np.float64)))
    right = np.dot(np.transpose(X_train), y_train)
    theta = np.dot(left, right)
    return theta