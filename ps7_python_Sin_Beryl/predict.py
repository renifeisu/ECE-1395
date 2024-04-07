import numpy as np
import math
from sigmoid import * # sigmoid function

# a function that uses assigned weights to return nn prediction
def predict(Theta1, Theta2, x):
    Theta1 = np.transpose(Theta1)
    Theta2 = np.transpose(Theta2)
    # m x n+1
    a_1 = np.insert(x, 0, 1, axis=1)
    # m x 40
    z_2 = np.matmul(a_1, Theta1)
    # m x 41
    a_2 = np.insert(sigmoid(z_2), 0, 1, axis=1)
    # m x 3
    z_3 = np.matmul(a_2, Theta2)
    # m x 3
    a_3 = sigmoid(z_3)
    h_x = a_3
    p = np.argmax(a_3, axis=1)
    p = p + 1
    return [p, h_x]