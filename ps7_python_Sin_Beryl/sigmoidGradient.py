import numpy as np
from sigmoid import * # sigmoid function

# function to compute the gradient for the sigmoid function
# g(z) = 1 / (1 + e^-z)
# g'(z) = g(z)(1 - g(z))
def sigmoidGradient(z):
    gradient = sigmoid(z) * (1 - sigmoid(z))
    return gradient