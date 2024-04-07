import numpy as np

# function to compute the sigmoid function
# g(z) = 1 / (1 + e^-z)
def sigmoid(z):
    gz = 1 / (1 + np.exp(-z))
    
    return gz