import numpy as np
import math
from sklearn.utils import shuffle

# a function that generates random samples from data
def randomizer(matrix_x, vector_y, number):
    # rows and cols of x
    row, col = matrix_x.shape
    # create and shuffle temp_vector
    temp_vector = range(row)
    temp_vector = shuffle(temp_vector)
    # define X_random and y_random
    X_random = np.empty((number, col), 'float')
    y_random = np.empty((number, col), 'float')
    # used shuffled vector to determine which rows from matrix_x become X_random
    # and which rows from vector_y become y_random
    X_random = matrix_x[temp_vector[0:number], :]
    y_random = vector_y[temp_vector[0:number], :]
    
    return [X_random, y_random]