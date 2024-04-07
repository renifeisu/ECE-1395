import numpy as np
import math
from sklearn.utils import shuffle

# a function that splits data into training and testing sets
def dataSplitter(matrix_x, vector_y, ratio):
    # rows and cols of x
    row, col = matrix_x.shape
    # create and shuffle temp_vector
    temp_vector = range(row)
    temp_vector = shuffle(temp_vector)
    # number of training/testing samples
    train = math.ceil(row * ratio)
    test = row - train
    # define X_train and X_test
    X_train = np.empty((train, col), 'float')
    X_test = np.empty((test, col), 'float')
    # define y_train and y_test
    y_train = np.empty((train, 1), 'float')
    y_test = np.empty((test, 1), 'float')
    # used shuffled vector to determine which rows from matrix_x become X_train and X_test
    # and which rows from vector_y become y_train and y_test
    X_train = matrix_x[temp_vector[0:train], :]
    y_train = vector_y[temp_vector[0:train], :]
    X_test = matrix_x[temp_vector[train:], :]
    y_test = vector_y[temp_vector[train:], :]
    
    return [X_train, X_test, y_train, y_test]