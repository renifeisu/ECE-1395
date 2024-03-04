import numpy as np
import math
from sklearn.utils import shuffle
import os
import shutil

# a function that splits data into training and testing sets
def imgSplitter(folder, ratio):
    path_all = 'input/all/'
    path_train = 'input/train/'
    path_test = 'input/test/'

    lst = os.listdir(path_all+'s'+str(folder)) # folder path
    # create and shuffle temp_vector
    temp_vector = range(1, len(lst)+1)
    temp_vector = shuffle(temp_vector)
    # number of training/testing samples
    train = math.ceil(len(lst) * ratio)
    for i in temp_vector[0:train]:
        filename = str(i)+'.pgm'
        shutil.copy(path_all+'s'+str(folder)+'/'+filename, 'input/train')
        os.replace(path_train+filename, path_train+'s'+str(folder)+str('-')+filename)
    for i in temp_vector[train:]:
        filename = str(i)+'.pgm'
        shutil.copy(path_all+'s'+str(folder)+'/'+filename, 'input/test')
        os.replace(path_test+filename, path_test+'s'+str(folder)+str('-')+filename)
