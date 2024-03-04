import numpy as np
from scipy.spatial.distance import cdist

def weightedKNN(X_train, y_train, X_test, sigma):
    # unique labels from y_train
    labels = np.unique(y_train)
    # weight vote vector
    weight_vote = np.zeros((X_test.shape[0], len(labels)))
    # y_pred
    y_pred = np.zeros((X_test.shape[0], ))
    for i in range(len(labels)):
        # find corresponding samples to each y
        y_temp = np.where(y_train == labels[i])[0] # indices corresponding to label
        X_temp = X_train[y_temp, :] # samples corresponding to label
        # find distance > weight of each y
        dist = cdist(X_temp, X_test, 'euclidean') # total distances
        weight = np.exp(-dist**2/(sigma**2)) # weight
        # append to votes
        weight_vote[:, i] = weight.sum(axis=0) # sum of weights 
    # find the index of max probability and use that for prediction
    for i in range(weight_vote.shape[0]):
        pred_val = labels[weight_vote.argmax(axis=1)[i]]
        y_pred[i] = pred_val
    return y_pred