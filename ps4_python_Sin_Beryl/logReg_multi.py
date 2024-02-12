import numpy as np
from sklearn.linear_model import LogisticRegression

def logReg_multi(X_train, y_train, X_test):
    # unique labels from y_train
    labels = np.unique(y_train)
    # probability vector
    proba_c = np.zeros((X_test.shape[0], len(labels)))
    # y_pred
    y_pred = np.zeros((X_test.shape[0], ))
    for i in range(len(labels)):
        y_temp = (y_train == labels[i]).astype(int) # binary classification (label[i] vs not label[i])
        mdl = LogisticRegression(random_state=0).fit(X_train, y_temp) # model
        proba_c[:, i] = mdl.predict_proba(X_test)[:, 1] # probability 
    # find the index of max probability and use that for prediction
    for i in range(proba_c.shape[0]):
        pred_val = labels[proba_c.argmax(axis=1)[i]]
        y_pred[i] = pred_val
    return y_pred