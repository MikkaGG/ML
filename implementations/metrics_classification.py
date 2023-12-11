import numpy as np

def confusion_matrix(y_test, y_pred):
    TN, FN, FP, TP = 0, 0, 0, 0
    TN += np.count_nonzero((y_test == 1 ) & (y_pred == 1))
    FN += np.count_nonzero((y_test == 1) & (y_pred == 0))
    FP += np.count_nonzero((y_test == 0) & (y_pred == 1))
    TP += np.count_nonzero((y_test == 0) & (y_pred == 0))
    return TN, FN, FP, TP

def accuracy(y_test, y_pred):
    TN, FN, FP, TP = confusion_matrix(y_test, y_pred)
    return((TP + TN) / (TP + TN + FP + FN))

def precision(y_test, y_pred):
    TN, FN, FP, TP = confusion_matrix(y_test, y_pred)
    return(TP / (TP + FP))

def recall(y_test, y_pred):
    TN, FN, FP, TP = confusion_matrix(y_test, y_pred)
    return(TP / (TP + FN))

def F1(y_test, y_pred):
    return(2 / (1 / precision(y_test, y_pred) + 1 / recall(y_test, y_pred)))
