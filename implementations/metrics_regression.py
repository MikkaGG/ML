import numpy as np

def MAE(y, y_pred):
    n = len(y)
    summa = np.sum(abs(y - y_pred))
    return summa/n

def MSE(y, y_pred):
    n = len(y)
    summa = np.sum((y - y_pred) ** 2)
    return summa/n

def RMSE(y, y_pred):
    return np.sqrt(MSE(y, y_pred))

def MAPE(y, y_pred):
    n = len(y)
    summa = np.sum(abs((y - y_pred)/y))
    return summa/n

def R2(y, y_pred):
    n = len(y)
    mean_y = np.mean(y)
    summa = np.sum((y - mean_y) ** 2)
    return 1 - MSE(y, y_pred)/(summa/n)
