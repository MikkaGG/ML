import numpy as np
import sys
sys.path.append('D:\ML\myLib')
import metrics_regression

class LinerRegression():
    def __init__(self, alpha = 1.0):
        self.alpha = alpha
        self.w = None
 
    def fit(self, X, y):
        X = X.copy()
        X = self.add_ones(X)
        I = np.identity(X.shape[1])
        I[0][0] = 0
        self.w = np.linalg.inv(X.T.dot(X) + self.alpha * I).dot(X.T).dot(y)
 
    def predict(self, X):
        X = X.copy()
        X = self.add_ones(X)
        return np.dot(X, self.w)

    def add_ones(self, X):
        return np.c_[np.ones((len(X), 1)), X]
