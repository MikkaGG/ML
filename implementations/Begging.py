import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

class BeggingClassifier:
    def __init__(self, n_folds=None, model=None):
        self.n_folds = n_folds
        self.model = model
    
    def Data_split(self, X_train_clf, y_train_clf):
        train = pd.concat([X_train_clf, y_train_clf], axis=1)
        data = pd.DataFrame()
        for fold in range(self.n_folds):
            index = np.random.randint(0, len(train), 1000)
            subdata = pd.DataFrame()
            for fold in index:
                subdata = subdata.append(train.iloc[[fold]])
            data = data.append(subdata)
        return data

    def Algorithm(self, data, X_test):
        predicts = []
        for fold in range(0, len(data), 1000):
            train = data.iloc[fold:fold+1000]
            self.model.fit(train.iloc[:,:-1], train.iloc[:,-1])
            y = self.model.predict(X_test)
            predicts.append(y.reshape(1,-1))
        return predicts

    def Predict(self, predict):
        y = sum(np.round(np.mean(predict, axis=0)))
        return y
