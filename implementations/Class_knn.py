import math
import numpy as np
 
class KNN:
    @staticmethod
    def e_distance(data1, data2):
        square = np.square(data1 - data2)
        sum_sq = np.sum(square)
        distance = np.sqrt(sum_sq)
        return distance

    @staticmethod
    def prediction(cls):
        cl0 = 0
        cl1 = 0
        cl2 = 0
    #     if cls.count(0) >= cls.count(1) & cls.count(0) >= cls.count(2):
    #         obj_class += 0

    #     elif cls.count(1) >= cls.count(0) & cls.count(1) >= cls.count(2):
    #         obj_class += 1

    #     elif cls.count(2) >= cls.count(1) & cls.count(2) >= cls.count(0):
    #         obj_class += 2
        for i in cls:
            if i == 2.:
                cl2 += 1
            elif i == 1.:
                cl1 += 1
            elif i == 0.:
                cl0 += 1
        if cl0 >= cl2+cl1:
            return 0.
        elif cl1 >= cl2 + cl0:
            return 1.
        elif cl2 >= cl1 + cl0:
            return 2.
        return 0.


    def get_neighbors(self, X_train, X_test, y_train):
        y_pred=[]
        distances = []
        DistAndClass = {}
        y_train = np.array(y_train)
        for i in range(0, len(X_test)):
            DistAndClass = dict(sorted(DistAndClass.items()))
            neighbors = list(DistAndClass.keys())
            cl = list(DistAndClass.values())
            neighbors = neighbors[:5]
            cl = cl[:5]

            y_pred.append(self.prediction(cl))

            distances = []
            DistAndClass = {}
            for j in range(len(X_train)):
                distances = (self.e_distance(X_train[j],X_test[i]))
                DistAndClass[distances] =  y_train[j]

        return y_pred
