import math
import numpy as np

class Naive_Bayes:
    def __init__(self):
        pass
    
    @staticmethod
    def class_data(data):
        statistics = {}
        for i in range(len(data)):
            vector = data[i]
            if (vector[-1] not in statistics):
                statistics[vector[-1]] = []
            statistics[vector[-1]].append(vector)
        return statistics
    
    def mean(figure):
        return sum(figure)/float(len(figure))
    
    def stdev(figure):
        mean = Naive_Bayes.mean(figure)
        dispersion = sum([pow(x - mean, 2) for x in figure])/float(len(figure))
        return np.sqrt(dispersion)
    
    @staticmethod
    def addition(data):
        additions = [(Naive_Bayes.mean(x), Naive_Bayes.stdev(x)) for x in zip(*data)]
        del additions[-1]
        return additions
    
    @staticmethod
    def addition_for_class(data):
        statistics = Naive_Bayes.class_data(data)
        additions = {}
        for class_value, meaning in statistics.items():
            additions[class_value] = Naive_Bayes.addition(meaning)
        return additions
    
    @staticmethod
    def probability(data, mean, stdev):
        if (2 * math.pow(stdev, 2)) == 0:
            return 0
        else:
            return (1 / (math.sqrt(2 * math.pi) * stdev)) * (math.exp(-(math.pow(data - mean, 2)/(2 * math.pow(stdev, 2)))))
    
    @staticmethod
    def class_probabilities(additions, inputVector):
        probabilities = {}
        for class_value, meaning in additions.items():
            probabilities[class_value] = 1
            for i in range(len(meaning)):
                mean, stdev = meaning[i]
                x = inputVector[i]
                probabilities[class_value] *= Naive_Bayes.probability(x, mean, stdev)
        return probabilities
    
    def predict(additions, inputVector):
        probabilities = Naive_Bayes.class_probabilities(additions, inputVector)
        bestLabel, bestProb = None, -1
        for class_value, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = class_value
        return bestLabel
    
    def get_predict(additions, test):
        predictions = []
        for i in range(len(test)):
            result = Naive_Bayes.predict(additions, test[i])
            predictions.append(result)
        return predictions
    
    def accuracy(test, predictions):
        correct = 0
        for i in range(len(test)):
            if test[i][-1] == predictions[i]:
                correct += 1
        return (correct/float(len(test))) * 100.0
    