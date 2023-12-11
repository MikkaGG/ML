import numpy as np
import math as m

class Naive_Bayes:
    def __init__(self):
        pass
    
    @staticmethod
    def separateByClass(dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        # print(separated)
        return separated

    def mean(numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(numbers):
        avg = Naive_Bayes.mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return np.sqrt(variance)
    
    @staticmethod
    def summarize(dataset):
        summaries = [(Naive_Bayes.mean(attribute), Naive_Bayes.stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries
    
    @staticmethod
    def summarizeByClass(dataset):
        separated = Naive_Bayes.separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = Naive_Bayes.summarize(instances)
#         print(summaries)
        return summaries

    @staticmethod
    def calculateProbability(x, mean, stdev):
        try:
            exponent = m.exp(-(m.pow(x - mean, 2) / (2 * m.pow(stdev, 2))))
        except ZeroDivisionError:
            exponent = 0
        return (1 / (m.sqrt(2 * m.pi) * stdev)) * exponent
    
    @staticmethod
    def calculateClassProbabilities(summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= Naive_Bayes.calculateProbability(x, mean, stdev)
        return probabilities

    @staticmethod
    def predict(summaries, inputVector):
        probabilities = Naive_Bayes.calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel
    
    @staticmethod
    def getPredictions(summaries, testSet):
        predictions = []
        for i in range(len(testSet)):
            result = Naive_Bayes.predict(summaries, testSet[i])
            predictions.append(result)
        return predictions

    def getAccuracy(testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0
