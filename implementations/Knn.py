import math

class KNN:
    @staticmethod
    def euclidean_distance(data1, data2):
        distance = 0
        for i in range (len(data1) - 1):
            distance += (data1[i] - data2[i]) ** 2
        return math.sqrt(distance)

    def neighbors(self, train, test):
        k = 5
        distances = []    
        for i in range(len(train)):
            distances.append((0.0, self.euclidean_distance(train[i], test)))
        
        distances.sort(key=lambda elem: elem[1])
        
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][-1])
            
        return neighbors

    def prediction(self, neighbors):
        count = {}
        for instance in neighbors:
            if instance in count:
                count[instance] += 1
            else:
                count[instance] = 1
                             
        target = max(count.items(), key=lambda x: x[1])[0]
        return target

    def accuracy(self, test, test_prediction):
        correct = 0
        for i in range (len(test)):
            if test[i][-1] == test_prediction[i]:
                correct += 1
        return (correct / len(test))
    
    def append(self, np1):
        np1 = np1.tolist()
        return list.append(np1)
