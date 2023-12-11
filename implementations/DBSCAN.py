import numpy as np

def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit(self, X):
        labels = np.zeros(len(X))
        cluster_label = 1
        
        for point_idx in range(len(X)):
            if not (labels[point_idx] == 0):
                continue
                
            if self.installing_cluster(X, labels, point_idx, cluster_label):
                cluster_label += 1
        return labels
        
    def installing_cluster(self, X, labels, point_idx, cluster_label):
        neighbors = self.check_neighbors(X, point_idx)

        if len(neighbors) < self.min_samples:
            labels[point_idx] = -1
            return True

        labels[point_idx] = cluster_label
        for neighbor_index in neighbors:
            if labels[neighbor_index] == 0:
                labels[neighbor_index] = cluster_label
                self.installing_cluster(X, labels, neighbor_index, cluster_label)
        return True

    def check_neighbors(self, X, point_idx):
        neighbors = []
        for i in range(len(X)):
            if euclidean_distance(X[i], X[point_idx]) < self.eps:
                neighbors.append(i)
        return neighbors
