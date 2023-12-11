import numpy as np

def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.clusters = [[] for i in range(self.k)]
        self.centroids = []
        
    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        random_sample_idsx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idsx]
        for i in range(self.max_iters):
            self.clusters = self.create_clusters(self.centroids)
            centroids_old = self.centroids
            self.centroids = self.recalculation_centroids(self.clusters)
            
            if self.coverage(centroids_old, self.centroids):
                break
        
        return self.cluster_labels(self.clusters)
    
    def create_clusters(self, centroids):
        clusters = [[] for i in range(self.k)]
        for idx in range(len(self.X)):
            centroids_idx = self.closest_centroid(self.X[idx], centroids)
            clusters[centroids_idx].append(idx)
        return clusters
    
    def closest_centroid(self, point, centroids):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
        
    def recalculation_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx in range(len(clusters)):
            cluster_mean = np.mean(self.X[clusters[cluster_idx]], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
                       
    def coverage(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) < 1.1e-09
    
    def cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx in range(len(clusters)):
            for sample_idx in clusters[cluster_idx]:
                labels[sample_idx] = cluster_idx
        return labels
