import numpy as np

class PCA:
    def __init__(self, n_feature=2):
        self.n_feature = n_feature
    
    def centered(self, X):
        X.apply(lambda x: x - x.mean())
        X.apply(lambda x: x - x.std())
        return X
    
    def sort(self, eigen_values, eigen_vectors):
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        return sorted_eigenvectors
    
    def fit(self, X):
        X = self.centered(X)
        data = np.array(X)
        cov_matrix = np.cov(data, bias=True, rowvar = False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        sorted_eigenvectors = self.sort(eigen_values, eigen_vectors)
        eigenvector_subset = sorted_eigenvectors[:,0:self.n_feature]
        newData = np.dot(eigenvector_subset.transpose(), data.transpose()).transpose()
        return newData 
    