from typing import List
import numpy as np
from scipy.optimize import minimize

from classifiers.LearningClass import LearningClass 

# rng = np.random.default_rng(12345)
rng = np.random.default_rng()        

itteration = 0
def save_weights(w):
    """
    Callback function to save weights during optimization.
    """
    np.save(f"weights_{itteration}.npy", w)
    print("Weights saved:", w)

class KNearestNeighbor(LearningClass):
    def __init__(self, data: List[str], labels: List[int], split:float = 0.8):
        elements = data.shape[0]
        if(elements != labels.shape[0]):
            raise ValueError("Data and labels must have the same number of elements")
        
        print(f"KNN classifier, {elements} elements, {data.shape[1]} features, split={split}")
        self.output_classes = np.unique(labels)
        
        self.X = data
        self.Y = labels
        
        mask = rng.uniform(size=elements) < split

        self.train_X = self.X[mask == True] 
        self.train_Y = self.Y[mask == True]
        
        self.test_X = self.X[mask == False]
        self.test_Y = self.Y[mask == False]
        
        # TODO: Add optimizing weight function, issue: training locally takes too long
        # print(f"Optimizing Neighest Neighbor weights")
        # self.weights = self.optimize_weights()
        # print("Weights:", self.weights)
    
    def weighted_euclidean_distance(self, x1: np.ndarray, x2: np.ndarray, w: np.ndarray) -> float:
        diff = x1 - x2
        return np.sqrt(diff.T @ w @ diff)
    
    def cost_function(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_param: float):
        """
        Compute the cost function:
        Cost(w) = λ * sum over similar pairs - (1 - λ) * sum over dissimilar pairs
        """
        w = np.maximum(w, 1e-6)
        
        weights = np.diag(w)
        
        n = len(X)
        similar_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if y[i] == y[j]]
        dissimilar_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if y[i] != y[j]]

        # Randomly sample 1/2% of pairs
        # sample_ratio = 0.005
        sample_ratio = 0.00005
        sample_similar_pairs = rng.choice(len(similar_pairs), size=int(sample_ratio * len(similar_pairs)), replace=False)
        sample_dissimilar_pairs = rng.choice(len(dissimilar_pairs), size=int(sample_ratio * len(dissimilar_pairs)), replace=False)
        
        print(len(sample_similar_pairs), len(sample_dissimilar_pairs))
        
        sum_similar = sum(self.weighted_euclidean_distance(X[i], X[j], weights) for i, j in np.array(similar_pairs)[sample_similar_pairs])
        sum_dissimilar = sum(self.weighted_euclidean_distance(X[i], X[j], weights) for i, j in np.array(dissimilar_pairs)[sample_dissimilar_pairs])

        print("Done with itteration of cost function")
        return lambda_param * sum_similar - (1 - lambda_param) * sum_dissimilar

    def optimize_weights(self, lambda_param=0.5, max_iter=20):
        """
        Optimize weight vector w for nearest neighbor search.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Class labels (n_samples,)
        - lambda_param: Weighting parameter for similar vs dissimilar pairs
        - max_iter: Maximum iterations for optimization
        
        Returns:
        - Optimized weight vector w
        """
        n_features = self.train_X.shape[1]
        w_init = np.ones(n_features)

        print("Initial weights:", w_init.shape)

        result = minimize(
            self.cost_function, 
            w_init, 
            args=(self.train_X, self.train_Y, lambda_param), 
            method='L-BFGS-B',
            bounds=[(1e-6, None)] * n_features, 
            options={'maxiter': max_iter},
            callback=save_weights
        )

        # save the weights to an npy file
        np.save("weights.npy", result.x)

        return result.x
    
    def predict(self, X: str):
        distances = np.linalg.norm(self.train_X - X, axis=1)
        nearest = np.argmin(distances)
        
        return self.train_Y[nearest]