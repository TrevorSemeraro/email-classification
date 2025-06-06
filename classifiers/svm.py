from typing import List
import numpy as np
from classifiers.LearningClass import LearningClass 
from scipy.optimize import minimize
from tqdm import tqdm

class SupportVectorMachine(LearningClass):
    def __init__(self, data: List[str], labels: List[int]):
        elements = data.shape[0]
        if(elements != labels.shape[0]):
            raise ValueError("Data and labels must have the same number of elements")
        
        print(f"Support Vector Machine classifier, {elements} elements, {data.shape[1]} features")
                
        self.X = data
        self.Y = labels
                
        self.features = self.X.shape[1]
        self.gamma = 0.1
        self.C = 1
        
    def _rbf_kernel(self, X):
        # Vectorized RBF kernel computation
        sq_norms = np.sum(X ** 2, axis=1)
        K = -2 * X @ X.T + sq_norms[:, None] + sq_norms[None, :]
        return np.exp(-self.gamma * K)

    def train(self):
        n_samples = self.X.shape[0]
        y = self.Y.astype(np.float64)

        K = self._rbf_kernel(self.X)
        Y = y[:, None] * y[None, :] * K

        def objective(alpha):
            return 0.5 * alpha @ Y @ alpha - np.sum(alpha)

        def gradient(alpha):
            return Y @ alpha - np.ones_like(alpha)

        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
        bounds = [(0, self.C)] * n_samples
        initial_alpha = np.zeros(n_samples)

        # Track iterations with tqdm
        pbar = tqdm(desc="Optimizing dual SVM", unit="iter")
        def callback(alpha):
            pbar.update(1)

        result = minimize(
            objective, initial_alpha, jac=gradient,
            bounds=bounds, constraints=constraints,
            callback=callback,
            options={"maxiter": 500}
        )
        pbar.close()
        
        self.alpha = result.x
        self.support_ = self.alpha > 1e-5
        self.support_vectors_ = self.X[self.support_]
        self.alpha_support_ = self.alpha[self.support_]
        self.y_support_ = y[self.support_]

        # Compute bias term b using support vectors
        K_sv = self._rbf_kernel(self.support_vectors_)
        decision = np.sum(self.alpha_support_ * self.y_support_ * K_sv, axis=1)
        self.b = np.mean(self.y_support_ - decision)

    def project(self, X):
        # Project input points to decision function value
        K = self._rbf_kernel_predict(X)
        return (self.alpha_support_ * self.y_support_) @ K.T + self.b

    def predict(self, X):
        return np.sign(self.project(X))

    def _rbf_kernel_predict(self, X_new):
        X_train = self.support_vectors_
        sq_norm_train = np.sum(X_train ** 2, axis=1)
        sq_norm_new = np.sum(X_new ** 2, axis=1)
        K = -2 * X_new @ X_train.T + sq_norm_new[:, None] + sq_norm_train[None, :]
        return np.exp(-self.gamma * K)