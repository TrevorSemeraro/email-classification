from typing import List
import numpy as np
import scipy
import pickle

from classifiers.LearningClass import LearningClass 

# rng = np.random.default_rng(12345)
rng = np.random.default_rng()

class NaiveBayes(LearningClass):
    def __init__(self, data: List[str], labels: List[int]):
        elements = data.shape[0]
        if(elements != labels.shape[0]):
            raise ValueError("Data and labels must have the same number of elements")
        
        print(f"Naive Bayes classifier, {elements} elements, {data.shape[1]} features")        
        self.output_classes = np.unique(labels)
                
        self.train_X = data
        self.train_Y = labels
        
        self.indices = []
        self.separate = {}
        
        for c in self.output_classes:
            self.separate[c] = data[np.where(labels == c)]
            vars = np.var(self.separate[c], axis=0)
            
            self.indices.append(np.where(vars > 0.0000001))
        
        self.indices = np.intersect1d(self.indices[0], self.indices[1])

        self.train_X = self.train_X[:, self.indices]
        
        print(f"Reduced to {len(self.indices)} features")

    def train(self):
        self.mean = {}
        self.variance = {}
        self.class_prior = {}
        
        for c in self.output_classes:            
            current = self.separate[c][:, self.indices]
        
            self.mean[c] = np.average(current, axis=0)
            self.variance[c] = np.var(current, axis=0)
            self.class_prior[c] = np.mean(self.train_Y == c)
        
        print(self.mean[0].shape, self.variance[0].shape)
    
    def save_model(self, directory):
        model_data = {
            'mean': self.mean,
            'variance': self.variance,
            'class_prior': self.class_prior,
            'indices': self.indices,
            'output_classes': self.output_classes
        }
        with open(f"{directory}/naive_bayes.pkl", "wb") as f:
            pickle.dump(model_data, f)
    
    def load_model(self, directory):
        with open(f"{directory}/naive_bayes.pkl", "rb") as f:
            model_data = pickle.load(f)
            self.mean = model_data['mean']
            self.variance = model_data['variance']
            self.class_prior = model_data['class_prior']
            self.indices = model_data['indices']
            self.output_classes = model_data['output_classes']
    
    def gaussian_likelihood(self, X, c: int):
        mean = self.mean[c]
        variance = np.sqrt(self.variance[c])
        
        return scipy.stats.norm(mean, variance).pdf(X)
    
    def predict(self, X: np.ndarray):   
        X = X[self.indices]
        log_probs = {}

        for c in self.output_classes:
            log_prob = np.log(self.class_prior[c])
            
            likelihood = self.gaussian_likelihood(X, c)
            log_prob += np.log(likelihood).sum()
            
            log_probs[c] = log_prob
        
        return max(log_probs, key=log_probs.get)