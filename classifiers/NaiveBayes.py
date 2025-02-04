from typing import List
import numpy as np
import scipy 

# rng = np.random.default_rng(12345)
rng = np.random.default_rng()

class NaiveBayes:
    def __init__(self, data: List[str], labels: List[int], split:float = 0.8):
        elements = data.shape[0]
        if(elements != labels.shape[0]):
            raise ValueError("Data and labels must have the same number of elements")
        
        print(f"Naive Bayes classifier, {elements} elements, {data.shape[1]} features, split={split}")
        self.output_classes = np.unique(labels)
        
        self.X = data
        self.Y = labels
        
        mask = rng.uniform(size=elements) < split

        self.train_X = self.X[mask] 
        self.train_Y = self.Y[mask]
        
        self.test_X = self.X[mask == False]
        self.test_Y = self.Y[mask == False]
        
        self.indices = []
        self.separate = {}
        
        for c in self.output_classes:
            self.separate[c] = data[np.where(labels == c)]
            vars = np.var(self.separate[c], axis=0)
            
            self.indices.append(np.where(vars > 0.0000001))
        
        self.indices = np.intersect1d(self.indices[0], self.indices[1])

        self.train_X = self.train_X[:, self.indices]
        self.test_X = self.test_X[:, self.indices]
        
        print(f"Reduced to {len(self.indices)} features")

        self.mean = {}
        self.variance = {}
        self.class_prior = {}
        
        for c in self.output_classes:            
            current = self.separate[c][:, self.indices]
        
            self.mean[c] = np.average(current, axis=0)
            self.variance[c] = np.var(current, axis=0)
            self.class_prior[c] = np.mean(labels == c)
        
        print(self.mean[0].shape, self.variance[0].shape)
                        
    def gaussian_likelihood(self, X, c: int):
        mean = self.mean[c]
        variance = np.sqrt(self.variance[c])
        
        return scipy.stats.norm(mean, variance).pdf(X)
    
    def predict(self, X: str):                
        log_probs = {}

        for c in self.output_classes:
            log_prob = np.log(self.class_prior[c])
            
            likelihood = self.gaussian_likelihood(X, c)
            log_prob += np.log(likelihood).sum()
            
            log_probs[c] = log_prob
        
        return max(log_probs, key=log_probs.get)

    def evaluate(self):
        correct = 0
        total = 0
        print(f"Evaluating with {self.test_X.shape[0]} samples")
        
        for i, x in enumerate(self.test_X):
            prediction = self.predict(x)
            # print(f"Predicted {prediction}, actual {self.test_Y[i]}")
            if prediction == self.test_Y[i]:
                correct += 1
            total += 1
        
        print(f"Accuracy: {correct / total}")
        return correct, total