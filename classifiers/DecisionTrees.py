from collections import deque
from typing import List
import numpy as np
from scipy.optimize import minimize
import pickle
from classifiers.LearningClass import LearningClass 

rng = np.random.default_rng()

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, parent=None, left=None, right=None, indicies= None, value=None, depth=0):
        self.feature = feature  # Index of the feature to split on
        self.threshold = threshold  # Value of the feature to split on
        
        self.parent = parent
        self.left = left  # Left child node
        self.right = right  # Right child node
        
        self.indicies = indicies  # Indices of the samples in this node
        self.value = value  # Class label (if leaf node)
        
        self.depth = depth  # Depth of the node in the tree
    
    def is_leaf(self):
        return self.value is not None

class DecisionTrees(LearningClass):
    def __init__(self, data: List[str], labels: List[int], split:float = 0.8):
        elements = data.shape[0]
        if(elements != labels.shape[0]):
            raise ValueError("Data and labels must have the same number of elements")
        
        print(f"Decision Tree classifier, {elements} elements, {data.shape[1]} features, split={split}")
        self.output_classes = np.unique(labels)
        
        self.X = data
        self.Y = labels
        
        mask = rng.uniform(size=elements) < split

        self.train_X = self.X[mask == True] 
        self.train_Y = self.Y[mask == True]
        
        self.test_X = self.X[mask == False]
        self.test_Y = self.Y[mask == False]
        
        self.features = self.X.shape[1]
        
        # with open('decision_tree.pkl', 'rb') as f:
        #     self.root = pickle.load(f)
        
        self.root = DecisionTreeNode(indicies=np.arange(len(self.train_X)), depth=0)
        current_level = deque([self.root])
        
        min_sample_split = 32
        
        while len(current_level) > 0:
            node : DecisionTreeNode = current_level.popleft()
            
            print(f"Node depth: {node.depth}, indicies: {np.sum(node.indicies)}")
            
            if node.is_leaf():
                continue
            
            if len(node.indicies) < min_sample_split:
                node.value = np.argmax(np.bincount(self.train_Y[node.indicies]))
                continue
            
            best_feature, best_threshold = self.optimal_split(self.train_X[node.indicies], self.train_Y[node.indicies])
            print(f"Best feature: {best_feature}, Best threshold: {best_threshold}")
            
            if best_feature is None:
                node.value = np.argmax(np.bincount(self.train_Y[node.indicies]))
                continue
            
            mask = np.zeros(len(self.train_X), dtype=bool)
            mask[node.indicies] = True
            
            left_indices = (self.train_X[:, best_feature] < best_threshold) & mask
            right_indices = (self.train_X[:, best_feature] >= best_threshold) & mask
            
            node.left = left_indices
            node.right = right_indices
            
            node.feature = best_feature
            node.threshold = best_threshold
            
            node.left = DecisionTreeNode(parent=node, indicies=left_indices, depth=node.depth + 1)
            node.right = DecisionTreeNode(parent=node, indicies=right_indices, depth=node.depth + 1)
            
            current_level.append(node.left)
            current_level.append(node.right)
        
        print("Finished building the decision tree")
        print(self.root)
        
        with open('decision_tree.pkl', 'wb') as f:
            pickle.dump(self.root, f)
                    
    def optimal_split(self, X, Y):
        best_feature = None
        best_threshold = None
        best_loss = float('inf')
        
        min_samples_leaf = 16
        
        for feature in range(self.features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                
                if len(Y[left_indices]) <= min_samples_leaf or len(Y[right_indices]) <= min_samples_leaf:
                    continue
                
                current_uncertainty = self.calc_uncertainty(Y)
                loss = current_uncertainty - (
                    len(Y[left_indices]) / len(Y) * self.calc_uncertainty(Y[left_indices]) +
                    len(Y[right_indices]) / len(Y) * self.calc_uncertainty(Y[right_indices])
                )
                
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def calc_uncertainty(self, Y):
        if len(Y) == 0:
            return 0

        # Calculate the Gini impurity
        # class_counts = np.bincount(Y)
        # probabilities = class_counts / len(Y)
        # gini = 1 - np.sum(probabilities ** 2)
        # return gini
        
        # entropy loss
        class_counts = np.bincount(Y)
        probabilities = class_counts / len(Y)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
    
    def predict(self, X: np.ndarray) -> int:
        node: DecisionTreeNode = self.root
        
        while not node.is_leaf():
            if X[node.feature] >= node.threshold:
                node = node.right
            else:
                node = node.left
        
        return node.value