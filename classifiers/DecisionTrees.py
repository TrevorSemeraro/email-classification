from collections import deque
from typing import List
import numpy as np
from scipy.optimize import minimize
import pickle
from tqdm import tqdm
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
    def __init__(self, data: List[str], labels: List[int], max_depth=10):
        elements = data.shape[0]
        if(elements != labels.shape[0]):
            raise ValueError("Data and labels must have the same number of elements")
        
        print(f"Decision Tree classifier, {elements} elements, {data.shape[1]} features")
        self.output_classes = np.unique(labels)
        
        self.train_X = data
        self.train_Y = labels
        
        self.max_depth = max_depth
                
        print(f"Using {self.train_X.shape[1]} features after selection")
        print(self.train_X.shape)
        
        self.features = self.train_X.shape[1]
        self.indicies = np.ones(len(self.train_X), dtype=bool)

        self.root = DecisionTreeNode(indicies=self.indicies, depth=0)
    
    def save_model(self, directory):
        with open(f"{directory}/decision_tree.pkl", "wb") as f:
            pickle.dump(self.root, f)
    
    def load_model(self, directory):
        with open(f"{directory}/decision_tree.pkl", "rb") as f:
            self.root = pickle.load(f)
    
    def train(self):
        current_level = deque([self.root])
        
        # min_sample_split = max(50, int(0.01 * len(self.train_X)))  # More conservative
        min_sample_split = 32
        
        while len(current_level) > 0:
            node : DecisionTreeNode = current_level.popleft()
            
            print(f"Node depth: {node.depth}, indicies: {np.sum(node.indicies)}")
            
            if node.is_leaf() or np.sum(node.indicies) < 10:
                continue
            
            # Stop if max depth reached
            if node.depth >= self.max_depth:
                node.value = np.argmax(np.bincount(self.train_Y[node.indicies]))
                continue
            
            if len(node.indicies) < min_sample_split:
                node.value = np.argmax(np.bincount(self.train_Y[node.indicies]))
                continue
            
            # Check if node is pure enough (early stopping)
            class_counts = np.bincount(self.train_Y[node.indicies])
            if len(class_counts) == 1 or np.max(class_counts) / len(node.indicies) > 0.95:
                node.value = np.argmax(class_counts)
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
        best_gain = -float('inf')  # Changed to maximize gain
        
        min_samples_leaf = max(32, int(0.01 * len(X)))
        
        for feature in range(self.features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                
                if len(Y[left_indices]) <= min_samples_leaf or len(Y[right_indices]) <= min_samples_leaf:
                    continue
                
                # Calculate information gain properly
                parent_entropy = self.calc_uncertainty(Y)
                left_entropy = self.calc_uncertainty(Y[left_indices])
                right_entropy = self.calc_uncertainty(Y[right_indices])
                
                # Weighted average of child entropies
                n_left = len(Y[left_indices])
                n_right = len(Y[right_indices])
                n_total = len(Y)
                
                weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
                
                # Information gain = parent entropy - weighted child entropy
                gain = parent_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def calc_uncertainty(self, Y):
        if len(Y) == 0:
            return 0

        # entropy loss
        class_counts = np.bincount(Y)
        probabilities = class_counts / len(Y)
        return -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    def predict(self, X: np.ndarray) -> int:
        node: DecisionTreeNode = self.root
        
        while not node.is_leaf():
            node = node.right if X[node.feature] >= node.threshold else node.left
        
        return node.value