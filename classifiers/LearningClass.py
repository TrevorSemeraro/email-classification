from abc import abstractmethod
from tqdm import tqdm
import pickle
class LearningClass:
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    def evaluate(self, X, Y):
        correct = 0
        total = 0
        print(f"Evaluating with {X.shape[0]} samples")
        
        ham_correct, ham_total = 0, 0
        spam_correct, spam_total = 0, 0
        
        for i, x in tqdm(enumerate(X), total=len(X), desc="Evaluating"):
            prediction = self.predict(x)
            if Y[i] == 0:
                ham_total += 1
            else:
                spam_total += 1
            
            if prediction == Y[i]:
                if Y[i] == 0:
                    ham_correct += 1
                else:
                    spam_correct += 1
                correct += 1
            total += 1
        
        self.spam_correct = spam_correct
        self.spam_total = spam_total
        
        self.ham_correct = ham_correct
        self.ham_total = ham_total
        
        return correct, total

    @abstractmethod
    def save_model(self, directory):
        pass

    @abstractmethod
    def load_model(self, directory):
        pass