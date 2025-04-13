from abc import abstractmethod

class LearningClass:
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    def evaluate(self):
        correct = 0
        total = 0
        print(f"Evaluating with {self.test_X.shape[0]} samples")
        
        for i, x in enumerate(self.test_X):
            prediction = self.predict(x)
            if prediction == self.test_Y[i]:
                correct += 1
            total += 1
        
        print(f"Accuracy: {correct / total}")
        return correct, total