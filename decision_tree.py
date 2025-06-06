import numpy as np
from sklearn.model_selection import train_test_split
from classifiers.DecisionTrees import DecisionTrees
    
if __name__ == '__main__':
    print("Loading datasets")    
    X = np.load('./inputs/decision_trees/X.npy')
    Y = np.load('./inputs/decision_trees/Y.npy')
    
    X = X[:, :200]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    classifier = DecisionTrees(X_train, Y_train)
    classifier.train()
    correct, total = classifier.evaluate(X_test, Y_test)
        
    spam_recall = classifier.spam_correct / classifier.spam_total
    ham_recall = classifier.ham_correct / classifier.ham_total

    print("Ham Recall:", ham_recall)
    print("Spam Recall:", spam_recall)
    print("Accuracy:", correct/total)
    
    classifier.save_model('models')