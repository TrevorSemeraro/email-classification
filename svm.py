import numpy as np
from classifiers.NaiveBayes import NaiveBayes
from sklearn.model_selection import train_test_split
    
if __name__ == '__main__':
    print("Loading datasets")    
    X = np.load('./inputs/svm/X.npy')
    Y = np.load('./inputs/svm/Y.npy')
    
    # expensive to solve when d is large, therefore we will be use the most important features
    X = X[:, :200]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    classifier = NaiveBayes(X_train, Y_train)
    classifier.train()
    
    correct, total = classifier.evaluate(X_test, Y_test)
    
    spam_recall = classifier.spam_correct / classifier.spam_total
    ham_recall = classifier.ham_correct / classifier.ham_total

    print("Ham Recall:", ham_recall)
    print("Spam Recall:", spam_recall)
    print("Accuracy:", correct/total)
    
    classifier.save_model('models')