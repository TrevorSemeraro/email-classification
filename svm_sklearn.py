import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

if __name__ == '__main__':
    print("Loading datasets")    
    X = np.load('./inputs/decision_trees/X.npy')
    Y = np.load('./inputs/decision_trees/Y.npy')
    
    # Use subset of features for faster computation
    X = X[:, :200]
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale the features for better SVM performance
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create SVM classifier with RBF kernel (soft margin)
    print("Training SVM classifier...")
    start_time = time.time()
    
    # Using default parameters: C=1.0, gamma='scale'
    svm_classifier = SVC(
        kernel='rbf',
        C=1.0,  # Regularization parameter (soft margin)
        gamma='scale',  # Kernel coefficient
        random_state=42
    )
    
    svm_classifier.fit(X_train_scaled, Y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Number of support vectors: {svm_classifier.n_support_}")
    print(f"Total support vectors: {np.sum(svm_classifier.n_support_)}")
    
    # Make predictions
    print("Making predictions...")
    start_time = time.time()
    Y_pred = svm_classifier.predict(X_test_scaled)
    prediction_time = time.time() - start_time
    
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    
    # Calculate metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Calculate spam and ham recall
    cm = confusion_matrix(Y_test, Y_pred)
    
    # Assuming 0 = ham, 1 = spam
    if len(np.unique(Y_test)) == 2:
        tn, fp, fn, tp = cm.ravel()
        ham_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        ham_recall = spam_recall = 0
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Ham Recall: {ham_recall:.4f}")
    print(f"Spam Recall: {spam_recall:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(Y_test, Y_pred))
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Try different C values for comparison
    print("\n" + "="*50)
    print("TESTING DIFFERENT C VALUES")
    print("="*50)
    
    C_values = [0.1, 1.0, 10.0, 100.0]
    
    for C in C_values:
        print(f"\nTesting C = {C}")
        svm_temp = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
        svm_temp.fit(X_train_scaled, Y_train)
        Y_pred_temp = svm_temp.predict(X_test_scaled)
        acc_temp = accuracy_score(Y_test, Y_pred_temp)
        print(f"  Accuracy: {acc_temp:.4f}")
        print(f"  Support vectors: {np.sum(svm_temp.n_support_)}") 