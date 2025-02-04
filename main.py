from nltk.stem import PorterStemmer
from collections import Counter
from typing import Dict, List
import numpy as np
import os
import re
from classifiers.NaiveBayes import NaiveBayes

class Vectorizer:
    def __init__(self, dictionaries: List[Dict[str, int]]):
        self.vocab = set()
        
        for d in dictionaries:
            self.vocab.update(d.keys())

    def transform(self, X: Dict[str, int]) -> np.ndarray:
        base = np.zeros(len(self.vocab))
        
        for i, word in enumerate(self.vocab):
            base[i] = X.get(word, 0)
        
        return base
    
if __name__ == '__main__':
    folders = ['ham', 'spam']
    
    emails = []
    labels = []
    
    ps = PorterStemmer()
    
    for folder_index, folder in enumerate(folders):
        files = os.listdir(os.path.join('data', 'emails', folder))
        for file_index, file in enumerate(files):            
            filepath = os.path.join('data', 'emails', folder, file)
            with open(filepath, 'r', encoding="utf8", errors="ignore") as f:
                content = []
                
                for line in f.readlines():
                    words = line.split()
                    
                    for word in words:    
                        word = re.sub(r'\d+', '', word)
                        word = ps.stem(word)
                        content += word
                
                labels.append(1 if folder == 'spam' else 0)    
                email = Counter(words)
                emails.append(email)
    
    print(f"Loaded {len(emails)} emails")
    vectorizer = Vectorizer(emails)
    
    print("Finsished vectorizing emails")
        
    X = np.array([vectorizer.transform(email) for email in emails])
    Y = np.array(labels)
    
    classifier = NaiveBayes(X, Y)
    classifier.evaluate()