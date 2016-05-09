"""
StackingClassifier: Simple example

Copyright 2016, 
Jesse Myrberg (jesse.myrberg@aalto.fi)
"""
from stacking_classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
np.random.seed(1234)

def load_data():
    data = pd.read_csv('./Data/winequality-white.csv', sep=';').as_matrix()
    data[data[:,11]==9] = 8
    x_train, y_train = data[:3000,:11], data[:3000,11]
    x_test, y_true = data[3000:,:11], data[3000:,11]
    return(x_train, y_train, x_test, y_true)

def main():
    
    # Load data
    x_train, y_train, x_test, y_true = load_data()
    
    # Create model
    clf = StackingClassifier(clfs = [RandomForestClassifier(), ExtraTreesClassifier()],
                             meta_clfs = [LogisticRegression()])
    
    
    # Fit and predict
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    
    # Evaluate
    print('Accuracy: %f' % ((y_pred==y_true).sum() / y_pred.shape[0]))
    
if __name__ == '__main__':
    main()