'''
Created on 1.5.2016

@author: Jesse
'''
from sklearn.base import BaseEstimator, ClassifierMixin

class WrapperClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, clf, fe_set, fe):
        self.clf = clf
        self.fe = fe
        self.fe_set = fe_set
        
    def fit(self, x_train, y_train):
        x_train = self.fe.get(self.fe_set,x_train)
        self.clf.fit(x_train, y_train)
        return(self.clf)
    
    def predict_proba(self,x_test):
        x_test = self.fe.get(self.fe_set,x_test)
        y_pred = self.clf.predict_proba(x_test)
        return(y_pred)
    
    def predict(self, x_test):
        x_test = self.fe.get(self.fe_set,x_test)
        y_pred = self.clf.predict(x_test)
        return(y_pred)