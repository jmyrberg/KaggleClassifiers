"""
Copyright 2016, 
Jesse Myrberg (jesse.myrberg@aalto.fi)
"""
from stacking_classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def main():
    
    # Load data
    print('\nLoading data...')
    x_train = pd.read_csv('./Data/train.csv').as_matrix()
    x_test = x_train[:1000]
    x_train = x_train[1000:]
    y_train = x_train[:,11].astype(np.int32)
    x_train = x_train[:,:11]
    y_test = x_test[:,11].astype(np.int32)
    x_test = x_test[:,:11]
    y_train[y_train==1] = 2
    print('Original data shapes:',x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    
    # Models
    print('\nGenerating models...')
    l0_1 = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=1234)
    l0_2 = ExtraTreesClassifier(n_estimators=50, n_jobs=1, random_state=1234)
    l0_3 = AdaBoostClassifier(n_estimators=10, random_state=1234)
    l1_1 = LogisticRegression(C=1,random_state=1234)
    l1_2 = LogisticRegression(C=0.8,random_state=1234)
    l1_3 = LogisticRegression(C=1.2,random_state=1234)
    
    clf = StackingClassifier(
                            clfs=[l0_1,l0_2,l0_3], 
                            meta_clfs=[l1_1,l1_2,l1_3],
                            n_blend_folds=5,
                            stratified=True,
                            stack_original_features=True,
                            combine_folds_method='fold_score',
                            combine_probas_method='fold_avg_pow_50_score', 
                            combine_meta_probas_method='weighted', 
                            weights = {'combine_meta_probas':[0.5,0.3,0.2]}, 
                            save_blend_sets='myStacker',
                            verbose=0,
                            compute_scores = True,
                            scoring = accuracy_score,
                            seed=1234
                            )
    
    # Class prediction
    print('\nClass prediction...')
    clf2 = clf
    clf2.fit(x_train,y_train)
    y_pred = clf2.predict(x_test)
    print('Scores: %s' % clf2.scores_)
    print('Classes: %s' % clf2.classes_)
    print('Prediction shape:', y_pred.shape)
    print('Accuracy: %.6f' % round(accuracy_score(y_test, y_pred),6))
    
    # Probability prediction
    print('\nProbability prediction...')
    clf3 = clf
    clf3.fit(x_train,y_train)
    y_pred_proba = clf3.predict_proba(x_test)
    print('Scores: %s' % clf3.scores_)
    print('Classes: %s' % clf2.classes_)
    print('Prediction shape:',y_pred_proba.shape)
    print('Accuracy: %.6f' % round(accuracy_score(y_test, clf3.classes_[np.argmax(y_pred_proba, axis=1)]),6))
    
    # Validation
    print('\nCross validating...')
    cv = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy', n_jobs=4)
    print('Accuracy: %.6f' % round(np.mean(cv),6))
    
    # Open predictions
    print('\nLoading saved predictions...')
    blend_train = np.load('myStacker_blend_train.npy')
    blend_test = np.load('myStacker_blend_test.npy')
    y_pred_raw = np.load('myStacker_blend_pred_raw.npy')
    y_pred = np.load('myStacker_blend_pred.npy')
    print(blend_train.shape,blend_test.shape,y_pred_raw.shape,y_pred.shape)
    
    
    print('\nAll ok!')
    

if __name__ == '__main__':
    main()