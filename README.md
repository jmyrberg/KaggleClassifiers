# KaggleClassifiers

## StackingClassifier

###Stacked generalization
The basic idea behind stacked generalization is to use a pool of base classifiers (level 0 classifiers), and then combine their predictions by using another set of classifiers, meta-classifiers, with the aim of reducing the generalization error.

See: [Kaggle Ensembling Guide](http://mlwave.com/kaggle-ensembling-guide/)
    
    
###How StackingClassifier works:
1. StackingClassifier takes level0 classifiers as an input. These level0 classifiers 'clfs' are trained over all but one fold at a time on the original training set, where the non-trained fold is always left out for out-of-fold probability predictions. Each time the level0 classifiers are trained, also predictions for stacking_test set are made. 

2. Test set predictions are combined as determined by the 'combine_folds_method'. 

3. After this, the full original training set and test set can be represented with out-of-fold predictions. Both training and stacking_test set probabilities for each classifier are combined as determined by the 'combine_probas_method'. Original features can be stacked for both sets, as controlled by 'stack_original_features'.

4. The new combined training and test sets are called blend_train and blend_test,and they are used for training and predicting class probabilities with meta-classifiers 'meta_clfs'. The method to combine predictions of multiple meta-classifiers is controlled by 'combine_meta_probas_method'.

5. Depending on the used prediction method (predict() or predict_proba()), the output is either class labels or class probabilities for the original test set.


###Parameters
* **clfs** :  list (default=[RandomForestClassifier(), ExtraTreesClassifier()])

  Level0 classifiers to use. Classifiers with no .predict_proba() method are used with .predict() method, the result is rounded into nearest class, and the probability for that class is set to 1.*

* **meta_clfs** : list (default=[LogisticRegression()])

  Meta-level classifiers with either method .predict() or .predict_proba().

* **n_blend_folds** : int (default=5)

  Number of folds to produce out-of-fold predictions.

* **stratified** : boolean (default=True)

  Whether to use stratified folds or not.

* **stack_original_features** : boolean (default=False)

  Whether to stack original features with level0 probabilities or not.

* **combine_folds_method** : string (default='fold_score')

  Method for combining out-of-fold predictions:
    - 'mean' : take the mean over folds, separately for each predicted class
    - 'median' : take the median over folds, separately for each predicted class
    - 'fold_score' : use fold weights from out-of-fold score (based on scoring function in 'scoring')

* **combine_probas_method** : string (default='blended')

  Method for combining level0 probability predictions:
    - 'stacked' : stack all probabilities for all classes and classifiers in columns
    - 'mean' : take the mean over all classifiers, separately for each predicted class
    - 'median' : take the median over all classifiers, separately for each predicted class
    - 'weighted' : use custom weight for each classifier as determined in 'weights' 
    - 'fold_avg_score' : use average fold scores as weights for each classifier (based on scoring function in 'scoring')
    - 'fold_geomavg_score' : use geometric average fold scores as weights for each classifier (based on scoring function in 'scoring')
    - 'fold_avg_pow_X_score' : use average fold score to the power of X for each classifier, such as w1^X + w2^X + ...

* **combine_meta_probas_method** : string (default='mean')

  Method for combining predicted meta-classifier probabilities (.predict_proba() -method):
    - 'mean' : take the mean over all meta-classifiers, separately for each predicted class
    - 'median' : take the mean over all meta-classifiers, separately for each predicted class
    - 'min' : take the minimum over all meta-classifiers, separately for each predicted class
    - 'max' : take the maximum over all meta-classifiers, separately for each predicted class
    - 'weighted' : use custom weight for each classifier as determined in 'weights'
    - 'class_majority' : turn max probability into class and use majority vote
    - 'class_mean_round' : turn max probability into class and use rounded mean as class
    - 'class_median_round' : turn max probability into class and use rounded median as class
    - 'class_min' : turn max probability into class and use minimum of classes as class
    - 'class_max' : turn max probability into class and use maximum of classes as class

* **weights** : dict (default={})

    - If 'weighted' method is used in 'combine_probas_method', then dict should have key 'combine_probas' with numeric weights as a list. Number of elements in list should be equal to the number of classifiers. The sum of weights does not have to be 1.
    - If 'weighted' method is used in 'combine_meta_probas_method', then dict should have key 'combine_meta_probas' with numeric weights as a list. Number of elements in list should be equal to the number of classifiers. The sum of weights does not have to be 1.
          
* **save_blend_sets** : None or string (default=None)

    - If None, results are not saved on disk.
    - If string, level0 class out-of-fold probability predictions are saved as follows:
      - blended train set is saved in "string + _blend_train.npy"
      - blended test set predictions are saved in "string + _blend_test.npy"
      - meta-classifier probability predictions are saved in "string + _blend_pred_raw.npy"
      - meta-classifier output (StackingClassifier output), either class or probability is saved in "string + blend_pred.npy"

* **verbose** : 0, 1, or 2 (default=0)

  Print the training/prediction progress. The higher this is, the more is printed.
        
* **compute_scores** : boolean (default=False)

  Whether to compute out-of-fold scores with function in 'scoring'. The scores can be found in 'StackedClassifier().scores_' after fit. Other parameters, such as 'verbose', 'combine_folds_method', 'combine_probas_method' may override this parameter to be True.

* **scoring** : function (default=sklearn.metrics.accuracy_score)

  Function to maximize and use for out-of-fold scores if 'compute_scores'=True. If the metric needs to be minimized, use a custom function that takes parameters (y_true, y_pred) as input, and returns a numeric score. If the score needs to be minimized, one can use for example a custom function with "return(1-original_function(y_pred,y_true)".
        
* **seed** : int (default=1234)

  Seed for k-fold iterations. Level 0 classifier and meta-classifier seeds should be set manually.
                
                
###Attributes
TODO
                
                
###Example
####Imports
```
from stacking_classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
```
  
#####Load data
```
x_train = pd.read_csv('./Data/train.csv').as_matrix()
x_test = x_train[:1000]
x_train = x_train[1000:]
y_train = x_train[:,11].astype(np.int32)
x_train = x_train[:,:11]
y_test = x_test[:,11].astype(np.int32)
x_test = x_test[:,:11]
y_train[y_train==1] = 2
print('Original data shapes:',x_train.shape,x_test.shape,y_train.shape,y_test.shape)
```

#####Level 0 classifiers
```
l0_1 = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=1234)
l0_2 = ExtraTreesClassifier(n_estimators=50, n_jobs=1, random_state=1234)
l0_3 = AdaBoostClassifier(n_estimators=10, random_state=1234)
```

#####Meta-classifiers
```
l1_1 = LogisticRegression(C=1,random_state=1234)
l1_2 = LogisticRegression(C=0.8,random_state=1234)
l1_3 = LogisticRegression(C=1.2,random_state=1234)
```

#####Create new StackingClassfier
```
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
```

#####Class prediction
```
clf2 = clf
clf2.fit(x_train,y_train)
y_pred = clf2.predict(x_test)
print('Scores: %s' % clf2.scores_)
print('Classes: %s' % clf2.classes_)
print('Prediction shape:', y_pred.shape)
print('Accuracy: %.6f' % round(accuracy_score(y_test, y_pred),6))
```

#####Probability prediction
```
print('\nProbability prediction...')
clf3 = clf
clf3.fit(x_train,y_train)
y_pred_proba = clf3.predict_proba(x_test)
print('Scores: %s' % clf3.scores_)
print('Classes: %s' % clf2.classes_)
print('Prediction shape:',y_pred_proba.shape)
print('Accuracy: %.6f' % round(accuracy_score(y_test, clf3.classes_[np.argmax(y_pred_proba, axis=1)]),6))
```

#####Cross validation using sklearn
```
cv = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy', n_jobs=4)
print('Accuracy: %.6f' % round(np.mean(cv),6))
```

#####Open saved sets
```
print('\nLoading saved predictions...')
blend_train = np.load('myStacker_blend_train.npy')
blend_test = np.load('myStacker_blend_test.npy')
y_pred_raw = np.load('myStacker_blend_pred_raw.npy')
y_pred = np.load('myStacker_blend_pred.npy')
print(blend_train.shape,blend_test.shape,y_pred_raw.shape,y_pred.shape)
```
