"""

Copyright 2016, Jesse Myrberg.
BSD license, 3 clauses.

"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from scipy.stats import mode


class StackingClassifier(BaseEstimator, ClassifierMixin):
    """ 
    Base class for classifiers that use stacked generalization.
    
    Stacked generalization
    ----------------------
    Stacked generalization as described in http://mlwave.com/kaggle-ensembling-guide/:
    The basic idea behind stacked generalization is to use a pool of base classifiers, then using another 
    classifier to combine their predictions, with the aim of reducing the generalization error.
    
    This classifier implements stacked generalization with API similar to sklearn classifiers. Some
    extra features are added in order to control how level 0 and meta-level probabilities are aggregated.
    
    
    How StackingClassifier works:
    -----------------------------
    
        1. StackingClassifier only takes level0 classifiers clfs with predict_proba() -method 
        as an input. These level0 classifiers clfs are trained over all but one fold at a time 
        on the original training set, where the last fold is always left out for out-of-fold 
        probability predictions. Each time the classifiers are trained, also predictions for 
        test set are made. 
        
        2. The test set predictions are combined as determined by the combine_folds_method. 
        
        3. After this, the full original training set and test set can be represented as 
        out-of-fold predictions. Training and test set probabilities for each classifier may 
        be combined as determined by the combine_probas_method, which defaults to 'blended' 
        (blend the probabilities in columns).
        
        4. The new training and test sets are used for training and predicting with the 
        meta-classifiers meta_clfs. The output from meta-classifiers may be probabilities
        or labels. The method to combine predictions from meta-classifiers is either
        combine_meta_probas_method (all meta-classifiers must have .predict_proba() method) 
        or combine_meta_class_method (all meta-classifiers must have .predict() method).
    
        NOTE: fit() -method only sets training set and training targets in the classifier.
        NOTE: No error handling is provided.
    """
    
    def __init__(self, clfs=[RandomForestClassifier()],meta_clfs=[LogisticRegression()], 
                 n_blend_folds=5, stratified=True, stack_original_features=False,
                 combine_folds_method='fold_score', combine_probas_method='blended', 
                 combine_meta_probas_method='mean', combine_meta_class_method='majority',
                 weights = {}, save_preds = None, verbose=0, compute_scores = False,
                 scoring = accuracy_score, seed=1234):
        """
        Stacked generalization
        ----------------------
        Stacked generalization as described in http://mlwave.com/kaggle-ensembling-guide/:
        The basic idea behind stacked generalization is to use a pool of base classifiers, then using another 
        classifier to combine their predictions, with the aim of reducing the generalization error.
        
        
        Parameters
        ----------
        clfs :  list (default=[RandomForestClassifier()])
                Level0 classifiers with method .predict_proba().
        
        meta_clfs : list (default=[LogisticRegression()])
                Meta-level classifiers with either method .predict() or .predict_proba().
        
        n_blend_folds : int (default=5)
                Number of folds to produce out-of-fold predictions.
        
        stratified : boolean (default=True)
                Whether to use stratified folds or not.
        
        stack_original_features : boolean (default=False)
                Whether to blend original features with probabilities or not.
        
        combine_folds_method : string (default='fold_score')
                Method for combining out-of-fold predictions:
                - 'mean' : take the mean of folds, individually for each predicted class
                - 'median' : take the median of folds, individually for each predicted class
                - 'fold_score' : use fold weights from out-of-fold score (based on scoring 
                  function in 'scoring')
        
        combine_probas_method : string (default='blended')
                Method for combining level0 probability predictions:
                - 'blended' : blend all probabilities for all classes in columns
                - 'mean' : take the mean over all classifiers, individually for each predicted class
                - 'weighted' : use custom weights determined in 'weights' for each classifier
                - 'fold_avg_score' : use average fold scores of classifiers as weights 
                  (based on scoring function in 'scoring')
        
        combine_meta_probas_method : string (default='mean')
                Method for combining predicted meta-classifier probabilities (.predict_proba() -method):
                - 'blended' : blend all probabilities for all classes in columns
                - 'mean' : take the mean over all meta-classifiers, individually for each predicted class
                - 'weighted' : use custom weights determined in 'weights' for each classifier
        
        combine_meta_class_method : string (default='majority')
                Method for combining predicted meta-classifier classes (.predict() -method):
                - 'majority' : take the mode over all classifiers
                - 'median' : take the median class over all classifiers
                - 'min' : take the minimum class over all classifiers
                - 'max' : take the maximum class over all classifiers
                - 'mean_round' : take the mean of classes over all classifiers and round the result to integer
        
        weights : dict (default={})
                - If 'weighted' method is used in 'combine_probas_method', then dict should have key
                  'combine_probas' with numeric weights as a list. Number of elements in list should
                  be equal to the number of classifiers. The sum of weights does not have to be 1.
                - If 'weighted' method is used in 'combine_meta_probas_method', then dict should have key
                  'combine_meta_probas' with numeric weights as a list. Number of elements in list 
                  should be equal to the number of classifiers. The sum of weights does not have to be 1.
                  
        save_preds : None or string (default=None)
                - If None, results are not saved on disk.
                - If string, level0 class out-of-fold probability predictions for train set are saved in
                  "string + '_blend_train.npy'", test set predictions are saved in "string + '_blend_test.npy'",
                  and meta-classifier predictions (class or probability) are saved in "string + '_blend_train.npy'"
        
        verbose : 0, 1, or 2 (default=0)
                Print the training/prediction progress. The higher, the more is printed.
                
        compute_scores : boolean (default=False)
                Whether to compute out-of-fold scores with function in 'scoring'. The scores can be found in
                'StackedClassifier().scores_'. Other parameters, such as 'verbose', 'combine_folds_method', 
                'combine_probas_method' may override this to be True.
        
        scoring : function (default=sklearn.metrics.accuracy_score)
                Function to use for out-of-fold scores if 'compute_scores'=True. May be custom function that
                takes parameters (y_true, y_pred) as input, and returns a numeric score.
                
        seed : int (default=1234)
                Seed for k-fold iterations. Level 0 classifier and meta-classifier seeds should be set manually.
                
                
                
        Example
        ----------        
        from blend_classifier import StackingClassifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.cross_validation import cross_val_score
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import numpy as np
            
        # Load data
        print('Loading data...')
        x_train = pd.read_csv('./Data/train.csv').as_matrix()
        y_train = pd.read_csv('./Data/y_train.csv').as_matrix()
        x_test = pd.read_csv('./Data/test.csv').as_matrix()
        
        # Level 0 models
        print('\nGenerating models...')
        l0_1 = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=1234)
        l0_2 = ExtraTreesClassifier(n_estimators=50, n_jobs=1, random_state=1234)
        l1_1 = LogisticRegression(C=1,random_state=1234)
        l1_2 = LogisticRegression(C=0.5,random_state=1234)
        l1_3 = LogisticRegression(C=0.25,random_state=1234)
        
        clf = StackingClassifier(
                                clfs=[l0_1,l0_2], 
                                meta_clfs=[l1_1,l1_2,l1_3],
                                n_blend_folds=5,
                                stratified=True,
                                stack_original_features=True,
                                combine_folds_method='fold_score',
                                combine_probas_method='weighted', 
                                combine_meta_probas_method='mean', 
                                combine_meta_class_method='median',
                                weights={'combine_probas':[0.3,0.7]}, 
                                save_preds='myStackRun',
                                verbose=2,
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
        
        # Load saved predictions
        print('\nLoading saved predictions...')
        blend_train = np.load('myStackRun_blend_train.npy')
        blend_test = np.load('myStackRun_blend_test.npy')
        blend_pred = np.load('myStackRun_blend_pred.npy')
        print(blend_train.shape,blend_test.shape,blend_pred.shape)
        
        print('\nDone!')
        """
        self.clfs = clfs
        self.meta_clfs = meta_clfs
        self.n_blend_folds = n_blend_folds
        self.stratified = stratified
        self.stack_original_features = stack_original_features
        self.combine_folds_method = combine_folds_method
        self.combine_probas_method = combine_probas_method
        self.combine_meta_probas_method = combine_meta_probas_method
        self.combine_meta_class_method = combine_meta_class_method
        self.weights = weights
        self.save_preds = save_preds
        self.verbose = verbose
        self.compute_scores = compute_scores
        self.scoring = scoring
        self.seed = seed
        self.must_compute_scores_ = self._must_compute_score()
        
    def _save_lvl0_predictions(self,blend_train,blend_test):
        np.save(self.save_preds + '_blend_train.npy', blend_train)
        np.save(self.save_preds + '_blend_test.npy', blend_test)
        
    def _save_meta_predictions(self,y_pred):
        for i,pred in enumerate(y_pred.values()):
            if i == 0:
                out = pred
            else:
                out = np.column_stack((out,pred))
        np.save(self.save_preds + '_blend_pred.npy', out)
        
    def _clear_mem(self):
        del self.x_train
        del self.y_train
        
    def _must_compute_score(self):
        if self.compute_scores:
            return(True)
        if self.verbose > 1:
            return(True)
        if self.combine_folds_method in ['fold_score']:
            return(True)
        if self.combine_probas_method in ['fold_score']:
            return(True)
        
    def _get_score(self, y_true, y_pred):
        score = self.scoring(y_true, y_pred)
        return(score)
        
    def _combine_folds(self, blend_test):
        if self.combine_folds_method == 'mean':
            d = {}
            for clf,iv in blend_test.items():
                for fold,jv in iv.items():
                    if fold == 0:
                        d[clf] = np.zeros(jv.shape)
                    d[clf] += jv
                    if fold == self.n_blend_folds-1:
                        d[clf] /= self.n_blend_folds
        
        elif self.combine_folds_method == 'median':
            d_tmp = {}
            d = {}
            for clf,iv in blend_test.items():
                for fold,jv in iv.items():
                    if fold == 0:
                        n_obs = jv.shape[0]
                        d_tmp[clf] = []
                    d_tmp[clf].append(jv)
                d[clf] = np.zeros((n_obs,self.n_classes_))
                for cls in range(self.n_classes_):
                    res = np.zeros((n_obs,self.n_blend_folds))
                    for fold in range(self.n_blend_folds):
                        res[:,fold] = d_tmp[clf][fold][:,cls]
                    res = np.median(res, axis=1)
                    d[clf][:,cls] = res
                d[clf] /= d[clf].sum(axis=1)[:,None]    
        
        elif self.combine_folds_method == 'fold_score':
            d = {}
            for clf,iv in blend_test.items():
                for fold,jv in iv.items():
                    if fold == 0:
                        d[clf] = np.zeros(jv.shape)
                    w = np.array(self.scores_[clf])
                    w /= w.sum()
                    d[clf] += w[fold]*jv
        return(d)
        
    def _combine_probas(self, blend_train, blend_test):
        blend_test = self._combine_folds(blend_test)
        blend_trains = list(blend_train.values())
        blend_tests = list(blend_test.values())
        
        if self.combine_probas_method == 'stacked':
            blend_train = np.concatenate(blend_trains, axis=1)
            blend_test = np.concatenate(blend_tests, axis=1)
        
        elif self.combine_probas_method == 'mean':
            tmp_train = np.zeros(blend_trains[0].shape)
            tmp_test = np.zeros(blend_tests[0].shape)
            n_clfs = len(self.clfs)
            for i in range(n_clfs):
                tmp_train += blend_trains[i]
                tmp_test += blend_tests[i]
            blend_train = tmp_train / n_clfs
            blend_test = tmp_test / n_clfs
        
        elif self.combine_probas_method == 'weighted':
            w = np.array(self.weights['combine_probas'])
            w /= w.sum()
            tmp_train = np.zeros(blend_trains[0].shape)
            tmp_test = np.zeros(blend_tests[0].shape)
            n_clfs = len(self.clfs)
            for i in range(n_clfs):
                tmp_train += w[i]*blend_trains[i]
                tmp_test += w[i]*blend_tests[i]
            blend_train = tmp_train
            blend_test = tmp_test
        
        elif self.combine_probas_method == 'fold_avg_score':
            w = np.array([np.mean(e) for e in self.scores_.values()])
            w /= w.sum()
            tmp_train = np.zeros(blend_trains[0].shape)
            tmp_test = np.zeros(blend_tests[0].shape)
            n_clfs = len(self.clfs)
            for i in range(n_clfs):
                tmp_train += w[i]*blend_trains[i]
                tmp_test += w[i]*blend_tests[i]
            blend_train = tmp_train
            blend_test = tmp_test
        return(blend_train, blend_test)
    
    def _combine_meta_probas(self, y_pred_meta):
        if self.combine_meta_probas_method == 'mean':
            y_pred = np.zeros(y_pred_meta[0].shape)
            for v in y_pred_meta.values():
                y_pred += v
            y_pred /= len(y_pred_meta)
        
        elif self.combine_meta_probas == 'stacked':
            for i,v in enumerate(y_pred_meta.values()):
                if i == 0:
                    y_pred = v
                else:
                    y_pred = np.column_stack((y_pred,v))
        
        elif self.combine_meta_probas == 'weighted':
            w = np.array(self.weights['combine_meta_probas'])
            w /= w.sum()
            y_pred = np.zeros(y_pred_meta[0].shape)
            for i,v in enumerate(y_pred_meta.values()):
                y_pred += w[i]*v
            y_pred /= len(y_pred_meta)
        return(y_pred)
            
    def _combine_meta_class(self, y_pred_meta):
        for i,(k,v) in enumerate(y_pred_meta.items()):
                if i == 0:
                    y_pred_tmp = np.zeros((y_pred_meta[k].shape[0],len(y_pred_meta)))
                y_pred_tmp[:,i] = v
        
        if self.combine_meta_class_method == 'majority':
            y_pred = mode(y_pred_tmp, axis=1)[0].ravel()
        
        elif self.combine_meta_class_method == 'median':
            y_pred = np.median(y_pred_tmp, axis=1)
        
        elif self.combine_meta_class_method == 'min':
            y_pred = y_pred_tmp.min(axis=1)
        
        elif self.combine_meta_class_method == 'max':
            y_pred = y_pred_tmp.max(axis=1)
        
        elif self.combine_meta_class_method == 'mean_round':
            y_pred = np.round(y_pred_tmp.mean(axis=1))
        return(y_pred)    
        
    def _predict_lvl0(self, x_test): # Level 0
        
        if self.stratified:
            kf = StratifiedKFold(self.y_train, n_folds=self.n_blend_folds, shuffle=True, random_state=self.seed)
        else:
            kf = KFold(n_folds=self.n_blend_folds, shuffle=True, random_state=self.seed)
        
        blend_train = {}
        blend_test = {}
        self.classes_ = np.sort(pd.unique(self.y_train))
        self.n_classes_ = len(self.classes_)
        n_clfs = len(self.clfs)
        n_train = self.x_train.shape[0]
        
        if self.must_compute_scores_:
            self.scores_ = {}
        
        for i,clf in enumerate(self.clfs):
            
            if self.verbose > 0:
                print('Training level 0 classifier %d/%d: %s' % (i+1,n_clfs,clf.__repr__()))
            
            blend_train[i] = np.zeros((n_train,self.n_classes_))
            blend_test[i] = {}
            
            for j,(tr_ind,te_ind) in enumerate(kf):
                blend_test[i][j] = np.zeros((n_train,self.n_classes_))
                x_train_fold, x_test_fold, y_train_fold = self.x_train[tr_ind], self.x_train[te_ind], self.y_train[tr_ind]
                clf.fit(x_train_fold, y_train_fold)
                y_pred_fold = clf.predict_proba(x_test_fold)
                blend_train[i][te_ind] = y_pred_fold
                blend_test[i][j] = clf.predict_proba(x_test)
                
                if self.must_compute_scores_:
                    y_test_fold = self.y_train[te_ind]
                    y_pred_max_proba = [self.classes_[e] for e in np.argmax(y_pred_fold, axis=1)]
                    score = self._get_score(y_test_fold, y_pred_max_proba)
                    if j == 0:
                        self.scores_[i] = []
                    
                    if self.verbose > 1:
                        print('-- Fold %d/%d score: %.6f' % (j+1,self.n_blend_folds,round(score,6)))
                    self.scores_[i].append(score)
                    if self.verbose > 1 and j == self.n_blend_folds-1:
                        avg_score = np.mean(self.scores_[i])
                        print('-- Average score: %.6f' % round(avg_score,6))
        
        blend_train, blend_test = self._combine_probas(blend_train, blend_test)
        
        if self.save_preds is not None:
            self._save_lvl0_predictions(blend_train,blend_test)
        
        if self.stack_original_features:
            blend_train = np.column_stack((self.x_train,blend_train))
            blend_test = np.column_stack((x_test,blend_test))
        return(blend_train, blend_test)
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return(self)
        
    def predict_proba(self, x_test): # Level 1 probability output
        blend_train, blend_test = self._predict_lvl0(x_test)
        y_pred = {}
        for i,clf in enumerate(self.meta_clfs):
            if self.verbose > 0:
                print('Training meta classifier %d/%d: %s' % (i+1,len(self.meta_clfs),clf.__repr__()))
            clf.fit(blend_train, self.y_train)
            y_pred[i] = clf.predict_proba(blend_test)
        
        if self.save_preds is not None:
            self._save_meta_predictions(y_pred)
        
        y_pred = self._combine_meta_probas(y_pred)
        self._clear_mem()
        return(y_pred)
    
    def predict(self, x_test): # Level 1 class output
        blend_train, blend_test = self._predict_lvl0(x_test)
        y_pred = {}
        for i,clf in enumerate(self.meta_clfs):
            if self.verbose > 0:
                print('Training meta classifier %d/%d: %s' % (i+1,len(self.meta_clfs),clf.__repr__()))
            clf.fit(blend_train, self.y_train)
            y_pred[i] = clf.predict(blend_test)
        
        if self.save_preds is not None:
            self._save_meta_predictions(y_pred)
        
        y_pred = self._combine_meta_class(y_pred)
        self._clear_mem()
        return(y_pred)
            
