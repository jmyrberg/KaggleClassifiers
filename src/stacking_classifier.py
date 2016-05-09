"""
StackingClassifier

Copyright 2016, 
Jesse Myrberg (jesse.myrberg@aalto.fi)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from scipy.stats import mode, gmean
from copy import copy


class StackingClassifier(BaseEstimator, ClassifierMixin):
    """ 
    Base class for stacked generalization classifier.
    """
    
    def __init__(self, clfs=[RandomForestClassifier(), ExtraTreesClassifier()], meta_clfs=[LogisticRegression()], 
                 n_blend_folds=3, stratified=True, stack_original_features=False,
                 combine_folds_method='fold_score', combine_probas_method='stacked', 
                 combine_meta_probas_method='mean', weights = {}, save_blend_sets = None, 
                 verbose=0, compute_scores = False, scoring = accuracy_score, seed=None):
        """
        Stacked generalization
        ----------------------
        The basic idea behind stacked generalization is to use a pool of base classifiers 
        (level 0 classifiers), and then combine their predictions by using another set of 
        classifiers, meta-classifiers, with the aim of reducing the generalization error.
        
        
        Parameters
        ----------
        clfs :  list (default=[RandomForestClassifier(), ExtraTreesClassifier()])
                Level0 classifiers to use. Classifiers with no .predict_proba() method
                are used with .predict() method, the result is rounded into nearest class, 
                and the probability for that class is set to 1.
        
        meta_clfs : list (default=[LogisticRegression()])
                Meta-level classifiers with either method .predict() or .predict_proba().
        
        n_blend_folds : int (default=5)
                Number of folds to produce out-of-fold predictions.
        
        stratified : boolean (default=True)
                Whether to use stratified folds or not.
        
        stack_original_features : boolean (default=False)
                Whether to stack original features with level0 probabilities or not.
        
        combine_folds_method : string (default='fold_score')
                Method for combining out-of-fold predictions:
                - 'mean' : take the mean over folds, separately for each predicted class
                - 'median' : take the median over folds, separately for each predicted 
                  class
                - 'fold_score' : use fold weights from out-of-fold score (based on scoring 
                  function in 'scoring')
        
        combine_probas_method : string (default='blended')
                Method for combining level0 probability predictions:
                - 'stacked' : stack all probabilities for all classes and classifiers in columns
                - 'mean' : take the mean over all classifiers, separately for each predicted class
                - 'median' : take the median over all classifiers, separately for each predicted class
                - 'weighted' : use custom weight for each classifier as determined in 'weights' 
                - 'fold_avg_score' : use average fold scores as weights for each classifier
                  (based on scoring function in 'scoring')
                - 'fold_geomavg_score' : use geometric average fold scores as weights for each 
                  classifier (based on scoring function in 'scoring')
                - 'fold_avg_pow_X_score' : use average fold score to the power of X for each
                  classifier, such as w1^X + w2^X + ...
        
        combine_meta_probas_method : string (default='mean')
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
        
        weights : dict (default={})
                - If 'weighted' method is used in 'combine_probas_method', then dict should have key
                  'combine_probas' with numeric weights as a list. Number of elements in list should
                  be equal to the number of classifiers. The sum of weights does not have to be 1.
                - If 'weighted' method is used in 'combine_meta_probas_method', then dict should have key
                  'combine_meta_probas' with numeric weights as a list. Number of elements in list 
                  should be equal to the number of classifiers. The sum of weights does not have to be 1.
                  
        save_blend_sets : None or string (default=None)
                - If None, results are not saved on disk.
                - If string, level0 class out-of-fold probability predictions are saved as follows:
                    o train set is saved in "string + _blend_train.npy"
                    o stacking_full_example set predictions are saved in "string + _blend_test.npy"
                    o meta-classifier probability predictions are saved in "string + _blend_pred_raw.npy"
                    o meta-classifier output (StackingClassifier output), either class or probability 
                      is saved in "string + blend_pred.npy"
        
        verbose : 0, 1, or 2 (default=0)
                Print the training/prediction progress. The higher this is, the more is printed.
                
        compute_scores : boolean (default=False)
                Whether to compute out-of-fold scores with function in 'scoring'. The scores can be found in
                'StackedClassifier().scores_' after fit. Other parameters, such as 'verbose', 
                'combine_folds_method', 'combine_probas_method' may override this parameter to be True.
        
        scoring : function (default=sklearn.metrics.accuracy_score)
                Function to maximize and use for out-of-fold scores if 'compute_scores'=True. If the
                metric needs to be minimized, use a custom function that takes parameters (y_true, y_pred)
                as input, and returns a numeric score. If the score needs to be minimized, one can use
                for example a custom function with "return(1-original_function(y_pred,y_true)".
                
        seed : int (default=1234)
                Seed for k-fold iterations. Level 0 classifier and meta-classifier seeds should be set manually.
                
                
        Attributes
        ----------
        
                
                
        Example
        ----------        
        from stacking_classifier import StackingClassifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.cross_validation import cross_val_score
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import numpy as np
            
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
        """
        self.clfs = clfs
        self.meta_clfs = meta_clfs
        self.n_blend_folds = n_blend_folds
        self.stratified = stratified
        self.stack_original_features = stack_original_features
        self.combine_folds_method = combine_folds_method
        self.combine_probas_method = combine_probas_method
        self.combine_meta_probas_method = combine_meta_probas_method
        self.weights = weights
        self.save_blend_sets = save_blend_sets
        self.verbose = verbose
        self.compute_scores = compute_scores
        self.scoring = scoring
        self.seed = seed
        
    def _set_clf_names(self):
        self.clf_names_ = {}
        for i,clf in enumerate(self.clfs):
            self.clf_names_[i] = clf.__repr__()
        self.meta_clf_names_ = {}
        for j,meta_clf in enumerate(self.meta_clfs):
            self.meta_clf_names_[j] = meta_clf.__repr__()    
    
    def _copy_lvl0_clfs(self):
        self.clfs_cv_ = {}
        for i,clf in enumerate(self.clfs):
            self.clfs_cv_[i] = []
            for _ in range(self.n_blend_folds):
                self.clfs_cv_[i].append(copy(clf))
    
    def _save_set(self, array, name):
        np.save(self.save_blend_sets + name + '.npy', array)
        
    def _must_compute_score(self):
        if self.compute_scores:
            return(True)
        if self.verbose > 1:
            return(True)
        if self.combine_folds_method in ['fold_score']:
            return(True)
        if self.combine_probas_method in ['fold_avg_score']:
            return(True)
        
    def _get_score(self, y_true, y_pred):
        score = self.scoring(y_true, y_pred)
        return(score)
    
    def _find_nearest(self, array, value):
        ind = (np.abs(array-value)).argmin()
        return(ind)
    
    def _get_proba(self, clf, x):
        if hasattr(clf,'predict_proba'):
            y_pred = clf.predict_proba(x)
        else:
            y_pred = np.zeros((x.shape[0],self.n_classes_))
            y_pred_tmp = clf.predict(x)
            inds = np.array([self._find_nearest(self.classes_,ind) for ind in y_pred_tmp])
            y_pred[np.arange(x.shape[0]),inds] = 1
        return(y_pred)
    
    def _combine_probas(self, blend_set):
        # Input size: (n_clfs, n_obs, n_classes)
        if self.combine_probas_method == 'stacked':
            comb = blend_set.reshape((blend_set.shape[1],blend_set.shape[2]*blend_set.shape[0]))
        
        elif self.combine_probas_method == 'mean':
            comb = blend_set.mean(axis=0)
            
        elif self.combine_probas_method == 'median':
            comb = np.median(blend_set, axis=0)
            comb /= comb.sum(axis=1)[:,None]
        
        elif self.combine_probas_method == 'weighted':
            w = np.array(self.weights['combine_probas'])
            w /= w.sum()
            comb = (blend_set*w[:,None,None]).sum(axis=0)
        
        elif self.combine_probas_method == 'fold_avg_score':
            w = np.array([np.mean(e) for e in self.scores_.values()])
            w /= w.sum()
            comb = (blend_set*w[:,None,None]).sum(axis=0)
            
        elif self.combine_probas_method == 'fold_geomavg_score':
            w = np.array([gmean(e) for e in self.scores_.values()])
            w /= w.sum()
            comb = (blend_set*w[:,None,None]).sum(axis=0)
            
        elif self.combine_probas_method.startswith('fold_avg_pow_'):
            power = eval(self.combine_probas_method.split('_')[-2])
            w = np.array([np.mean(e) for e in self.scores_.values()])
            w **= power
            w /= w.sum()
            comb = (blend_set*w[:,None,None]).sum(axis=0)
        return(comb)
        
    def _combine_folds(self, blend_test):
        # Input size: (n_clfs, n_cv, n_obs, n_classes)
        if self.combine_folds_method == 'mean':
            comb = blend_test.mean(axis=1)
        
        elif self.combine_folds_method == 'median':
            comb = np.median(blend_test, axis=1)
            comb /= comb.sum(axis=2)[:,:,None]
        
        elif self.combine_folds_method == 'fold_score':
            w = np.array([e for e in self.scores_.values()])
            w /= w.sum(axis=0)
            comb = (blend_test*w[:,:,None,None]).sum(axis=1)
            comb /= comb.sum(axis=2)[:,:,None]
        return(comb)
    
    def _combine_meta_probas(self, y_pred_meta):
        # Input size: (n_clfs, n_obs, n_classes)
        if self.combine_meta_probas_method == 'mean':
            comb = y_pred_meta.mean(axis=0)
        
        elif self.combine_meta_probas_method == 'median':
            comb = np.median(y_pred_meta, axis=0)
            comb /= comb.sum(axis=1)[:,None]
            
        elif self.combine_meta_probas_method == 'min':
            comb = y_pred_meta.min(axis=0)
            comb /= comb.sum(axis=1)[:,None]
            
        elif self.combine_meta_probas_method == 'max':
            comb = y_pred_meta.max(axis=0)
            comb /= comb.sum(axis=1)[:,None]
        
        elif self.combine_meta_probas_method == 'weighted':
            w = np.array(self.weights['combine_meta_probas'])
            w /= w.sum()
            comb = (y_pred_meta*w[:,None,None]).sum(axis=0)
            
        elif self.combine_meta_probas_method.startswith('class'):
            n_obs = y_pred_meta.shape[1]
            y_pred_meta = self.classes_[y_pred_meta.argmax(axis=2)]
            
            if self.combine_meta_probas_method == 'class_majority':
                y_pred_meta = mode(y_pred_meta, axis=0)[0].ravel()
                
            elif self.combine_meta_probas_method == 'class_mean_round':
                y_pred_meta = np.round(y_pred_meta.mean(axis=0))
                
            elif self.combine_meta_probas_method == 'class_median_round':
                y_pred_meta = np.round(np.median(y_pred_meta, axis=0))
                
            elif self.combine_meta_probas_method == 'class_min':
                y_pred_meta = y_pred_meta.min(axis=0)
                
            elif self.combine_meta_probas_method == 'class_max':
                y_pred_meta = y_pred_meta.max(axis=0)
            
            comb = np.zeros((n_obs,self.n_classes_))
            inds = np.array([np.where(self.classes_==cls) for cls in y_pred_meta]).ravel()
            comb[np.arange(n_obs),inds] = 1
        return(comb)   
        
    def _fit_lvl0_clfs(self, x_train, y_train):
        if self.stratified:
            kf = StratifiedKFold(y_train, n_folds=self.n_blend_folds, shuffle=True, random_state=self.seed)
        else:
            kf = KFold(n_folds=self.n_blend_folds, shuffle=True, random_state=self.seed)
        self.kf_ = kf
        
        blend_train = np.zeros((self.n_clfs_,self.n_train_,self.n_classes_))
        
        if self.must_compute_scores_:
            self.scores_ = {}
        
        for i in range(self.n_clfs_):
            
            if self.verbose > 0:
                print('Fitting level 0 classifier %d/%d: %s' % (i+1,self.n_clfs_,self.clf_names_[i]))
            
            for j,(tr_ind,te_ind) in enumerate(self.kf_):
                x_train_fold, x_test_fold, y_train_fold = x_train[tr_ind], x_train[te_ind], y_train[tr_ind]
                self.clfs_cv_[i][j].fit(x_train_fold, y_train_fold)
                y_pred_fold = self._get_proba(self.clfs_cv_[i][j], x_test_fold)
                blend_train[i,te_ind,:] = y_pred_fold
                
                if self.must_compute_scores_:
                    y_test_fold = y_train[te_ind]
                    y_pred_max_proba = [self.classes_[e] for e in np.argmax(y_pred_fold, axis=1)]
                    score = self._get_score(y_test_fold, y_pred_max_proba)
                    if j == 0:
                        self.scores_[i] = []
                    self.scores_[i].append(score)
                    
                    if self.verbose > 1:
                        print('-- Fold %d/%d score: %.6f' % (j+1,self.n_blend_folds,round(score,6)))
                        if j == self.n_blend_folds-1:
                            avg_score = np.mean(self.scores_[i])
                            print('-- Average score: %.6f' % round(avg_score,6))
                       
        blend_train = self._combine_probas(blend_train)
        
        if self.save_blend_sets is not None:
            self._save_set(blend_train,'_blend_train')
        
        if self.stack_original_features:
            blend_train = np.column_stack((x_train,blend_train))
        
        return(blend_train)
        
    def _fit_meta_clfs(self, blend_train, y_train):
        for i in range(self.n_meta_clfs_):
            if self.verbose > 0:
                print('Fitting meta classifier %d/%d: %s' % (i+1,self.n_meta_clfs_,self.meta_clf_names_[i]))
            self.meta_clfs[i].fit(blend_train, y_train)
        
    def _predict_lvl0(self, x_test): # Level 0
        
        blend_test = np.zeros((self.n_clfs_,self.n_blend_folds,x_test.shape[0],self.n_classes_))
        
        for i in range(self.n_clfs_):
            for j in range(self.n_blend_folds):
                blend_test[i,j,:,:] = self._get_proba(self.clfs_cv_[i][j], x_test)
                
        blend_test = self._combine_folds(blend_test)
        blend_test = self._combine_probas(blend_test)
        
        if self.save_blend_sets is not None:
            self._save_set(blend_test,'_blend_test')
        
        if self.stack_original_features:
            blend_test = np.column_stack((x_test,blend_test))
            
        return(blend_test)
    
    def _predict_meta(self, x_test):
        blend_test = self._predict_lvl0(x_test)
        
        y_pred = np.zeros((self.n_meta_clfs_,x_test.shape[0],self.n_classes_))
        for i in range(self.n_meta_clfs_):
            if self.verbose > 0:
                print('Training meta classifier %d/%d: %s' % (i+1,self.n_meta_clfs_,self.meta_clf_names_[i]))
            y_pred[i,:,:] = self._get_proba(self.meta_clfs[i], blend_test)
        
        y_pred = self._combine_meta_probas(y_pred)
        
        if self.save_blend_sets is not None:
            self._save_set(y_pred,'_blend_pred_raw')
            
        return(y_pred)
    
    def _fit(self, x_train, y_train):
        self._set_clf_names()
        self._copy_lvl0_clfs()
        self.must_compute_scores_ = self._must_compute_score()
        self.classes_ = np.sort(np.unique(y_train))
        self.n_classes_ = len(self.classes_)
        self.n_clfs_ = len(self.clfs)
        self.n_meta_clfs_ = len(self.meta_clfs)
        self.n_train_ = x_train.shape[0]
    
    def fit(self, x_train, y_train):
        self._fit(x_train,y_train)
        blend_train = self._fit_lvl0_clfs(x_train,y_train)
        self._fit_meta_clfs(blend_train,y_train)
        return(self)
        
    def predict_proba(self, x_test): # Meta probability output
        y_pred = self._predict_meta(x_test)
        
        if self.save_blend_sets is not None:
            self._save_set(y_pred,'_blend_pred')
            
        return(y_pred)
    
    def predict(self, x_test): # Meta class output
        y_pred_inds = self._predict_meta(x_test).argmax(axis=1)
        y_pred = self.classes_[y_pred_inds]
        
        if self.save_blend_sets is not None:
            self._save_set(y_pred,'_blend_pred')
        return(y_pred)
            
