"""
StackingClassifier

Copyright 2016
Jesse Myrberg (jesse.myrberg@aalto.fi)
https://github.com/jmyrberg/KaggleClassifiers/
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
    """Stacked generalization classifier.
    
    The basic idea behind stacked generalization is to use a pool of
    base classifiers (level 0 classifiers), and then combine their 
    predictions by using another set of classifiers, meta-classifiers, 
    with the aim of reducing the generalization error.
    """
    
    def __init__(self,clfs=[RandomForestClassifier(), ExtraTreesClassifier()],
                 meta_clfs=[LogisticRegression()], n_blend_folds=3, stratified=True,
                 stack_original_features=False, combine_fold_probas_method='fold_score',
                 combine_lvl0_probas_method='stacked', combine_meta_probas_method='mean',
                 weights=None, compute_scores=False, scoring=accuracy_score,
                 save_blend_sets=None, verbose=0, seed=None):
        """
        Parameter explanations currently at:
        https://github.com/jmyrberg/KaggleClassifiers/
        """
        self.clfs = clfs
        self.meta_clfs = meta_clfs
        self.n_blend_folds = n_blend_folds
        self.stratified = stratified
        self.stack_original_features = stack_original_features
        self.combine_fold_probas_method = combine_fold_probas_method
        self.combine_lvl0_probas_method = combine_lvl0_probas_method
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
        self.clfs_ = {}
        for i,clf in enumerate(self.clfs):
            self.clfs_[i] = []
            for _ in range(self.n_blend_folds):
                self.clfs_[i].append(copy(clf))
    
    def _save_array(self, array, name):
        np.save(self.save_blend_sets + '_' + name + '.npy', array)
        
    def _is_function(self,f):
        if hasattr(f, '__call__'):
            return(True)
        else:
            return(False)
        
    def _check_custom_functions(self):
        self.combine_fold_probas_custom_ = False
        self.combine_lvl0_probas_custom_ = False
        self.combine_meta_probas_custom_ = False
        if self._is_function(self.combine_fold_probas_method):
            self.combine_fold_probas_custom_ = True
        if self._is_function(self.combine_lvl0_probas_method):
            self.combine_lvl0_probas_custom_ = True
        if self._is_function(self.combine_meta_probas_method):
            self.combine_meta_probas_custom_ = True  
        
    def _must_compute_score(self):
        if self.compute_scores:
            return(True)
        if self.verbose > 1:
            return(True)
        if self.combine_fold_probas_method in ['fold_score']:
            return(True)
        if self.combine_lvl0_probas_method in ['fold_avg_score']:
            return(True)
        
    def _get_score(self, y_true, y_pred):
        score = self.scoring(y_true, y_pred)
        return(score)
    
    def _find_nearest_ind(self, array, value):
        ind = (np.abs(array-value)).argmin()
        return(ind)
    
    def _get_proba(self, clf, x, lvl):
        if hasattr(clf,'predict_proba'):
            y_pred = clf.predict_proba(x)
        else:
            y_pred = np.zeros((x.shape[0],self.n_classes_))
            y_pred_tmp = clf.predict(x)
            inds = np.array([self._find_nearest_ind(self.classes_,ind) for ind in y_pred_tmp])
            y_pred[np.arange(x.shape[0]),inds] = 1
        return(y_pred)
    
    def _combine_lvl0_probas(self, blend_set):
        # Input size: (n_clfs, n_obs, n_classes)
        if not self.combine_lvl0_probas_custom_:
        
            if self.combine_lvl0_probas_method == 'stacked':
                comb = blend_set.reshape((blend_set.shape[1],blend_set.shape[2]*blend_set.shape[0]))
            
            elif self.combine_lvl0_probas_method == 'mean':
                comb = blend_set.mean(axis=0)
                
            elif self.combine_lvl0_probas_method == 'median':
                comb = np.median(blend_set, axis=0)
                comb /= comb.sum(axis=1)[:,None]
            
            elif self.combine_lvl0_probas_method == 'weighted':
                w = np.array(self.weights['combine_probas'])
                w /= w.sum()
                comb = (blend_set*w[:,None,None]).sum(axis=0)
            
            elif self.combine_lvl0_probas_method == 'fold_avg_score':
                w = np.array([np.mean(e) for e in self.scores_.values()])
                w /= w.sum()
                comb = (blend_set*w[:,None,None]).sum(axis=0)
                
            elif self.combine_lvl0_probas_method == 'fold_geomavg_score':
                w = np.array([gmean(e) for e in self.scores_.values()])
                w /= w.sum()
                comb = (blend_set*w[:,None,None]).sum(axis=0)
                
            elif self.combine_lvl0_probas_method.startswith('fold_avg_pow_'):
                power = eval(self.combine_lvl0_probas_method.split('_')[-2])
                w = np.array([np.mean(e) for e in self.scores_.values()])
                w **= power
                w /= w.sum()
                comb = (blend_set*w[:,None,None]).sum(axis=0)
        else:
            comb = self.combine_lvl0_probas_method(blend_set)
        return(comb)
        
    def _combine_fold_probas(self, blend_test):
        # Input size: (n_clfs, n_cv, n_obs, n_classes)
        if not self.combine_fold_probas_custom_:
            if self.combine_fold_probas_method == 'mean':
                comb = blend_test.mean(axis=1)
            
            elif self.combine_fold_probas_method == 'median':
                comb = np.median(blend_test, axis=1)
                comb /= comb.sum(axis=2)[:,:,None]
            
            elif self.combine_fold_probas_method == 'fold_score':
                w = np.array([e for e in self.scores_.values()])
                w /= w.sum(axis=0)
                comb = (blend_test*w[:,:,None,None]).sum(axis=1)
                comb /= comb.sum(axis=2)[:,:,None]
        else:
            comb = self.combine_fold_probas_method(blend_test)
        return(comb)
    
    def _combine_meta_probas(self, y_pred_meta):
        # Input size: (n_clfs, n_obs, n_classes)
        if not self.combine_meta_probas_custom_:
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
        else:
            comb = self.combine_meta_probas_method(y_pred_meta)
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
                self.clfs_[i][j].fit(x_train_fold, y_train_fold)
                y_pred_fold = self._get_proba(self.clfs_[i][j], x_test_fold, 'lvl0')
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
                       
        blend_train = self._combine_lvl0_probas(blend_train)
        
        if self.save_blend_sets is not None:
            self._save_array(blend_train,'blend_train')
        
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
                blend_test[i,j,:,:] = self._get_proba(self.clfs_[i][j], x_test, 'fold')
                
        blend_test = self._combine_fold_probas(blend_test)
        blend_test = self._combine_lvl0_probas(blend_test)
        
        if self.save_blend_sets is not None:
            self._save_array(blend_test,'blend_test')
        
        if self.stack_original_features:
            blend_test = np.column_stack((x_test,blend_test))
            
        return(blend_test)
    
    def _predict_meta(self, x_test):
        blend_test = self._predict_lvl0(x_test)
        
        y_pred = np.zeros((self.n_meta_clfs_,x_test.shape[0],self.n_classes_))
        for i in range(self.n_meta_clfs_):
            if self.verbose > 0:
                print('Training meta classifier %d/%d: %s' % (i+1,self.n_meta_clfs_,self.meta_clf_names_[i]))
            y_pred[i,:,:] = self._get_proba(self.meta_clfs[i], blend_test, 'meta')
        
        y_pred = self._combine_meta_probas(y_pred)
        
        if self.save_blend_sets is not None:
            self._save_array(y_pred,'blend_pred_raw')
            
        return(y_pred)
    
    def _fit(self,x_train,y_train):
        self._set_clf_names()
        self._copy_lvl0_clfs()
        self._check_custom_functions()
        self.must_compute_scores_ = self._must_compute_score()
        self.classes_ = np.sort(np.unique(y_train))
        self.n_classes_ = len(self.classes_)
        self.n_clfs_ = len(self.clfs)
        self.n_meta_clfs_ = len(self.meta_clfs)
        self.n_train_ = x_train.shape[0]
    
    def fit(self,x_train,y_train):
        """Fit model to training data X with training targets y.
        """
        self._fit(x_train,y_train)
        blend_train = self._fit_lvl0_clfs(x_train,y_train)
        self._fit_meta_clfs(blend_train,y_train)
        return(self)
        
    def predict_proba(self,x_test): # Meta probability output
        """Predict class probabilities on samples in X.
        """
        y_pred = self._predict_meta(x_test)
        if self.save_blend_sets is not None:
            self._save_array(y_pred,'blend_pred')
        return(y_pred)
    
    def predict(self,x_test): # Meta class output
        """Perform classification on samples in X.
        """
        y_pred_inds = self._predict_meta(x_test).argmax(axis=1)
        y_pred = self.classes_[y_pred_inds]
        if self.save_blend_sets is not None:
            self._save_array(y_pred,'blend_pred')
        return(y_pred)
            
