#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.linear_model import LogisticRegressionCV , ElasticNetCV, LinearRegression , LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import scipy.stats as ss
import statsmodels.stats.multitest as mt
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import ast
import pandas as pd
import time
from scipy import stats, linalg
from pycombat import Combat

def runCombat(X,batch,covs):
    model = Combat()
    X_combat = model.fit_transform(Y=X,b=batch,C=covs)
    
    return X_combat

def makeFakeNets(num_subjects,num_nodes,num_Covs,sparsity=0.95,withCov=True,vectorization=True):
    X = np.zeros((num_subjects,num_nodes,num_nodes))
    X_vec = np.zeros((num_subjects, int((1+num_nodes)*num_nodes/2)))
    np.random.seed(550) 
    y = np.random.randn(num_subjects)
    
    if withCov:
        covs = np.zeros((num_subjects,num_Covs))
        for c in range(num_Covs):
            np.random.seed(c)
            covs[:,c] = np.random.randn(num_subjects)
            
    for i in range(num_subjects):
        tmp = make_sparse_spd_matrix(dim=num_nodes, alpha=sparsity,random_state=i)
        if vectorization:
            X_vec[i,:] = tmp[np.triu_indices(num_nodes)]
        elif ~vectorization:
            X[i,:,:] = tmp
    
    if withCov and vectorization:
        return X_vec, y,covs
    elif withCov and ~vectorization:
        return X, y, covs
    elif ~withCov and vectorization:
        return X_vec, y
    elif ~withCov and ~vectorization:
        return X, y

def decomf_covariates(X_vec,y,mode,covs):  
    if mode == 'X1y1':
        X1 = np.zeros((np.shape(X_vec)[0],np.shape(X_vec)[1]))
        y1 = np.zeros((np.shape(y)[0],1))
        for column in range(np.shape(X_vec)[1]): 
            betaX = linalg.lstsq(covs, X_vec[:,column])[0]
            tmp = X_vec[:,column] - covs.dot(betaX)
            X1[:,column] = tmp.reshape(np.shape(X_vec)[0],)

        betay = linalg.lstsq(covs,y)
        y1 = y - covs.dot(betay[0])
        
        return X1,y1
    
    elif mode == 'X1y0':
        X1 = np.zeros((np.shape(X_vec)[0],np.shape(X_vec)[1]))
        for column in range(np.shape(X_vec)[1]): 
            betaX = linalg.lstsq(covs, X_vec[:,column])[0]
            tmp = X_vec[:,column] - covs.dot(betaX)
            X1[:,column] = tmp.reshape(np.shape(X_vec)[0],)
        return X1, y
    
    elif mode == 'X0y1':
        y1 = np.zeros((np.shape(y)[0],1))
        betay = linalg.lstsq(covs,y)
        y1 = y - covs.dot(betay[0])
        return X_vec, y1
    
    elif mode == 'X0y0':
        return X_vec, y
    
def enlr(X,y,num_folds=10,random_state=1,n_jobs=1,verbose=True,mode='classification',standardization=True):
    #
    num_subj = len(y)
    num_features = np.shape(X)[1]
    num_folds = num_folds
    pos = np.max(y)
    neg = np.min(y)
    
    #
    Css = ([ .01,.05,.1, .3, .5, .7, .9,  .99])
    L1s = ([.1, .5, .7, .9, .95, .99, 1])

    # Normalization
    if standardization:
        sample_x_zscore = np.zeros((num_subj,num_features))
        for i in range(num_features):
            vec = X[:,i]
            m = np.mean(vec)
            s = np.std(vec)
            sample_x_zscore[:,i] = (vec - m) / s
    else:
        sample_x_zscore = X

    # 10 fold crossvalidation
    if mode == 'classification':
        y = np.where(y > (pos+neg)/2,1,0)
        kf = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=random_state)
        fold=0
        acc,sen,ppv,f1,spe = np.zeros((num_folds,)),np.zeros((num_folds,)),np.zeros((num_folds,)),np.zeros((num_folds,)),np.zeros((num_folds,))

        cnt=0
        coefs =np.zeros((num_folds,num_features))
        for train_idx, test_idx in kf.split(sample_x_zscore,y):
            start = time.time()
            fold = fold + 1
            tp, fp, tn, fn = 0, 0, 0, 0
            x_train ,x_test = sample_x_zscore[train_idx,:], sample_x_zscore[test_idx,:]
            y_train ,y_test = y[train_idx], y[test_idx] 
            x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

            regr = LogisticRegressionCV(cv=10,
                                        random_state=random_state,
                                        penalty='elasticnet',
                                        solver='saga',
                                        l1_ratios =L1s,
                                        Cs = Css,
                                        max_iter=int(1e5),tol=1e-5,
                                        n_jobs=n_jobs
                                        )
            regr.fit(x_train,y_train)
            y_pred = regr.predict(x_test)

            acc[cnt] = accuracy_score(y_test,y_pred)
            sen[cnt] = recall_score(y_test,y_pred)
            ppv[cnt] = precision_score(y_test,y_pred)
            f1[cnt] = f1_score(y_test,y_pred)
            for i in range(np.shape(y_test)[0]):
                if y_test[i] == 1 and y_pred[i] == 1: tp = tp +1
                if y_test[i] == 1 and y_pred[i] == 0: fn = fn +1
                if y_test[i] == 0 and y_pred[i] == 1: fp = fp +1
                if y_test[i] == 0 and y_pred[i] == 0: tn = tn +1
            spe[cnt] = (tn) / (tn + fp)

            coefs[cnt,:] = regr.coef_[0]

            if verbose:
                print(y_test)
                print(y_pred)
                print(coefs[cnt,:])
                print('Fold '+ str(fold) + ' accuracy = ' + str(acc[cnt]))
                print("Spent time ... ", time.time() - start)

            cnt = cnt + 1

        print('')
        print('##===================result======================##')
        print('mean accuracy : ' + str(np.mean(acc)))
        print('mean sensitivity : ' + str(np.mean(sen)))
        print('mean specificity : ' + str(np.mean(spe)))
        print('mean f1 score : ' + str(np.mean(f1)))
        print('mean positive predicted value : ' + str(np.mean(ppv)))
        print(acc)
        print('##===============================================##')
        model_eval = np.zeros((5,))
        model_eval[0] = np.mean(acc)
        model_eval[1] = np.mean(sen)
        model_eval[2] = np.mean(spe)
        model_eval[3] = np.mean(f1)
        model_eval[4] = np.mean(ppv)

        return np.mean(coefs,axis=0), model_eval
    
    elif mode == 'regression':
        percentiles = np.percentile(y, [0,25,50,75,100])
        inds = np.digitize(y,percentiles)
        kf = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=random_state)
        fold=0
        mae, r, p = np.zeros((num_folds,)),np.zeros((num_folds,)),np.zeros((num_folds,))

        cnt=0
        coefs =np.zeros((num_folds,num_features))
        for train_idx, test_idx in kf.split(sample_x_zscore,inds):
            start = time.time()
            fold = fold + 1
            x_train ,x_test = sample_x_zscore[train_idx,:], sample_x_zscore[test_idx,:]
            y_train ,y_test = y[train_idx], y[test_idx] 
            x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

            regr = ElasticNetCV(cv=10,
                                random_state=random_state,
                                l1_ratio =L1s,
                                alphas = Css,
                                max_iter=int(1e5),tol=1e-5,
                                n_jobs=n_jobs
                                )
            regr.fit(x_train,y_train)
            y_pred = regr.predict(x_test)

            mae[cnt] = mean_squared_error(y_test,y_pred)
            r[cnt] = pearsonr(y_test,y_pred)[0]
            p[cnt] = pearsonr(y_test,y_pred)[1]
            coefs[cnt,:] = regr.coef_[0]

            if verbose:
                print(y_test)
                print(y_pred)
                print(coefs[cnt,:])
                print('Fold '+ str(fold) + ' MAE = ' + str(mae[cnt]))
                print('Fold '+ str(fold) + ' Pearson r = ' + str(r[cnt]))
                print("Spent time ... ", time.time() - start)

            cnt = cnt + 1

        print('')
        print('##===================result======================##')
        print('mean MAE : ' + str(np.mean(mae)))
        print('Pearson correlation coefficients : ', end='')
        print(r)
        print('Statistical significance : ', end='')
        print(p)

        print('##===============================================##')
        model_eval = np.zeros((3,))
        model_eval[0] = np.mean(mae)
        model_eval[1] = p
        model_eval[2] = r

        return np.mean(coefs,axis=0), model_eval


