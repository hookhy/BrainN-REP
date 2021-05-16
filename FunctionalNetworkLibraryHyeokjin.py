#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pandas as pd
import numpy as np
from scipy.linalg import inv, sqrtm, logm

def make_sparse_matrix(mat,thr=90,mode='prop'):
    num_nodes = np.shape(mat)[-1]
    posmat = (mat >=  0) * mat
    
    if mode=='value':
        thr = np.percentile(posmat.reshape(num_nodes*num_nodes,), thr)
        sparse_mat = (mat >=  thr) * mat

    elif mode=='prop':
        thr_per = 50
        while True:
            thr = np.percentile(posmat.reshape(num_nodes*num_nodes,), thr_per)
            tmp = (posmat >=  thr) * posmat
            numsurv = len( np.where( tmp > 0 )[0] )
            if numsurv <= (num_nodes**2)*(1- thr/100 ) or thr_per >= 99 : 
                break
            else:
                thr_per += 1
        sparse_mat = tmp 
        
    return sparse_mat

def tangentCalc(ref_mat,X,mode='mat2mat'):
    if mode == 'mat2mat':
        mat = X
        factor = inv( sqrtm(ref_mat) )
        tmp = np.matmul(factor,mat)
        tmp = np.matmul(tmp,factor)
        tan_vec = logm(tmp)

        return tan_vec

    elif mode == 'mat2vec':
        mat = X
        factor = inv( sqrtm(ref_mat) )
        tmp = np.matmul(factor,mat)
        tmp = np.matmul(tmp,factor)
        tan_vec = logm(tmp)

        num_nodes = np.shape(ref_mat)[0]
        tan_vec_vec = tan_vec[np.triu_indices(num_nodes)]
        return tan_vec_vec

    elif mode == 'vec2mat':

        mat = triu_transition(X,mode='vec2mat')
        factor = inv( sqrtm(ref_mat) )
        tmp = np.matmul(factor,mat)
        tmp = np.matmul(tmp,factor)
        tan_vec = logm(tmp)

        return tan_vec

    elif mode == 'vec2vec':

        mat = triu_transition(X,mode='vec2mat')
        factor = inv( sqrtm(ref_mat) )
        tmp = np.matmul(factor,mat)
        tmp = np.matmul(tmp,factor)
        tan_vec = logm(tmp)

        num_nodes = np.shape(ref_mat)[0]
        tan_vec_vec = tan_vec[np.triu_indices(num_nodes)]
        return tan_vec_vec

def triu_transition(input_ndarray,mode='vec2mat'):
    if mode == 'vec2mat':
        num_triuel = len(input_ndarray)
        num_nodes = int(( -1 + np.sqrt(1+8*num_triuel) ) / 2)
        mat_ret_upper = np.zeros((num_nodes,num_nodes))
        mat_ret_upper[np.triu_indices(num_nodes)] = input_ndarray
        mat_ret = np.maximum( mat_ret_upper, mat_ret_upper.transpose() )

        return mat_ret
    elif mode == 'mat2vec':
        num_nodes = np.shape(input_ndarray)[-1]
        vec_trans = input_ndarray[np.triu_indices(num_nodes)]
        return vec_trans

def set_network_params(parcel, fc_mode, ref_mode):
    # out2 : parcel
    if parcel == 'SHEN':
        n_features = 268
        chr_parcel = 'S'
    elif parcel == 'AAL':
        n_features = 94
        chr_parcel = 'A'
    elif parcel == 'HCP':
        n_features = 374
        chr_parcel = 'H'
    elif parcel == 'FIND':
        n_features = 90
        chr_parcel = 'F'
    else: print('Unknown option argument : parcel')
    
    # out3 : edge
    if fc_mode == 'Tikhonov':
        chr_fc = 'T'
    elif fc_mode == 'l1_GGM':
        chr_fc = 'G'
    elif fc_mode == 'pearson':
        chr_fc = 'P'
    elif fc_mode == 'LW':
        chr_fc = 'L'
    else: print('Unknown option argument : fc_mode')
    
    # out4 : ref_mode
    if ref_mode == 'mean':
        chr_ref = 'M'
    elif ref_mode == 'SVD':
        chr_ref = 'S'
    else: print('Unknown option argument : ref_mode')
        
    #
    chr_netmat = chr_parcel + chr_ref + chr_fc
    return n_features, chr_netmat

def get_netmat(mat_path,demo,subj_key,phenotype,parcel = 'AAL',fc_mode = 'pearson',ref_mode = 'mean'):
    #
    n_features, chr_netmat = set_network_params(parcel=parcel, fc_mode=fc_mode, ref_mode=ref_mode)
    # load the demo 
    demo_fil = demo[np.isfinite(demo[phenotype])]
    subj_list = demo_fil[subj_key].to_numpy()
    n_subjects = len(subj_list)

    # dataset  - x  
    all_mats = np.zeros((n_subjects,n_features,n_features))
    for i in range(n_subjects):
        s = str(subj_list[i])
        if fc_mode == 'Tikhonov': 
            all_mats[i,:,:] = pd.read_table(mat_path + '/' + s + '_' + chr_netmat + '.txt',header=None,sep=',').to_numpy()
        else:
            all_mats[i,:,:] = np.load(mat_path + '/' + s + '_' + chr_netmat + '.npy')
            
    # dataset - y
    y = demo_fil[phenotype].to_numpy()
    
    return all_mats, y

def standardize_netmat(mats,mode='minmax'):
    cnt = 0
    n_features = np.shape(mats)[-1]
    n_subjects = np.shape(mats)[0]
    for r in range(n_features):
        for c in range(n_features):
            if r > c:
                feavec = mats[:,r,c]
                m = np.mean(feavec)
                s = np.std(feavec)
                feavec_z = (feavec-m)/s
                mats[:,r,c] = feavec_z
                mats[:,c,r] = feavec_z
            elif r == c:
                mats[:,r,c] = 1
    if mode is not None:
        for i in range(n_subjects):
            mats[i,:,:] = normalize_netmat(mats[i,:,:],mode=mode)
    return mats

def normalize_netmat(netmat,mode='z'):
    n = np.shape(netmat)[0]
    tri_upper_diag_vec  = netmat[np.triu_indices(n)]
    
    if mode == 'minmax':
        maxi = np.max(tri_upper_diag_vec)
        mini = np.min(tri_upper_diag_vec)
        norm_netmat = (netmat - mini) / (maxi-mini)
        for r in range(n):
            for c in range(n):
                if r == c : norm_netmat[r,c] = 1
                    
    elif mode == 'z':
        ss = np.std(tri_upper_diag_vec)
        mm = np.mean(tri_upper_diag_vec)
        if ss != 0:
            norm_netmat = (netmat - mm) / ss
            for r in range(n):
                for c in range(n):
                    if r == c : norm_netmat[r,c] = 1
        else: print('Matrix has 0 std..Exit')
    
    else: print('Unknown option argument : mode')
    return norm_netmat


# In[ ]:




