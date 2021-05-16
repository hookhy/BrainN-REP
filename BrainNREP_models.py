#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python

import os, sys
import math
import pandas as pd
import smogn
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.metrics import mean_absolute_error as mae
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data.dataset
from scipy.stats import pearsonr
from scipy.stats import rankdata
import copy

from MachineLearningLibraryHyeokjin import *
from FunctionalNetworkLibraryHyeokjin import *

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes,example,bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes,planes,(1,self.d),bias=bias)
        torch.nn.init.kaiming_normal_(self.cnn1.weight)
        self.cnn2 = torch.nn.Conv2d(in_planes,planes,(self.d,1),bias=bias)
        torch.nn.init.kaiming_normal_(self.cnn2.weight)
        
    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d,3)+torch.cat([b]*self.d,2)

# Normalizing the adjacency matrix by a degree matrix
def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

# GCN model for batch graph data
# H1 = act( norm(G)*H0*W )
class batchedGCN(torch.nn.Module):
    def __init__(self, in_planes, planes):
        super(batchedGCN,self).__init__()
        # W
        self.proj = torch.nn.Linear(in_planes, planes)
        
    def forward(self,h,g):
        # h&g - batch x channel x node x node 
        h_ones, g_ones = torch.split(h,1,dim=0),torch.split(g,1,dim=0)
        o_hs = []
        for batch in range(len(h_ones)):
            # h_one&g_one - node x node
            h_one, g_one = h_ones[batch].squeeze(0).squeeze(0),g_ones[batch].squeeze(0).squeeze(0)
            
            # preprocessing : g_one
            g_one = (g_one > 0).float() # binarization
            g = norm_g(g_one) # normalization
            
            # calculate : norm(G)*H0*W
            h = torch.matmul(g,h_one)
            h = self.proj(h)
            o_hs.append(h)
            
        hs = torch.stack(o_hs,0).unsqueeze(1) # stack along the batch & add the channel
        
        return hs

# top-k pooling using the attention score
class top_k_pool(torch.nn.Module):
    def __init__(self, k):
        super(top_k_pool,self).__init__()
        # Nodes which have an upper k score will be survived
        self.k = k
        
    def forward(self,h,g,scores):
        # h&g - batch x channel x node x node 
        # score - batch x 1 x node x 1
        h_ones, g_ones, score_ones = torch.split(h,1,dim=0),torch.split(g,1,dim=0),torch.split(scores,1,dim=0)
        h_pool,g_pool,score_pool = [],[],[]
        for batch in range(len(h_ones)):
            # h_one&g_one - node x node
            h_one, g_one = h_ones[batch].squeeze(0).squeeze(0),g_ones[batch].squeeze(0).squeeze(0)
            
            # score - node x 1
            score = score_ones[batch].squeeze(0).squeeze(0)
            values, idx = torch.topk(score.squeeze(),self.k,dim=0)
            
            # top-k selection
            new_h = h_one[idx,:] # k x channel
            new_g = g_one[idx,:] 
            new_g = new_g[:,idx] # k x k
            new_score = score[idx] # k x 1
            
            h_pool.append(new_h)
            g_pool.append(new_g)
            score_pool.append(new_score)
        
        # stack along the batch & add the channel
        hs, gs = torch.stack(h_pool,0).unsqueeze(1), torch.stack(g_pool,0).unsqueeze(1)
        ss = torch.stack(score_pool,0).unsqueeze(1)
        
        return hs, gs, ss
    
def ReadOut(h):
    # input h : batch x channel x node x 1
    # maxpooling : batch x channel x 1
    max_out, _ = torch.max(h,dim=2)
    # meanpooling : batch x channel x 1
    mean_out = torch.mean(h,dim=2)
    # concatenate : batch x 2*channel x 1
    new_h = torch.cat((max_out,mean_out),dim=1)
    
    return new_h
    
#
class GaGCN_identity(torch.nn.Module):
    def __init__(self, example):
        super(GaGCN_identity, self).__init__()
        # flexible shape
        self.in_planes = example.size(1)
        self.d = example.size(3)
        
        # 
        self.lReLu = torch.nn.LeakyReLU(0.33)
        self.ReLu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.sigmoid = torch.nn.Sigmoid()
        
        # init : feature extractor
        self.e2econv1 = E2EBlock(1,128,example,bias=True)
        self.e2econv2 = E2EBlock(128,256,example,bias=True)
        self.E2N = torch.nn.Conv2d(256,8,(1,self.d))
        torch.nn.init.kaiming_normal_(self.E2N.weight)
        
        # init : attention module
        self.attention11 = batchedGCN(8,1)
        self.attention12 = batchedGCN(1,1)
        
        self.gcn1 = batchedGCN(8,8)
        self.attention21 = batchedGCN(8,1)
        self.attention22 = batchedGCN(1,1)
        
        # top-k pooling layer
        self.topkpool1 = top_k_pool(int(self.d/2))
        self.topkpool2 = top_k_pool(int(self.d/4))
        
        # init : predictor (FCN)
        self.dense1 = torch.nn.Linear(16,128)
        self.dense2 = torch.nn.Linear(128,64)
        self.dense3 = torch.nn.Linear(64,1)
        
    def forward(self, x):
        # feature extractor : E2E
        out = self.lReLu(self.e2econv1(x))
        out = self.lReLu(self.e2econv2(out))
        out = self.lReLu(self.E2N(out))
        
        # Node attention score
        weight = self.lReLu(self.attention11(out.permute(0,3,2,1),x))
        weight = self.sigmoid(self.attention12(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool1(out.permute(0,3,2,1),x,weight)
        out = out.permute(0,3,2,1)
        out = out*weight + out
        sum1 = ReadOut(out)
        
        # gcn layer1
        out = self.gcn1(out.permute(0,3,2,1),x).permute(0,3,2,1)
        
        # Node attention score
        weight = self.lReLu(self.attention21(out.permute(0,3,2,1),x))
        weight = self.sigmoid(self.attention22(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool2(out.permute(0,3,2,1),x,weight)
        out = out.permute(0,3,2,1)
        out = out*weight + out
        sum2 = ReadOut(out)
        
        # summary the readout results
        out = torch.add(sum1,sum2)
        
        # predictor
        out = out.view(out.size(0), -1)
        out = self.dropout(self.lReLu(self.dense1(out)))
        out = self.dropout(self.lReLu(self.dense2(out)))
        out = self.dense3(out)
        
        return out
    
# Update the adjacency matrix
class embedding_adj(torch.nn.Module):
    def __init__(self,in_plane,example,device,mode):
        super(embedding_adj,self).__init__()
        assert mode in ["e2e", "avg"]
        self.e2econv = E2EBlock(in_plane,1,example,bias=True)
        self.Sigmoid = torch.nn.Sigmoid()
        self.d = example.size(3)
        self.mode=mode
        
    def forward(self,g):
        if self.mode == 'e2e': # A single E2E layer will be used for new adj
            new_g = self.Sigmoid(self.e2econv(g)) # 0~1
            new_g_t = new_g.permute(0,1,3,2) # transpose

            g_sym = torch.add(new_g,new_g_t) / 2 # for symmetricity
            # adding self connections
            imat = torch.eye(self.d).reshape((1,1, self.d, self.d)).repeat(new_g.size(0),1,1,1).to(device)
            g_sym_sc = torch.add(g_sym,imat)
            
            # binarization
            g_sym_bin = (g_sym_sc >= 0.5).float()

        elif self.mode == 'avg': # simple averaging the channel embeddings
            new_g = self.Sigmoid(torch.mean(g,1)).unsqueeze(1)
            new_g_t = new_g.permute(0,1,3,2) # transpose

            g_sym = torch.add(new_g,new_g_t) / 2 # for symmetricity
            # adding self connections
            imat = torch.eye(self.d).reshape((1,1, self.d, self.d)).repeat(new_g.size(0),1,1,1).to(device)
            g_sym_sc = torch.add(g_sym,imat)
            
            # binarization
            g_sym_bin = (g_sym_sc >= 0.5).float()
            
        return g_sym_bin
            

# 
class GaGCN_avg(torch.nn.Module):
    def __init__(self, example,device):
        super(GaGCN_avg, self).__init__()
        # flexible shape
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.device = device
        # 
        self.lReLu = torch.nn.LeakyReLU(0.33)
        self.ReLu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.sigmoid = torch.nn.Sigmoid()
        
        # init : feature extractor
        self.e2econv1 = E2EBlock(1,128,example,bias=True)
        self.e2econv2 = E2EBlock(128,256,example,bias=True)
        self.embedding = embedding_adj(256,example,self.device,mode='avg')
        self.E2N = torch.nn.Conv2d(256,8,(1,self.d))
        torch.nn.init.kaiming_normal_(self.E2N.weight)
        
        # init : attention module
        self.attention11 = batchedGCN(8,1)
        self.attention12 = batchedGCN(1,1)
        
        self.gcn1 = batchedGCN(8,8)
        self.attention21 = batchedGCN(8,1)
        self.attention22 = batchedGCN(1,1)
        
        # top-k pooling layer
        self.topkpool1 = top_k_pool(int(self.d/2))
        self.topkpool2 = top_k_pool(int(self.d/4))
        
        # init : predictor (FCN)
        self.dense1 = torch.nn.Linear(16,128)
        self.dense2 = torch.nn.Linear(128,64)
        self.dense3 = torch.nn.Linear(64,1)
        
    def forward(self, x):
        # feature extractor : E2E
        out = self.lReLu(self.e2econv1(x))
        out = self.lReLu(self.e2econv2(out)) 
        x = self.embedding(out)
        out = self.lReLu(self.E2N(out))
        
        # Node attention score
        weight = self.lReLu(self.attention11(out.permute(0,3,2,1),x))
        weight = self.sigmoid(self.attention12(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool1(out.permute(0,3,2,1),x,weight)
        out = out.permute(0,3,2,1)
        out = out*weight + out
        sum1 = ReadOut(out)
        
        # gcn layer1
        out = self.gcn1(out.permute(0,3,2,1),x).permute(0,3,2,1)
        
        # Node attention score
        weight = self.lReLu(self.attention21(out.permute(0,3,2,1),x))
        weight = self.sigmoid(self.attention22(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool2(out.permute(0,3,2,1),x,weight)
        out = out.permute(0,3,2,1)
        out = out*weight + out
        sum2 = ReadOut(out)
        
        # summary the readout results
        out = torch.add(sum1,sum2)
        
        # predictor
        out = out.view(out.size(0), -1)
        out = self.dropout(self.lReLu(self.dense1(out)))
        out = self.dropout(self.lReLu(self.dense2(out)))
        out = self.dense3(out)
        
        return out
    
# 
class GaGCN_e2e(torch.nn.Module):
    def __init__(self, example,device):
        super(GaGCN_e2e, self).__init__()
        # flexible shape
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.device = device
        # 
        self.lReLu = torch.nn.LeakyReLU(0.33)
        self.ReLu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.sigmoid = torch.nn.Sigmoid()
        
        # init : feature extractor
        self.e2econv1 = E2EBlock(1,128,example,bias=True)
        self.e2econv2 = E2EBlock(128,256,example,bias=True)
        self.embedding = embedding_adj(256,example,self.device,mode='e2e')
        self.E2N = torch.nn.Conv2d(256,8,(1,self.d))
        torch.nn.init.kaiming_normal_(self.E2N.weight)
        
        # init : attention module
        self.attention11 = batchedGCN(8,1)
        self.attention12 = batchedGCN(1,1)
        
        self.gcn1 = batchedGCN(8,8)
        self.attention21 = batchedGCN(8,1)
        self.attention22 = batchedGCN(1,1)
        
        # top-k pooling layer
        self.topkpool1 = top_k_pool(int(self.d/2))
        self.topkpool2 = top_k_pool(int(self.d/4))
        
        # init : predictor (FCN)
        self.dense1 = torch.nn.Linear(16,128)
        self.dense2 = torch.nn.Linear(128,64)
        self.dense3 = torch.nn.Linear(64,1)
        
    def forward(self, x):
        # feature extractor : E2E
        out = self.lReLu(self.e2econv1(x))
        out = self.lReLu(self.e2econv2(out)) 
        x = self.embedding(out)
        out = self.lReLu(self.E2N(out))
        
        # Node attention score
        weight = self.lReLu(self.attention11(out.permute(0,3,2,1),x))
        weight = self.sigmoid(self.attention12(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool1(out.permute(0,3,2,1),x,weight)
        out = out.permute(0,3,2,1)
        out = out*weight + out
        sum1 = ReadOut(out)
        
        # gcn layer1
        out = self.gcn1(out.permute(0,3,2,1),x).permute(0,3,2,1)
        
        # Node attention score
        weight = self.lReLu(self.attention21(out.permute(0,3,2,1),x))
        weight = self.sigmoid(self.attention22(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool2(out.permute(0,3,2,1),x,weight)
        out = out.permute(0,3,2,1)
        out = out*weight + out
        sum2 = ReadOut(out)
        
        # summary the readout results
        out = torch.add(sum1,sum2)
        
        # predictor
        out = out.view(out.size(0), -1)
        out = self.dropout(self.lReLu(self.dense1(out)))
        out = self.dropout(self.lReLu(self.dense2(out)))
        out = self.dense3(out)
        
        return out

def init_weights_he(m):
    #https://keras.io/initializers/#he_uniform
    print(m)
    if type(m) == torch.nn.Linear:
        fan_in = net.dense1.in_features
        he_lim = np.sqrt(6) / fan_in
        m.weight.data.uniform_(-he_lim,he_lim)
        
def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def context_prediction_data(X, limit_search_depth=2, r1=1, r2=4, max_neighbour=10):
    pos_subgraphs, pos_congraphs = [], []
    anchor_nodelist =  []
    pos_nodelist = []
    num_sub = np.shape(X)[0]
    num_nodes = np.shape(X)[-1]

    # subject-wise Breath First Search
    for sub in range(num_sub):
        in_mat = X[sub,:,:]
        # for every node,
        for start in range(num_nodes):
            center_node_idx = np.array([start])
            sub_list, con_list = [], []
            # search depth for subgraph
            for depth in range(limit_search_depth):
                for n in range(len(center_node_idx)):
                    node_idx = center_node_idx[n]
                    for i in range(num_nodes):
                        if in_mat[node_idx,i] > 0.5:
                            sub_list.append(i)
                    center_node_idx = sub_list
                if depth==r1-1: first_neihbours = list(np.unique(center_node_idx))
            sub_list = list(np.unique(sub_list)) # get subgraph list

            # search depth for r2 (for outer boundary of context graph)
            center_node_idx = np.array([start])
            for depth in range(r2):
                for n in range(len(center_node_idx)):
                    node_idx = center_node_idx[n]
                    for i in range(num_nodes):
                        if in_mat[node_idx,i] > 0.5:
                            con_list.append(i)
                    center_node_idx = con_list
            con_list = list(np.unique(con_list)) # get subgraph list
            # get context graph list & anchor node list
            anchor_nodelist_tmp = sub_list
            for first in range(len(first_neihbours)):
                con_list.remove(first_neihbours[first])
                anchor_nodelist_tmp.remove(first_neihbours[first])

            # if the present node has an apropriate number of neightbours, 
            if len(sub_list) <= max_neighbour and len(sub_list) >= 3:
                # make subgraph adjacency matrix by masking
                sub_graph, con_graph = in_mat, in_mat
                submask, conmask = np.zeros_like(in_mat), np.zeros_like(in_mat)
                submask[sub_list,:], conmask[con_list,:] = 1, 1
                submask[:,sub_list], conmask[:,con_list] = 1, 1
                sub_graph, con_graph = np.multiply(sub_graph, submask), np.multiply(con_graph, conmask)
                # stack the subgraphs
                pos_subgraphs.append(sub_graph)
                pos_congraphs.append(con_graph)

                # memorize the index of the samples
                tmp_vec_con = np.zeros((1,num_nodes))
                for i in range(len(anchor_nodelist_tmp)): 
                    tmp_vec_con[0,anchor_nodelist_tmp[i]] = 1
                pos_nodelist.append([sub,start]) # memory the subject number & node number
                anchor_nodelist.append(tmp_vec_con)
    return pos_subgraphs, pos_congraphs, pos_nodelist, anchor_nodelist