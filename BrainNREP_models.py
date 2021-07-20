#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python

import os, sys
import math
import pandas as pd
#import smogn
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
from scipy.special import expit

from MachineLearningLibraryHyeokjin import *
from FunctionalNetworkLibraryHyeokjin import *

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes,example,bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes,planes,(1,self.d),bias=bias)
        torch.nn.init.xavier_uniform(self.cnn1.weight)
        self.cnn2 = torch.nn.Conv2d(in_planes,planes,(self.d,1),bias=bias)
        torch.nn.init.xavier_uniform(self.cnn2.weight)
        
    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d,3)+torch.cat([b]*self.d,2)

# Normalizing the adjacency matrix by a degree matrix
def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    g[g != g] = 0 # for zero degree : nan_to_zero
    return g

# GCN model for batch graph data
# H1 = act( norm(G)*H0*W )
class batchedGCN(torch.nn.Module):
    def __init__(self, in_planes, planes):
        super(batchedGCN,self).__init__()
        # W
        self.proj = torch.nn.Linear(in_planes, planes)
        torch.nn.init.xavier_uniform(self.proj.weight)
        
    def forward(self,h,g):
        # g - batch x channel x node x node 
        # h - batch x channel x node x 1
        h_ones, g_ones = torch.split(h,1,dim=0),torch.split(g,1,dim=0)
        o_hs = []
        for batch in range(len(h_ones)):
            # g_one - node x node
            # h_one - channel x node
            h_one, g_one = h_ones[batch].squeeze(0).squeeze(-1),g_ones[batch].squeeze(0).squeeze(0)
            
            # preprocessing : g_one
            g_one = (g_one > 0).float() # binarization
            g = norm_g(g_one) # normalization
            
            # calculate : norm(G)*H0*W
            h_embed = torch.matmul(g,h_one.permute(1,0)) # ( node x node ) x ( node x channel ) -> ( node x channel )
            h_embed = self.proj(h_embed) # ( node x channel_embed )
            o_hs.append(h_embed.permute(1,0)) # batch x channel_embed x node
            
        hs = torch.stack(o_hs,0).unsqueeze(-1) # stack along the batch & add the channel
        
        return hs # batch x channel_embed x node x 1
    
# GIN layer 
class batchedGIN(torch.nn.Module):
    def __init__(self, in_planes, planes):
        super(batchedGIN,self).__init__()
        # W
        self.proj = torch.nn.Linear(in_planes, planes)
        torch.nn.init.xavier_uniform(self.proj.weight)
        
    def forward(self,h,g):
        # g - batch x channel x node x node 
        # h - batch x channel x node x 1
        h_ones, g_ones = torch.split(h,1,dim=0),torch.split(g,1,dim=0)
        o_hs = []
        for batch in range(len(h_ones)):
            # g_one - node x node
            # h_one - node x channel
            h_one, g_one = h_ones[batch].squeeze(0).squeeze(-1).permute(1,0),g_ones[batch].squeeze(0).squeeze(0)
            
            # preprocessing : g_one
            g_one = (g_one > 0).float() # binarization
            
            # calculate 
            h_embed = self.proj(h_one + g_one @ h_one) # node x channel_embed
            o_hs.append(h_embed.permute(1,0)) # batch x channel_embed x node
            
        hs = torch.stack(o_hs,0).unsqueeze(-1) # stack along the batch & add the channel
        
        return hs # batch x channel_embed x node x 1
    
class maskedSoftmax(torch.nn.Module):
    def __init__(self):
        super(maskedSoftmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x[x == 0] = float('-inf')
        result = self.softmax(x)
        return result

# GAT layer 
class batchedGAT(torch.nn.Module):
    def __init__(self, in_planes, planes):
        super(batchedGAT,self).__init__()
        #
        self.in_planes = in_planes
        self.planes = planes 
        self.W = torch.nn.Parameter(torch.empty(size=(in_planes, planes)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.empty(size=(2*planes, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(0.33)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, h,g):
        #
        num_nodes = g.shape[-1]
        # g - batch x 1 x node x node 
        # h - batch x channel x node x 1
        h_ones, g_ones = torch.split(h,1,dim=0),torch.split(g,1,dim=0)
        o_hs = []
        # batch-wise operation
        for batch in range(len(h_ones)):
            # g_one - node x node
            # h_one - node x channel
            h_one, g_one = h_ones[batch].squeeze(0).squeeze(-1).permute(1,0),g_ones[batch].squeeze(0).squeeze(0)

            # preprocessing : g_one
            g_one = (g_one > 0).float() # binarization

            Wh = torch.mm(h_one, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
            a_input = self._prepare_attentional_mechanism_input(Wh)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(g_one > 0, e, zero_vec)
            attention = self.softmax(attention)
            h_prime = torch.matmul(attention, Wh)       

            o_hs.append(h_prime.permute(1,0)) 
        hs = torch.stack(o_hs,0).unsqueeze(-1) # stack along the batch & add the channel
        return hs # batch x channel_embed x node x 1

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2 * self.planes)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_planes) + ' -> ' + str(self.planes) + ')'

# top-k pooling using the attention score
class top_k_pool(torch.nn.Module):
    def __init__(self, k):
        super(top_k_pool,self).__init__()
        # Nodes which have an upper k score will be survived
        self.k = k
        
    def forward(self,h,g,scores):
        # g - batch x 1 x node x node 
        # h - batch x channel x node x 1
        # score - batch x 1 x node x 1
        h_ones, g_ones, score_ones = torch.split(h,1,dim=0),torch.split(g,1,dim=0),torch.split(scores,1,dim=0)
        h_pool,g_pool,score_pool = [],[],[]
        for batch in range(len(h_ones)):
            # g_one - node x node
            # h_one - channel x node
            h_one, g_one = h_ones[batch].squeeze(0).squeeze(-1),g_ones[batch].squeeze(0).squeeze(0)
            
            # score - node x 1
            score = score_ones[batch].squeeze(0).squeeze(0)
            values, idx = torch.topk(score.squeeze(-1),self.k,dim=0) # idx : k x 1
            idx = idx.squeeze() # idx : k
            
            # top-k selection
            new_h = torch.index_select(h_one, 1, idx)  # channel x k 
            new_g = torch.index_select(g_one, 0, idx)  # k x node
            new_g = torch.index_select(new_g, 1, idx)  # k x k
            new_score = torch.index_select(score, 0, idx)  # k x 1
            
            h_pool.append(new_h) # batch x channel x k
            g_pool.append(new_g) # batch x k x k 
            score_pool.append(new_score) # batch x k x 1
        
        # stack along the batch & add the channel
        hs, gs = torch.stack(h_pool,0).unsqueeze(-1), torch.stack(g_pool,0).unsqueeze(1)
        ss = torch.stack(score_pool,0).unsqueeze(1)
        
        # hs - batch x channel x k x 1
        # gs - batch x 1 x k x k 
        # ss - batch x 1 x k x 1
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
        weight = self.lReLu(self.attention11(out,x))
        weight = self.sigmoid(self.attention12(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool1(out,x,weight)
        out = out*weight + out
        sum1 = ReadOut(out)
        
        # gcn layer1
        out = self.gcn1(out,x)
        
        # Node attention score
        weight = self.lReLu(self.attention21(out,x))
        weight = self.sigmoid(self.attention22(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool2(out,x,weight)
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
        weight = self.lReLu(self.attention11(out,x))
        weight = self.sigmoid(self.attention12(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool1(out,x,weight)
        out = out*weight + out
        sum1 = ReadOut(out)
        
        # gcn layer1
        out = self.gcn1(out,x)
        
        # Node attention score
        weight = self.lReLu(self.attention21(out,x))
        weight = self.sigmoid(self.attention22(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool2(out,x,weight)
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
        weight = self.lReLu(self.attention11(out,x))
        weight = self.sigmoid(self.attention12(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool1(out,x,weight)
        out = out*weight + out
        sum1 = ReadOut(out)
        
        # gcn layer1
        out = self.gcn1(out,x)
        
        # Node attention score
        weight = self.lReLu(self.attention21(out,x))
        weight = self.sigmoid(self.attention22(weight,x))
        
        # top k pooling
        out, x, weight = self.topkpool2(out,x,weight)
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
        print(sub,end=':')
        in_mat = X[sub,:,:]
        in_adj = in_mat > 0
        if sub == 5:
            sns.heatmap(in_mat)
            plt.show()
            sns.heatmap(in_adj)
            plt.show()
        # for every node,
        for start in range(num_nodes):
            center_node_idx = np.array([start])
            sub_list, con_list = [], []
            # search depth for subgraph
            for depth in range(limit_search_depth):
                for n in range(len(center_node_idx)):
                    node_idx = center_node_idx[n]
                    for i in range(num_nodes):
                        if in_adj[node_idx,i] > 0:
                            sub_list.append(i)
                    center_node_idx = sub_list
                if depth==r1-1: first_neihbours = list(np.unique(center_node_idx))
            sub_list = list(np.unique(sub_list)) # get subgraph list
            # if the present node has an apropriate number of neightbours, 
            if len(sub_list) <= max_neighbour and len(sub_list) >= 3:
                
                # search depth for r2 (for outer boundary of context graph)
                center_node_idx = np.array([start])
                for depth in range(r2):
                    for n in range(len(center_node_idx)):
                        node_idx = center_node_idx[n]
                        for i in range(num_nodes):
                            if in_adj[node_idx,i] > 0:
                                con_list.append(i)
                        center_node_idx = con_list
                con_list = list(np.unique(con_list)) # get subgraph list
                # get context graph list & anchor node list
                anchor_nodelist_tmp = sub_list
                for first in range(len(first_neihbours)):
                    con_list.remove(first_neihbours[first])
                    anchor_nodelist_tmp.remove(first_neihbours[first])
            
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
        #print(len(pos_subgraphs))
    return pos_subgraphs, pos_congraphs, pos_nodelist, anchor_nodelist

def attribute_masking_data(X, masking_ratio=0.15, random_state=11):
    num_sub = np.shape(X)[0]
    num_nodes = np.shape(X)[-1]
    masked_graphs, masked_atts = [],[]
    masked_idx = []

    for sub in range(num_sub):
        in_mat = X[sub,:,:]
        # find connected edges
        nonzero_edge_idx = np.array(np.where(in_mat > 0)) # 2 x num_samples numpy array
        num_samples = np.shape(nonzero_edge_idx)[1]

        # DELETE the lower traiangular elements & diagonal self connections
        del_idx = []
        for sample in range(num_samples):
            one_pair = nonzero_edge_idx[:,sample] # 1 x 2
            if one_pair[0] >= one_pair[1]: del_idx.append(sample)
        nonzero_edge_idx = np.delete(nonzero_edge_idx,del_idx, axis = 1)

        # new num_samples after filtering 
        num_samples = np.shape(nonzero_edge_idx)[1]

        # number of selected edges
        num_masking = round(num_samples * masking_ratio)
        np.random.seed(random_state)
        randidx = np.random.choice(num_samples, num_masking,replace=False) # (num_masking,)
        selected_edge_idx = nonzero_edge_idx[:,randidx] # 2 x num_masking

        for mask in range(num_masking):
            masked_atts.append(in_mat[selected_edge_idx[0,mask],selected_edge_idx[1,mask]])
            tmp_mat = in_mat
            tmp_mat[selected_edge_idx[0,mask],selected_edge_idx[1,mask]] = 0
            tmp_mat[selected_edge_idx[1,mask],selected_edge_idx[0,mask]] = 0
            masked_idx.append([sub, selected_edge_idx[0,mask], selected_edge_idx[1,mask] ])
            masked_graphs.append(tmp_mat)
    return masked_graphs, masked_atts, masked_idx