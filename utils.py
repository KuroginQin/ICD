import numpy as np
import math
from munkres import Munkres
from sklearn.metrics import f1_score
import scipy as sp
import torch

def get_adj(edges, num_nodes, zero_idx_flag=False):
    '''
    Function to get adjacency matrix according to the edge list
    :param edges: edge list
    :param num_nodes: number of nodes
    :param zero_idx_flag: flag to represent whether node indices start from 0 (=true)
    :return: adj: adjacency matrix
    '''
    # ====================
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges) # Number of edges
    for i in range(num_edges):
        if not zero_idx_flag: # Node indices start from 1
            src = int(edges[i][0])-1
            dst = int(edges[i][1])-1
        else: # Node indices start from 0
            src = int(edges[i][0])
            dst = int(edges[i][1])
        adj[src, dst] = 1
        adj[dst, src] = 1
    return adj

def get_NMI(A, B):
    '''
    Function to get the NMI metric
    :param A: the label sequence A
    :param B: the label sequence B
    :return: MIhat: NMI metric
    '''
    # ====================
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # ==========
    MI = 0 # Mutual information
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # ==========
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy) # Normalized mutual information

    return MIhat

def get_AC(gnd_seq, pred_seq):
    '''
    Function to get the AC metric
    :param gnd_seq: label sequence of ground-truth
    :param pred_seq: label sequence of the partitioning (clustering) result
    :return: AC metric
    '''
    # ====================
    res_map = best_map(gnd_seq, np.array(pred_seq))
    AC = f1_score(gnd_seq, res_map, average='micro')

    return AC

def best_map(L1, L2):
    '''
    Function to get the best membership map from label sequence L1 to L2 for AC metric
    :param L1: label sequence L1
    :param L2: label sequence L2
    :return: the best map membership
    '''
    # ====================
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    # ==========
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]

    return newL2

def get_mod_metric_gpu(adj_tnr, labels, num_clus, device):
    '''
    Function to get the modularity metric (speed up via GPU)
    :param adj_tnr: tensor of adjacency matrix
    :param labels: label sequence of partitioning result
    :param num_clus: number of clusters
    :return: modularity metric
    '''
    # ====================
    num_nodes, _ = adj_tnr.shape # Get number of nodes
    degs = torch.sum(adj_tnr, dim=1) # Node degree sequence
    degs = torch.reshape(degs, (1, num_nodes))
    wei_sum = torch.sum(degs) # Twice of the sum of edge weights
    prop = torch.mm(degs.t(), degs)/wei_sum
    mod_tnr = adj_tnr - prop # (Tensor of) modularity matrix
    # ==========
    mem_ind = np.zeros((num_nodes, num_clus)) # Partitioning membership indicator for modularity
    for i in range(num_nodes):
        idx = labels[i]
        mem_ind[i, idx] = 1
    mem_ind_tnr = torch.FloatTensor(mem_ind).to(device) # Tensor of membership indicator
    # =========
    # Derive the modularity metric via matrix multiplication (speed up via GPU)
    metric_tnr = torch.trace(torch.mm(torch.mm(mem_ind_tnr.t(), mod_tnr), mem_ind_tnr))/wei_sum

    if torch.cuda.is_available():
        return metric_tnr.cpu().data.numpy()
    else:
        return metric_tnr.data.numpy()

def get_NCut_metric_gpu(adj_tnr, labels, num_clus, device):
    '''
    Function to get the NCut metric (speed up via GPU)
    :param adj_tnr: tensor of adjacency matrix
    :param labels: label sequence of partitioning result
    :param num_clus: number of clusters
    :return: NCut metric
    '''
    # ====================
    num_nodes, _ = adj_tnr.shape # Get number of nodes
    degs = torch.sum(adj_tnr, dim=1) # Node degree sequence
    lap_tnr = torch.diag(degs) - adj_tnr # (Tensor of) Laplacian matrix w.r.t. the adjacency matrix input
    # ==========
    mem_ind = np.zeros((num_nodes, num_clus)) # Partitioning membership indicator
    for i in range(num_nodes):
        if min(labels) == 0:
            label_idx = int(labels[i])
        else:
            label_idx = int(labels[i]) - 1
        mem_ind[i, label_idx] = 1
    # ==========
    mem_ind_tnr = torch.FloatTensor(mem_ind).to(device) # (Tensor of) partitioning membership indicator for NCut
    vol = torch.diag(torch.mm(torch.mm(mem_ind_tnr.t(), adj_tnr), mem_ind_tnr))
    vol = torch.max(vol, 1e-1*torch.ones(num_clus).to(device))
    vol_inv_sqrt = torch.sqrt(torch.reciprocal(vol))
    mem_ind_tnr = torch.mm(mem_ind_tnr, torch.diag(vol_inv_sqrt))
    # ==========
    # Derive the NCut metric via matrix multiplication (speed up via GPU)
    metric_tnr = torch.trace(torch.mm(torch.mm(mem_ind_tnr.t(), lap_tnr), mem_ind_tnr))/2.0
    if torch.cuda.is_available():
        return metric_tnr.cpu().data.numpy()
    else:
        return metric_tnr.data.numpy()

def get_gnn_sup(adj):
    '''
    Function to get the support (normalized adjacency matrix with self-connected edges) for GNN
    :param adj: adjacency matrix
    :return: sup: support
    '''
    # ====================
    num_nodes, _ = adj.shape # Get number of nodes
    adj_ = adj + np.eye(num_nodes) # Derive the adjacency matrix with self-connected edges
    degs = np.sum(adj_, axis=1)
    degs_sqrt = np.sqrt(degs)
    sup = adj_ # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs_sqrt[i]
        sup[:, i] /= degs_sqrt[i]
    #for j in range(num_nodes):
    #    sup[:, j] /= degs_sqrt[j]

    return sup

def sparse_to_tuple(sparse_mx):
    '''
    Function to transfer sparse matrix to tuple format
    :param sparse_mx: original sparse matrix
    :return: corresponding tuple format
    '''
    # ====================
    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx): # sp.sparse.isspmatrix_coo(mx)
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def get_mod_GPU(adj_tnr, num_nodes):
    '''
    Funtion to get the modularity matrix based on adjacency matrix (speed up via GPU)
    :param adj_tnr: tensor of adjacency matrix
    :param num_nodes: number of nodes
    :return: tensor of modularity matrix
    '''
    degs = torch.sum(adj_tnr, dim=1) # Node degree vector
    wei_sum = torch.sum(degs) # Twice of the sum of edges
    degs = torch.reshape(degs, (1, num_nodes))
    prop = torch.mm(degs.t(), degs)/wei_sum
    mod_tnr = adj_tnr - prop # Tensor of modularity matrix

    return mod_tnr

def get_mar_GPU(adj_tnr, num_nodes, device):
    '''
    Function to get Markov matrix (i.e., normalized adjacency matrix) based on adjacency matrix (speed up via GPU)
    :param adj_tnr: tensor of adjacency matrix
    :param num_nodes: number of nodes
    :return: tensor of Markov matrix
    '''
    degs = torch.sum(adj_tnr, dim=1) # Node degree vector
    degs = torch.max(degs, 1e-1*torch.ones(num_nodes).to(device))
    degs = torch.sqrt(torch.reciprocal(degs))
    degs = torch.diag(degs)
    mar_tnr = torch.mm(degs, torch.mm(adj_tnr, degs)) # Tensor of Markov matrix

    return mar_tnr

def m_norm(s):
    '''
    Function to normalize a variable to the range [-1, 1]
    :param s: vector/matrix to be normalized
    :return: normalized vector/matrix
    '''
    max_elem = np.max(s)
    min_elem = np.min(s)
    s_ = (s-min_elem)/max(max_elem-min_elem, 1e-10)
    s_ = 2*s_-1

    return s_

def max_min_norm(s):
    '''
    Function to conduct the max-min normalization
    :param s: vector/matrix to be normalized
    :return: the normalized vector/matrix
    '''
    max_elem = np.max(s)
    min_elem = np.min(s)
    s_ = (s - min_elem) / max(max_elem - min_elem, 1e-10)

    return s_
