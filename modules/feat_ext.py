import numpy as np
import torch
import random

# Functions to define the operations of the feature extraction module

def feat_coar_gpu(edge_map, feat_tnr, num_nodes, red_dim, device):
    '''
    Function to derived the reduced feature input (speed up via GPU)
    :param edge_map: reweighted edge map (i.e., (src, dst) -> weight) w.r.t. the extracted neighbor-induced features
    :param feat_tnr: tensor of neighbor-induced feature
    :param num_nodes: number of nodes
    :param red_dim: dimensionality of the reduced feature
    :param device: device type
    :return: tensor of reduced feature input
    '''
    # ====================
    if num_nodes==red_dim: # If the number of nodes = dimensionality of reduced features
        return feat_tnr # Directly return the extracted neighbor-induced feature
    elif num_nodes<red_dim: # If the number of nodes is smaller than the required dimension
        # Concatenate feature matrix with zeros & return the result
        return torch.cat((feat_tnr, torch.zeros(num_nodes, red_dim-num_nodes).to(device)), axis=1)

    # =====================
    node_map_list = [] # List to recode the supernode membership of each level
    # ==========
    # For Level-0
    # Sort the reweigthed edge list based on the weights
    edge_list_hyper = sorted(edge_map.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    node_map_hyper, node_count = SHEM(edge_list_hyper, num_nodes, red_dim) # Apply SHEM to the sorted edge list
    node_map_list.append(node_map_hyper)
    while node_count>red_dim:
        # ==========
        # Rearrange the edge set of the derived supergraph
        edge_coar = {}
        for edge in edge_list_hyper:
            src = node_map_hyper[edge[0][0]]
            dst = node_map_hyper[edge[0][1]]
            if src == dst:
                continue
            if src > dst:
                temp = src
                src = dst
                dst = temp
            wei = edge[1]
            if (src, dst) not in edge_coar:
                edge_coar[(src, dst)] = wei
            else:
                edge_coar[(src, dst)] += wei
        # ==========
        # For next level
        # Sort the reweigthed edge list based on the weights
        edge_list_hyper = sorted(edge_coar.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        node_map_hyper, node_count = SHEM(edge_list_hyper, node_count, red_dim) # Apply SHEM to the sorted edge list
        node_map_list.append(node_map_hyper)

    # ====================
    # Derive the tensor of the reduced features
    num_levels = len(node_map_list) # Number of levels
    coar_mem = [] # Supernode membership list w/ coar_mem[i] as the member list of supernde i
    for r in range(red_dim):
        coar_mem.append([])
    for i in range(num_nodes): # For each node i
        map_idx = i
        for d in range(num_levels): # Search the supernode to which node i is mapped
            map_idx = node_map_list[d][map_idx]
        coar_mem[map_idx].append(i)
    # ==========
    # Derive the sparse tensor of coarsening matrix
    idxs = []
    vals = []
    for r in range(red_dim):
        mem_list = coar_mem[r]
        mem_num = len(mem_list)
        sqrt_vol = np.sqrt(mem_num)
        for i in mem_list:
            idxs.append([i, r])
            vals.append(sqrt_vol)
    idxs = torch.LongTensor(idxs).to(device)
    vals = torch.FloatTensor(vals).to(device)
    # Sparse tensor of coarsening matrix
    coar_mat_tnr = torch.sparse.FloatTensor(idxs.t(), vals, torch.Size([num_nodes, red_dim])).float().to(device)
    coar_feat_tnr = torch.spmm(coar_mat_tnr.t(), feat_tnr).t() # Tensor of reduced features

    return coar_feat_tnr

def feat_coar(edge_map, feat, node_num, red_dim):
    # ====================
    if node_num==red_dim: # If the number of nodes is equal to the required dimension
        return feat # Directly return the result
    elif node_num<red_dim: # If the number of nodes is smaller than the required dimension
        # Concatenate the original feature matrix with zero, and return the result
        return np.concatenate((feat, np.zeros((node_num, red_dim-node_num))), axis=1)
    # ====================
    node_map_list = []
    edge_list_hyper = sorted(edge_map.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    node_map_hyper, node_count = SHEM(edge_list_hyper, node_num, red_dim)
    node_map_list.append(node_map_hyper)
    while node_count>red_dim:
        # ==========
        edge_coar = {}
        for edge in edge_list_hyper:
            src = node_map_hyper[edge[0][0]]
            dst = node_map_hyper[edge[0][1]]
            if src == dst:
                continue
            if src > dst:
                temp = src
                src = dst
                dst = temp
            wei = edge[1]
            if (src, dst) not in edge_coar:
                edge_coar[(src, dst)] = wei
            else:
                edge_coar[(src, dst)] += wei
        edge_list_hyper = sorted(edge_coar.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        node_map_hyper, node_count = SHEM(edge_list_hyper, node_count, red_dim)
        node_map_list.append(node_map_hyper)

    # ==========
    level_num = len(node_map_list)
    coar_mem = []
    for r in range(red_dim):
        coar_mem.append([])
    for i in range(node_num):
        map_idx = i
        for d in range(level_num):
            map_idx = node_map_list[d][map_idx]
        coar_mem[map_idx].append(i)
    # ==========
    coar_feat = feat
    col_del_list = []
    for r in range(red_dim):
        mem_list = coar_mem[r]
        mem_num = len(mem_list)
        if mem_num>1:
            sqrt_vol = np.sqrt(mem_num)
            coar_feat[:, mem_list[0]] /= sqrt_vol
            for e in mem_list[1:]:
                coar_feat[:, mem_list[0]] += coar_feat[:, e]/sqrt_vol
            col_del_list.extend(mem_list[1:])
    coar_feat = np.delete(coar_feat, col_del_list, axis=1)

    return coar_feat

def SHEM(edge_list, num_nodes, red_dim):
    '''
    Function to define the SHEM procedure
    :param edge_list: edge list of a (super)graph
    :param num_nodes: number of nodes
    :param red_dim: dimensionality of reduced feature
    :return: merged membership of supernodes & number of (super)nodes (in the merged supergraph)
    '''
    # ====================
    node_map = {} # Merged supernode membership (original node -> supdenode)
    node_count = num_nodes # Number of (super)nodes (in the merged supergraph)
    node_idx = 0 # New index of a new (super)node in the current supergraph
    for edge in edge_list:
        src = edge[0][0]
        dst = edge[0][1]
        if src==dst: # Skip the self-connected edges
            continue
        if (src not in node_map) and (dst not in node_map):
            node_map[src] = node_idx
            node_map[dst] = node_idx
            node_idx += 1
            node_count -= 1
        if node_count==red_dim: # Stop when the number of (super)nodes = dimensionality of reduced features
            break
    # ====================
    # Directly add the rest (supde)nodes into the node set of current supergraph
    for i in range(num_nodes):
        if i not in node_map:
            node_map[i] = node_idx
            node_idx += 1

    return node_map, node_count

def rand_edge_map(edges, feat, zero_idx_flag=False):
    '''
    Function to extract the reweighted edge list w.r.t. the neighbor-induced feature
    :param edges: edge list
    :param feat: extracted neighbor-induced feature
    :param zero_idx_flag: flag to represent whether node indices start from 0 (=true)
    :return: reweighted edge map (i.e., (src, dst) -> weight)
    '''
    # ====================
    rand_edges = edges
    rand_idxs = [e for e in range(len(rand_edges))]
    random.shuffle(rand_idxs)
    rand_edges = rand_edges[rand_idxs] # Random shuffle the edge list
    # ===========
    edge_map = {} # Reweighted edge map
    for edge in rand_edges:
        if not zero_idx_flag: # Node indices start from 1
            src = int(edge[0])-1
            dst = int(edge[1])-1
        else: # Node indices start from 0
            src = int(edge[0])
            dst = int(edge[1])
        if src>dst:
            temp = src
            src = dst
            dst = temp
        edge_map[(src, dst)] = feat[src, dst]

    return edge_map