from utils import *
import igraph

# Generate the GN synthetic benchmark

# ====================
num_graphs = 2000 # Number of (independent) graphs, i.e., T
num_nodes = 5000 # Number of nodes, i.e., N
# ==========
num_clus = 250 # Number of clusters, i.e., K
p_in = 0.5 # 0.5, 0.4, 0.3
data_name = 'GN-%.1f'%p_in # Name of the dataset

# ====================
prob_mat = np.zeros((num_clus, num_clus)) # Probability matrix (K*K)
for i in range(num_clus):
    for j in range(num_clus):
        if i == j:
            prob_mat[i, j] = p_in # Probability (for each node) to generate an edge between nodes in the same cluster
        else:
            prob_mat[i, j] = (1-p_in)/(num_clus-1) # # Probability (for each node) to generate an edge between nodes in different clusters
prob_mat = prob_mat.tolist()
# ==========
block_sizes = [] # List to record the number of nodes in each cluster
for i in range(num_clus):
    block_sizes.append(round(1.0*num_nodes/num_clus))

# ====================
# Independently generate T synthetic graphs
for t in range(num_graphs):
    # ==========
    # Generate an independent graph via SBM, based on the aforementioned parameter settings
    g = igraph.Graph.SBM(num_nodes, prob_mat, block_sizes, directed=False, loops=False)
    adj = g.get_adjacency() # Adjacency matrix of the current generated graph
    # ==========
    # Randomly map the original node indices to new indices
    rand_idxs = [e for e in range(num_nodes)]
    np.random.shuffle(rand_idxs)
    adj_mat = np.zeros((num_nodes, num_nodes)) # Target adjacnecy matrix w.r.t. new node indices
    for i in range(num_nodes):
        for j in range(i):
            if adj[i, j]>0 and i!=j:
                i_idx = rand_idxs[i]
                j_idx = rand_idxs[j]
                adj_mat[i_idx, j_idx] = 1
                adj_mat[j_idx, i_idx] = 1
    del adj
    adj = adj_mat
    # ===========
    # Generate the partitioning ground-truth
    gnd = [-1]*num_nodes # Label sequence
    iter = 0
    for c in range(num_clus):
        for _ in range(block_sizes[c]):
            idx = rand_idxs[iter]
            gnd[idx] = c
            iter += 1
    gnd = np.array(gnd)

    # ====================
    # Save the edge list
    f_output = open('%s/edge_%d.txt' % (data_name, t+1), 'w')
    for i in range(num_nodes):
        for j in range(i):
            if adj[i, j]>0:
                f_output.write('%d %d\n' % (i+1, j+1)) # Node indices start from 1
    f_output.close()
    # ===========
    # Save the partitioning ground-truth
    f_output = open('%s/gnd_%d.txt' % (data_name, t+1), 'w')
    for i in range(num_nodes):
        f_output.write('%d\n' % (gnd[i]+1))
    f_output.close()

    print('-Fin %d/%d' % (t+1, num_graphs))
