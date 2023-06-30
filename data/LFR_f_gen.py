import numpy as np
import os

# Generate the L(f, mu) synthetic benchmark
# # Using the open-source implmentation of the LFR benchmark from
# #   https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp
# # To use this implementation, enter data/LFR & use the command:
# #   make

# =====================
num_graphs = 2000 # Number of graphs, i.e., T
num_nodes = 5000 # Number of nodes, i.e., N
mu = 0.3 # 0.3, 0.6
data_name = 'L(f,%.1f)' % mu # Name of the dataset

# =====================
for t in range(num_graphs):
    # ===========
    # Generate an independent graph snapshot of the LFR-Net
    command_res = os.popen('LFR/lfrbench_udwov -N %d -k 10 -maxk 100 '
                           '-mut %.1f -muw 0.1 -beta 1 -minc 10 -maxc 200 '
                           '-on 0 -om 0 -cnl 1 -name LFR/buffer/LFR_f_bf' % (num_nodes, mu)).readlines()
    # ==========
    # Extract the topology structure
    adj = np.zeros((num_nodes, num_nodes)) # Adjacency matrix of the current graph
    f_input = open('LFR/buffer/LFR_f_bf.nse', 'r')
    for line in f_input.readlines():
        if line[0] == '#':
            continue
        record = line.strip().split('	')
        src_idx = int(record[0])
        dst_idx = int(record[1])
        adj[src_idx-1, dst_idx-1] = 1
        adj[dst_idx-1, src_idx-1] = 1
    f_input.close()
    # ===========
    # Save the edge list
    f_output = open('%s/edge_%d.txt' % (data_name, t+1), 'w')
    for i in range(num_nodes):
        for j in range(i):
            if adj[i, j]>0:
                f_output.write('%d %d\n' % (i+1, j+1)) # Node indices start from 1
    f_output.close()
    # ===========
    # Extract the partitioning ground-truth
    gnd = [-1]*num_nodes
    clus_idx = 1
    f_input = open('LFR/buffer/LFR_f_bf.cnl', 'r')
    for line in f_input.readlines():
        record = line.strip().split(' ')
        for idx in record:
            gnd[int(idx)-1] = clus_idx
        clus_idx += 1
    f_input.close()
    gnd = np.array(gnd)
    num_clus = clus_idx-1 # Number of clusters
    # ===========
    # Save the ground-truth
    f_output = open('%s/gnd_%d.txt' % (data_name, t+1), 'w')
    for i in range(num_nodes):
        f_output.write('%d\n' % (gnd[i]))
    f_output.close()

    print('-Fin %d/%d' % (t+1, num_graphs))