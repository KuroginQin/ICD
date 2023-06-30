import numpy as np

# Save the LFR synthetic benchmark as the .npy format

# ====================
data_name = 'L(f,0.3)' # L(f,0.3), L(f,0.6), L(n,0.3), L(n,0.6)
num_graphs = 2000 # Number of nodes, i.e., T

# =====================
edge_seq = []
gnd_seq = []
for t in range(num_graphs):
    # ===========
    # Read edge list of the current graph
    edge = np.loadtxt('%s/edge_%d.txt' % (data_name, t+1))
    l = len(edge)
    for i in range(l):
        if edge[i][0]>edge[i][1]:
            temp = edge[i][0]
            edge[i][0] = edge[i][1]
            edge[i][1] = temp
    edge_seq.append(edge)
    # ==========
    # Read ground-truth of the current graph
    gnd = np.loadtxt('%s/gnd_%d.txt' % (data_name, t+1))
    gnd_seq.append(gnd)
    # ==========
    print('-Fin %d/%d' % (t+1, num_graphs))

# ====================
# Save as the .npy format
np.save('%s_edge_seq.npy' % (data_name), edge_seq)
np.save('%s_gnd_seq.npy' % (data_name), gnd_seq)
