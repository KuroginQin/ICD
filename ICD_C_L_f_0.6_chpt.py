# Evaluation of ICD-C checkpoint trained on L(f,0.6)

from utils import *
from modules.models import *
from modules.feat_ext import *
from modules.loss import *
import time
import numpy as np
from sklearn.cluster import KMeans
import random

#torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
#setup_seed(0)

# ====================
data_name = 'L(f,0.6)'
num_graphs = 2000 # Number of graph snapshots
feat_dim = 4096 # Dimensionality of reduced feature input
epoch_idx = 60 # Index of the assigned epoch (in the offline training)

# ====================
train_rate = 0.80 # Ratio of snapshots in the training set
val_rate = 0.10 # Ratio of snapshots in the validation set
num_train = int(num_graphs*train_rate) # Number of snapshots in the training set
num_val = int(num_graphs*val_rate) # Number of snapshots in the validation set
num_test = num_graphs-(num_train+num_val) # Number of snapshots in the test set

# ===================
# Read the dataset
edge_list = np.load('data/%s_edge_list.npy' % (data_name), allow_pickle=True)
gnd_list = np.load('data/%s_gnd_list.npy' % (data_name), allow_pickle=True)
# ==========
edge_test_list = edge_list[num_train+num_val: ] # Edge list of the test set
gnd_test_list = gnd_list[num_train+num_val: ] # Ground-truth list of the test set

# ====================
# Load saved check point w.r.t. the assigned epoch
gen_net = torch.load('chpt/ICD-C_gen_%s_%d.pkl' % (data_name, epoch_idx)).to(device)
gen_net.eval()

# ====================
NMI_list = []
AC_list = []
mod_list = []
ncut_list = []
time_list = []
for s in range(num_test):
    # ====================
    gnd = gnd_test_list[s] # Ground-truth
    num_nodes = len(gnd) # Number of nodes
    edges = edge_test_list[s] # Edge list
    adj = get_adj(edges, num_nodes) # Adjacency matrix
    adj_tnr = torch.FloatTensor(adj).to(device)
    if min(gnd)==0: # Number of clusters
        num_clus = int(max(gnd))+1
    else:
        num_clus = int(max(gnd))
    # ==========
    # Extract the node label sequence from ground-truth (for evaluation)
    if min(gnd)==0:
        labels_ = np.array(gnd)
    else:
        labels_ = np.array(gnd-1)

    # ====================
    sup_topo = sp.sparse.coo_matrix(get_gnn_sup(adj)) # Normalized sparse adjacency matrix of the original graph
    sup_topo_sp = sparse_to_tuple(sup_topo)
    idxs = torch.LongTensor(sup_topo_sp[0].astype(float)).to(device)
    vals = torch.FloatTensor(sup_topo_sp[1]).to(device)
    sup_topo_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_topo_sp[2]).float().to(device)
    # ==========
    # Extract the neighbor-induced feature (i.e., Markov matrix)
    time_start = time.time()
    mar_tnr = get_mar_GPU(adj_tnr, num_nodes, device)
    if torch.cuda.is_available():
        mar = mar_tnr.cpu().data.numpy()
    else:
        mar = mar_tnr.data.numpy()
    # ==========
    # Derive the reduced feature input
    edge_map = rand_edge_map(edges, mar)
    feat_coar_tnr = feat_coar_gpu(edge_map, mar_tnr, num_nodes, feat_dim, device) # Tensor of the reduced features
    # ==========
    time_end = time.time()
    feat_time = time_end-time_start # Runtime of feature extraction

    # =====================
    # Derive the (inductive) graph embedding
    time_start = time.time()
    _, emb, _ = gen_net(sup_topo_tnr, sup_topo_tnr, feat_coar_tnr, train_flag=False)
    time_end = time.time()
    prop_time = time_end-time_start # Runtime of one feedforward propagation
    if torch.cuda.is_available():
        emb = emb.cpu().data.numpy()
    else:
        emb = emb.data.numpy()

    # =====================
    time_start = time.time()
    kmeans = KMeans(n_clusters=num_clus, n_init=10).fit(emb.astype(np.float64))
    clus_res = kmeans.labels_
    time_end = time.time()
    clus_time = time_end-time_start # Runtime of downstream clustering
    runtime = feat_time+prop_time+clus_time # Total runtime

    # ====================
    # Compute the quality metric of current partitioning result
    NMI = get_NMI(clus_res, labels_)
    AC = get_AC(labels_, clus_res)
    # ===========
    #mod_metric = get_mod_metric(adj, clus_res, num_clus)
    #ncut_metric = get_NCut_metric(adj, clus_res, num_clus)
    mod_metric = get_mod_metric_gpu(adj_tnr, clus_res, num_clus, device)
    ncut_metric = get_NCut_metric_gpu(adj_tnr, clus_res, num_clus, device)
    # ===========
    NMI_list.append(NMI)
    AC_list.append(AC)
    mod_list.append(mod_metric)
    ncut_list.append(ncut_metric)
    time_list.append(runtime)
# ====================
NMI_mean = np.mean(NMI_list)
NMI_std = np.std(NMI_list)
AC_mean = np.mean(AC_list)
AC_std = np.std(AC_list)
mod_mean = np.mean(mod_list)
mod_std = np.std(mod_list)
ncut_mean = np.mean(ncut_list)
ncut_std = np.std(ncut_list)
time_mean = np.mean(time_list)
time_std = np.std(time_list)
print('ICD-M Epoch %d Test NMI %f (%f) AC %f (%f) Mod %f (%f) NCut %f (%f) Time %f (%f)'
          % (epoch_idx, NMI_mean, NMI_std, AC_mean, AC_std, mod_mean, mod_std,
             ncut_mean, ncut_std, time_mean, time_std))
