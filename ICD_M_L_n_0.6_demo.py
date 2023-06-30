# Demonstration of ICD-M on L(n,0.6)

import torch.optim as optim
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
data_name = 'L(n,0.6)'
num_graphs = 2000 # Number of graph snapshots
feat_dim = 4096 # Dimensionality of reduced feature input
# ==========
# Layer configurations
gen_dims = [feat_dim, 2048, 512, 256] # Layer configuration of generator
hid_dim = gen_dims[-1] # Dimensionality of graph embedding
disc_dims = [hid_dim, 128, 64, 16, 1] # Layer configuration of discriminator
# ==========
# Hyper-parameter settings
alpha = 1
beta = 3
m = 1
p = 1000 # Number of training samples in each epoch
gen_lr = 1e-4 # Learning rate of generator
disc_lr = 1e-4 # Learning rate of discriminator
feat_norm_flag = True # Flag for whether to normalize the neighbor-induced feature
# =========
save_flag = True # Flag to save the check points (=True) for each epoch
test_eva_flag = True # Flag for evaluation on test set (=True) for each epoch

# ====================
train_rate = 0.80 # Ratio of snapshots in the training set
val_rate = 0.10 # Ratio of snapshots in the validation set
num_train = int(num_graphs*train_rate) # Number of snapshots in the training set
num_val = int(num_graphs*val_rate) # Number of snapshots in the validation set
num_test = num_graphs-(num_train+num_val) # Number of snapshots in the test set
# ==========
dropout_rate = 0.0 # Dropout rate

# ===================
# Read the dataset
edge_list = np.load('data/%s_edge_list.npy' % (data_name), allow_pickle=True)
gnd_list = np.load('data/%s_gnd_list.npy' % (data_name), allow_pickle=True)
# ==========
edge_train_list = edge_list[0: num_train] # Edge list of the training set
edge_val_list = edge_list[num_train: num_train+num_val] # Edge list of the validation set
edge_test_list = edge_list[num_train+num_val: ] # Edge list of the test set
# ==========
gnd_train_list = gnd_list[0: num_train] # Ground-truth list of the training set
gnd_val_list = gnd_list[num_train: num_train+num_val] # Ground-truth list of the validation set
gnd_test_list = gnd_list[num_train+num_val: ] # Ground-truth list of the test set

# ====================
# Define the models
gen_net = GenNet(gen_dims, dropout_rate, device) # Generator
disc_net = DiscNet(disc_dims, dropout_rate, device) # Discriminator
# ==========
# Define the optimizers
gen_opt = optim.RMSprop(gen_net.parameters(), lr=gen_lr, weight_decay=1e-5)
disc_opt = optim.RMSprop(disc_net.parameters(), lr=disc_lr, weight_decay=1e-5)
#gen_opt = optim.Adam(gen_net.parameters(), lr=gen_lr, weight_decay=1e-5)
#disc_opt = optim.Adam(disc_net.parameters(), lr=disc_lr, weight_decay=1e-5)

# ====================
num_train_epochs = 100 # Number of training epochs
for epoch in range(num_train_epochs):
    # ====================
    # Train the model
    gen_net.train()
    disc_net.train()
    # ==========
    # Lists to record the loss functions
    gen_loss_list = []
    disc_loss_list = []
    # ===========
    for q in range(p): # Randomly select p snapshots from the training set to optimize the model
        s = random.randint(0, num_train-1) # Randomly select a node index
        # ====================
        gnd = gnd_train_list[s] # Ground-truth
        num_nodes = len(gnd) # Number of nodes
        edges = edge_train_list[s] # Edge list
        adj = get_adj(edges, num_nodes) # Adjacency Matrix
        if int(min(gnd)) == 0: # Number of clusters
            num_clus = int(max(gnd))+1
        else:
            num_clus = int(max(gnd))
        # ==========
        # Construct the partitioning membership indicator
        label_onehot = np.zeros((num_nodes, num_clus)) # One-hot representation of the ground-truth
        for i in range(num_nodes):
            if min(gnd) == 0:
                label_idx = int(gnd[i])
            else:
                label_idx = int(gnd[i])-1
            label_onehot[i, label_idx] = 1

        # ====================
        sup_topo = sp.sparse.coo_matrix(get_gnn_sup(adj)) # Sparse GNN support of the original graph
        sup_topo_sp = sparse_to_tuple(sup_topo)
        idxs = torch.LongTensor(sup_topo_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(sup_topo_sp[1]).to(device)
        sup_topo_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_topo_sp[2]).float().to(device)
        # ==========
        adj_gnd = np.matmul(label_onehot, np.transpose(label_onehot)) # Adjacency matrix of the auxiliary label-induced graph
        sup_gnd = sp.sparse.coo_matrix(get_gnn_sup(adj_gnd)) # Sparse GNN support of the auxiliary label-induced graph
        sup_gnd_sp = sparse_to_tuple(sup_gnd)
        idxs = torch.LongTensor(sup_gnd_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(sup_gnd_sp[1]).to(device)
        sup_gnd_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_gnd_sp[2]).float().to(device)
        # ====================
        adj_tnr = torch.FloatTensor(adj).to(device)
        gnd_tnr = torch.FloatTensor(label_onehot).to(device) # Tensor of the partitioning membership indicator

        # ====================
        # Extract the neighbor-induced feature (i.e., modularity matrix)
        mod_tnr = get_mod_GPU(adj_tnr, num_nodes)
        if torch.cuda.is_available():
            mod = mod_tnr.cpu().data.numpy()
        else:
            mod = mod_tnr.data.numpy()
        if feat_norm_flag:
            mod_ = m_norm(mod) # Get the normalized modularity matrix (w/ value range [-1, 1])
        else:
            mod_ = mod
        feat_norm_tnr = torch.FloatTensor(mod_).to(device) # Tensor of the 'normalized' neighbor-induced feature
        # ==========
        # Derive the reduced feature input
        edge_map = rand_edge_map(edges, mod)
        #edge_map = rand_edge_map(edges, max_min_norm(mod))
        feat_coar_tnr = feat_coar_gpu(edge_map, mod_tnr, num_nodes, feat_dim, device) # Tensor of the reduced features

        # ====================
        if epoch==0: # No parameter update when epoch==0
            feat_rec, emb, emb_gnd = gen_net(sup_topo_tnr, sup_gnd_tnr, feat_coar_tnr, train_flag=True)
            disc_fake = disc_net(emb)
            disc_real = disc_net(emb_gnd)
            gen_loss = get_gen_loss(disc_fake, feat_norm_tnr, feat_rec, gnd_tnr, alpha, beta) # Loss of generator
            disc_loss = get_disc_loss(disc_fake, disc_real) # Loss of discriminator
        else: # Update model parameters for epoch>0
            # ==========
            for _ in range(m):
                # ==========
                # Train discriminator
                _, emb, emb_gnd = gen_net(sup_topo_tnr, sup_gnd_tnr, feat_coar_tnr, train_flag=True)
                disc_fake = disc_net(emb)
                disc_real = disc_net(emb_gnd)
                disc_loss = get_disc_loss(disc_fake, disc_real) # Loss of discriminator
                disc_opt.zero_grad()
                disc_loss.backward() # Update parameters of discriminator
                disc_opt.step()
                # ==========
                # Train generator
                feat_rec, emb, _ = gen_net(sup_topo_tnr, sup_gnd_tnr, feat_coar_tnr, train_flag=False)
                disc_fake = disc_net(emb)
                gen_loss = get_gen_loss(disc_fake, feat_norm_tnr, feat_rec, gnd_tnr, alpha, beta) # Loss of generator
                gen_opt.zero_grad()
                gen_loss.backward() # Update parameters of generator
                gen_opt.step()
        # ==========
        gen_loss_list.append(gen_loss.item())
        disc_loss_list.append(disc_loss.item())
        if q%100 == 0:
            print('Train Epoch %d Sample %d / %d' % (epoch, q, p))
    # ====================
    gen_loss_sum = np.sum(gen_loss_list)
    disc_loss_sum = np.sum(disc_loss_list)
    print('#%d Train G-Loss %f D-Loss %f' % (epoch, gen_loss_sum, disc_loss_sum))
    # ==========
    if save_flag: # If save_flag=True, save the trained model for current epoch
        torch.save(gen_net, 'chpt/ICD-M_gen_%s_%d.pkl' % (data_name, epoch))
        torch.save(disc_net, 'chpt/ICD-M_disc_%s_%d.pkl' % (data_name, epoch))

    # ====================
    # Validate the model
    # ==========
    gen_net.eval()
    disc_net.eval()
    # ==========
    # Lists to record the quality metrics and runtime w.r.t. each validation snapshot
    NMI_list = []
    time_list = []
    for s in range(num_val):
        # ==========
        gnd = gnd_val_list[s] # Ground-truth
        num_nodes = len(gnd) # Number of nodes
        edges = edge_val_list[s] # Edge list
        adj = get_adj(edges, num_nodes) # Adjacency matrix
        adj_tnr = torch.FloatTensor(adj).to(device)
        if min(gnd) == 0: # Number of clusters
            num_clus = int(max(gnd))+1
        else:
            num_clus = int(max(gnd))
        # ===========
        # Extract the node label sequence from ground-truth (for evaluation)
        if min(gnd)==0:
            labels_ = np.array(gnd)
        else:
            labels_ = np.array(gnd-1)

        # ====================
        sup_topo = sp.sparse.coo_matrix(get_gnn_sup(adj)) # Sparse GNN support of the original graph
        sup_topo_sp = sparse_to_tuple(sup_topo)
        idxs = torch.LongTensor(sup_topo_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(sup_topo_sp[1]).to(device)
        sup_topo_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_topo_sp[2]).float().to(device)
        # ==========
        # Extract the neighbor-induced feature (i.e., modularity matrix)
        time_start = time.time()
        mod_tnr = get_mod_GPU(adj_tnr, num_nodes)
        if torch.cuda.is_available():
            mod = mod_tnr.cpu().data.numpy()
        else:
            mod = mod_tnr.data.numpy()
        # ==========
        # Derive the reduced feature input
        edge_map = rand_edge_map(edges, mod)
        #edge_map = rand_edge_map(edges, max_min_norm(mod))
        feat_coar_tnr = feat_coar_gpu(edge_map, mod_tnr, num_nodes, feat_dim, device) # Tensor of the reduced features
        # ==========
        time_end = time.time()
        feat_time = time_end-time_start # Runtime of feature extraction

        # =====================
        # Derive the (inductive) graph embedding
        time_start = time.time()
        _, emb, _ = gen_net(sup_topo_tnr, sup_topo_tnr, feat_coar_tnr, train_flag=False)
        time_end = time.time()
        prop_time = time_end-time_start # Runtime of feedforward propagation
        if torch.cuda.is_available():
            emb = emb.cpu().data.numpy()
        else:
            emb = emb.data.numpy()

        # ====================
        # Applied downstream kmeans algorithm to the graph embedding
        time_start = time.time()
        kmeans = KMeans(n_clusters=num_clus, n_init=10).fit(emb.astype(np.float64))
        clus_res = kmeans.labels_
        time_end = time.time()
        clus_time = time_end-time_start # Runtime of downstream clustering
        runtime = feat_time + prop_time + clus_time # Total runtime

        # =====================
        # Compute the quality metric of current partitioning result
        NMI = get_NMI(clus_res, labels_)
        NMI_list.append(NMI)

    # ====================
    NMI_mean = np.mean(NMI_list)
    NMI_std = np.std(NMI_list)
    time_mean = np.mean(time_list)
    time_std = np.std(time_list)
    print('#%d Val NMI %f (%f) Time %f (%f)'
          % (epoch, NMI_mean, NMI_std, time_mean, time_std))
    # ====================
    f_output = open('res/ICD-M_%s_val.txt' % (data_name), 'a+')
    f_output.write('#%d NMI %.8f %.8f Time %.8f %.8f\n'
            % (epoch, NMI_mean, NMI_std, time_mean, time_std))
    f_output.close()

    # ====================
    # Test the model
    # ==========
    if not test_eva_flag: # If test_eva_flag=True, conduct evaluation on test set for current epoch
        print()
        continue
    gen_net.eval()
    disc_net.eval()
    # ==========
    # Lists to record the quality metrics and runtime w.r.t. each test snapshot
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
        if min(gnd) == 0: # Number of clusters
            num_clus = int(max(gnd))+1
        else:
            num_clus = int(max(gnd))
        # ===========
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
        # Extract the neighbor-induced feature (i.e., modularity matrix)
        time_start = time.time()
        mod_tnr = get_mod_GPU(adj_tnr, num_nodes)
        if torch.cuda.is_available():
            mod = mod_tnr.cpu().data.numpy()
        else:
            mod = mod_tnr.data.numpy()
        # ==========
        # Derive the reduced feature input
        edge_map = rand_edge_map(edges, mod)
        #edge_map = rand_edge_map(edges, max_min_norm(mod))
        feat_coar_tnr = feat_coar_gpu(edge_map, mod_tnr, num_nodes, feat_dim, device) # Tensor of the reduced features
        # ==========
        time_end = time.time()
        feat_time = time_end-time_start # Runtime of feature extraction

        # ====================
        # Derive the (inductive) graph embedding
        time_start = time.time()
        _, emb, _ = gen_net(sup_topo_tnr, sup_topo_tnr, feat_coar_tnr, train_flag=False)
        time_end = time.time()
        prop_time = time_end-time_start # Runtime of feedforward propagation
        if torch.cuda.is_available():
            emb = emb.cpu().data.numpy()
        else:
            emb = emb.data.numpy()

        # ====================
        # Applied downstream kmeans algorithm to the graph embedding
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
        # ==========
        #mod_metric = get_mod_metric(adj, clus_res, num_clus)
        #ncut_metric = get_NCut_metric(adj, clus_res, num_clus)
        mod_metric = get_mod_metric_gpu(adj_tnr, clus_res, num_clus, device)
        ncut_metric = get_NCut_metric_gpu(adj_tnr, clus_res, num_clus, device)
        # ==========
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
    print('#%d Test NMI %f (%f) AC %f (%f) Mod %f (%f) NCut %f (%f) Time %f (%f)'
          % (epoch, NMI_mean, NMI_std, AC_mean, AC_std, mod_mean, mod_std, ncut_mean, ncut_std, time_mean, time_std))
    # ====================
    f_output = open('res/ICD-M_%s_test.txt' % (data_name), 'a+')
    f_output.write('#%d NMI %.8f %.8f AC %.8f %.8f Mod %.8f %.8f NCut %.8f %.8f Time %.8f %.8f\n'
            % (epoch, NMI_mean, NMI_std, AC_mean, AC_std, mod_mean, mod_std, ncut_mean, ncut_std, time_mean, time_std))
    f_output.close()
    print()
