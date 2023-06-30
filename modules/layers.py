import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphNeuralNetwork(Module):
    '''
    Class to define a single GNN layer
    '''

    def __init__(self, input_dim, output_dim, dropout_rate, device):
        super(GraphNeuralNetwork, self).__init__()
        # ====================
        self.device = device
        # ====================
        self.input_dim = input_dim # Dimensionality of the input features
        self.output_dim = output_dim # Dimensionality of the output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize the model parameters via the Xavier algorithm
        self.agg_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(input_dim, output_dim))) # Aggregation weight matrix
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        self.param.to(self.device)
        # ==========
        self.topo_dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat_topo, sup_topo, feat_gnd, sup_gnd, train_flag=False):
        '''
        Rewrite the forward function
        :param feat_topo: feature input of original graph
        :param sup_topo: sparse support (normalized adjacency matrix) of original graph
        :param feat_gnd: feature input of auxiliary label-induced graph
        :param sup_gnd: sparse support (normalized adjacency matrix) of auxiliary label-induced graph
        :param train_flag: flag for training model (=true), i.e., whether to derive auxiliary label-induced embedding
        :return: learned embedding & auxiliary label-induced embedding
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        feat_topo_agg = torch.spmm(sup_topo, feat_topo)
        agg_topo_output = torch.tanh(torch.mm(feat_topo_agg, self.param[0]))
        agg_topo_output = F.normalize(agg_topo_output, dim=1, p=2)
        agg_topo_output = self.topo_dropout_layer(agg_topo_output)
        # ==========
        agg_gnd_output = None
        if train_flag: # For the training model, derive auxiliary label-induced embedding
            feat_gnd_agg = torch.spmm(sup_gnd, feat_gnd)
            agg_gnd_output = torch.tanh(torch.mm(feat_gnd_agg, self.param[0]))
            agg_gnd_output = F.normalize(agg_gnd_output, dim=1, p=2)  # l2-normalization

        return agg_topo_output, agg_gnd_output