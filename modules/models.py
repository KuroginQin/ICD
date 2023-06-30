from .layers import *

class GenNet(nn.Module):
    '''
    Class to define the generative network
    '''
    def __init__(self, EN_dims, dropout_rate, device):
        super(GenNet, self).__init__()
        # ====================
        self.device = device
        # =========
        self.EN_dims = EN_dims # Layer configurations of the encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.num_EN_layers = len(EN_dims)-1 # Number of encoder layers
        self.GNNs = nn.ModuleList()
        for l in range(self.num_EN_layers):
            self.GNNs.append(GraphNeuralNetwork(self.EN_dims[l], self.EN_dims[l+1],
                                                dropout_rate=self.dropout_rate, device=self.device))

    def forward(self, sup_topo, sup_gnd, feat, train_flag=False):
        '''
        Rewrite the forward function
        :param feat: extracted feature input
        :param sup_topo: sparse support (normalized adjacency matrix) of original graph
        :param sup_gnd: sparse support (normalized adjacency matrix) of auxiliary label-induced graph
        :param train_flag: flag for training model (=true), i.e., whether to derive auxiliary label-induced embedding
        :return: reconstructed neighbor-induced feature, learned embedding, & auxiliary label-induced embedding
        '''
        # ====================
        # The encoder
        feat_topo = feat # Feature input of original graph
        feat_gnd = feat # Feature input of auxiliary label-induced graph
        output_topo = None
        output_gnd = None
        for GNN in self.GNNs:
            output_topo, output_gnd = GNN(feat_topo, sup_topo, feat_gnd, sup_gnd, train_flag)
            feat_topo = output_topo
            feat_gnd = output_gnd
        emb = output_topo # Learned embedding
        emb_gnd = output_gnd # Auxiliary label-induced embedding
        # ==========
        # The decoder
        feat_rec = torch.tanh(torch.mm(emb, emb.t())) # Reconstruct the neighbor-induced feature

        return feat_rec, emb, emb_gnd

class DiscNet(nn.Module):
    '''
    Class to define the discriminative network
    '''
    def __init__(self, disc_dims, dropout_rate, device):
        super(DiscNet, self).__init__()
        # ====================
        self.device = device
        # ==========
        self.disc_dims = disc_dims # Layer configurations of the discriminative network
        self.dropout_rate = dropout_rate  # Dropout rate
        # ==========
        self.num_disc_layers = len(disc_dims)-1 # Number of full-connected layers
        # =====================
        # Initialize the model parameters
        self.weis = nn.ParameterList()
        self.biass = nn.ParameterList()
        self.dropout_layers = nn.ModuleList()
        for l in range(self.num_disc_layers):
            wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(self.disc_dims[l], self.disc_dims[l+1])))
            self.weis.append(wei)
            bias = Parameter(torch.zeros(self.disc_dims[l+1]))
            self.biass.append(bias)
            self.dropout_layers.append(nn.Dropout(p=self.dropout_rate))
        # ==========
        self.weis.to(self.device)
        self.biass.to(self.device)

    def forward(self, emb):
        '''
        Rewrite the forward function
        :param emb: embedding input
        :return: probability vector
        '''
        input = emb
        for l in range(self.num_disc_layers-1):
            output = F.relu(torch.mm(input, self.weis[l])+self.biass[l])
            output = self.dropout_layers[l](output)
            input = output
        output = torch.sigmoid(torch.mm(input, self.weis[-1])+self.biass[-1])
        output = self.dropout_layers[-1](output)
        # ==========
        #output = F.relu(torch.mm(input, self.weis[-1]) + self.biass[-1])
        #output = torch.sigmoid(output)

        return output
