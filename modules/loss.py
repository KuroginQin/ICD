import torch

# Functions to return the training loss of G & D

def get_gen_loss(fake_prob, feat_real, feat_rec, mem, alpha, beta):
    '''
    Function to return the loss of generative network G
    :param fake_prob: probability vector w.r.t. the learned embedding (given by D)
    :param feat_real: extracted neighbor-induced feature
    :param feat_rec: reconstructed neighbor-induced feature (from G)
    :param mem: partitioning membership indicator (i.e., 'ground-truth' label information)
    :param alaph, beta: hyper-parameters
    :return: loss of generative network G
    '''
    epsilon = 1e-15
    loss = -torch.mean(torch.log(fake_prob+epsilon)) # AL loss
    loss += alpha*torch.norm((feat_rec-feat_real), p='fro')**2 # FR loss
    loss -= beta*torch.trace(torch.mm(mem.t(), torch.mm(feat_rec, mem))) # CR loss

    return loss

def get_disc_loss(fake_prob, real_prob):
    '''
    Function to return the loss of discriminative network D
    :param fake_prob: probability vector w.r.t. the leaned embedding
    :param real_prob: probability vector w.r.t. the auxiliary label-induced embedding
    :return: loss of the discriminative network D
    '''
    epsilon = 1e-15 # 1e-3
    #loss = torch.mean(fake_prob) - torch.mean(real_prob)
    loss = torch.mean(torch.log(fake_prob+epsilon)) - torch.mean(torch.log(real_prob+epsilon))
    #loss = - torch.mean(torch.log(1-fake_prob+epsilon)) - torch.mean(torch.log(real_prob+epsilon))

    return loss

