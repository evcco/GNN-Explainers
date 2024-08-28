import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn import ModuleList, ReLU
from torch_geometric.nn import CGConv, APPNP, SAGEConv, GCNConv, GraphConv, TAGConv
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch.nn import Linear

from torch_geometric.utils import (negative_sampling, remove_self_loops, degree,
                                   add_self_loops, batched_negative_sampling)

def gen_mnist_attr(edge_index, pos):
    assert pos is not None
    max_value = 9

    (row, col) = edge_index

    cart = pos[col] - pos[row]
    cart = cart.view(-1, 1) if cart.dim() == 1 else cart
    cart = cart / (2 * max_value) + 0.5

    return cart


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
                ('lin1', nn.Linear(in_channels, hidden_channels)),
                ('act', act),
                ('lin2', nn.Linear(hidden_channels, out_channels))
                ]))
    def forward(self, x):
        return self.mlp(x)


class VGAE(torch.nn.Module):

    threshold = 0.55
    def __init__(self, in_channels, out_channels, e_feature, 
                num_units=2):
        super(VGAE, self).__init__()


        # encoder conv
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.convs = ModuleList()
        for i in range(num_units):
           self.convs.append(CGConv(out_channels, dim=e_feature))

        self.conv_mu = CGConv(out_channels, dim=e_feature)
        self.conv_log_var = CGConv(out_channels, dim=e_feature)

        self.e_fc1 = nn.Linear(4 * out_channels, out_channels)
        self.e_fc2 = nn.Linear(out_channels, e_feature)
        self.mlp = MLP(4 * out_channels, out_channels, 1, act=nn.ReLU())

        # conditional conv
        self.cond_convs = ModuleList()
        self.cond_lin = torch.nn.Linear(in_channels, out_channels)
        for i in range(num_units-1):
           self.cond_convs.append(CGConv(out_channels, dim=e_feature))
        self.last_conv = CGConv(out_channels, dim=e_feature)
        

    def conditional_rep(self, x, in_edge_index, in_edge_attr=None, reparameterize=True):

        x = F.relu(self.cond_lin(x))
        for conv in self.cond_convs:
            x = F.relu(conv(x, in_edge_index, in_edge_attr))
        x = self.last_conv(x, in_edge_index, in_edge_attr)
        return None, None, x

    def encode(self, x, in_edge_index, in_edge_attr=None, reparameterize=True):

        x = F.relu(self.lin(x))
        # generate the representation via num_unit-layer CGConv
        for conv in self.convs:
            x = F.relu(conv(x, in_edge_index, in_edge_attr))
        
        # generate the mean via 1-layer CGConv
        mu = self.conv_mu(x, in_edge_index, in_edge_attr)
        
        # generate the variance via 1-layer CGConv
        log_var = self.conv_log_var(x, in_edge_index, in_edge_attr)

        # reparameterize
        if self.training and reparameterize:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return mu, log_var, z

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(log_var / 10)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index, pos=None):
            
        row, col = edge_index
        z_normalized = z / z.norm(dim=1, keepdim=True)
        row_e, col_e = z_normalized[row], z_normalized[col]
    
        # 1. reconstruct the edge index
        e = torch.cat([row_e, col_e], dim=1)
        edge_prob = self.mlp(e).view(-1)

        edge_prob = torch.sigmoid(edge_prob)
        # 2. reconstruct the edge attribute
        if pos is None:
            # for one-hot edge attr
            edge_attr = F.relu(self.e_fc1(e))
            edge_attr = self.e_fc2(edge_attr).softmax(dim=1)

        else:
            # use pos feature as edge attr
            edge_attr = gen_mnist_attr(edge_index, pos)

        return edge_prob, edge_attr

    def decode_generate(self, z, top_edge_size, sigmoid=True):
        num_nodes = z.size()[0]
        # Calculate the fractional adjacency matrix
        adj_prob = torch.matmul(z, z.t())
        if sigmoid:
            adj_prob = torch.sigmoid(adj_prob)

        # Exclude the self-connection
        adj_prob.fill_diagonal_(-1)

        # Get the top k edges
        tmp = adj_prob.reshape(-1)
        top_edge_prob, top_edge_index = torch.topk(tmp, top_edge_size, sorted=True)
        top_edge_index = torch.cat([(top_edge_index / num_nodes), (top_edge_index % num_nodes)], dim=0)

        return top_edge_index, top_edge_prob

    def forward(self, z, out_edge_index, batch, pos=None, neg_edge_index=None):
        # Reconstruction: Return the target de_edge_index with the probability
        out_edge_prob, out_edge_attr = self.decode(z, out_edge_index, pos=pos)

        if neg_edge_index is not None:
            neg_edge_prob, neg_edge_attr = self.decode(z, neg_edge_index, pos=pos)
            fake_edge_index, fake_edge_prob, fake_edge_attr = \
                self.generate_new(out_edge_index, out_edge_prob, out_edge_attr,
                                  neg_edge_index, neg_edge_prob, neg_edge_attr, batch)
            return out_edge_prob, neg_edge_prob, fake_edge_index, fake_edge_prob, fake_edge_attr

        return out_edge_prob, None, None, None

    def fill(self, z, preserved_edge_index, 
            preserved_edge_ratio, batch, pos=None, 
            neg_edge_index=None, untouched=False, threshold=False):

        preserved_edge_prob, preserved_edge_attr = self.decode(z, preserved_edge_index, pos=pos)
        preserved_nodes = torch.unique(preserved_edge_index).tolist()
        
        split = degree(batch[preserved_edge_index[0]], dtype=torch.long).tolist()
        edge_indices = torch.split(preserved_edge_index, split, dim=1)
        num_edges = [e.size(1) for e in edge_indices]
        num_nodes = degree(batch, dtype=torch.long)
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])

        fake_edge_probs, fake_edge_indices, fake_edge_attrs = [], [], []
        fake_edge_probs.append(preserved_edge_prob)
        fake_edge_indices.append(preserved_edge_index)
        fake_edge_attrs.append(preserved_edge_attr)
        for edge_index, N, C, E, R in zip(edge_indices, num_nodes.tolist(),
                                    cum_nodes.tolist(), num_edges, preserved_edge_ratio.tolist()):
            if E == 0:
                continue
            num_edge_fill = int(E / R  - E)
            num_neg_samples = int(min(N * N - E, 3*num_edge_fill))
            neg_edge_index = negative_sampling(edge_index - C, N, num_neg_samples) + C
            neg_edge_index, _ = remove_self_loops(neg_edge_index)
            neg_edge_prob, neg_edge_attr = self.decode(z, neg_edge_index, pos=pos)
            if untouched:
                edge_filter = [False if (neg_edge_index[0][i] in preserved_nodes) and \
                     (neg_edge_index[1][i] in preserved_nodes) \
                         else True for i in range(neg_edge_index.size(1))]
                neg_edge_prob = neg_edge_prob[edge_filter]
                neg_edge_index = neg_edge_index[:, edge_filter]
                neg_edge_attr = neg_edge_attr[edge_filter]

            if threshold:
                top_index = (neg_edge_prob > self.threshold)#top_index: bool type
                neg_top_edge_prob = neg_edge_prob[top_index]
            else:
                num_edge_fill = neg_edge_prob.size(0) if neg_edge_prob.size(0) < num_edge_fill else num_edge_fill
                neg_top_edge_prob, top_index = torch.topk(neg_edge_prob, num_edge_fill, sorted=False)
            fake_edge_indices.append(neg_edge_index[:, top_index])
            fake_edge_probs.append(neg_top_edge_prob)
            fake_edge_attrs.append(neg_edge_attr[top_index])
        fake_edge_indices = torch.cat(fake_edge_indices, dim=1)   
        fake_edge_probs = torch.cat(fake_edge_probs, dim=0)
        fake_edge_attrs = torch.cat(fake_edge_attrs, dim=0)
        
        return fake_edge_indices, fake_edge_probs, fake_edge_attrs, neg_edge_prob

    def get_contrastive_loss(self, z, y, batch, tau=0.1):
        
        c = global_mean_pool(z, batch)
        c_normalized = c / c.norm(dim=1, keepdim=True)
        mat = F.relu(torch.mm(c_normalized, c_normalized.T))
        unique_graphs = torch.unique(batch)

        # mat = torch.exp(mat / tau)
        ttl_scores = torch.sum(mat, dim=1) # size = (num_graphs)
        pos_scores = torch.tensor([mat[i, y == y[i]].sum() for i in unique_graphs]).to(z.device)
        neg_scores = ttl_scores - pos_scores
        
        # contrastive_loss = - torch.log(torch.sum(pos_scores / ttl_scores, dim=0))
        # contrastive_loss = - torch.logsumexp(pos_scores / (tau * ttl_scores), dim=0)
        # contrastive_loss = - torch.logsumexp((pos_scores - ttl_scores) / tau, dim=0)
        contrastive_loss = - torch.logsumexp((pos_scores - neg_scores) / tau, dim=0)
        
        return contrastive_loss


    def generate_new(self, out_edge_index, out_edge_prob, out_edge_attr,
                     neg_edge_index, neg_edge_prob, neg_edge_attr, batch, threshold=False):
        # Generation: Return the edge index with top edge probabilities.
        tmp_prob = torch.cat([out_edge_prob, neg_edge_prob], dim=0)
        tmp_index = torch.cat([out_edge_index, neg_edge_index], dim=1)
        tmp_attr = torch.cat([out_edge_attr, neg_edge_attr], dim=0)

        if threshold:
            select_flag = (tmp_prob > self.threshold)
            fake_edge_prob = tmp_prob[select_flag]
            fake_edge_index = (tmp_index.T[select_flag]).T
            fake_edge_attr = tmp_attr[select_flag]
        else:
            fake_edge_prob, topK_index = torch.topk(tmp_prob, out_edge_prob.size()[0], sorted=False)
            fake_edge_attr = tmp_attr[topK_index]
            fake_edge_index = (tmp_index.T[topK_index]).T

        return fake_edge_index, fake_edge_prob, fake_edge_attr

    @staticmethod
    def get_link_labels(pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def get_reconstruct_loss_1(self, z, edge_index, edge_attr, batch, pos, tau=0.1):

        pos_edge_index = edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, num_nodes=z.size(0),
            num_neg_samples=pos_edge_index.size(1))
        
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
        edge_prob, edge_attr = self.decode(z, edge_index, pos=pos)

        reconstruct_loss = F.binary_cross_entropy_with_logits(edge_prob, edge_labels)
        
        return reconstruct_loss

    def get_reconstruct_loss_2(self, z, edge_index, edge_attr, batch, pos, tau=0.1):

        row, col = edge_index
        z = z / z.norm(dim=1, keepdim=True)

        # 1. get the positive scores of every node
        unique_nodes, remap_row = torch.unique(row, return_inverse=True)
        edge_prob, decode_edge_attr = self.decode(z, edge_index, pos=pos)
        pos_scores = torch.zeros(len(unique_nodes)).to(z.device)
        pos_scores.scatter_add_(0, remap_row, edge_prob)

        # 1. get the total scores of every node
        # .... sum all scores related to each node up
        ttl_scores = torch.sum(F.relu(torch.mm(z, z.T)), dim=1)  # size = (num_nodes)
        ttl_scores = ttl_scores[unique_nodes]

        # 2. scale the positive & total scores via temperature
        # 3. calculate the contrastive learning loss
        reconstruct_loss = -torch.logsumexp((pos_scores - ttl_scores) / tau, dim=0)

        # criterion = nn.MSELoss(reduction='sum')
        # attr_loss = criterion(decode_edge_attr, edge_attr)

        return (reconstruct_loss) / len(batch.unique())

    @staticmethod
    def get_kl_div(mu, log_var):
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_div
