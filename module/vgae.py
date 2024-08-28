import os.path as osp

import argparse
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, CGConv
from torch_geometric.utils import train_test_split_edges


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, e_feature):
        super(VariationalGCNEncoder, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.conv1 = CGConv(out_channels, dim=e_feature) 
        self.conv_mu = CGConv(out_channels, dim=e_feature) 
        self.conv_logstd = CGConv(out_channels, dim=e_feature) 

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        x = self.conv1(x, edge_index, edge_attr).relu()
        return self.conv_mu(x, edge_index, edge_attr), self.conv_logstd(x, edge_index, edge_attr)
