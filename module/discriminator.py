import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import ModuleList, LeakyReLU
from torch_geometric.utils import degree
from torch_geometric.nn import CGConv, TransformerConv
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Linear


class DisGNN(nn.Module):
    def __init__(self, in_channels, out_channels, e_feature, n_classes):
        super(DisGNN, self).__init__()

        self.lin = nn.Linear(in_channels + n_classes, out_channels)
        self.conv1 = CGConv(out_channels, dim=e_feature)
        self.conv2 = CGConv(out_channels, dim=e_feature)
        self.fc = nn.Sequential(
            nn.Linear(out_channels + n_classes, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.n_classes = n_classes

    def __set_masks__(self, mask):

        for module in self.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = mask

    def __clear_masks__(self):

        for module in self.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def forward(self, x, y, edge_index, edge_attr, batch=None, edge_mask=None, tau=0.5):
        
        one_hot_label = torch.zeros(x.size(0), self.n_classes).to(x.device)
        num_nodes = degree(batch, dtype=torch.long)
        one_hot_label[range(x.size(0)), y[batch]] = 1.
        x = torch.cat([x, one_hot_label], dim=1)

        x = F.leaky_relu(self.lin(x))
        if edge_mask is not None:
            self.__set_masks__(edge_mask/tau)

        if batch is None:
            batch = torch.zeros(x.size(0)).long().to(x.device)
        
        new_x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        new_x = self.conv2(new_x, edge_index, edge_attr)

        new_x = torch.cat([new_x, one_hot_label], dim=1)
        new_x = global_mean_pool(new_x, batch)

        out = self.fc(new_x)
        self.__clear_masks__()

        return out.sigmoid()


