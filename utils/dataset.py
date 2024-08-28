import torch
import numpy as np

from gnn.mutag_gnn import MutagNet
from gnn.tox21_gnn import Tox21Net
from gnn.ba2motif_gnn import BA2MotifNet
from gnn.ba3motif_gnn import BA3MotifNet
from gnn.reddit5k_gnn import Reddit5kNet
from gnn.tr3motif_gnn import TR3MotifNet
from gnn.vtr3motif_gnn import VTR3MotifNet

import torch_geometric.transforms as T
from datasets import *
from torch_geometric.datasets import MNISTSuperpixels

from torch_geometric.data import Dataset

class MNISTTransform(object):
    
    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[col] - pos[row]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            max_value = cart.abs().max() if self.max is None else self.max
            cart = cart / (2 * max_value) + 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart
            
        row, col = data.edge_index
        data.ground_truth_mask = (data.x[row] > 0).view(-1).bool() * (data.x[col] > 0).view(-1).bool()
        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)


def get_datasets(args):
    if args.dataset == "mutag":
        folder = 'data/MUTAG'
        train_dataset = Mutagenicity(folder, mode='training')
        test_dataset = Mutagenicity(folder, mode='testing')
        val_dataset = Mutagenicity(folder, mode='evaluation')
    elif args.dataset == "reddit5k":
        folder = 'data/Reddit5k'
        train_dataset = Reddit5k(folder, mode='training')
        test_dataset = Reddit5k(folder, mode='testing')
        val_dataset = Reddit5k(folder, mode='evaluation')
    elif args.dataset == "ba3":
        folder = 'data/BA3'
        train_dataset = BA3Motif(folder, mode='training')
        test_dataset = BA3Motif(folder, mode='testing')
        val_dataset = BA3Motif(folder, mode='evaluation')
    elif args.dataset == "tr3":
        folder = 'data/TR3'
        train_dataset = TR3Motif(folder, mode='training')
        test_dataset = TR3Motif(folder, mode='testing')
        val_dataset = TR3Motif(folder, mode='evaluation')
    elif args.dataset == "vtr3":
        folder = 'data/VTR3'
        train_dataset = VTR3Motif(folder, mode='training')
        test_dataset = VTR3Motif(folder, mode='testing')
        val_dataset = VTR3Motif(folder, mode='evaluation')
    elif args.dataset == "mnist":
        data_path = 'data/MNIST'
        transform = MNISTTransform(cat=False, max_value=9)#T.Cartesian(cat=False, max_value=9)
        train_dataset = MNISTSuperpixels(data_path, True, transform=transform)
        test_dataset = MNISTSuperpixels(data_path, False, transform=transform)
        # Reduced dataset
        train_dataset = train_dataset[:6000]
        val_dataset = test_dataset[1000:2000]
        test_dataset = test_dataset[:1000]
    elif args.dataset == 'graphsst2':
        import sys
        sys.path.append('..')
        from dataset.graphss2_dataset import get_dataset
        train_dataset = get_dataset(dataset_dir='data/', dataset_name='Graph_SST2', task=None)
        val_dataset = None; test_dataset = None
    else:
        raise ValueError


    return train_dataset, val_dataset, test_dataset
