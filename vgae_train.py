

import torch
import argparse
import os.path as osp
from torch_geometric.data import DataLoader
from dataset.graphss2_dataset import get_dataset, get_dataloader  

from utils import *
from utils.dataset import *
from utils.logger import Logger
from utils.parser import args_print


from module.vgae import VariationalGCNEncoder
from torch_geometric.data import Data, Batch
from datasets.graphss2_dataset import get_dataset, get_dataloader  
from torch_geometric.utils import train_test_split_edges
from utils.get_subgraph import bool_vec
from module.vgae import *
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops, batched_negative_sampling)

n_classes_dict = {'mutag': 2, 'mnist': 10, 'ba3': 3, 'tr3': 3, 'graphsst2': 2}

def parse_args():
    parser = argparse.ArgumentParser(description="VGAE Training.")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='param/vgae/',
                        help='path to save model.')
    parser.add_argument('--dataset', type=str, default='tr3',
                        help='One of [tr3, mnist, graphsst2]')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--masked_ratio', type=float, default=0.5,
                        help='Create broken graphs with some edges masked.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Device NO. of cuda.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--out_channels', type=int, default=128,
                        help='Out channels for CG.')

    return parser.parse_args()


if __name__ == '__main__':

    # set parameters
    args = parse_args()
    edge_ratio = 1 - args.masked_ratio
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    logger = Logger.init_logger(filename="log/vgae_%s" % args.dataset)
    args_print(args, logger)

    # ================================================================== #
    #                         Load Datasets                              #
    # ================================================================== #
    set_seed(19930819)
    print('loading dataset...')
    datasets = []
    dataset_name = args.dataset
    n_classes = n_classes_dict[dataset_name]
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    out_channels = args.out_channels
    n_feature = train_dataset[0].num_node_features
    e_feature = train_dataset[0].num_edge_features
    if args.dataset == 'graphsst2':
        train_dataset.shuffle()
        dataloader = get_dataloader(train_dataset,  # data_loader: dict, following the structure {'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset}
                                    batch_size=args.batch_size,                 # batch_size: int
                                    random_split_flag=True,                     # random_split_flag：bool, True when randomly split the dataset into training, deving and testing datasets.
                                    data_split_ratio=[0.8, 0.1, 0.1],           # data_split_ratio: list, the ratio of data in training, deving and testing datasets.
                                    seed=2)    
        train_loader = dataloader['train']
        val_loader = dataloader['eval'] 
        test_loader = dataloader['test']
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print('Load # graphs %5d, %5d, %5d ' % (len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))

    # ================================================================== #
    #                        Initialize Model                            #
    # ================================================================== #
    model = VGAE(VariationalGCNEncoder(n_feature, out_channels, e_feature))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    gnn_path = 'param/gnns/%s_net.pt' % dataset_name

    def train(g):
        model.train()
        optimizer.zero_grad()
        idx = bool_vec(length=g.num_edges, r_True=edge_ratio)
        train_pos_edge_index = g.edge_index[:, idx]
        z = model.encode(g.x, train_pos_edge_index, g.edge_attr[idx])
        neg_edge_index = batched_negative_sampling(
            edge_index=g.edge_index, 
            batch=g.batch
            )
        loss = model.recon_loss(z, train_pos_edge_index, neg_edge_index)
        loss = loss + (1 / g.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)


    def test(g):
        model.eval()
        neg_edge_index = batched_negative_sampling(
            edge_index=g.edge_index, 
            batch=g.batch
            )
        with torch.no_grad():
            z = model.encode(g.x, g.edge_index, g.edge_attr)
        return model.test(z, g.edge_index, neg_edge_index)


    for epoch in range(args.epoch):
        all_loss, all_auc, all_ap = [], [], []
        for i, g in enumerate(train_loader):
            loss = train(g)
            auc, ap = test(g)
            all_loss.append(loss)
            all_auc.append(auc); all_ap.append(ap)
        loss = np.array(all_loss).mean()
        auc = np.array(all_auc).mean()
        ap = np.array(all_ap).mean()
        logger.info('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))

    torch.save(model.cpu(), osp.join(args.model_path, '%s.pt' % args.dataset))




