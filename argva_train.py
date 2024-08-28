# adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py
import argparse
import os.path as osp

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ARGVA
from torch_geometric.data import DataLoader
from torch_geometric.utils import batched_negative_sampling

from utils import *
from utils.dataset import *
from utils.logger import Logger
from utils.parser import args_print
from utils.get_subgraph import bool_vec
from datasets.graphss2_dataset import get_dataloader  

n_classes_dict = {'mutag': 2, 'mnist': 10, 'ba3': 3, 'tr3': 3, 'graphsst2': 2}


def parse_args():
    parser = argparse.ArgumentParser(description="ARGVA Training.")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='param/argva/',
                        help='path to save model.')
    parser.add_argument('--dataset', type=str, default='tr3',
                        help='One of [tr3, mnist, graphsst2]')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--masked_ratio', type=float, default=0.5,
                        help='Create broken graphs with some edges masked.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Device NO. of cuda.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--out_channels', type=int, default=64,
                        help='Out channels for CG.')

    return parser.parse_args()


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)

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

    encoder = Encoder(n_feature, hidden_channels=64, out_channels=args.out_channels).to(device)
    discriminator = Discriminator(in_channels=args.out_channels, hidden_channels=args.out_channels,
                                out_channels=args.out_channels)
    model = ARGVA(encoder, discriminator).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                            lr=args.lr)
        
    def train(data):
        model.train()
        encoder_optimizer.zero_grad()
        idx = bool_vec(length=g.num_edges, r_True=edge_ratio)
        train_pos_edge_index = g.edge_index[:, idx]
        z = model.encode(data.x, train_pos_edge_index)
        neg_edge_index = batched_negative_sampling(
            edge_index=g.edge_index, 
            batch=g.batch
            )
        # We optimize the discriminator more frequently than the encoder.
        for _ in range(5):
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        loss = model.recon_loss(z, train_pos_edge_index, neg_edge_index)
        loss = loss + model.reg_loss(z)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        encoder_optimizer.step()
        return float(loss)


    def test(g):
        model.eval()
        neg_edge_index = batched_negative_sampling(
            edge_index=g.edge_index, 
            batch=g.batch
            )
        with torch.no_grad():
            z = model.encode(g.x, g.edge_index)
        return model.test(z, g.edge_index, neg_edge_index)

    for epoch in range(args.epoch):
        all_loss, all_auc, all_ap = [], [], []
        for i, g in enumerate(train_loader):
            g = g.to(device)
            loss = train(g)
            auc, ap = test(g)
    
            all_loss.append(loss)
            all_auc.append(auc)
            all_ap.append(ap)
        loss = np.array(all_loss).mean()
        auc = np.array(all_auc).mean()
        ap = np.array(all_ap).mean()
        logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, AUC: {auc:.3f}, '
           f'AP: {ap:.3f}'
           )

    torch.save(model.cpu(), osp.join(args.model_path, '%s.pt' % args.dataset))





