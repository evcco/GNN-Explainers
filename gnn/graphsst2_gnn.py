import argparse
import os.path as osp
import random
import time
import sys
import torch
from functools import wraps
from torch.nn import Linear, ELU, ModuleList, Softmax, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import ARMAConv, BatchNorm, MessagePassing, global_mean_pool ,global_max_pool
from torch_geometric.data import DataLoader
from collections import OrderedDict
import torch.nn.functional as F
from .overloader import overload
import sys
sys.path.append('..')
from utils import set_seed, Gtrain, Gtest
from datasets.graphss2_dataset import get_dataset, get_dataloader  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Graph-SST2 Model")
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default= 1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    return parser.parse_args()


class GraphSST2Net(torch.nn.Module):
    def __init__(self):
        super(GraphSST2Net, self).__init__()

        self.conv1 = ARMAConv(768, 128)
        self.conv2 = ARMAConv(128, 128)
        self.to_logit = torch.nn.Sequential(OrderedDict([
                ('lin1', Linear(128, 32)),
                ('elu',  ELU()),
                ('lin2', Linear(32, 2)),
                ]))
                
    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        node_x = self.conv2(x, edge_index, edge_weight)
        return node_x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.to_logit(graph_x)
        self.readout = pred.softmax(dim=1)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


if __name__ == '__main__':

    args = parse_args()
    set_seed(0)
    dataset = get_dataset(dataset_dir='../data/', dataset_name='Graph_SST2', task=None)
    dataloader = get_dataloader(dataset,                           # data_loader: dict, following the structure {'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset}
                                batch_size=args.batch_size,                     # batch_size: int
                                random_split_flag=True,            # random_split_flag：bool, True when randomly split the dataset into training, deving and testing datasets.
                                data_split_ratio=[0.8, 0.1, 0.1],  # data_split_ratio: list, the ratio of data in training, deving and testing datasets.
                                seed=2)    
    train_loader = dataloader['train']
    val_loader = dataloader['eval'] 
    test_loader = dataloader['test']

    model = GraphSST2Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=5,
                                  min_lr=1e-4
                                  )
    min_error = None
    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']

        loss = Gtrain(train_loader,
                      model,
                      optimizer,
                      criterion=CrossEntropyLoss()
                      )

        _, train_acc = Gtest(train_loader,
                             model,
                             criterion=CrossEntropyLoss()
                             )

        val_error, val_acc = Gtest(val_loader,
                                   model,
                                   criterion=CrossEntropyLoss()
                                   )
        test_error, test_acc = Gtest(test_loader,
                                     model,
                                     criterion=CrossEntropyLoss()
                                     )
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader,
                                         model,
                                         criterion=CrossEntropyLoss()
                                         )
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, '
                  'Test acc: {:.5f}'.format(epoch, t3 - t1, lr, loss, test_error, test_acc))
            continue

        print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, '
              'Validation acc: {:5f}'.format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc))

        torch.cuda.empty_cache()
    save_path = 'graphsst2_net.pt'
    torch.save(model, osp.join(args.model_path, save_path))

