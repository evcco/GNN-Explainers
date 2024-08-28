from tqdm import tqdm
from pathlib import Path
from utils import *
from utils.saver import *
from utils.dataset import *
from utils.get_subgraph import *
from utils.logger import Logger

import os, os.path as osp
import torch.autograd as autograd
from torch.autograd import Variable
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

from module.generator2 import VGAE
from module.discriminator import DisGNN
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops, batched_negative_sampling)
from torch_geometric.data import Data, Batch
from datasets.graphss2_dataset import get_dataset, get_dataloader  

from gnn.tr3motif_gnn import TR3MotifNet
from gnn.vtr3motif_gnn import VTR3MotifNet
from gnn.mnist_gnn import MNISTNet
from gnn.graphsst2_gnn import GraphSST2Net
from module.gmm import GaussianMixture

warnings.filterwarnings("ignore")

n_classes_dict = {'mutag': 2, 'mnist': 10, 'ba3': 3, 'tr3': 3, 'graphsst2': 2}
Gd_dict = {'mutag': False, 'mnist': True, 'ba3': False, 'tr3': True, 'graphsst2': False}

last_ID_perf = -1e6
MAX_DIAM = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Counterfactual Generation via Adversarial Training.")
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='param/',
                        help='path to save model.')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment Name.')
    parser.add_argument('--dataset', type=str, default='tr3',
                        choices=['tr3', 'mnist', 'graphsst2', 'mutag', 'vg'])
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Device NO. of cuda.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate.')
    parser.add_argument('--val_loop', type=int, default=1,
                        help='Number of loops for testing.')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Temperature.')
    parser.add_argument('--out_channels', type=int, default=128,
                        help='Out channels for CG.')
    parser.add_argument('--masked_ratio', type=float, default=0.3,
                        help='Create broken graphs with some edges masked.')
    parser.add_argument('--regs', nargs='?', default='[3.,5.,3.,1e-4]',
                        help='Hyper-parameters: reg_rec, reg_penalty, reg_cts, reg_kl')
    parser.add_argument('--reconstruction_type', type=str, default='BCE',
                        help='Reconstrction Type for Generator Training',
                        choices=['BCE', 'CTS'])

    return parser.parse_args()


def reset_grad(G_optimizer, D_optimizer_y):
    G_optimizer.zero_grad()
    D_optimizer_y.zero_grad()


def get_gradient_penalty(D, g, pos_edge_prob, 
                        fake_edge_index, fake_edge_attr, fake_edge_prob, 
                        divide=0.5):
    alpha1 = (torch.rand(g.edge_index.size(1)) > divide).bool()
    alpha2 = (torch.rand(fake_edge_index.size(1)) < divide).bool()
    mix_edge_index = torch.cat([g.edge_index[:, alpha1], fake_edge_index[:, alpha2]], dim=1)
    mix_edge_prob = torch.cat([pos_edge_prob[alpha1], fake_edge_prob[alpha2]], dim=0)
    mix_edge_attr = torch.cat([g.edge_attr[alpha1], fake_edge_attr[alpha2]], dim=0)
    D_grad = D(g.x, g.y, mix_edge_index, mix_edge_attr, g.batch, mix_edge_prob)[0, 0]
    gradients = autograd.grad(outputs=D_grad, inputs=mix_edge_prob, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    D_penalty = ((gradients.norm(2) - 1) ** 2).mean()
    return D_penalty


if __name__ == '__main__':
    
    args = parse_args()
    flag = 1 if args.dataset in ['mnist'] else 0
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    reparameterize = 1 - flag
    ground_truth_avaliable =  Gd_dict[args.dataset] 

    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    logger = Logger.init_logger(filename="log/%s" % args.experiment_name)
    args_print(args, logger)
    
    # set parameters
    edge_ratio = 1 - args.masked_ratio
    ckpt_dir = Path(args.model_path) / 'cg' / f'{args.experiment_name}' 
    G_folder = ckpt_dir / 'generator'
    D_folder = ckpt_dir / 'discriminator'

    # a list of regularization:
    # .... loss of generator, min reconstruct_loss + reg2 * kl_div + reg1 * (- disc_fake_loss - reg3 * disc_penalty)
    # .... loss of discriminator, min -( disc_real_loss - dis_fake_loss - reg3 * disc_penalty)
    regs = eval(args.regs)
    reg_rec, reg_penalty, reg_cts, reg_kl = regs

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
        dataloader = get_dataloader(train_dataset,
                                    batch_size=args.batch_size,
                                    random_split_flag=True, 
                                    data_split_ratio=[0.8, 0.1, 0.1],   
                                    seed=2)    
        train_loader = dataloader['train']
        val_loader = dataloader['eval'] 
        test_loader = dataloader['test']
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print('Load # graphs %5d, %5d, %5d ' % (len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    # ================================================================== #
    #                        Initialize Model                            #
    # ================================================================== #
    print('initializing models...')
    G = VGAE(in_channels=n_feature, out_channels=out_channels, e_feature=e_feature).to(device)
    D = DisGNN(in_channels=n_feature, out_channels=out_channels, e_feature=e_feature, n_classes=n_classes).to(device)
    
    # lr=2e-4 for TR
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, weight_decay=1e-5)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, weight_decay=1e-5)

    
    G_scheduler = ReduceLROnPlateau(G_optimizer, mode='max', 
                                    factor=0.5, patience=10,
                                    min_lr=1e-6)
    D_scheduler = ReduceLROnPlateau(D_optimizer, mode='min', 
                                    factor=0.5, patience=10,
                                    min_lr=1e-6)
    criterion = nn.BCELoss()

    # ================================================================== #
    #             Load Pretrained GMM and GNN for Testing                #
    # ================================================================== #

    gnn_path = 'param/gnns/%s_net.pt' % dataset_name
    emb_path = 'param/emb/%s_graph_emb.pt' % dataset_name
    gmm_path = 'param/gmm/%s.pt' % dataset_name

    gnn = torch.load(gnn_path).to(device)
    gnn.eval()
    if not ground_truth_avaliable:
        if osp.exists(gmm_path):
            gmm = torch.load(gmm_path).to(device)
        else:
            save_train_emb(emb_path, gnn, train_loader.dataset, device)
            training_emb = torch.load(emb_path)[0]
            training_emb = torch.stack(training_emb).to(device)
            gmm = GaussianMixture(n_components=n_classes_dict[args.dataset]+1, n_features=training_emb.size(1)).to(device)
            gmm.fit(training_emb, n_iter=1000)
            gmm.eval()
            torch.save(gmm, gmm_path)
    print("begin training...")
    for epoch in range(args.epoch):
        D.train()
        G.train()
        # Initialize Loggers
        logger_neg_prob = []
        logger_D_loss, logger_G_loss = [], []
        logger_D_perf, logger_G_perf = [], []
        logger_real_score, logger_fake_score = [], []
        logger_full_likelihood, logger_filled_likelihood = [], []

        G_lr = G_scheduler.optimizer.param_groups[0]['lr']
        D_lr = D_scheduler.optimizer.param_groups[0]['lr']
        tmp = torch.FloatTensor([]).to(device)
        for i, g in enumerate(train_loader):
            g.to(device)
            # full_emb = gnn.get_graph_rep(g.x, g.edge_index, g.edge_attr, g.batch)
            # train_log_likelihood = gmm.score_samples(full_emb)
            # tmp = torch.cat([train_log_likelihood, tmp])

            pos = g.pos if flag else None
            real_labels = torch.ones(len(g.y), 1).to(device)
            fake_labels = torch.zeros(len(g.y), 1).to(device)

            # ================================================================== #
            #           Generate Fake Graphs based on Full Graphs                #
            # ================================================================== #
            # # 1. Encode the broken input graphs & treat all edges as positive edges
            broken_edge_index, broken_edge_attr, out_edge_ratio = get_broken_graph(g, edge_ratio, connectivity=False)
            mu, log_var, z = G.encode(
                x=g.x, in_edge_index=broken_edge_index, 
                in_edge_attr=broken_edge_attr, reparameterize=reparameterize
                )
            _, _, cond_z = G.encode(
                x=g.x, in_edge_index=g.edge_index, 
                in_edge_attr=g.edge_attr, reparameterize=reparameterize
                )
            z = torch.cat([z, cond_z], dim=1)#torch.cat([z, cond_z, cond_z-z], dim=1)
            # mu, log_var, z = G.encode(x=g.x, in_edge_index=g.edge_index, in_edge_attr=g.edge_attr)

            # # 2. get possibility for positive edges
            pos_edge_prob, _ = G.decode(z, g.edge_index, pos=pos)

            # 3. Generate fake graphs by merging positive edges (i.e., g.edge_index) & negative edges
            # .... neg_edge_prob for neg_edge_index
            fake_edge_index, fake_edge_prob, fake_edge_attr, _ = \
                G.fill(
                    z=z, preserved_edge_index=broken_edge_index, preserved_edge_ratio=out_edge_ratio,
                    batch=g.batch, pos=pos, neg_edge_index=None, threshold=False
                    )
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #
            # 1. Compute BCE_Loss using real graphs where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # .... Second term of the loss is always zero since real_labels == 1
            reset_grad(G_optimizer, D_optimizer)
            real_outputs = D(g.x, g.y, g.edge_index, g.edge_attr, g.batch, pos_edge_prob)
            D_loss_real = criterion(real_outputs, real_labels)
            logger_real_score.extend(real_outputs.view(-1).detach())

            # 2. Compute BCELoss using fake graphs
            # .... First term of the loss is always zero since fake_labels == 0
            fake_outputs = D(g.x, g.y, fake_edge_index, fake_edge_attr, g.batch, fake_edge_prob)
            D_loss_fake = criterion(fake_outputs, fake_labels)

            logger_fake_score.extend(fake_outputs.view(-1).detach())

            # 3. Compute penalty on gradients
            D_penalty = get_gradient_penalty(D, g, pos_edge_prob, fake_edge_index, fake_edge_attr, fake_edge_prob)
            
            # 4. Backprop and optimize
            D_loss = D_loss_real + \
                D_loss_fake + \
                reg_penalty * D_penalty

            D_perf = D_loss_real + D_loss_fake
            D_loss.backward()
            D_optimizer.step()
            
            logger_D_perf.append(D_perf)
            logger_D_loss.append(D_loss)

        
            reset_grad(G_optimizer, D_optimizer)
            # ================================================================== #
            #           Generate Fake Graphs based on Broken Graphs              #
            # ================================================================== #
            # 1. Encode the broken input graphs & treat partial edges as positive edges
            broken_edge_index, broken_edge_attr, out_edge_ratio = get_broken_graph(g, edge_ratio, connectivity=False)
            mu, log_var, z = G.encode(
                x=g.x, in_edge_index=broken_edge_index, 
                in_edge_attr=broken_edge_attr, reparameterize=reparameterize
                )
            _, _, cond_z = G.encode(
                x=g.x, in_edge_index=g.edge_index, 
                in_edge_attr=g.edge_attr, reparameterize=reparameterize
                )
            z = torch.cat([z, cond_z], dim=1)#torch.cat([z, cond_z, cond_z-z], dim=1)
            # mu, log_var, z = G.encode(x=g.x, in_edge_index=g.edge_index, in_edge_attr=g.edge_attr)

            # 2. Generate fake graphs by merging positive edges (i.e., broken_edge_index) & negative edges
            # .... pos_edge_prob for full edges (g.edge_index), neg_edge_prob for neg_edge_index
            
            pos_edge_prob, fake_pos_egde_attr = G.decode(z, g.edge_index, pos=pos)
            fake_edge_index, fake_edge_prob, fake_edge_attr, neg_prob2reg = \
                G.fill(
                    z=z, preserved_edge_index=broken_edge_index, preserved_edge_ratio=out_edge_ratio,
                    batch=g.batch, pos=pos, neg_edge_index=None, threshold=False
                    )


            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #
            # 1. Compute the VAE loss
            if args.reconstruction_type == 'BCE':
                reconstruct_loss = G.get_reconstruct_loss_1(
                    z=z, edge_index=g.edge_index, 
                    edge_attr=fake_pos_egde_attr,
                    batch=g.batch, pos=pos, tau=args.tau
                    )
            elif args.reconstruction_type == 'CTS':
                reconstruct_loss = G.get_reconstruct_loss_2(
                    z=z, edge_index=g.edge_index, 
                    edge_attr=fake_pos_egde_attr,
                    batch=g.batch, pos=pos, tau=args.tau
                    )
            contrastive_loss = G.get_contrastive_loss(
                z=z, y=g.y, batch=g.batch, tau=args.tau
                )
            kl_div = G.get_kl_div(mu=mu, log_var=log_var)

            # 2. Compute the loss of discriminating whether the fake graph is in distribution
            outputs = D(g.x, g.y, fake_edge_index, fake_edge_attr, g.batch, fake_edge_prob)
            G_loss_fake = criterion(outputs, real_labels)

            # 3. Compute penalty on gradients
            D_penalty = get_gradient_penalty(D, g, pos_edge_prob, fake_edge_index, fake_edge_attr, fake_edge_prob)
            
            # 4. Backprop and optimize
            G_loss = (G_loss_fake-reg_penalty * D_penalty)  + \
                reg_rec * reconstruct_loss + \
                reg_cts * contrastive_loss
            if reparameterize:
                G_loss += reg_kl * kl_div

            G_perf = reg_rec * reconstruct_loss + reg_cts * contrastive_loss
            logger_neg_prob.extend(neg_prob2reg.view(-1).detach())
            G_loss.backward()
            G_optimizer.step()

            logger_G_perf.append(G_perf)
            logger_G_loss.append(G_loss)
        logger_neg_prob = torch.tensor(logger_neg_prob)
        logger_G_perf, logger_D_perf = torch.tensor(logger_G_perf), torch.tensor(logger_D_perf)
        logger_real_score, logger_fake_score = torch.tensor(logger_real_score), torch.tensor(logger_fake_score)
        logger.info('Epoch [{}/{}] D_loss:{:.2f}, D_perf:{:.2f} | G_loss:{:.2f}, G_perf:{:.2f}| '
            'D(x):{:.2f}, D(G(z)):{:.2f} | P(E):{:.2f}, Max:{:.2f}, Min:{:.2f}'
            .format(epoch, args.epoch, D_loss.mean().item(), 
                    logger_D_perf.mean().item(), 
                    G_loss.mean().item(), 
                    logger_G_perf.mean().item(), 
                    logger_real_score.mean().item(), 
                    logger_fake_score.mean().item(),
                    logger_neg_prob.mean().item(), 
                    logger_neg_prob.max().item(), 
                    logger_neg_prob.min().item()))
    
        # ================================================================== #
        #                        Test the generator                          #
        # ================================================================== #
        with torch.no_grad():
            G.eval()
            D.eval()
            val_filled_likelihood = []
            ID_perf = torch.tensor([]).to(device)
            if ground_truth_avaliable:
                for g in val_loader:
                    g.to(device)
                    pos = g.pos if flag else None
                    broken_edge_index, broken_edge_attr, out_edge_ratio = get_ground_truth_graph(args, g)
                    mu, log_var, z = G.encode(
                        x=g.x, in_edge_index=broken_edge_index, 
                        in_edge_attr=broken_edge_attr, reparameterize=reparameterize
                        )
                    _, _, cond_z = G.encode(
                        x=g.x, in_edge_index=g.edge_index, 
                        in_edge_attr=g.edge_attr, reparameterize=reparameterize
                        )
                    z = torch.cat([z, cond_z], dim=1)
                    for inner_loop in range(args.val_loop):
                        fake_edge_index, fake_edge_prob, fake_edge_attr, _ = \
                            G.fill(
                                z=z, preserved_edge_index=broken_edge_index, 
                                preserved_edge_ratio=out_edge_ratio,
                                batch=g.batch, pos=pos, neg_edge_index=None, threshold=False
                                )
                        relabel_x, relabel_edge_index, relabel_batch, relabel_pos = relabel(g.x, fake_edge_index, g.batch, pos)
                        if flag:
                            new_g = Batch(batch=relabel_batch, x=relabel_x, edge_index=relabel_edge_index, edge_attr=fake_edge_attr, pos=relabel_pos)
                            readout = gnn(data=new_g)
                        else:
                            readout = gnn(
                                relabel_x,
                                relabel_edge_index,
                                fake_edge_attr,
                                relabel_batch
                                )
                        id_acc = (g.y == readout.argmax(dim=1)).view(-1).float()
                        ID_perf = torch.cat([ID_perf, id_acc])
                        
                ID_perf = ID_perf.mean().item() * 100
            else:
                for g in val_loader:
                    g.to(device)
                    pos = g.pos if flag else None
                    broken_edge_index, broken_edge_attr, out_edge_ratio = get_broken_graph(g, edge_ratio, connectivity=False)
                    mu, log_var, z = G.encode(
                        x=g.x, in_edge_index=broken_edge_index, 
                        in_edge_attr=broken_edge_attr, reparameterize=reparameterize
                        )
                    _, _, cond_z = G.encode(
                        x=g.x, in_edge_index=g.edge_index, 
                        in_edge_attr=g.edge_attr, reparameterize=reparameterize
                        )
                    z = torch.cat([z, cond_z], dim=1)
                    for inner_loop in range(args.val_loop):
                        
                        fake_edge_index, fake_edge_prob, fake_edge_attr, _ = \
                        G.fill(
                                z=z, preserved_edge_index=broken_edge_index, 
                                preserved_edge_ratio=out_edge_ratio,
                                batch=g.batch, pos=pos, neg_edge_index=None, threshold=False
                                )
                        relabel_x, relabel_edge_index, relabel_batch, relabel_pos = relabel(g.x, fake_edge_index, g.batch, pos)
                        full_emb = gnn.get_graph_rep(relabel_x, relabel_edge_index, fake_edge_attr, relabel_batch)
                        # relabel_x, relabel_edge_index, relabel_batch, relabel_pos = relabel(g.x, broken_edge_index, g.batch, pos)
                        # full_emb = gnn.get_graph_rep(relabel_x, relabel_edge_index, broken_edge_attr, relabel_batch)
                        log_likelihood = gmm.score_samples(full_emb)
                        ID_perf = torch.cat([ID_perf, log_likelihood])
                ID_perf = torch.tensor(ID_perf).mean().item()
            
            logger.info('Couterfactual Graph Performance {:.2f}'.format(ID_perf))
            # Track early stopping values with respect to ACC.
            if ID_perf > last_ID_perf:
                last_ID_perf = ID_perf
                save_model(G, G_folder, epoch, device=device)
                save_model(D, D_folder, epoch, device=device)
            
            # Update lrs
            G_scheduler.step(ID_perf)
            D_scheduler.step(logger_D_perf.mean().item())
            if (epoch + 1) % 20 == 0:
                logger.info('---------------Generator LR:{:.3f}  Discriminator LR: {:.3f}------------'.format(1e3 * G_lr, 1e3 * D_lr))

    save_model(G, G_folder, args.epoch, device=device, cover=False)
    save_model(D, D_folder, args.epoch, device=device, cover=False)