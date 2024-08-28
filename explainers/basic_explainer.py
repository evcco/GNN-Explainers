import os
import copy
import math
import time
import random
import warnings
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import re
import requests
import scipy as sp
import torch.nn as nn
import torch_geometric
from torch.autograd import Variable

import torch
import os.path as osp
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image
from torch_geometric.utils import negative_sampling, remove_self_loops
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
ifd_pert = 0.1
n_class_dict = {
    'MutagNet': 2, 'Tox21Net': 2, 
    'Reddit5kNet': 5, 'VGNet': 5, 
    'BA2MotifNet': 2, 'BA3MotifNet': 3, 
    'TR3MotifNet': 3, 'MNISTNet':10}
vis_dict = {
    'MutagNet': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'Tox21Net': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'BA3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'TR3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 5},
    'GraphSST2Net': {'node_size': 400, 'linewidths': 1, 'font_size': 12, 'width': 3},
    'MNISTNet': {'node_size': 100, 'linewidths': 1, 'font_size': 10, 'width': 2},
    'defult': {'node_size': 200, 'linewidths': 1, 'font_size': 10, 'width': 2}
}
chem_graph_label_dict = {'MutagNet': {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'},
                         'Tox21Net': {0: 'O', 1: 'C', 2: 'N', 3: 'F', 4: 'Cl', 5: 'S', 6: 'Br', 7: 'Si',
                                      8: 'Na', 9: 'I', 10: 'Hg', 11: 'B', 12: 'K', 13: 'P', 14: 'Au',
                                      15: 'Cr', 16: 'Sn', 17: 'Ca', 18: 'Cd', 19: 'Zn', 20: 'V', 21: 'As',
                                      22: 'Li', 23: 'Cu', 24: 'Co', 25: 'Ag', 26: 'Se', 27: 'Pt', 28: 'Al',
                                      29: 'Bi', 30: 'Sb', 31: 'Ba', 32: 'Fe', 33: 'H', 34: 'Ti', 35: 'Tl',
                                      36: 'Sr', 37: 'In', 38: 'Dy', 39: 'Ni', 40: 'Be', 41: 'Mg', 42: 'Nd',
                                      43: 'Pd', 44: 'Mn', 45: 'Zr', 46: 'Pb', 47: 'Yb', 48: 'Mo', 49: 'Ge',
                                      50: 'Ru', 51: 'Eu', 52: 'Sc'}
                         }
rec_color = ['cyan', 'mediumblue', 'deeppink', 'darkorange', 'gold', 'chartreuse', 'lightcoral', 'darkviolet', 'teal',
             'lightgrey', ]

def sentence_layout(sentence, length, margin=0.2):
    num_token = len(sentence)
    pos = {}; height = []; width = []
    
    right_margin = len(sentence[-1]) * 0.05
    gap = (length-right_margin) / (num_token-1)
    start = 0
    for i in range(num_token):
        pos[i] = np.array([start + gap*i, gap/5 * pow(-1, i)])
        width.append(len(sentence[i]) * 0.04)
        height.append(gap/3)
    return pos, np.array(width), np.array(height)
    
    
class Explainer(object):

    def __init__(self, gnn_model_path, gen_model_path=None):
        self.model = torch.load(gnn_model_path).to(device)
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__

        self.path = gnn_model_path
        self.last_result = None
        self.vis_dict = None
        if gen_model_path:
            self.vgae = torch.load(gen_model_path, map_location=device)
            self.vgae.eval()
            
    def explain_graph(self, graph, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs:
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    def get_cxplain_scores(self, graph):
        # initialize the ranking list with cxplain.
        y = graph.y
        orig_pred = self.model(graph)[0, y]

        scores = []
        for e_id in range(graph.num_edges):
            edge_mask = torch.ones(graph.num_edges, dtype=torch.bool)
            edge_mask[e_id] = False
            masked_edge_index = graph.edge_index[:, edge_mask]
            masked_edge_attr = graph.edge_attr[edge_mask]

            masked_pred = self.model(graph)[0, y]

            scores.append(orig_pred - masked_pred)
            # scores.append(orig_pred - masked_pred)
        scores = torch.tensor(scores)
        return scores.cpu().detach().numpy()

    @staticmethod
    def get_rank(lst, r=1):

        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        # _min = np.min(imp)
        # _max = np.max(imp) + 1e-16
        # imp = (imp - _min)/(_max - _min)
        # return imp
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def __relabel__(self, g, edge_index):

        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except:
            pass

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos
    
    def __reparameterize__(self, log_alpha, beta=0.1, training=True):

        if training:
            random_noise = torch.rand(log_alpha.size()).to(device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs
    
    def pack_explanatory_subgraph(self, top_ratio=0.2, 
                                  graph=None, imp=None, relabel=True):

        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, 'length mismatch'
        
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x
        if relabel:
            exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos = \
                self.__relabel__(exp_subgraph, exp_subgraph.edge_index)
        
        return exp_subgraph

    def evaluate_recall(self, topk=10):

        graph, imp = self.last_result
        E = graph.num_edges
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        index = np.argsort(-imp)[:topk]
        values = graph.ground_truth_mask[index]
        return float(values.sum()) / float(graph.ground_truth_mask.sum())
        

    def evaluate_precision(self, topk=10):

        graph, imp = self.last_result
        E = graph.num_edges
        index = np.argsort(-imp)[:topk]
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        values = graph.ground_truth_mask[index]
        #
        return float(values.sum()) / topk
    
    def evaluate_CounterSup_acc(self, top_ratio_list, K=10, return_fake_ratio=1.0, reweight=False):
        assert self.last_result is not None
        assert self.vgae is not None
        
        g, imp = self.last_result
        rank = np.argsort(-imp)
        acc = np.array([[]])
        prob = np.array([[]])
        neg_edge_index_list = []
        
        for idx, top_ratio in enumerate(top_ratio_list):
            
            topk = max(math.floor(top_ratio * g.num_edges), 1)
            pos_idx = rank[:topk]
            broken_edge_index = g.edge_index[:, pos_idx]
            broken_edge_attr = g.edge_attr[pos_idx, :]
            out_edge_ratio = torch.tensor([float(topk) / g.num_edges]).to(g.x.device)
            mu, log_var, z = self.vgae.encode(
                x=g.x, in_edge_index=broken_edge_index, 
                in_edge_attr=broken_edge_attr, reparameterize=False
            )
            _, _, cond_z = self.vgae.encode(
                    x=g.x, in_edge_index=g.edge_index, 
                    in_edge_attr=g.edge_attr, reparameterize=False
                    )
            z = torch.cat([z, cond_z], dim=1)
            G_acc = []
            G_prob = []
            for _ in range(K):
                fake_edge_index, fake_edge_prob, fake_edge_attr, _ = \
                self.vgae.fill(z=z, preserved_edge_index=broken_edge_index, preserved_edge_ratio=out_edge_ratio,
                            batch=g.batch, neg_edge_index=None, threshold=False)
                
                relabel_x, relabel_edge_index, relabel_batch, relabel_pos = self.__relabel__(g, fake_edge_index)
                
                tmp_g = g.clone()
                tmp_g.x, tmp_g.edge_index, tmp_g.edge_attr, tmp_g.batch, tmp_g.pos = \
                        relabel_x, relabel_edge_index, fake_edge_attr, relabel_batch, relabel_pos
                
                log_logits = self.model(tmp_g)
                if reweight:
                    # adjust for g' in Equation (1)
                    eps = 0.1
                    g_start_rep = self.model.get_graph_rep(tmp_g)
                    weight = F.cosine_similarity(g_rep, g_start_rep).item()
                    weight = (1 + weight + eps * (1 - weight)) / 2.0  # rescale to (eps,1)
                else:
                    weight = 1.
                G_acc.append((g.y == log_logits.argmax(dim=1)).detach().cpu().float().numpy())
                G_prob.append(self.model.readout[0, g.y].detach().cpu().float().numpy() / weight)
                
            G_acc = np.array([np.mean(G_acc)]).reshape(-1, 1)
            G_prob = np.array([np.mean(G_prob)]).reshape(-1, 1)
            
            acc = np.concatenate([acc, G_acc], axis=1)
            prob = np.concatenate([prob, G_prob], axis=1)
        index = torch.argsort(-fake_edge_prob)[:int(return_fake_ratio * g.num_edges)]
        print(fake_edge_prob[index], self.model.readout[0, g.y])
        
        return acc, prob, fake_edge_index[:, index]
    
    def evaluate_acc(self, top_ratio_list, graph=None, imp=None):
        
        if graph is None:
            assert self.last_result is not None
        acc = np.array([[]])
        prob = np.array([[]])
        for idx, top_ratio in enumerate(top_ratio_list):

            exp_subgraph = self.pack_explanatory_subgraph(top_ratio, 
                                                          graph=graph, imp=imp)
            self.model(exp_subgraph)
            res_acc = (exp_subgraph.y == self.model.readout.argmax(dim=1)).detach().cpu().float().view(-1, 1).numpy()
            res_prob = self.model.readout[0, exp_subgraph.y].detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res_acc], axis=1)
            prob = np.concatenate([prob, res_prob], axis=1)
        return acc, prob
    
    def evaluate_contrastivity(self, cts_ratio=0.2):

        assert self.last_result is not None
        graph, imp = self.last_result
        idx = np.argsort(-imp)[int(cts_ratio*graph.num_edges):]
        _imp = copy.copy(imp)
        _imp[idx] = 0
        counter_graph = graph.clone()
        counter_classes = [i for i in range(n_class_dict[self.model_name])]
        counter_classes.pop(graph.y)
        counter_accumulate = 0
        for c in counter_classes:
            counter_graph.y = torch.LongTensor([c]).cuda()
            if self.name == "Screener" and \
                    isinstance(graph.name[0], str) and \
                    "reddit" in graph.name[0]:
                counter_imp, _ = self.explain_graph(counter_graph, large_scale=True)
            elif self.name == "Screener":
                counter_imp, _ = self.explain_graph(counter_graph)
            else:
                counter_imp = self.explain_graph(counter_graph)
            counter_imp = self.norm_imp(counter_imp)
            idx = np.argsort(-counter_imp)[int(cts_ratio*graph.num_edges):]
            counter_imp[idx] = 0
            tmp = scipy.stats.spearmanr(counter_imp, _imp)[0]

            if np.isnan(tmp):
                tmp = 1
            counter_accumulate += abs(tmp)
        self.last_result = graph, imp  # may be unnecessary

        return counter_accumulate / len(counter_classes)

    def evaluate_infidelity(self, N=5, p0=0.25):

        assert self.last_result is not None
        graph, imp = self.last_result

        imp = torch.FloatTensor(imp + 1e-8).cuda()
        imp = imp / imp.sum()
        ps = p0 * torch.ones_like(imp)

        self.model(graph)
        ori_pred = self.model.readout[0, graph.y]
        lst = []
        for _ in range(N):
            p0 = torch.bernoulli(ps)
            edge_mask = (1.0 - p0).bool()
            self.model(graph)
            pert_pred = self.model.readout[0, graph.y]
            infd = pow(sum(p0 * imp) - (ori_pred - pert_pred), 2).cpu().detach().numpy()
            lst.append(infd)
        lst = np.array(lst)
        return lst.mean()

    def visualize(self, graph=None, edge_imp=None, 
                  counter_edge_index=None ,vis_ratio=0.2, 
                  save=False, layout=False, name=None):
        
        if graph is None:
            assert self.last_result is not None
            graph, edge_imp = self.last_result
        
        topk = max(int(vis_ratio * graph.num_edges), 1)
        idx = np.argsort(-edge_imp)[:topk]
        G = nx.DiGraph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
        
        if not counter_edge_index==None:
            G.add_edges_from(list(counter_edge_index.cpu().numpy().T))
        if self.vis_dict is None:
            self.vis_dict = vis_dict[self.model_name] if self.model_name in vis_dict.keys() else vis_dict['defult']
        
        folder = Path(r'image/%s' % (self.model_name))
        if save and not os.path.exists(folder):
            os.makedirs(folder)
                

        edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
        edge_pos_mask[idx] = True
        vmax = sum(edge_pos_mask)
        node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
        node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
        node_pos_mask[node_pos_idx] = True
        node_neg_mask[node_neg_idx] = True
        
        if self.model_name == "GraphSST2Net":
            plt.figure(figsize=(10, 4), dpi=100)
            ax = plt.gca()
            node_imp = np.zeros(graph.num_nodes)
            row, col = graph.edge_index[:, edge_pos_mask].cpu().numpy()
            node_imp[row] += edge_imp[edge_pos_mask]
            node_imp[col] += edge_imp[edge_pos_mask]
            node_alpha = node_imp / max(node_imp)
            pos, width, height = sentence_layout(graph.sentence_tokens[0], length=2)
            
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index.cpu().numpy().T),
                                   edge_color='whitesmoke',
                                   width=self.vis_dict['width'], arrows=True,
                                   connectionstyle="arc3,rad=0.2"  # <-- THIS IS IT
                                   )
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('Greys'),
                                   edge_vmin=-vmax, edge_vmax=vmax,
                                   arrows=True, connectionstyle="arc3,rad=0.2" 
                                   )
            
            for i in node_pos_idx:
                patch = Rectangle(
                    xy=(pos[i][0]-width[i]/2, pos[i][1]-height[i]/2) ,width=width[i], height=height[i],
                    linewidth=1, color='orchid', alpha=node_alpha[i], fill=True, label=graph.sentence_tokens[0][i])
                ax.add_patch(patch)
                
            nx.draw_networkx_labels(G, pos=pos,
                                    labels={i: graph.sentence_tokens[0][i] for i in range(graph.num_nodes)},
                                    font_size=self.vis_dict['font_size'],
                                    font_weight='bold', font_color='k'
                                    )
            if not counter_edge_index==None:
                nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(counter_edge_index.cpu().numpy().T),
                                   edge_color='mediumturquoise',
                                   width=self.vis_dict['width']/2.0,
                                   arrows=True, connectionstyle="arc3,rad=0.2" 
                                   )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        if 'Motif' in self.model_name:
            plt.figure(figsize=(8, 6), dpi=100)
            ax = plt.gca()
            pos = graph.pos[0]
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx},
                                   nodelist=node_pos_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color=graph.z[0][node_pos_idx],
                                   alpha=1, cmap='winter',
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='red',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
                                   )
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_neg_idx},
                                   nodelist=node_neg_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color=graph.z[0][node_neg_idx],
                                   alpha=0.2, cmap='winter',
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='whitesmoke',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
                                   )
            nx.draw_networkx_edges(G, pos=pos,
                                       edgelist=list(graph.edge_index.cpu().numpy().T),
                                       edge_color='whitesmoke',
                                       width=self.vis_dict['width'],
                                       arrows=False
                                       )
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   # np.ones(len(edge_imp[edge_pos_mask])),
                                   width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('bwr'),
                                   edge_vmin=-vmax, edge_vmax=vmax,
                                   arrows=False
                                   )
            if not counter_edge_index==None:
                nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(counter_edge_index.cpu().numpy().T),
                                   edge_color='mediumturquoise',
                                   width=self.vis_dict['width']/3.0,
                                   arrows=False
                                   )
            
        if 'Mutag' in self.model_name:
            idx = [int(i/2) for i in idx]
            x = graph.x.detach().cpu().tolist()
            edge_index = graph.edge_index.T.detach().cpu().tolist()
            edge_attr = graph.edge_attr.detach().cpu().tolist()
            mol = graph_to_mol(x, edge_index, edge_attr)
            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            hit_at = np.unique(graph.edge_index[:,idx].detach().cpu().numpy()).tolist()
            def add_atom_index(mol):
                atoms = mol.GetNumAtoms()
                for i in range( atoms ):
                    mol.GetAtomWithIdx(i).SetProp(
                        'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
                return mol

            hit_bonds=[]
            for (u, v) in graph.edge_index.T[idx]:
                hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, highlightAtoms=hit_at, highlightBonds=hit_bonds,
                highlightAtomColors={i:(0, 1, 0) for i in hit_at},
                highlightBondColors={i:(0, 1, 0) for i in hit_bonds})
            d.FinishDrawing()
            bindata = d.GetDrawingText()
            iobuf = io.BytesIO(bindata)
            image = Image.open(iobuf)
            image.show()
            if save:
                if name:
                    d.WriteDrawingText('image/%s/%s-%d-%s.png' % (self.model_name, name, int(graph.y[0]), self.name)) 
                else:
                    d.WriteDrawingText('image/%s/%s-%d-%s.png' % (self.model_name, str(graph.name[0]), int(graph.y[0]), self.name)) 
            return 
            
            
        if 'MNIST' in self.model_name:
            plt.figure(figsize=(6, 6), dpi=100)
            ax = plt.gca()
            pos = graph.pos.detach().cpu().numpy()
            row, col = graph.edge_index
            z = np.zeros(graph.num_nodes)
            for i in idx:
                z[row[i]] += edge_imp[i]
                z[col[i]] += edge_imp[i]
            z = z / max(z)

            row, col = graph.edge_index
            pos = graph.pos.detach().cpu().numpy()
            z = graph.x.detach().cpu().numpy()
            edge_mask = torch.tensor(graph.x[row].view(-1) * graph.x[col].view(-1), dtype=torch.bool).view(-1)

            nx.draw_networkx_edges(
                    G, pos=pos,
                    edgelist=list(graph.edge_index.cpu().numpy().T),
                    edge_color='whitesmoke',
                    width=self.vis_dict['width'],
                    arrows=False
                )
            nx.draw_networkx_edges(
                    G, pos=pos,
                    edgelist=list(graph.edge_index[:,edge_mask].cpu().numpy().T),
                    edge_color='black',
                    width=self.vis_dict['width'],
                    arrows=False
                )
            nx.draw_networkx_nodes(G, pos=pos,
                                   node_size=self.vis_dict['node_size'],
                                   node_color='black', alpha=graph.x, 
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='black'
                                   )
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('YlOrRd'),
                                   edge_vmin=-vmax, edge_vmax=vmax,
                                   arrows=False
                                   )
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx},
                                   nodelist=node_pos_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color='brown', alpha=z[node_pos_idx], 
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='black'
                                   )
            if not counter_edge_index==None:
                nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(counter_edge_index.cpu().numpy().T),
                                   edge_color='mediumturquoise',
                                   width=self.vis_dict['width']/3.0,
                                   arrows=False
                                   )
        if self.model_name == "VGNet":
            from visual_genome import local as vgl
            topk = 1
            idx = np.argsort(-edge_imp)[:topk]
            top_edges = graph.edge_index[:, idx]
            all = graph.edge_index

            scene_graph = vgl.get_scene_graph(image_id=int(graph.name),
                                              images='visual_genome/raw',
                                              image_data_dir='visual_genome/raw/by-id/',
                                              synset_file='visual_genome/raw/synsets.json')
            # scene_graph = api.get_scene_graph_of_image(id=int(graph.id))
            r = 0.95  # transparency
            img = Image.open("data/VG/raw/%d-%d.jpg" % (graph.name, graph.y))
            data = list(img.getdata())
            ndata = list(
                [(int((255 - p[0]) * r + p[0]), int((255 - p[1]) * r + p[1]), int((255 - p[2]) * r + p[2])) for p in
                 data])
            mode = img.mode
            width, height = img.size
            edges = list(top_edges.T)
            for i, (u, v) in enumerate(edges[::-1]):
                r = 1.0 - 1.0 / len(edges) * (i + 1)
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                for obj in [obj1, obj2]:
                    for x in range(obj.x, obj.width + obj.x):
                        for y in range(obj.y, obj.y + obj.height):
                            ndata[y * width + x] = (int((255 - data[y * width + x][0]) * r + data[y * width + x][0]),
                                                    int((255 - data[y * width + x][1]) * r + data[y * width + x][1]),
                                                    int((255 - data[y * width + x][2]) * r + data[y * width + x][2]))

            img = Image.new(mode, (width, height))
            img.putdata(ndata)

            plt.imshow(img)
            ax = plt.gca()
            for i, (u, v) in enumerate(edges):
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                ax.annotate("", xy=(obj2.x, obj2.y), xytext=(obj1.x, obj1.y),
                            arrowprops=dict(width=topk - i, color='wheat', headwidth=5))
                for obj in [obj1, obj2]:
                    ax.text(obj.x, obj.y - 8, str(obj), style='italic',
                            fontsize=13,
                            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3,
                                  'edgecolor': rec_color[i % len(rec_color)]}
                            )
                    ax.add_patch(Rectangle((obj.x, obj.y),
                                           obj.width,
                                           obj.height,
                                           fill=False,
                                           edgecolor=rec_color[i % len(rec_color)],
                                           linewidth=1.5))
            plt.tick_params(labelbottom='off', labelleft='off')
            plt.axis('off')
        if save:
            if name:
                plt.savefig(folder / Path(r'%s-%d-%s.png' % (name, int(graph.y[0]), self.name)), dpi=500,
                                bbox_inches='tight')
            else:
                if isinstance(graph.name[0], str):
                    plt.savefig(folder / Path(r'%s-%d-%s.png' % (str(graph.name[0]), int(graph.y[0]), self.name)), dpi=500,
                                bbox_inches='tight')
                else:
                    plt.savefig(folder / Path(r'%d-%d-%s.png' % (int(graph.name[0]), int(graph.y[0]), self.name)), dpi=500,
                                bbox_inches='tight')
        
        plt.show()