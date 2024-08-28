import torch
from torch_geometric.utils import (negative_sampling, remove_self_loops, degree,
                                   add_self_loops, batched_negative_sampling)
                                   
class RandomGenerator(object):

    def fill(self, g, edge_ratio):

        preserved_edge_index = g.edge_index
        preserved_edge_attr = g.edge_attr
        n_feature = g.edge_attr.size(-1)
        mean, var = g.edge_attr.mean(), g.edge_attr.var()

        split = degree(g.batch[preserved_edge_index[0]], dtype=torch.long).tolist()
        edge_indices = torch.split(preserved_edge_index, split, dim=1)
        num_edges = [e.size(1) for e in edge_indices]
        num_nodes = degree(g.batch, dtype=torch.long)
        cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])

        fake_edge_indices, fake_edge_attrs = [], []
        fake_edge_indices.append(preserved_edge_index)
        fake_edge_attrs.append(preserved_edge_attr)

        for edge_index, N, C, E, R in zip(edge_indices, num_nodes.tolist(),
                                    cum_nodes.tolist(), num_edges, edge_ratio.tolist()):
            if E == 0:
                continue
            num_edge_fill = int(E / R  - E)
            neg_edge_index = negative_sampling(edge_index - C, N, num_edge_fill) + C
            neg_edge_index, _ = remove_self_loops(neg_edge_index)
            neg_edge_attr = torch.ones((neg_edge_index.size(1), n_feature)).to(g.x.device) * mean +\
                            torch.rand((neg_edge_index.size(1), n_feature)).to(g.x.device) * var
            fake_edge_indices.append(neg_edge_index)
            fake_edge_attrs.append(neg_edge_attr)
        fake_edge_indices = torch.cat(fake_edge_indices, dim=1)
        fake_edge_attrs = torch.cat(fake_edge_attrs, dim=0)

        filled_g = g.clone()
        filled_g.edge_index = fake_edge_indices
        filled_g.edge_attr = fake_edge_attrs

        return filled_g