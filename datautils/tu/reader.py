import glob
import os
import os.path as osp

import numpy as np
import torch
from torch_sparse import coalesce

from torch_geometric.io import read_txt_array
from datautils.graph import GraphBatch

# names = [
#     'A', 'graph_indicator', 'node_labels', 'node_attributes'
#     'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
# ]


def read_graph(folder, d_name):
    files = glob.glob(osp.join(folder, f'{d_name}_*.txt'))
    names = [f.split(os.sep)[-1][len(d_name) + 1:-4] for f in files]

    edge_index = read_file(folder, d_name, 'A', torch.long).t() - 1
    node_batch = read_file(folder, d_name, 'graph_indicator', torch.long) - 1

    num_nodes = 0
    node_attr = None
    if 'node_attributes' in names:
        node_attr = read_file(folder, d_name, 'node_attributes')
        if node_attr.dim() == 1:
            node_attr = node_attr.unsqueeze(-1)
            num_nodes = node_attr.size(0)

    node_label = None
    if 'node_labels' in names:
        node_label = read_file(folder, d_name, 'node_labels', torch.long)
        if node_label.dim() == 1:
            node_label = node_label.unsqueeze(-1)
        node_label = node_label - node_label.min(dim=0)[0]
        if num_nodes == 0:
            num_nodes = node_label.size(0)
        else:
            assert num_nodes == node_label.size(0)

    edge_attr = None
    if 'edge_attributes' in names:
        edge_attr = read_file(folder, d_name, 'edge_attributes')
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

    edge_label = None
    if 'edge_labels' in names:
        edge_label = read_file(folder, d_name, 'edge_labels', torch.long)
        if edge_label.dim() == 1:
            edge_label = edge_label.unsqueeze(-1)
        edge_label = edge_label - edge_label.min(dim=0)[0]

    graph_attr = None
    if 'graph_attributes' in names:  # Regression problem.
        graph_attr = read_file(folder, d_name, 'graph_attributes')
    graph_label = None
    if 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, d_name, 'graph_labels', torch.long)
        _, graph_label = y.unique(sorted=True, return_inverse=True)

    if num_nodes == 0:
        num_nodes = edge_index.max().item() + 1

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    _edge_index = edge_index
    if edge_attr is not None:
        edge_index, edge_attr = coalesce(_edge_index, edge_attr,
                                         num_nodes, num_nodes)
    edge_index, edge_label = coalesce(_edge_index, edge_label,
                                      num_nodes, num_nodes)

    node_slice = torch.cumsum(torch.from_numpy(np.bincount(node_batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = edge_index
    edge_batch = node_batch[row]
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(edge_batch)), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    if 'REDDIT' in d_name:
        node_label = torch.ones((num_nodes, 1), dtype=torch.long)
    dataset = GraphBatch(
        num_nodes, edge_index, node_attr, node_label,
        edge_attr, edge_label, graph_attr, graph_label,
        node_slice, edge_slice, node_batch, edge_batch,
        batch_size=node_batch.max().item() + 1,
        validate=True
    )
    slices = {'node': node_slice, 'edge': edge_slice}
    sizes = dataset.get_sizes()
    return dataset, slices, sizes


def read_file(folder, d_name, name, dtype=None):
    path = osp.join(folder, f'{d_name}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)
