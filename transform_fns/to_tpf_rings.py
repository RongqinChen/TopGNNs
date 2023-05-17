"""
# Truncated Path Tree

## Node Table

| Name      | Attribute             |
| :-------: | :-------------------: |
| node      | label, attr           |
| edge      | label, attr, source   |
| node_1    | source                |
| node_2    | source                |
| ...       | source                |


## Edge Table

| Name                           | Attribute                    |  Note    |
| :-----------------------------:| :--------------------------: | :------: |
| 'node',   'n2n',    'node'     |                              |          |
| 'node_1', 'edge_1', 'node'     | source                       | height-1 |
| 'node_2', 'edge_2', 'node_1'   | source                       | height-2 |
| 'node_3', 'edge_3', 'node_2'   | source                       | height-3 |
| ......                         | source                       | ......   |


"""


from typing import List

import torch

from datautils.graph import Graph
from datautils.path_tree import PathTree
from datautils.path_tree_rings import PathTreeWithRings
from transform_fns.tpf import make_TPF_rawdata
from transform_fns.rings import make_rings_rawdata


def to_TPFwithRings(graph: Graph, height: int, max_ring_size: int) \
        -> PathTreeWithRings:

    num_nodes = graph.num_nodes
    src_nodes, dst_nodes = graph.edge_index

    n2n_src: List[int] = src_nodes.tolist()
    n2n_dst: List[int] = dst_nodes.tolist()
    dst_dict, source_dict, num_nodes_dict = \
        make_TPF_rawdata(height, num_nodes, n2n_src, n2n_dst)

    edge_image = source_dict['edge']
    n2e_dst = dst_dict['n2e']
    edge_attr = None
    if graph.edge_attr is not None:
        if edge_image.numel() > 0:
            edge_attr = graph.edge_attr[edge_image, :]
        else:
            edge_attr = graph.edge_attr
    edge_label = None
    if graph.edge_label is not None:
        if edge_image.numel() > 0:
            edge_label = graph.edge_label[edge_image, :]
        else:
            edge_label = graph.edge_label

    n_image_list = [None] + \
        [source_dict[f'node_{k}'] for k in range(1, height+1)]
    e_image_list = [None] + \
        [source_dict[f'edge_{k}'] for k in range(1, height+1)]
    tree_e_dst_list = [None] + [dst_dict[f'edge_{k}']
                                for k in range(1, height+1)]
    num_edges = num_nodes_dict['edge']
    num_tree_nodes_list = [num_nodes] + \
        [num_nodes_dict[f'node_{k}'] for k in range(1, height+1)]
    pt = PathTree(height, num_nodes, num_edges, num_tree_nodes_list,
                  src_nodes, dst_nodes, n2e_dst,
                  graph.node_attr, graph.node_label,
                  edge_attr, edge_label,
                  graph.graph_attr, graph.graph_label,
                  n_image_list, e_image_list,
                  tree_e_dst_list,
                  )
    relation_dict, num_nodes_dict = make_rings_rawdata(
        num_nodes, n2n_src, n2n_dst, max_ring_size)

    ptr = PathTreeWithRings(pt, relation_dict, num_nodes_dict)
    return ptr


def test():
    edge_index = torch.LongTensor(size=(2, 0))
    graph = Graph(2, edge_index,
                  None, None, None, None, None, None)
    ptr = to_TPFwithRings(graph, 4, 8)
    print(ptr)
    batch = ptr.collate([ptr, ptr])
    print(batch)

    srcs = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 7, 8]
    dsts = [1, 3, 4, 0, 2, 4, 5, 1, 3, 6, 0, 2, 7, 0, 1, 8, 1, 2, 3, 4]
    edge_index = torch.LongTensor((srcs, dsts))
    height = 4
    edge_attr = torch.randint(0, 16, (len(srcs), 16))
    graph = Graph(edge_index.max().item()+1, edge_index,
                  None, None, edge_attr, None, None, None)
    ptr = to_TPFwithRings(graph, height, 8)
    print(ptr)
    print('num_edges')
    print(ptr.num_edges)
    print('n2e', ptr.n2e_dst)
    print('tree_e_dst_list')
    print(*ptr.tree_e_dst_list, sep='\n')
    print('tree_node_image_list')
    print(*ptr.tree_node_image_list, sep='\n')
    print('tree_edge_image_list')
    print(*ptr.tree_edge_image_list, sep='\n')
    import os
    import networkx as nx
    import matplotlib.pyplot as plt
    nx_edge_list = edge_index.T.tolist()

    nx_g = nx.Graph()
    nx_g.add_edges_from(nx_edge_list)
    node_label_dict = {idx: str(idx) for idx in range(ptr.num_nodes)}
    node_list = list(range(ptr.num_nodes))

    plt.figure(figsize=(48, 12))
    plt.subplot(211)
    pos = nx.nx_agraph.graphviz_layout(nx_g, prog="dot")
    nx.draw_networkx(nx_g, pos, with_labels=False)
    nx.draw_networkx_nodes(nx_g, pos,
                           nodelist=node_list)
    nx.draw_networkx_edges(nx_g, pos,
                           edgelist=nx_edge_list)
    nx.draw_networkx_labels(nx_g, pos, labels=node_label_dict)
    plt.subplot(212)
    nx_edge_list = []

    node_label_dict = {idx: str(idx) for idx in range(ptr.num_nodes)}
    for k in range(1, height+1):
        tree_e_dst_list = ptr.tree_e_dst_list[k].tolist()
        offset_1 = sum(ptr.num_tree_nodes_list[:k-1]) if k > 0 else 0
        offset_2 = sum(ptr.num_tree_nodes_list[:k])
        nx_edge_list += [
            (offset_1+tree_dst, offset_2+idx)
            for idx, tree_dst in enumerate(tree_e_dst_list)]

        tree_node_image_list = ptr.tree_node_image_list[k].tolist()
        node_label_dict |= {idx + offset_2: f"{k}|{tree_node_image_list[idx]}"
                            for idx in range(len(tree_e_dst_list))}

    print(nx_edge_list)
    print(node_label_dict)
    nx_g = nx.Graph()
    nx_g.add_edges_from(nx_edge_list)
    node_list = list(range(sum(ptr.num_tree_nodes_list)))
    pos = nx.nx_agraph.graphviz_layout(nx_g, prog="dot")
    nx.draw_networkx(nx_g, pos, with_labels=False)
    nx.draw_networkx_nodes(nx_g, pos,
                           nodelist=node_list)
    nx.draw_networkx_edges(nx_g, pos,
                           edgelist=nx_edge_list)
    nx.draw_networkx_labels(nx_g, pos, labels=node_label_dict)

    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/to_tpfwithRings.pdf')


if __name__ == "__main__":
    test()
