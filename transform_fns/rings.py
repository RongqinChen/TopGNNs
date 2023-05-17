from itertools import combinations
from typing import Dict, List

import networkx as nx
import torch as th
from torch import LongTensor


def make_node2edge(num_nodes: int, srcs: List[int], dsts: List[int]):
    nnhash_list = [
        u * num_nodes + v if u < v else v * num_nodes + u
        for u, v in zip(srcs, dsts)
    ]
    unique_hash_list = [
        u * num_nodes + v
        for u, v in zip(srcs, dsts) if u < v
    ]
    nnhash2edge = {
        hashval: idx
        for idx, hashval in enumerate(unique_hash_list)
    }
    n2e_dst = [nnhash2edge[hashval] for hashval in nnhash_list]
    return nnhash2edge, n2e_dst


def search_chordless_rings(srcs: List[int], dsts: List[int],
                           max_ring_size: int):
    edge_list = [(u, v) for u, v in zip(srcs, dsts) if u < v]
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge_list)
    ring2nodes = list(nx.chordless_cycles(nx_graph, max_ring_size))
    return ring2nodes


def make_ring2edges(
    num_nodes: int, ring2nodes: List[List[int]],
    nnhash2bond: Dict[int, int]
):
    ring2nodePairHashs = [
        [
            u * num_nodes + v if u < v else v * num_nodes + u
            for u, v in combinations(nodes, 2)
        ]
        for nodes in ring2nodes
    ]
    ring2edges = [
        [
            nnhash2bond[nodePairHash]
            for nodePairHash in nodePairHashs
            if nodePairHash in nnhash2bond
        ]
        for nodePairHashs in ring2nodePairHashs
    ]
    return ring2edges


def make_rings_rawdata(
    num_nodes: int,
    n2n_src: List[int], n2n_dst: List[int],
    max_ring_size: int
):
    if len(n2n_src) < max_ring_size:
        ring2nodes = []
    else:
        ring2nodes = search_chordless_rings(n2n_src, n2n_dst, max_ring_size)

    # node2ring relation
    n2r_src, n2r_dst = [], []
    for ring_idx, nodes in enumerate(ring2nodes):
        n2r_src.extend(nodes)
        n2r_dst += ([ring_idx] * len(nodes))
    n2r_src = LongTensor(n2r_src)
    n2r_dst = LongTensor(n2r_dst)

    nnhash2edge, n2e_dst = make_node2edge(num_nodes, n2n_src, n2n_dst)

    # ring_2_edge relation
    ring2edges = make_ring2edges(num_nodes, ring2nodes, nnhash2edge)

    # edge_2_ring relation
    e2r_tuples = [
        (edge, ring)
        for ring, edges in enumerate(ring2edges) for edge in edges
    ]
    if len(e2r_tuples) > 0:
        e2r_tuples = LongTensor(e2r_tuples)
    else:
        e2r_tuples = th.LongTensor(size=(0, 2))
    e2r_src = e2r_tuples[:, 0]
    e2r_dst = e2r_tuples[:, 1]

    rnodes = set().union(*ring2nodes)
    # rn: node on ring
    n2rn = {n: rn for rn, n in enumerate(rnodes)}
    n2rn_src = list(n2rn.keys())
    n2rn_dst = LongTensor([n2rn[key] for key in n2rn_src])
    n2rn_src = LongTensor(n2rn_src)

    # nrt: Node-Ring-Tuple
    nrt_rn, nrt_r = [], []
    e2nrt_src, e2nrt_dst = [], []
    for ring, nodes in enumerate(ring2nodes):
        for u in nodes:
            # nt: node tuple
            nn_hash_list = [
                u * num_nodes + v if u < v else v * num_nodes + u
                for v in nodes
            ]
            edge_list = [
                nnhash2edge[nn_hash]
                for nn_hash in nn_hash_list if nn_hash in nnhash2edge
            ]
            assert len(edge_list) == 2
            e2nrt_src.extend(edge_list)
            e2nrt_dst.extend([len(nrt_r)] * len(edge_list))
            nrt_rn.append(n2rn[u])
            nrt_r.append(ring)

    num_nrt = len(nrt_rn)
    n2e_dst = LongTensor(n2e_dst)
    nrt_rn = LongTensor(nrt_rn)
    nrt_r = LongTensor(nrt_r)
    e2nrt_src = LongTensor(e2nrt_src)
    e2nrt_dst = LongTensor(e2nrt_dst)
    nrt = th.arange(num_nrt, dtype=th.long)

    relation_dict = {
        # ('node', 'n2e', 'edge'): (n2n_src, n2e_dst),
        ('node', 'n2r', 'ring'): (n2r_src, n2r_dst),
        ('edge', 'e2r', 'ring'): (e2r_src, e2r_dst),
        ('edge', 'e2nrt', 'nrt'): (e2nrt_src, e2nrt_dst),
        ('ring', 'r2nrt', 'nrt'): (nrt_r, nrt),
        ('nrt', 'nrt2rn', 'rn'): (nrt, nrt_rn),
        ('node', 'n2rn', 'rn'): (n2rn_src, n2rn_dst),
    }
    num_nodes_dict = {
        'ring': len(ring2nodes), 'nrt': num_nrt, 'rn': len(n2rn)
    }
    return relation_dict, num_nodes_dict
