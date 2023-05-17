from typing import List, Optional
import torch
from torch_sparse import SparseTensor

from datautils.path_tree import PathTree, PathTreeBatch

OptTensor = Optional[torch.Tensor]


"""
## Node Table

| Name      | Feature               |
| :-------: | :-------------------: |
| node      | label, attr           |
| edge      | label, attr           |
| node_1    | image                 |
| node_2    | image                 |
| ...       | image                 |


## Edge Table

| Name                         | Feature                      |  Note    |
| :---------------------------:| :--------------------------: | :------: |
| 'node',   'n2n',    'node'   |                              |          |
| 'node',   'n2e',    'edge'   |                              |          |
| 'node_1', 'edge_1', 'node'   | image, dst                   | height-1 |
| 'node_2', 'edge_2', 'node_1' | image, dst                   | height-2 |
| ......                       | image, dst                   | ......   |

"""


class PathTreeWithRings(PathTree):
    def __init__(self, pt: PathTree, rel_dict: dict, num_nodes_dict: dict,
                 validate=False):
        super().__init__(**pt.__dict__, validate=False)
        self.num_rings = num_nodes_dict['ring']
        self.num_nrts = num_nodes_dict['nrt']
        self.num_rns = num_nodes_dict['rn']
        self.n2r_src, self.n2r_dst = rel_dict[('node', 'n2r', 'ring')]
        self.e2r_src, self.e2r_dst = rel_dict[('edge', 'e2r', 'ring')]
        self.e2nrt_src, self.e2nrt_dst = rel_dict[('edge', 'e2nrt', 'nrt')]
        self.r2nrt_src, self.r2nrt_dst = rel_dict[('ring', 'r2nrt', 'nrt')]
        self.nrt2rn_src, self.nrt2rn_dst = rel_dict[('nrt', 'nrt2rn', 'rn')]
        self.n2rn_src, self.n2rn_dst = rel_dict[('node', 'n2rn', 'rn')]
        self._num_nodes_dict = num_nodes_dict
        if validate:
            self.validate()

    def validate(self):
        super().validate()
        if self.num_rings > 0:
            assert self.n2r_src.max().item() < self.num_nodes
            assert self.n2r_src.min().item() >= 0
            assert self.n2r_dst.max().item() < self.num_rings
            assert self.n2r_dst.min().item() >= 0
            assert self.e2r_src.max().item() < self.num_edges
            assert self.e2r_src.min().item() >= 0
            assert self.e2r_dst.max().item() < self.num_rings
            assert self.e2r_dst.min().item() >= 0
            assert self.e2nrt_src.max().item() < self.num_edges
            assert self.e2nrt_src.min().item() >= 0
            assert self.e2nrt_dst.max().item() < self.num_nrts
            assert self.e2nrt_dst.min().item() >= 0
            assert self.r2nrt_src.max().item() < self.num_rings
            assert self.r2nrt_src.min().item() >= 0
            assert self.r2nrt_dst.max().item() < self.num_nrts
            assert self.r2nrt_dst.min().item() >= 0
            assert self.nrt2rn_src.max().item() < self.num_nrts
            assert self.nrt2rn_src.min().item() >= 0
            assert self.nrt2rn_dst.max().item() < self.num_rns
            assert self.nrt2rn_dst.min().item() >= 0
            assert self.n2rn_src.max().item() < self.num_nodes
            assert self.n2rn_src.min().item() >= 0
            assert self.n2rn_dst.max().item() < self.num_rns
            assert self.n2rn_dst.min().item() >= 0

    def __repr__(self) -> str:
        repr = "{}(num_nodes={}, num_edges={}"
        repr = repr.format(self.__class__.__name__,
                           self.num_nodes, self.num_edges)
        if self.node_attr is not None:
            repr += ", node_attr={}".format(list(self.node_attr.shape))
        if self.node_label is not None:
            repr += ", node_label={}".format(list(self.node_label.shape))
        if self.edge_attr is not None:
            repr += ", edge_attr={}".format(list(self.edge_attr.shape))
        if self.edge_label is not None:
            repr += ", edge_label={}".format(list(self.edge_label.shape))
        if self.graph_attr is not None:
            repr += ", graph_attr={}".format(list(self.graph_attr.shape))
        if self.graph_label is not None:
            repr += ", graph_label={}".format(list(self.graph_label.shape))

        repr += ", tree_nodes={}".format(self.num_tree_nodes_list)
        for key, val in self._num_nodes_dict.items():
            repr += ", {}={}".format(key, val)
        repr += ")"
        return repr

    def collate(self, tree_list: List['PathTreeWithRings'], validate=False) \
            -> 'PathTreeWithRingsBatch':
        tree_batch = super().collate(tree_list)
        node_slice = tree_batch.node_slice
        edge_slice = tree_batch.edge_slice

        num_rings_list = [0] + [tree.num_rings for tree in tree_list]
        num_rings_whole = sum(num_rings_list)
        num_rings_tensor = torch.LongTensor(num_rings_list)
        ring_slice = torch.cumsum(num_rings_tensor, 0)
        ring_batch = torch.repeat_interleave(num_rings_tensor[1:])

        num_nrts_list = [0] + [tree.num_nrts for tree in tree_list]
        num_nrts_whole = sum(num_nrts_list)
        num_nrts_tensor = torch.LongTensor(num_nrts_list)
        nrt_slice = torch.cumsum(num_nrts_tensor, 0)
        nrt_batch = torch.repeat_interleave(num_nrts_tensor[1:])

        num_rns_list = [0] + [tree.num_rns for tree in tree_list]
        num_rns_whole = sum(num_rns_list)
        num_rns_tensor = torch.LongTensor(num_rns_list)
        rn_slice = torch.cumsum(num_rns_tensor, 0)
        rn_batch = torch.repeat_interleave(num_rns_tensor[1:])

        num_n2r_list = [0] + [tree.n2r_src.size(0) for tree in tree_list]
        num_n2r_tensor = torch.LongTensor(num_n2r_list)
        n2r_slice = torch.cumsum(num_n2r_tensor, 0)
        n2r_batch = torch.repeat_interleave(num_n2r_tensor[1:])
        n2r_src_list = [tree.n2r_src for tree in tree_list]
        n2r_dst_list = [tree.n2r_dst for tree in tree_list]
        n2r_src = torch.cat(n2r_src_list, dim=0)
        n2r_dst = torch.cat(n2r_dst_list, dim=0)
        n2r_src += node_slice[n2r_batch]
        n2r_dst += ring_slice[n2r_batch]

        num_e2r_list = [0] + [tree.e2r_src.size(0) for tree in tree_list]
        num_e2r_tensor = torch.LongTensor(num_e2r_list)
        e2r_slice = torch.cumsum(num_e2r_tensor, 0, dtype=torch.long)
        e2r_batch = torch.repeat_interleave(num_e2r_tensor[1:])
        e2r_src_list = [tree.e2r_src for tree in tree_list]
        e2r_dst_list = [tree.e2r_dst for tree in tree_list]
        e2r_src = torch.cat(e2r_src_list, dim=0)
        e2r_dst = torch.cat(e2r_dst_list, dim=0)
        e2r_src += edge_slice[e2r_batch]
        e2r_dst += ring_slice[e2r_batch]

        num_e2nrt_list = [0] + [tree.e2nrt_src.size(0) for tree in tree_list]
        num_e2nrt_tensor = torch.LongTensor(num_e2nrt_list)
        e2nrt_slice = torch.cumsum(num_e2nrt_tensor, 0, dtype=torch.long)
        e2nrt_batch = torch.repeat_interleave(num_e2nrt_tensor[1:])
        e2nrt_src_list = [tree.e2nrt_src for tree in tree_list]
        e2nrt_dst_list = [tree.e2nrt_dst for tree in tree_list]
        e2nrt_src = torch.cat(e2nrt_src_list, dim=0)
        e2nrt_dst = torch.cat(e2nrt_dst_list, dim=0)
        e2nrt_src += edge_slice[e2nrt_batch]
        e2nrt_dst += nrt_slice[e2nrt_batch]

        num_r2nrt_list = [0] + [tree.r2nrt_src.size(0) for tree in tree_list]
        num_r2nrt_tensor = torch.LongTensor(num_r2nrt_list)
        r2nrt_slice = torch.cumsum(num_r2nrt_tensor, 0, dtype=torch.long)
        r2nrt_batch = torch.repeat_interleave(num_r2nrt_tensor[1:])
        r2nrt_src_list = [tree.r2nrt_src for tree in tree_list]
        r2nrt_dst_list = [tree.r2nrt_dst for tree in tree_list]
        r2nrt_src = torch.cat(r2nrt_src_list, dim=0)
        r2nrt_dst = torch.cat(r2nrt_dst_list, dim=0)
        r2nrt_src += ring_slice[r2nrt_batch]
        r2nrt_dst += nrt_slice[r2nrt_batch]

        num_nrt2rn_list = [0] + [tree.nrt2rn_src.size(0) for tree in tree_list]
        num_nrt2rn_tensor = torch.LongTensor(num_nrt2rn_list)
        nrt2rn_slice = torch.cumsum(num_nrt2rn_tensor, 0, dtype=torch.long)
        nrt2rn_batch = torch.repeat_interleave(num_nrt2rn_tensor[1:])
        nrt2rn_src_list = [tree.nrt2rn_src for tree in tree_list]
        nrt2rn_dst_list = [tree.nrt2rn_dst for tree in tree_list]
        nrt2rn_src = torch.cat(nrt2rn_src_list, dim=0)
        nrt2rn_dst = torch.cat(nrt2rn_dst_list, dim=0)
        nrt2rn_src += nrt_slice[nrt2rn_batch]
        nrt2rn_dst += rn_slice[nrt2rn_batch]

        n2rn_list = [0] + [tree.n2rn_src.size(0) for tree in tree_list]
        num_n2rn_tensor = torch.LongTensor(n2rn_list)
        n2rn_slice = torch.cumsum(num_n2rn_tensor, 0, dtype=torch.long)
        n2rn_batch = torch.repeat_interleave(num_n2rn_tensor[1:])
        n2rn_src_list = [tree.n2rn_src for tree in tree_list]
        n2rn_dst_list = [tree.n2rn_dst for tree in tree_list]
        n2rn_src = torch.cat(n2rn_src_list, dim=0)
        n2rn_dst = torch.cat(n2rn_dst_list, dim=0)
        n2rn_src += node_slice[n2rn_batch]
        n2rn_dst += rn_slice[n2rn_batch]

        ptrb = PathTreeWithRingsBatch(
            tree_batch, num_rings_whole, ring_slice, ring_batch,
            num_nrts_whole, nrt_slice, nrt_batch,
            num_rns_whole, rn_slice, rn_batch,
            n2r_slice, n2r_batch, n2r_src, n2r_dst,
            e2r_slice, e2r_batch, e2r_src, e2r_dst,
            e2nrt_slice, e2nrt_batch, e2nrt_src, e2nrt_dst,
            r2nrt_slice, r2nrt_batch, r2nrt_src, r2nrt_dst,
            nrt2rn_slice, nrt2rn_batch, nrt2rn_src, nrt2rn_dst,
            n2rn_slice, n2rn_batch, n2rn_src, n2rn_dst,
            batch_size=len(tree_list),
            validate=validate
        )

        return ptrb


class PathTreeWithRingsBatch(PathTreeWithRings, PathTreeBatch):
    def __init__(self, ptb: PathTreeBatch,
                 num_rings, ring_slice, ring_batch,
                 num_nrts, nrt_slice, nrt_batch,
                 num_rns, rn_slice, rn_batch,
                 n2r_slice, n2r_batch, n2r_src, n2r_dst,
                 e2r_slice, e2r_batch, e2r_src, e2r_dst,
                 e2nrt_slice, e2nrt_batch, e2nrt_src, e2nrt_dst,
                 r2nrt_slice, r2nrt_batch, r2nrt_src, r2nrt_dst,
                 nrt2rn_slice, nrt2rn_batch, nrt2rn_src, nrt2rn_dst,
                 n2rn_slice, n2rn_batch, n2rn_src, n2rn_dst, batch_size,
                 validate=False
                 ):

        PathTreeBatch.__init__(self, **ptb.__dict__, validate=False)

        self.num_rings = num_rings
        self.num_rns = num_rns
        self.num_nrts = num_nrts

        self.ring_slice = ring_slice
        self.ring_batch = ring_batch
        self.nrt_slice = nrt_slice
        self.nrt_batch = nrt_batch
        self.rn_slice = rn_slice
        self.rn_batch = rn_batch
        self.n2r_slice = n2r_slice
        self.n2r_batch = n2r_batch
        self.e2r_slice = e2r_slice
        self.e2r_batch = e2r_batch
        self.e2nrt_slice = e2nrt_slice
        self.e2nrt_batch = e2nrt_batch
        self.r2nrt_slice = r2nrt_slice
        self.r2nrt_batch = r2nrt_batch
        self.nrt2rn_slice = nrt2rn_slice
        self.nrt2rn_batch = nrt2rn_batch
        self.n2rn_slice = n2rn_slice
        self.n2rn_batch = n2rn_batch

        # self.n2e_adj_t = SparseTensor(
        #     row=self.n2e_dst, col=self.n2n_src,
        #     sparse_sizes=(self.num_edges, self.num_nodes))
        self.n2r_adj_t = SparseTensor(
            row=n2r_dst, col=n2r_src,
            sparse_sizes=(self.num_rings, self.num_nodes))
        self.e2r_adj_t = SparseTensor(
            row=e2r_dst, col=e2r_src,
            sparse_sizes=(self.num_rings, self.num_edges))
        self.e2nrt_adj_t = SparseTensor(
            row=e2nrt_dst, col=e2nrt_src,
            sparse_sizes=(self.num_nrts, self.num_edges))
        self.r2nrt_adj_t = SparseTensor(
            row=r2nrt_dst, col=r2nrt_src,
            sparse_sizes=(self.num_nrts, self.num_rings))
        self.nrt2rn_adj_t = SparseTensor(
            row=nrt2rn_dst, col=nrt2rn_src,
            sparse_sizes=(self.num_rns, self.num_nrts))
        self.n2rn_adj_t = SparseTensor(
            row=n2rn_dst, col=n2rn_src,
            sparse_sizes=(self.num_rns, self.num_nodes))

        if validate:
            self.validate()

    def validate(self):
        # super().validate()
        super(PathTreeWithRings, self).validate()
        if self.num_rings > 0:
            n2r_dst, n2r_src, _ = self.n2r_adj_t.coo()
            e2r_dst, e2r_src, _ = self.e2r_adj_t.coo()
            e2nrt_dst, e2nrt_src, _ = self.e2nrt_adj_t.coo()
            r2nrt_dst, r2nrt_src, _ = self.r2nrt_adj_t.coo()
            nrt2rn_dst, nrt2rn_src, _ = self.nrt2rn_adj_t.coo()
            n2rn_dst, n2rn_src, _ = self.n2rn_adj_t.coo()
            assert n2r_src.max().item() < self.num_nodes
            assert n2r_src.min().item() >= 0
            assert n2r_dst.max().item() < self.num_rings
            assert n2r_dst.min().item() >= 0
            assert e2r_src.max().item() < self.num_edges
            assert e2r_src.min().item() >= 0
            assert e2r_dst.max().item() < self.num_rings
            assert e2r_dst.min().item() >= 0
            assert e2nrt_src.max().item() < self.num_edges
            assert e2nrt_src.min().item() >= 0
            assert e2nrt_dst.max().item() < self.num_nrts
            assert e2nrt_dst.min().item() >= 0
            assert r2nrt_src.max().item() < self.num_rings
            assert r2nrt_src.min().item() >= 0
            assert r2nrt_dst.max().item() < self.num_nrts
            assert r2nrt_dst.min().item() >= 0
            assert nrt2rn_src.max().item() < self.num_nrts
            assert nrt2rn_src.min().item() >= 0
            assert nrt2rn_dst.max().item() < self.num_rns
            assert nrt2rn_dst.min().item() >= 0
            assert n2rn_src.max().item() < self.num_nodes
            assert n2rn_src.min().item() >= 0
            assert n2rn_dst.max().item() < self.num_rns
            assert n2rn_dst.min().item() >= 0

            self.ring_batch.max().item() == self.batch_size
            self.ring_slice.max().item() == self.batch_size
            self.nrt_batch.max().item() == self.batch_size
            self.nrt_slice.max().item() == self.batch_size
            self.rn_batch.max().item() == self.batch_size
            self.rn_slice.max().item() == self.batch_size
            self.n2r_batch.max().item() == self.batch_size
            self.n2r_slice.max().item() == self.batch_size
            self.e2r_batch.max().item() == self.batch_size
            self.e2r_slice.max().item() == self.batch_size
            self.e2nrt_batch.max().item() == self.batch_size
            self.e2nrt_slice.max().item() == self.batch_size
            self.r2nrt_batch.max().item() == self.batch_size
            self.r2nrt_slice.max().item() == self.batch_size
            self.nrt2rn_batch.max().item() == self.batch_size
            self.nrt2rn_slice.max().item() == self.batch_size
            self.n2rn_batch.max().item() == self.batch_size
            self.n2rn_slice.max().item() == self.batch_size

    def uncollate(self) -> List[PathTreeWithRings]:
        height = self.height
        node_slice = self.node_slice
        edge_slice = self.edge_slice
        ring_slice = self.ring_slice
        nrt_slice = self.nrt_slice
        rn_slice = self.rn_slice

        num_nodes_list = torch.diff(node_slice).tolist()
        num_edges_list = torch.diff(edge_slice).tolist()
        num_rings_list = torch.diff(ring_slice).tolist()
        num_nrts_list = torch.diff(nrt_slice).tolist()
        num_rns_list = torch.diff(rn_slice).tolist()
        num_ptr = len(num_nodes_list)

        n2n_slice = self.tree_node_slice_list[1]
        num_n2n_list = torch.diff(n2n_slice).tolist()
        n2r_slice = self.n2r_slice
        num_n2r_list = torch.diff(n2r_slice).tolist()
        e2r_slice = self.e2r_slice
        num_e2r_list = torch.diff(e2r_slice).tolist()
        e2nrt_slice = self.e2nrt_slice
        num_e2nrt_list = torch.diff(e2nrt_slice).tolist()
        r2nrt_slice = self.r2nrt_slice
        num_r2nrt_list = torch.diff(r2nrt_slice).tolist()
        nrt2rn_slice = self.nrt2rn_slice
        num_nrt2rn_list = torch.diff(nrt2rn_slice).tolist()
        n2rn_slice = self.n2rn_slice
        num_n2rn_list = torch.diff(n2rn_slice).tolist()

        n2n_batch = self.tree_node_batch_list[1]
        n2r_batch = self.n2r_batch
        e2r_batch = self.e2r_batch
        e2nrt_batch = self.e2nrt_batch
        r2nrt_batch = self.r2nrt_batch
        nrt2rn_batch = self.nrt2rn_batch
        n2rn_batch = self.n2rn_batch

        n2n_src = self.n2n_src - node_slice[n2n_batch]
        n2n_src_list = torch.split(n2n_src, num_n2n_list, dim=0)
        n2n_dst = self.n2n_dst - node_slice[n2n_batch]
        n2n_dst_list = torch.split(n2n_dst, num_n2n_list, dim=0)
        n2e_dst = self.n2e_dst - edge_slice[n2n_batch]
        n2e_dst_list = torch.split(n2e_dst, num_n2n_list, dim=0)

        n2r_dst, n2r_src, _ = self.n2r_adj_t.coo()
        n2r_dst_re = n2r_dst - ring_slice[n2r_batch]
        n2r_src_re = n2r_src - node_slice[n2r_batch]
        n2r_dst_list = torch.split(n2r_dst_re, num_n2r_list, dim=0)
        n2r_src_list = torch.split(n2r_src_re, num_n2r_list, dim=0)

        e2r_dst, e2r_src, _ = self.e2r_adj_t.coo()
        e2r_dst_re = e2r_dst - ring_slice[e2r_batch]
        e2r_src_re = e2r_src - edge_slice[e2r_batch]
        e2r_dst_list = torch.split(e2r_dst_re, num_e2r_list, dim=0)
        e2r_src_list = torch.split(e2r_src_re, num_e2r_list, dim=0)

        e2nrt_dst, e2nrt_src, _ = self.e2nrt_adj_t.coo()
        e2nrt_dst_re = e2nrt_dst - nrt_slice[e2nrt_batch]
        e2nrt_src_re = e2nrt_src - edge_slice[e2nrt_batch]
        e2nrt_dst_list = torch.split(e2nrt_dst_re, num_e2nrt_list, dim=0)
        e2nrt_src_list = torch.split(e2nrt_src_re, num_e2nrt_list, dim=0)

        r2nrt_dst, r2nrt_src, _ = self.r2nrt_adj_t.coo()
        r2nrt_dst_re = r2nrt_dst - nrt_slice[r2nrt_batch]
        r2nrt_src_re = r2nrt_src - ring_slice[r2nrt_batch]
        r2nrt_dst_list = torch.split(r2nrt_dst_re, num_r2nrt_list, dim=0)
        r2nrt_src_list = torch.split(r2nrt_src_re, num_r2nrt_list, dim=0)

        nrt2rn_dst, nrt2rn_src, _ = self.nrt2rn_adj_t.coo()
        nrt2rn_dst_re = nrt2rn_dst - rn_slice[nrt2rn_batch]
        nrt2rn_src_re = nrt2rn_src - nrt_slice[nrt2rn_batch]
        nrt2rn_dst_list = torch.split(nrt2rn_dst_re, num_nrt2rn_list, dim=0)
        nrt2rn_src_list = torch.split(nrt2rn_src_re, num_nrt2rn_list, dim=0)

        n2rn_dst, n2rn_src, _ = self.n2rn_adj_t.coo()
        n2rn_dst_re = n2rn_dst - rn_slice[n2rn_batch]
        n2rn_src_re = n2rn_src - node_slice[n2rn_batch]
        n2rn_dst_list = torch.split(n2rn_dst_re, num_n2rn_list, dim=0)
        n2rn_src_list = torch.split(n2rn_src_re, num_n2rn_list, dim=0)

        node_attr_list = [None] * num_ptr \
            if self.node_attr is None else \
            torch.split(self.node_attr, num_nodes_list, dim=0)
        node_label_list = [None] * num_ptr \
            if self.node_label is None else \
            torch.split(self.node_label, num_nodes_list, dim=0)
        edge_attr_list = [None] * num_ptr \
            if self.edge_attr is None else \
            torch.split(self.edge_attr, num_edges_list, dim=0)
        edge_label_list = [None] * num_ptr \
            if self.edge_label is None else \
            torch.split(self.edge_label, num_edges_list, dim=0)
        graph_attr_list = [None] * num_ptr \
            if self.graph_attr is None else \
            torch.split(self.graph_attr, 1, dim=0)
        graph_label_list = [None] * num_ptr \
            if self.graph_label is None else \
            torch.split(self.graph_label, 1, dim=0)

        num_tree_nodes_list_list = [[num] for num in num_nodes_list]
        tree_node_image_list_list = [[None] for _ in range(num_ptr)]
        tree_edge_image_list_list = [[None] for _ in range(num_ptr)]
        tree_e_dst_list_list = [[None] for _ in range(num_ptr)]
        for k in range(1, height + 1):
            tree_node_batch = self.tree_node_batch_list[k]
            num_tree_nodes = self.num_tree_nodes_list_list[k]
            tree_node_image = self.tree_node_image_list[k]
            tree_node_image_re = tree_node_image - node_slice[tree_node_batch]
            tree_node_image_list = torch.split(tree_node_image_re, num_tree_nodes) # noqa
            tree_edge_image = self.tree_edge_image_list[k]
            tree_edge_image_re = tree_edge_image - edge_slice[tree_node_batch]
            tree_edge_image_list = torch.split(tree_edge_image_re, num_tree_nodes) # noqa
            tree_e_dst = self.tree_e_dst_list[k]
            dst_slice = self.tree_node_slice_list[k - 1]
            tree_e_dst_re = tree_e_dst - dst_slice[tree_node_batch]
            tree_e_dst_list = torch.split(tree_e_dst_re, num_tree_nodes)

            for ptr_idx in range(num_ptr):
                num_tree_nodes_list_list[ptr_idx].append(
                    num_tree_nodes[ptr_idx])
                tree_node_image = tree_node_image_list[ptr_idx]
                tree_node_image_list_list[ptr_idx].append(tree_node_image)
                tree_edge_image = tree_edge_image_list[ptr_idx]
                tree_edge_image_list_list[ptr_idx].append(tree_edge_image)
                tree_e_dst = tree_e_dst_list[ptr_idx]
                tree_e_dst_list_list[ptr_idx].append(tree_e_dst)

        ptr_list = [
            PathTreeWithRings(
                PathTree(
                    height, num_nodes_list[ptr_idx], num_edges_list[ptr_idx],
                    num_tree_nodes_list_list[ptr_idx],
                    n2n_src_list[ptr_idx], n2n_dst_list[ptr_idx],
                    n2e_dst_list[ptr_idx],
                    node_attr_list[ptr_idx], node_label_list[ptr_idx],
                    edge_attr_list[ptr_idx], edge_label_list[ptr_idx],
                    graph_attr_list[ptr_idx], graph_label_list[ptr_idx],
                    tree_node_image_list_list[ptr_idx],
                    tree_edge_image_list_list[ptr_idx],
                    tree_e_dst_list_list[ptr_idx],
                ), {
                    ('node', 'n2r', 'ring'):
                        (n2r_src_list[ptr_idx], n2r_dst_list[ptr_idx]),
                    ('edge', 'e2r', 'ring'):
                        (e2r_src_list[ptr_idx], e2r_dst_list[ptr_idx]),
                    ('edge', 'e2nrt', 'nrt'):
                        (e2nrt_src_list[ptr_idx], e2nrt_dst_list[ptr_idx]),
                    ('ring', 'r2nrt', 'nrt'):
                        (r2nrt_src_list[ptr_idx], r2nrt_dst_list[ptr_idx]),
                    ('nrt', 'nrt2rn', 'rn'):
                        (nrt2rn_src_list[ptr_idx], nrt2rn_dst_list[ptr_idx]),
                    ('node', 'n2rn', 'rn'):
                        (n2rn_src_list[ptr_idx], n2rn_dst_list[ptr_idx]),
                }, {
                    'ring': num_rings_list[ptr_idx],
                    'nrt': num_nrts_list[ptr_idx],
                    'rn': num_rns_list[ptr_idx]
                }, validate=True
            )
            for ptr_idx in range(num_ptr)
        ]
        return ptr_list
