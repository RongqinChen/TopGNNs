from typing import List, Optional
import torch
from torch_sparse import SparseTensor

from datautils.graph import Graph, GraphBatch

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


class PathTree(Graph):
    def __init__(self, height: int, num_nodes, num_edges,
                 num_tree_nodes_list: List[int],
                 n2n_src: torch.Tensor,
                 n2n_dst: torch.Tensor, n2e_dst: torch.Tensor,
                 node_attr: OptTensor, node_label: OptTensor,
                 edge_attr: OptTensor, edge_label: OptTensor,
                 graph_attr: OptTensor, graph_label: OptTensor,
                 tree_node_image_list: List[torch.Tensor],
                 tree_edge_image_list: List[torch.Tensor],
                 tree_e_dst_list: List[torch.Tensor],
                 validate=False
                 ):
        """ PathTree

            num_tree_nodes_list (List[int]):
                num_tree_nodes_list[0] for node
                num_tree_nodes_list[k] for node_k
            tree_node_image_list (List[torch.Tensor]): _description_
                tree_node_image_list[0] is None
                tree_node_image_list[k] for node_k
            tree_edge_image_list (List[torch.Tensor]): _description_
                tree_edge_image_list[0] is None
                tree_edge_image_list[k] for edge_k
            tree_e_dst_list (List[torch.Tensor]): _description_
                tree_e_dst_list[0] is None
                tree_e_dst_list[k] for edge_k
        """

        self.height = height
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_tree_nodes_list = num_tree_nodes_list
        self.n2n_src = n2n_src
        self.n2n_dst = n2n_dst
        self.n2e_dst = n2e_dst
        self.node_attr = node_attr
        self.node_label = node_label
        self.edge_attr = edge_attr
        self.edge_label = edge_label
        self.graph_attr = graph_attr
        self.graph_label = graph_label
        self.tree_node_image_list = tree_node_image_list
        self.tree_edge_image_list = tree_edge_image_list
        self.tree_e_dst_list = tree_e_dst_list

        if validate:
            self.validate()

    def validate(self):
        assert self.height + 1 == len(self.num_tree_nodes_list) \
            == len(self.tree_node_image_list) \
            == len(self.tree_edge_image_list) == len(self.tree_e_dst_list)
        assert self.n2n_src.ndim == self.n2n_dst.ndim == self.n2e_dst.ndim == 1
        assert self.n2n_src.size(0) == self.n2n_dst.size(0) \
            == self.n2e_dst.size(0) == self.num_edges * 2
        if self.node_attr is not None:
            assert self.node_attr.size(0) == self.num_nodes
            if self.node_attr.ndim == 1:
                self.node_attr = self.node_attr.unsqueeze(1)
        if self.node_label is not None:
            assert self.node_label.size(0) == self.num_nodes
            if self.node_label.ndim == 1:
                self.node_label = self.node_label.unsqueeze(1)
        if self.edge_attr is not None:
            assert self.edge_attr.size(0) == self.num_edges
            if self.edge_attr.ndim == 1:
                self.edge_attr = self.edge_attr.unsqueeze(1)
        if self.edge_label is not None:
            assert self.edge_label.size(0) == self.num_edges
            if self.edge_label.ndim == 1:
                self.edge_label = self.edge_label.unsqueeze(1)
        if self.graph_attr is not None and self.graph_attr.ndim == 1:
            self.graph_attr = self.graph_attr.unsqueeze(0)
        if self.graph_label is not None and self.graph_label.ndim == 1:
            self.graph_label = self.graph_label.unsqueeze(0)
        assert self.num_nodes == self.num_tree_nodes_list[0]
        if self.num_edges > 0:
            for k in range(1, self.height + 1):
                num_nodes, node_image, edge_image, tree_dst = \
                    self.num_tree_nodes_list[k], self.tree_node_image_list[k],\
                    self.tree_edge_image_list[k], self.tree_e_dst_list[k]
                assert num_nodes == node_image.size(0) \
                    == edge_image.size(0) == tree_dst.size(0)
                if node_image.numel() > 0:
                    assert 0 <= node_image.max().item() < self.num_nodes
                    assert 0 <= edge_image.max().item() < self.num_edges
                    assert 0 <= tree_dst.max().item() < self.num_tree_nodes_list[k-1]  # noqa

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

        repr += ", height={}".format(self.height)
        repr += ", tree_nodes={}".format(self.num_tree_nodes_list)
        repr += ")"
        return repr

    def collate(self, tree_list: List['PathTree'], validate=False):
        height = tree_list[0].height
        num_nodes_list = [0] + [tree.num_nodes for tree in tree_list]
        num_nodes_whole = sum(num_nodes_list)
        num_nodes_tensor = torch.LongTensor(num_nodes_list)
        node_slice = torch.cumsum(num_nodes_tensor, 0)
        node_batch = torch.repeat_interleave(num_nodes_tensor[1:])
        num_edges_list = [0] + [tree.num_edges for tree in tree_list]
        num_edges_whole = sum(num_edges_list)
        num_edges_tensor = torch.LongTensor(num_edges_list)
        edge_slice = torch.cumsum(num_edges_tensor, 0)
        edge_batch = torch.repeat_interleave(num_edges_tensor[1:])
        if tree_list[0].node_attr is not None:
            node_attr_list = [tree.node_attr for tree in tree_list]
            node_attr_whole = torch.cat(node_attr_list, dim=0)
        else:
            node_attr_whole = None
        if tree_list[0].node_label is not None:
            node_label_list = [tree.node_label for tree in tree_list]
            node_label_whole = torch.cat(node_label_list, dim=0)
        else:
            node_label_whole = None
        if tree_list[0].edge_attr is not None:
            edge_attr_list = [tree.edge_attr for tree in tree_list]
            edge_attr_whole = torch.cat(edge_attr_list, dim=0)
        else:
            edge_attr_whole = None
        if tree_list[0].edge_label is not None:
            edge_label_list = [tree.edge_label for tree in tree_list]
            edge_label_whole = torch.cat(edge_label_list, dim=0)
        else:
            edge_label_whole = None
        if tree_list[0].graph_attr is not None:
            graph_attr_list = [tree.graph_attr for tree in tree_list]
            graph_attr_whole = torch.cat(graph_attr_list, dim=0)
        else:
            graph_attr_whole = None
        if tree_list[0].graph_label is not None:
            graph_label_list = [tree.graph_label for tree in tree_list]
            graph_label_whole = torch.cat(graph_label_list, dim=0)
        else:
            graph_label_whole = None

        num_tree_nodes_list_list = [num_nodes_list[1:]]
        # num_tree_nodes_list_list[k][tree_idx]
        #   is the number of nodes at k-th layer of tree_idx tree
        num_tree_node_whole_list = [num_nodes_whole]
        tree_node_slice_list = [node_slice]
        tree_node_image_whole_list = [None]
        tree_edge_image_whole_list = [None]
        tree_dst_whole_list = [None]
        tree_node_batch_list = [None]
        for k in range(1, height + 1):
            num_tree_nodes_list = [0] + [tree.num_tree_nodes_list[k]
                                         for tree in tree_list]
            num_tree_nodes_list_list.append(num_tree_nodes_list[1:])
            num_tree_node_whole_list.append(sum(num_tree_nodes_list))
            num_tree_node_tensor = torch.LongTensor(num_tree_nodes_list)
            tree_node_slice = torch.cumsum(num_tree_node_tensor, 0)
            tree_node_slice_list.append(tree_node_slice)
            tree_node_batch = torch.repeat_interleave(num_tree_node_tensor[1:])
            tree_node_batch_list.append(tree_node_batch)
            n_image_list = [tree.tree_node_image_list[k] for tree in tree_list]
            n_image_whole = torch.cat(n_image_list, dim=0)
            n_image_whole += node_slice[tree_node_batch]
            tree_node_image_whole_list.append(n_image_whole)
            e_image_list = [tree.tree_edge_image_list[k] for tree in tree_list]
            e_image_whole = torch.cat(e_image_list, dim=0)
            e_image_whole += edge_slice[tree_node_batch]
            tree_edge_image_whole_list.append(e_image_whole)
            tree_e_dst_list = [tree.tree_e_dst_list[k] for tree in tree_list]
            tree_dst_whole = torch.cat(tree_e_dst_list, dim=0)
            dst_slice = tree_node_slice_list[k - 1]
            tree_dst_whole += dst_slice[tree_node_batch]
            tree_dst_whole_list.append(tree_dst_whole)

        n2n_src_list = [tree.n2n_src for tree in tree_list]
        n2n_dst_list = [tree.n2n_dst for tree in tree_list]
        n2e_dst_list = [tree.n2e_dst for tree in tree_list]
        n2n_batch = tree_node_batch_list[1]
        n2n_src_whole = torch.cat(n2n_src_list, dim=0)
        n2n_dst_whole = torch.cat(n2n_dst_list, dim=0)
        n2e_dst_whole = torch.cat(n2e_dst_list, dim=0)
        n2n_src_whole += node_slice[n2n_batch]
        n2n_dst_whole += node_slice[n2n_batch]
        n2e_dst_whole += edge_slice[n2n_batch]
        path_tree_batch = PathTreeBatch(
            height, num_nodes_whole, num_edges_whole,
            num_tree_node_whole_list,
            n2n_src_whole, n2n_dst_whole, n2e_dst_whole,
            node_attr_whole, node_label_whole,
            edge_attr_whole, edge_label_whole,
            graph_attr_whole, graph_label_whole,
            tree_node_image_whole_list,
            tree_edge_image_whole_list, tree_dst_whole_list,
            node_batch, node_slice, edge_batch, edge_slice,
            tree_node_batch_list, tree_node_slice_list,
            num_tree_nodes_list_list,
            batch_size=len(tree_list), validate=validate
        )
        return path_tree_batch


class PathTreeBatch(PathTree, GraphBatch):
    def __init__(self, height: int, num_nodes: int, num_edges: int,
                 num_tree_nodes_list: List[int],
                 n2n_src: torch.Tensor, n2n_dst: torch.Tensor,
                 n2e_dst: torch.Tensor,
                 node_attr: OptTensor, node_label: OptTensor,
                 edge_attr: OptTensor, edge_label: OptTensor,
                 graph_attr: OptTensor, graph_label: OptTensor,
                 tree_node_image_list: List[torch.Tensor],
                 tree_edge_image_list: List[torch.Tensor],
                 tree_e_dst_list: List[torch.Tensor],
                 node_batch: torch.Tensor, node_slice: torch.Tensor,
                 edge_batch: torch.Tensor, edge_slice: torch.Tensor,
                 tree_node_batch_list: List[torch.Tensor],
                 tree_node_slice_list: List[torch.Tensor],
                 num_tree_nodes_list_list: List[List[int]],
                 batch_size: int,
                 validate=False, n2n_adj_t=None, n2e_adj_t=None,
                 ):

        super(PathTreeBatch, self).__init__(
            height, num_nodes, num_edges,
            num_tree_nodes_list,
            n2n_src, n2n_dst, n2e_dst,
            node_attr, node_label,
            edge_attr, edge_label,
            graph_attr, graph_label,
            tree_node_image_list,
            tree_edge_image_list, tree_e_dst_list,
            validate=False
        )

        self.node_batch = node_batch
        self.edge_batch = edge_batch
        self.node_slice = node_slice
        self.edge_slice = edge_slice
        self.tree_node_batch_list = tree_node_batch_list
        self.tree_node_slice_list = tree_node_slice_list
        self.num_tree_nodes_list_list = num_tree_nodes_list_list
        self.batch_size = batch_size
        self.n2n_adj_t = n2n_adj_t if n2n_adj_t is not None else \
            SparseTensor(row=n2n_dst, col=n2n_src,
                         sparse_sizes=(num_nodes, num_nodes))
        self.n2e_adj_t = n2e_adj_t if n2e_adj_t is not None else \
            SparseTensor(row=n2e_dst, col=n2n_src,
                         sparse_sizes=(num_edges, num_nodes))
        if validate:
            self.validate()

    def validate(self):
        super().validate()
        self.node_batch.max().item() == self.batch_size
        self.node_slice.max().item() == self.batch_size
        self.edge_batch.max().item() <= self.batch_size
        self.edge_slice.max().item() <= self.batch_size
        assert len(self.tree_node_batch_list) == \
            len(self.tree_node_slice_list) == \
            len(self.num_tree_nodes_list_list) == (self.height+1)

    def uncollate(self) -> List[PathTree]:
        height = self.height
        node_slice = self.node_slice
        edge_slice = self.edge_slice

        num_nodes_list = torch.diff(node_slice).tolist()
        num_edges_list = torch.diff(edge_slice).tolist()
        num_trees = len(num_nodes_list)
        n2n_slice = self.tree_node_slice_list[1]
        num_n2n_list = torch.diff(n2n_slice).tolist()
        n2n_batch = self.tree_node_batch_list[1]
        n2n_src = self.n2n_src - node_slice[n2n_batch]
        n2n_src_list = torch.split(n2n_src, num_n2n_list, dim=0)
        n2n_dst = self.n2n_dst - node_slice[n2n_batch]
        n2n_dst_list = torch.split(n2n_dst, num_n2n_list, dim=0)
        n2e_dst = self.n2e_dst - edge_slice[n2n_batch]
        n2e_dst_list = torch.split(n2e_dst, num_n2n_list, dim=0)

        node_attr_list = [None] * num_trees \
            if self.node_attr is None else \
            torch.split(self.node_attr, num_nodes_list, dim=0)
        node_label_list = [None] * num_trees \
            if self.node_label is None else \
            torch.split(self.node_label, num_nodes_list, dim=0)
        edge_attr_list = [None] * num_trees \
            if self.edge_attr is None else \
            torch.split(self.edge_attr, num_edges_list, dim=0)
        edge_label_list = [None] * num_trees \
            if self.edge_label is None else \
            torch.split(self.edge_label, num_edges_list, dim=0)
        graph_attr_list = [None] * num_trees \
            if self.graph_attr is None else \
            torch.split(self.graph_attr, 1, dim=0)
        graph_label_list = [None] * num_trees \
            if self.graph_label is None else \
            torch.split(self.graph_label, 1, dim=0)

        num_tree_nodes_list_list = [[num] for num in num_nodes_list]
        tree_node_image_list_list = [[None] for _ in range(num_trees)]
        tree_edge_image_list_list = [[None] for _ in range(num_trees)]
        tree_e_dst_list_list = [[None] for _ in range(num_trees)]
        for k in range(1, height + 1):
            tree_node_batch = self.tree_node_batch_list[k]
            num_tree_nodes = self.num_tree_nodes_list_list[k]
            tree_node_image = self.tree_node_image_list[k]
            tree_node_image = tree_node_image - node_slice[tree_node_batch]
            tree_node_image_list = torch.split(tree_node_image, num_tree_nodes)
            self.tree_node_image_list[k] = None
            tree_edge_image = self.tree_edge_image_list[k]
            tree_edge_image = tree_edge_image - edge_slice[tree_node_batch]
            tree_edge_image_list = torch.split(tree_edge_image, num_tree_nodes)
            self.tree_edge_image_list[k] = None
            tree_dst = self.tree_e_dst_list[k]
            last_slice = self.tree_node_slice_list[k - 1]
            tree_dst = tree_dst - last_slice[tree_node_batch]
            tree_e_dst_list = torch.split(tree_dst, num_tree_nodes)
            self.tree_e_dst_list[k] = None
            for tree_idx in range(num_trees):
                num_tree_nodes_list_list[tree_idx].append(num_tree_nodes[tree_idx])  # noqa
                tree_node_image = tree_node_image_list[tree_idx]
                tree_node_image_list_list[tree_idx].append(tree_node_image)
                tree_edge_image = tree_edge_image_list[tree_idx]
                tree_edge_image_list_list[tree_idx].append(tree_edge_image)
                tree_dst = tree_e_dst_list[tree_idx]
                tree_e_dst_list_list[tree_idx].append(tree_dst)

        tree_list = [
            PathTree(
                height, num_nodes_list[tree_idx], num_edges_list[tree_idx],
                num_tree_nodes_list_list[tree_idx],
                n2n_src_list[tree_idx], n2n_dst_list[tree_idx],
                n2e_dst_list[tree_idx],
                node_attr_list[tree_idx], node_label_list[tree_idx],
                edge_attr_list[tree_idx], edge_label_list[tree_idx],
                graph_attr_list[tree_idx], graph_label_list[tree_idx],
                tree_node_image_list_list[tree_idx],
                tree_edge_image_list_list[tree_idx],
                tree_e_dst_list_list[tree_idx], validate=True
            )
            for tree_idx in range(num_trees)
        ]
        return tree_list
