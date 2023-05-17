from typing import List, Optional
import torch
from torch_sparse import SparseTensor

OptTensor = Optional[torch.Tensor]


class Graph:
    def __init__(self, num_nodes: int, edge_index: torch.Tensor,
                 node_attr: OptTensor, node_label: OptTensor,
                 edge_attr: OptTensor, edge_label: OptTensor,
                 graph_attr: OptTensor, graph_label: OptTensor,
                 validate=False
                 ) -> None:

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.node_attr = node_attr
        self.node_label = node_label
        self.edge_attr = edge_attr
        self.edge_label = edge_label
        self.graph_attr = graph_attr
        self.graph_label = graph_label
        if self.graph_attr is not None and self.graph_attr.ndim < 2:
            self.graph_attr = self.graph_attr.view((1, -1))
        if self.graph_label is not None and self.graph_label.ndim == 1:
            self.graph_label = self.graph_label.unsqueeze(1)

        self.num_edges = edge_index.size(1)
        if validate:
            self.validate()

    @property
    def x(self, ):
        return None

    def validate(self):
        if self.num_edges > 0:
            assert self.num_nodes > self.edge_index.max().item()
            assert self.edge_index.min().item() >= 0
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

        repr += ")"
        return repr

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, key, val.to(device))
            elif isinstance(val, SparseTensor):
                setattr(self, key, val.to_device(device))
            elif isinstance(val, list):
                val = [v.to(device) if
                       isinstance(v, torch.Tensor) else v.to_device(device) if
                       isinstance(v, SparseTensor) else v
                       for v in val]
                setattr(self, key, val)
        return self

    def pin_memory(self):
        for key, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, key, val.contiguous().pin_memory())
            elif isinstance(val, SparseTensor):
                setattr(self, key, val.pin_memory())
            elif isinstance(val, list):
                val = [v.contiguous().pin_memory() if
                       isinstance(v, torch.Tensor) else v.pin_memory() if
                       isinstance(v, SparseTensor) else v
                       for v in val]
                setattr(self, key, val)
        return self

    def collate(self, graph_list: List['Graph'], validate=False):
        num_nodes_list = [0] + [graph.num_nodes for graph in graph_list]
        num_nodes_tensor = torch.LongTensor(num_nodes_list)
        node_slice = torch.cumsum(num_nodes_tensor, 0)
        num_nodes_whole = sum(num_nodes_list)
        node_batch = torch.repeat_interleave(num_nodes_tensor[1:])
        num_edges_list = [0] + [graph.num_edges for graph in graph_list]
        num_edges_tensor = torch.LongTensor(num_edges_list)
        edge_slice = torch.cumsum(num_edges_tensor, 0)
        edge_batch = torch.repeat_interleave(num_edges_tensor[1:])
        edge_index_list = [graph.edge_index for graph in graph_list]
        edge_index_whole = torch.cat(edge_index_list, dim=1)
        edge_index_whole += node_slice[edge_batch]
        if graph_list[0].node_attr is not None:
            node_attr_list = [graph.node_attr for graph in graph_list]
            node_attr_whole = torch.cat(node_attr_list, dim=0)
        else:
            node_attr_whole = None
        if graph_list[0].node_label is not None:
            node_label_list = [graph.node_label for graph in graph_list]
            node_label_whole = torch.cat(node_label_list, dim=0)
        else:
            node_label_whole = None
        if graph_list[0].edge_attr is not None:
            edge_attr_list = [graph.edge_attr for graph in graph_list]
            edge_attr_whole = torch.cat(edge_attr_list, dim=0)
        else:
            edge_attr_whole = None
        if graph_list[0].edge_label is not None:
            edge_label_list = [graph.edge_label for graph in graph_list]
            edge_label_whole = torch.cat(edge_label_list, dim=0)
        else:
            edge_label_whole = None
        if graph_list[0].graph_attr is not None:
            graph_attr_list = [graph.graph_attr for graph in graph_list]
            graph_attr_whole = torch.cat(graph_attr_list, dim=0)
        else:
            graph_attr_whole = None
        if graph_list[0].graph_label is not None:
            graph_label_list = [graph.graph_label for graph in graph_list]
            graph_label_whole = torch.cat(graph_label_list, dim=0)
        else:
            graph_label_whole = None
        graph_whole = GraphBatch(
            num_nodes_whole, edge_index_whole,
            node_attr_whole, node_label_whole,
            edge_attr_whole, edge_label_whole,
            graph_attr_whole, graph_label_whole,
            node_slice, edge_slice, node_batch, edge_batch,
            batch_size=len(graph_list),
            validate=validate
        )
        return graph_whole


class GraphBatch(Graph):
    def __init__(self, num_nodes: int, edge_index: torch.Tensor,
                 node_attr: OptTensor, node_label: OptTensor,
                 edge_attr: OptTensor, edge_label: OptTensor,
                 graph_attr: OptTensor, graph_label: OptTensor,
                 node_slice: torch.Tensor, edge_slice: torch.Tensor,
                 node_batch: torch.Tensor, edge_batch: torch.Tensor,
                 batch_size: int, validate=False
                 ) -> None:

        self.node_slice = node_slice
        self.edge_slice = edge_slice
        self.node_batch = node_batch
        self.edge_batch = edge_batch
        self.batch_size = batch_size
        self.n2n_adj_t = SparseTensor(
            row=edge_index[1, :], col=edge_index[0, :],
            sparse_sizes=(num_nodes, num_nodes))
        super().__init__(num_nodes, edge_index, node_attr, node_label,
                         edge_attr, edge_label, graph_attr, graph_label,
                         validate)
        if validate:
            self.validate()

    def uncollate(self):
        num_nodes_list = torch.diff(self.node_slice).tolist()
        num_edges_list = torch.diff(self.edge_slice).tolist()
        num_graphs = len(num_nodes_list)

        edge_index = self.edge_index - self.node_slice[self.edge_batch]
        edge_index_list = torch.split(edge_index, num_edges_list, dim=1)
        node_attr_list = [None] * num_graphs \
            if self.node_attr is None else \
            torch.split(self.node_attr, num_nodes_list, dim=0)
        node_label_list = [None] * num_graphs \
            if self.node_label is None else \
            torch.split(self.node_label, num_nodes_list, dim=0)
        edge_attr_list = [None] * num_graphs \
            if self.edge_attr is None else \
            torch.split(self.edge_attr, num_edges_list, dim=0)
        edge_label_list = [None] * num_graphs \
            if self.edge_label is None else \
            torch.split(self.edge_label, num_edges_list, dim=0)
        graph_attr_list = [None] * num_graphs \
            if self.graph_attr is None else \
            torch.split(self.graph_attr, 1, dim=0)
        graph_label_list = [None] * num_graphs \
            if self.graph_label is None else \
            torch.split(self.graph_label, 1, dim=0)

        graph_list = [
            Graph(num_nodes_list[idx], edge_index_list[idx],
                  node_attr_list[idx], node_label_list[idx],
                  edge_attr_list[idx], edge_label_list[idx],
                  graph_attr_list[idx], graph_label_list[idx],
                  validate=True
                  )
            for idx in range(num_graphs)
        ]
        return graph_list

    def __repr__(self) -> str:
        repr = super().__repr__()[:-1] + \
            f", batch_size={self.batch_size})"
        return repr

    def get_sizes(self,):
        num_node_attr = None
        if self.node_attr is not None:
            num_node_attr = self.node_attr.size(-1)
        num_node_label = None
        if self.node_label is not None:
            num_node_label = self.node_label.amax(0) + 1
            num_node_label = num_node_label.tolist()
        num_edge_attr = None
        if self.edge_attr is not None:
            num_edge_attr = self.edge_attr.size(-1)
        num_edge_label = None
        if self.edge_label is not None:
            num_edge_label = self.edge_label.amax(0) + 1
            num_edge_label = num_edge_label.tolist()
        num_graph_attr = None
        if self.graph_attr is not None:
            num_graph_attr = self.graph_attr.size(-1)
        num_graph_label = None
        if self.graph_label is not None:
            if self.graph_label.shape[1] > 1:
                num_graph_label = self.graph_label.shape[1]
            else:
                num_graph_label = self.graph_label.max().item() + 1
                if num_graph_label == 2:
                    num_graph_label = 1
        sizes = {
            'num_node_attributes': num_node_attr,
            'num_node_labels': num_node_label,
            'num_edge_attributes': num_edge_attr,
            'num_edge_labels': num_edge_label,
            'num_graph_attributes': num_graph_attr,
            'num_graph_labels': num_graph_label,
            'num_graphs': self.batch_size
        }
        return sizes

    @property
    def targets(self):
        if self.graph_label is not None:
            return self.graph_label
        return self.graph_attr


if __name__ == "__main__":

    graph_1 = Graph(
        num_nodes=3,
        edge_index=torch.LongTensor([[0, 0, 1], [1, 2, 0]]),
        node_attr=torch.randn((3,)),
        node_label=torch.randint(0, 10, (3, 11)),
        edge_attr=torch.randint(0, 10, (3, 11)),
        edge_label=torch.randint(0, 10, (3, 11)),
        graph_attr=torch.randint(0, 10, (5,)),
        graph_label=torch.randint(0, 10, (5,)),
    )

    print(graph_1)

    graph_2 = Graph(
        num_nodes=4,
        edge_index=torch.LongTensor([[0, 0, 1, 3], [1, 2, 0, 2]]),
        node_attr=torch.randn((4,)),
        node_label=torch.randint(0, 10, (4, 11)),
        edge_attr=torch.randint(0, 10, (4, 11)),
        edge_label=torch.randint(0, 10, (4, 11)),
        graph_attr=torch.randint(0, 10, (5,)),
        graph_label=torch.randint(0, 10, (5,)),
    )

    print(graph_2)

    graph_whole = graph_1.collate([graph_1, graph_2, graph_2])
    print(graph_whole)

    print(graph_whole.edge_index)
    graph_list = graph_whole.uncollate()
    for graph in graph_list:
        print(graph.edge_index)

    device = torch.device('cuda:0')
    graph_1.to(device)
    print(graph_1.node_attr.device)
    print(graph_whole.node_attr.device)

    graph_whole.to(device)
    for key, val in graph_whole.__dict__.items():
        if isinstance(val, torch.Tensor):
            print(key, val.device)
