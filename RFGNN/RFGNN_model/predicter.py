from typing import List, Optional

from torch import nn

from datautils.graph import GraphBatch
from datautils.path_tree import PathTreeBatch

from .aggregation import aggfn
from .bottomup import Tree_BottomUp
from .init import Tree_INIT
from .mlp import MultiModelSLP


class RFGNN_Predictor(nn.Module):
    def __init__(
        self,
        height: int,
        node_feat_dim: Optional[int] = None,
        node_label_sizes: Optional[List[int]] = None,
        edge_feat_dim: Optional[int] = None,
        edge_label_sizes: Optional[List[int]] = None,
        graph_feat_dim: Optional[int] = None,
        graph_label_size: Optional[List[int]] = None,
        hidden_dim: int = 64,
        dropout_p: float = 0.25,
        graph_dropout_p: float = 0.5,
        readout: str = 'sum',
        child_agg: str = 'sum',
        norm: str = 'batch',
        pred_level: str = 'graph',
        target_dim: int = 1,
    ) -> None:
        super().__init__()

        self.readout = readout
        self.pred_level = pred_level
        assert norm in {'batch', 'layer'}
        Norm = nn.BatchNorm1d if norm == 'batch' else nn.LayerNorm
        self.init = Tree_INIT(
            height, node_feat_dim, node_label_sizes,
            edge_feat_dim, edge_label_sizes, hidden_dim)
        self.bottomup = Tree_BottomUp(height, hidden_dim,
                                      self.init.has_edge_feat,
                                      dropout_p, child_agg, Norm)

        if pred_level == 'graph':
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(graph_dropout_p),
                nn.Linear(hidden_dim, graph_feat_dim or graph_label_size),
            )
        elif pred_level == 'edge':
            self.predictor = nn.Sequential(
                MultiModelSLP(hidden_dim, hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(graph_dropout_p),
                nn.Linear(hidden_dim, target_dim),
            )
        else:
            print(f"pred_level of {pred_level} is not supported yet!")

    def encode(self, ptb: PathTreeBatch):
        n_feat_list, e_feat_list = self.init.forward(ptb)
        top_repr = self.bottomup.forward(ptb, n_feat_list, e_feat_list)
        graph_repr = aggfn(top_repr, ptb.node_batch,
                           ptb.batch_size, self.readout)
        return graph_repr

    def forward(self, ptb: PathTreeBatch):
        graph_repr = self.encode(ptb)
        graph_pred = self.predictor(graph_repr)
        return graph_pred

    def edge_pred(self, ptb: PathTreeBatch, label_graph: GraphBatch):
        n_feat_list, e_feat_list = self.init.forward(ptb)
        top_repr = self.bottomup.forward(ptb, n_feat_list, e_feat_list)
        node_pair = label_graph.edge_index
        left_repr = top_repr.index_select(0, node_pair[0])
        right_repr = top_repr.index_select(0, node_pair[1])
        edge_pred = self.predictor((left_repr, right_repr))
        return edge_pred
