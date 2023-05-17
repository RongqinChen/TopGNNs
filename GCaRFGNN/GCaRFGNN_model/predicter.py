from typing import List, Optional

from torch import nn

from datautils.path_tree import PathTreeBatch

from .aggregation import aggfn
from .assign import Tree_Assign
from .bottomup import Tree_BottomUp
from .init import Tree_INIT
from .mlp import MultiModelSLP


class GCaRFGNN_Predictor(nn.Module):
    def __init__(
        self,
        hilayers: int, height: int,
        node_feat_dim: Optional[int] = None,
        node_label_sizes: Optional[List[int]] = None,
        edge_feat_dim: Optional[int] = None,
        edge_label_sizes: Optional[List[int]] = None,
        graph_feat_dim: Optional[int] = None,
        graph_label_size: Optional[List[int]] = None,
        hidden_dim: int = 64,
        dropout_p: float = 0.25, readout: str = 'sum',
        graph_dropout_p: float = 0.5,
        child_agg: str = 'sum',
        norm: str = 'batch',
    ) -> None:
        super().__init__()

        assert hilayers > 1
        self.hilayers = hilayers
        self.readout = readout
        assert norm in {'batch', 'layer'}
        Norm = nn.BatchNorm1d if norm == 'batch' else nn.LayerNorm
        self.init = Tree_INIT(
            height, node_feat_dim, node_label_sizes,
            edge_feat_dim, edge_label_sizes, hidden_dim)
        self.bottomup = Tree_BottomUp(
            height, hidden_dim, self.init.has_edge_feat,
            dropout_p, child_agg, Norm)
        self.himodules = nn.ModuleDict()
        for hi in range(1, hilayers):
            self.himodules[f"assign_{hi}"] = Tree_Assign(height)
            self.himodules[f"bottomup_{hi}"] = Tree_BottomUp(
                height, hidden_dim, True, dropout_p, child_agg, Norm)
        self.predictor = nn.Sequential(
            MultiModelSLP(hidden_dim, hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(graph_dropout_p),
            nn.Linear(hidden_dim, graph_feat_dim or graph_label_size),
        )

    def encode(self, ptb: PathTreeBatch):
        n_feat_list, e_feat_list = self.init.forward(ptb)
        top_repr, edge_repr = self.bottomup.forward(
            ptb, n_feat_list, e_feat_list)
        for hi in range(1, self.hilayers):
            node_agg = aggfn(top_repr, ptb.node_batch,
                             ptb.batch_size, self.readout)
            assign: Tree_Assign = self.himodules[f"assign_{hi}"]
            n_feat_list, e_feat_list = \
                assign.forward(ptb, top_repr, edge_repr, node_agg)
            bottomup: Tree_BottomUp = self.himodules[f"bottomup_{hi}"]
            top_repr, edge_repr = bottomup.forward(
                ptb, n_feat_list, e_feat_list)

        node_agg = aggfn(top_repr, ptb.node_batch,
                         ptb.batch_size, self.readout)
        edge_agg = aggfn(edge_repr, ptb.edge_batch,
                         ptb.batch_size, self.readout)
        return node_agg, edge_agg

    def forward(self, ptb: PathTreeBatch):
        node_agg, edge_agg = self.encode(ptb)
        graph_pred = self.predictor((node_agg, edge_agg))
        return graph_pred
