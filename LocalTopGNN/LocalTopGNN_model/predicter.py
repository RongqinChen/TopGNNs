from typing import List, Optional

from torch import nn
from torch_sparse import matmul

from datautils.path_tree_rings import PathTreeWithRingsBatch

from .aggregation import aggfn
from .bottomup import Tree_BottomUp
from .init import Tree_INIT
from .mlp import MultiModelMLP, MultiModelSLP


class LocalTopGNN_Predictor(nn.Module):
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
        dropout_p: float = 0.25, readout: str = 'sum',
        graph_dropout_p: float = 0.5,
        child_agg: str = 'sum',
        norm: str = 'batch',
    ) -> None:
        super().__init__()

        self.readout = readout
        assert norm in {'batch', 'layer'}
        Norm = nn.BatchNorm1d if norm == 'batch' else nn.LayerNorm
        self.init = Tree_INIT(
            height, node_feat_dim, node_label_sizes,
            edge_feat_dim, edge_label_sizes, hidden_dim)
        self.bottomup = Tree_BottomUp(height, hidden_dim,
                                      self.init.has_edge_feat,
                                      dropout_p, child_agg, Norm)
        self.n2e_update = nn.Sequential(
            MultiModelMLP(hidden_dim, hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.n2r_update = nn.Sequential(
            MultiModelMLP(hidden_dim, hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.predictor = nn.Sequential(
            MultiModelSLP(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(graph_dropout_p),
            nn.Linear(hidden_dim, graph_feat_dim or graph_label_size),
        )

    def encode(self, ptrb: PathTreeWithRingsBatch):
        n_feat_list, e_feat_list, r_feat_list = self.init.forward(ptrb)
        top_repr = self.bottomup.forward(
            ptrb, n_feat_list, e_feat_list, r_feat_list)

        n2e_feat = matmul(ptrb.n2e_adj_t, top_repr)
        edge_repr = self.n2e_update((e_feat_list[0], n2e_feat))
        n2r_feat = matmul(ptrb.n2r_adj_t, top_repr)
        ring_repr = self.n2r_update((r_feat_list[0], n2r_feat))

        node_agg = aggfn(top_repr, ptrb.node_batch,
                         ptrb.batch_size, self.readout)
        edge_agg = aggfn(edge_repr, ptrb.edge_batch,
                         ptrb.batch_size, self.readout)
        ring_agg = aggfn(ring_repr, ptrb.ring_batch,
                         ptrb.batch_size, self.readout)
        return node_agg, edge_agg, ring_agg

    def forward(self, ptrb: PathTreeWithRingsBatch):
        node_agg, edge_agg, ring_agg = self.encode(ptrb)
        graph_pred = self.predictor((node_agg, edge_agg, ring_agg))
        return graph_pred
