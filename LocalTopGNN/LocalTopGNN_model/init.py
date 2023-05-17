from typing import List, Optional

import torch
from torch import nn
from torch_sparse import matmul

from datautils.path_tree_rings import PathTreeWithRingsBatch

from .feat_encoder import FeatEncoder
from .mlp import MLP

OptTensor = Optional[torch.Tensor]


class Tree_INIT(nn.Module):
    def __init__(
            self, height: int,
            node_feat_dim: Optional[int] = None,
            node_label_sizes: Optional[List[int]] = None,
            edge_feat_dim: Optional[int] = None,
            edge_label_sizes: Optional[List[int]] = None,
            hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.height = height

        assert not (node_feat_dim is None and node_label_sizes is None)
        self.has_edge_feat = not (
            edge_feat_dim is None and edge_label_sizes is None)

        self.n_enc_list = nn.ModuleList([
            FeatEncoder(hidden_dim, node_label_sizes, node_feat_dim)
            for _ in range(height + 1)
        ])

        if self.has_edge_feat:
            self.e_enc_list = nn.ModuleList([
                FeatEncoder(hidden_dim, edge_label_sizes, edge_feat_dim)
                for _ in range(height + 1)
            ])
        else:
            self.e_update = nn.Sequential(
                MLP(hidden_dim, hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )

        self.r_enc_list = nn.ModuleList([
            nn.Sequential(
                MLP(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ) for _ in range(height + 1)
        ])

    def forward(self, ptrb: PathTreeWithRingsBatch) -> List[torch.Tensor]:
        node_labels = ptrb.node_label
        node_attr = ptrb.node_attr
        edge_labels = ptrb.edge_label
        edge_attr = ptrb.edge_attr

        n_image_list = ptrb.tree_node_image_list
        n_feat_list = [self.n_enc_list[0](node_labels, node_attr)]
        tn_feat_list = [n_feat_list[0]]
        for k in range(1, self.height + 1):
            n_feat = self.n_enc_list[k](node_labels, node_attr)
            n_feat_list.append(n_feat)
            tn_feat_list.append(n_feat.index_select(0, n_image_list[k]))

        e_image_list = ptrb.tree_edge_image_list
        if self.has_edge_feat:
            e_feat_list = [self.e_enc_list[0](edge_labels, edge_attr)]
            te_feat_list = [e_feat_list[0]]
            for k in range(1, self.height + 1):
                e_feat = self.e_enc_list[k](edge_labels, edge_attr)
                e_feat_list.append(e_feat)
                te_feat_list.append(e_feat.index_select(0, e_image_list[k]))

            tr_feat_list = []
            for k in range(self.height + 1):
                e2r_feat = matmul(ptrb.e2r_adj_t, e_feat_list[k])
                ring_feat = self.r_enc_list[k](e2r_feat)
                if k == 0:
                    tr_feat_list.append(ring_feat)
                else:
                    r2e_feat = matmul(ptrb.e2r_adj_t.t(), ring_feat)
                    k_r2e = r2e_feat.index_select(0, e_image_list[k])
                    tr_feat_list.append(k_r2e)
        else:
            n2e_feat = matmul(ptrb.n2e_adj_t, n_feat_list[0])
            edge_feat = self.e_update(n2e_feat)
            te_feat_list = [edge_feat]
            tr_feat_list = []
            for k in range(self.height + 1):
                n2r_feat = matmul(ptrb.n2r_adj_t, n_feat_list[k])
                ring_feat = self.r_enc_list[k](n2r_feat)
                if k == 0:
                    tr_feat_list.append(ring_feat)
                else:
                    r2e_feat = matmul(ptrb.e2r_adj_t.t(), ring_feat)
                    k_r2e = r2e_feat.index_select(0, e_image_list[k])
                    tr_feat_list.append(k_r2e)

        return tn_feat_list, te_feat_list, tr_feat_list
