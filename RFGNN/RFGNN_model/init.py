from typing import List, Optional

import torch
from torch import nn

from datautils.path_tree import PathTreeBatch

from .feat_encoder import FeatEncoder

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

    def forward(self, ptb: PathTreeBatch) -> List[torch.Tensor]:
        node_labels = ptb.node_label
        node_attr = ptb.node_attr
        edge_labels = ptb.edge_label
        edge_attr = ptb.edge_attr

        n_image_list = ptb.tree_node_image_list
        n_feat_list = [self.n_enc_list[0](node_labels, node_attr)]
        for k in range(1, self.height + 1):
            n_feat = self.n_enc_list[k](node_labels, node_attr)
            n_feat_list.append(n_feat.index_select(0, n_image_list[k]))

        if self.has_edge_feat:
            e_image_list = ptb.tree_edge_image_list
            e_feat_list = [self.e_enc_list[0](edge_labels, edge_attr)]
            for k in range(1, self.height + 1):
                e_feat = self.e_enc_list[k](edge_labels, edge_attr)
                e_feat_list.append(e_feat.index_select(0, e_image_list[k]))
        else:
            e_feat_list = None

        return n_feat_list, e_feat_list
