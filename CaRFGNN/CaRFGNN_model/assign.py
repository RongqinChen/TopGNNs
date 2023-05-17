from typing import Optional

import torch
from torch import nn

from datautils.path_tree import PathTreeBatch


OptTensor = Optional[torch.Tensor]


class Tree_Assign(nn.Module):

    def __init__(self, height) -> None:

        super().__init__()
        self.height = height

    def forward(self, ptb: PathTreeBatch,
                node_feat: torch.Tensor,
                edge_feat: torch.Tensor):

        height_p = self.height + 1
        n_image_list = ptb.tree_node_image_list
        e_image_list = ptb.tree_edge_image_list
        tn_feat_list = [node_feat] + [
            node_feat.index_select(0, n_image_list[k])
            for k in range(1, height_p)]
        te_feat_list = [edge_feat] + [
            edge_feat.index_select(0, e_image_list[k])
            for k in range(1, height_p)]
        return tn_feat_list, te_feat_list
