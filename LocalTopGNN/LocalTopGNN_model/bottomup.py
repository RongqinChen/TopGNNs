from typing import List, Optional

import torch
from torch import nn

from datautils.path_tree_rings import PathTreeWithRingsBatch

from .aggregation import aggfn
from .mlp import MultiModelSLP, MultiModelMLP

OptTensor = Optional[torch.Tensor]


class Tree_BottomUp(nn.Module):

    def __init__(self, height, hidden_dim: int, has_edge_feat: bool,
                 dropout_p: float, agg='sum', Norm=nn.Module) -> None:
        super().__init__()

        self.height = height
        self.hidden_dim = hidden_dim
        self.has_edge_feat = has_edge_feat
        self.agg = agg

        def new_e_update_module():
            if has_edge_feat:
                return MultiModelSLP(hidden_dim, hidden_dim,
                                     hidden_dim, hidden_dim)
            else:
                return MultiModelSLP(hidden_dim, hidden_dim, hidden_dim)

        self.e_update_dict = nn.ModuleDict({
            str(k): nn.Sequential(
                new_e_update_module(),
                nn.Dropout(dropout_p)
            )
            for k in range(self.height, 0, -1)
        })

        self.n_update_dict = nn.ModuleDict({
            str(k): nn.Sequential(
                MultiModelMLP(hidden_dim, hidden_dim, hidden_dim, Norm=Norm),
                Norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            )
            for k in range(self.height, 0, -1)
        })

    def forward(self, ptrb: PathTreeWithRingsBatch,
                n_feat_list: List[torch.Tensor],
                e_feat_list: List[torch.Tensor],
                r_feat_list: List[torch.Tensor],
                ) -> torch.Tensor:

        dst_list = ptrb.tree_e_dst_list
        num_nodes_list = ptrb.num_tree_nodes_list
        n_repr_list = [0] * (self.height+1)
        n_repr_list[self.height] = n_feat_list[self.height]
        for k in range(self.height, 0, -1):
            k_n_repr = n_repr_list[k]
            k_r_feat = r_feat_list[k]
            if self.has_edge_feat:
                k_e_feat = e_feat_list[k]
                k_e_repr = self.e_update_dict[str(k)]((
                    k_e_feat, k_r_feat, k_n_repr))
            else:
                k_e_repr = self.e_update_dict[str(k)]((k_r_feat, k_n_repr))
            agg = aggfn(k_e_repr, dst_list[k],
                        num_nodes_list[k-1], self.agg)
            dst_n_repr = n_feat_list[k-1]
            updated = self.n_update_dict[str(k)]((dst_n_repr, agg))
            n_repr_list[k-1] = updated

        top_repr = n_repr_list[0]
        return top_repr

    def extra_repr(self) -> str:
        return f"aggregation={self.agg}"
