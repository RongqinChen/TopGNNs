from typing import List, Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

OptTensor = Optional[torch.Tensor]


class FeatEncoder(nn.Module):
    def __init__(self, out_dim: int,
                 label_sizes: Optional[List[int]] = None,
                 attr_size: Optional[int] = None) -> None:
        super().__init__()

        self._label_sizes = label_sizes
        self._attr_size = attr_size
        self._out_dim = out_dim

        if label_sizes is not None:
            self.nns = nn.ModuleList()
            for in_dim in label_sizes:
                emb = nn.Embedding(in_dim, out_dim)
                self.nns.append(emb)

        if attr_size is not None:
            self.linear = nn.Linear(attr_size, out_dim)

    def reset_parameters(self):
        if self._label_sizes is not None:
            for emb in self.nns:
                xavier_uniform_(emb.weight.data)

        if self._attr_size is not None:
            xavier_uniform_(self.linear.weight.data)

    def forward(self, labels: OptTensor, attr: OptTensor):
        embed = 0.
        if self._label_sizes is not None:
            embed_list = []
            for idx, label in enumerate(labels.T):
                embed_list.append(self.nns[idx](label))

            embed_stack = torch.stack(embed_list, 1)
            embed = embed_stack.sum(1)

        if self._attr_size is not None:
            embed2 = self.linear(attr)
            embed += embed2

        return embed

    def __repr__(self) -> str:
        repr = f"FeatEncoder({self._label_sizes}, " + \
            f"{self._attr_size}) -> {self._out_dim}"
        return repr
