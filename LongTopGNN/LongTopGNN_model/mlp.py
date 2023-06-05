from typing import Optional, Tuple

from torch import Tensor, stack
from torch.nn import ReLU, BatchNorm1d, Linear, Module


class MLP(Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: Optional[int] = None,
                 Norm: Module = BatchNorm1d, bias: bool = True,
                 ):
        '''MLP
        Args:
            in_dim (int): in_dim
            out_dim (int): out_dim
            hidden_dim (Optional[int], optional): hidden_dim. Defaults to None.
        '''
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        self.out_dim = out_dim
        self.lin1 = Linear(in_dim, hidden_dim, bias=bias)
        self.norm1 = Norm(hidden_dim)
        self.nonlinear = ReLU()
        self.lin2 = Linear(hidden_dim, out_dim, bias=bias)
        self._repr_val = f'(in_dim: {in_dim}, hidden_dim: {hidden_dim}, out_dim: {out_dim})'  # noqa

    def extra_repr(self):
        return self._repr_val

    def forward(self, feat: Tensor):
        h1 = self.lin1(feat)
        h1_norm = self.norm1(h1)
        h1_nonl = self.nonlinear(h1_norm)
        h2 = self.lin2(h1_nonl)
        return h2


class MultiModelSLP(Module):
    def __init__(self, out_dim: int, *in_dims: Tuple[int, ...], bias: bool = True):  # noqa
        '''MultiModelSLP

        Args:
            out_dim (int): out_dim
            *in_dims (int, ...): in_dims

        '''
        super(MultiModelSLP, self).__init__()
        assert len(in_dims) > 1, '# in_dims should greater than 1, otherwise use SLP'  # noqa
        for idx, in_dim in enumerate(in_dims):
            linear = Linear(in_dim, out_dim, bias and idx == 0)
            self.add_module(f'lin_{idx}', linear)

        self._repr_val = f'(in_dims: {in_dims}, out_dim: {out_dim})'

    def extra_repr(self):
        return self._repr_val

    def forward(self, feats: Tuple[Tensor, ...]):
        num_in_feats = len(feats)
        h1_list = []
        for idx in range(num_in_feats):
            feat = feats[idx]
            h1_i = self.get_submodule(f'lin_{idx}')(feat)
            h1_list.append(h1_i)
        h1_stack = stack(h1_list, 1)
        h1_sum = h1_stack.sum(1)
        return h1_sum


class MultiModelMLP(Module):
    def __init__(self, out_dim: int, *in_dims: Tuple[int, ...],
                 hidden_dim: Optional[int] = None,
                 Norm: Module = BatchNorm1d, bias=True,
                 ):
        '''MultiModelMLP
        Args:
            out_dim (int): out_dim
            *in_dims (int, ...): in_dims
            hidden_dim (Optional[int], optional): hidden_dim. Defaults to None.

        '''
        super(MultiModelMLP, self).__init__()
        assert len(in_dims) > 1, 'The numsber of in_dims should greater than 1, otherwise use MLP'  # noqa
        if hidden_dim is None:
            hidden_dim = out_dim

        for idx, in_dim in enumerate(in_dims):
            linear = Linear(in_dim, hidden_dim, bias and idx == 0)
            self.add_module(f'lin1_{idx}', linear)

        self.norm1 = Norm(hidden_dim)
        self.nonlinear = ReLU()
        self.lin2 = Linear(hidden_dim, out_dim, bias)
        self._repr_val = f'(in_dims: {in_dims}, hidden_dim: {hidden_dim}, out_dim: {out_dim})'  # noqa

    def extra_repr(self):
        return self._repr_val

    def forward(self, feats: Tuple[Tensor, ...]):
        num_in_feats = len(feats)
        h1_list = []
        for idx in range(num_in_feats):
            h1_i = self.get_submodule(f'lin1_{idx}')(feats[idx])
            h1_list.append(h1_i)

        h1_stack = stack(h1_list, 1)
        h1_sum = h1_stack.sum(1)
        h1_norm = self.norm1(h1_sum)
        h1_nlin = self.nonlinear(h1_norm)
        h2 = self.lin2(h1_nlin)
        return h2
