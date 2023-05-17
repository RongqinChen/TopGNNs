from torch import Tensor, arange
from torch_sparse import matmul, SparseTensor


def aggfn(src_feat: Tensor, dst_idxs: Tensor,
          num_dst: int, agg: str = 'sum'):
    src_idxs = arange(dst_idxs.size(0)).to(dst_idxs.device)
    adj_t = SparseTensor(
        row=dst_idxs, col=src_idxs,
        sparse_sizes=(num_dst, dst_idxs.size(0)))
    dst_feat = matmul(adj_t, src_feat, agg)
    return dst_feat
