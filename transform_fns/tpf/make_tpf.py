import torch as th
try:
    from .tpf_pyx import pyx_make_TPF
except Exception:
    from tpf_pyx import pyx_make_TPF


def make_TPF_rawdata(height, num_nodes, n2n_src, n2n_dst, epath=True):
    """ n2n_src and n2n_dst should be ordered

    Returns:
        relation_data, source_dict, num_nodes_dict, nnhash_2_edge
    """

    if len(n2n_src) > 0:
        _dst_dict, _image_dict, _num_nodes_dict = \
            pyx_make_TPF(height, num_nodes, n2n_src, n2n_dst, epath)

        dst_dict = {key.decode('ascii'): th.LongTensor(val)
                    for key, val in _dst_dict.items()}
        source_dict = {key.decode('ascii'): th.LongTensor(val)
                       for key, val in _image_dict.items()}
        num_nodes_dict = {key.decode('ascii'): val
                          for key, val in _num_nodes_dict.items()}

        num_nodes_dict['edge'] = max(_image_dict[b'edge_1']) + 1
        num_nodes_dict['node_1'] = len(n2n_src)
        source_dict['node_1'] = th.LongTensor(n2n_dst)
        dst_dict['edge_1'] = th.LongTensor(n2n_src)
    else:
        dst_dict = {f'edge_{k}': th.empty((0,)).to(th.long)
                    for k in range(1, height+1)}
        dst_dict['n2e'] = th.empty((0,)).to(th.long)
        source_dict = {
            f'node_{k}': th.empty((0,)).to(th.long) for k in range(1, height+1)
        } | {
            f'edge_{k}': th.empty((0,)).to(th.long) for k in range(1, height+1)
        }
        source_dict['edge'] = th.empty((0,)).to(th.long)
        num_nodes_dict = {f'node_{k}': 0 for k in range(1, height+1)}
        num_nodes_dict['edge'] = 0

    num_nodes_dict['node'] = num_nodes
    return dst_dict, source_dict, num_nodes_dict
