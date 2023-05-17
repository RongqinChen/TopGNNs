import os
import os.path as osp
from typing import Callable, List, Optional, Any

import torch
from tqdm import tqdm

from torch.utils.data import Dataset
from torch_geometric.data import (download_url, extract_zip)
from datautils.graph import Graph


class PCQMContactDataset(Dataset):
    r"""The `"Long Range Graph Benchmark (LRGB)"
    <https://arxiv.org/abs/2206.08164>`_
    datasets which is a collection of 5 graph learning datasets with tasks
    that are based on long-range dependencies in graphs. See the original
    `source code <https://github.com/vijaydwivedi75/lrgb>`_ for more details
    on the individual datasets.

    +------------------------+-------------------+----------------------+
    | Dataset                | Domain            | Task                 |
    +========================+===================+======================+
    | :obj:`PCQM-Contact`    | Quantum Chemistry | Link Prediction      |
    +------------------------+-------------------+----------------------+

    Args:
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 15 10 10 10 10
        :header-rows: 1
        * - PCQM-Contact
          - 529,434
          - ~30.14
          - ~61.09
          - 1
    """

    names = [
        'pascalvoc-sp', 'coco-sp', 'pcqm-contact', 'peptides-func',
        'peptides-struct'
    ]

    urls = {
        'pcqm-contact':
        'https://www.dropbox.com/s/qdag867u6h6i60y/pcqmcontact.zip?dl=1',
    }

    dwnld_file_name = {
        'pcqm-contact': 'pcqmcontact',
    }

    def __init__(self, split,
                 transform_fn: Optional[Callable[[Graph, Any], Graph]] = None,
                 transform_fn_kwargs: Optional[dict] = None,
                 reload: bool = False):

        assert split in {'train', 'val', 'test'}
        self._split = split
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        super().__init__()
        self.name = 'pcqm-contact'
        self.config_path()
        self.download()
        self.input_graph_list, self.label_graph_list, self.data_sizes =\
            self.process()
        # self.collate = self.input_graph_list[0].collate

    def config_path(self):
        self.root_dir = f"datasets/{self.name}"
        self.raw_dir = f"{self.root_dir}/raw"
        self.raw_fpath = f"{self.raw_dir}/{self._split}.pt"
        os.makedirs(self.raw_dir, exist_ok=True)
        proc_dir = f"{self.root_dir}/processed"
        os.makedirs(proc_dir, exist_ok=True)
        self.raw_graph_path = f"{proc_dir}/{self._split}.raw.pt"

        if self._transform_fn is not None:
            transed_fname = f"{self._transform_fn.__name__}"
            if len(self._transform_fn_kwargs) > 0:
                t_arg_list = [f"{key}={val}" for key, val in
                              self._transform_fn_kwargs.items()]
                t_args = ".".join(t_arg_list)
                transed_fname = f"{transed_fname}.{t_args}"
            self.transed_path = f"{proc_dir}/{self._split}.{transed_fname}.pt"
        else:
            self.transed_path = ''

    def download(self):
        if os.path.exists(self.raw_fpath):
            return
        path = download_url(self.urls[self.name], self.root_dir)
        extract_zip(path, self.root_dir)
        os.rename(osp.join(self.root_dir, self.dwnld_file_name[self.name]),
                  self.raw_dir)
        os.unlink(path)

    def process(self):
        if os.path.exists(self.transed_path):
            print(f'Loading data from {self.transed_path}',
                  end='\t......\t', flush=True)
            tr_graph_batch, label_graph_batch, data_sizes = \
                torch.load(self.transed_path)
            tr_graph_list = tr_graph_batch.uncollate()
            label_graph_list = label_graph_batch.uncollate()
            print('Done')
            del tr_graph_batch
            del label_graph_batch
            return tr_graph_list, label_graph_list, data_sizes

        if os.path.exists(self.raw_graph_path):
            print(f'Loading data from {self.raw_graph_path}',
                  end='\t......\t', flush=True)
            raw_graph_batch, label_graph_batch, data_sizes = \
                torch.load(self.raw_graph_path)
            raw_graph_list = raw_graph_batch.uncollate()
            label_graph_list = label_graph_batch.uncollate()
            print('Done')
        else:
            raw_graph_list, label_graph_list = self.process_raw_graphs()
            raw_graph_batch = raw_graph_list[0].collate(raw_graph_list)
            label_graph_batch = label_graph_list[0].collate(label_graph_list)
            data_sizes = raw_graph_batch.get_sizes()
            torch.save(
                (raw_graph_batch, label_graph_batch, data_sizes),
                self.raw_graph_path
            )

        del raw_graph_batch

        if self._transform_fn is not None:
            tr_graph_list = self.transform(raw_graph_list)
            tr_graph_batch = tr_graph_list[0].collate(tr_graph_list)
            torch.save(
                (tr_graph_batch, label_graph_batch, data_sizes),
                self.transed_path
            )
            del tr_graph_batch
            del label_graph_batch
            return tr_graph_list, label_graph_list, data_sizes
        else:
            del label_graph_batch
            return raw_graph_list, label_graph_list, data_sizes

    def process_raw_graphs(self) -> List[Graph]:
        print(f'Loading {self.raw_fpath}',
              end='\t......\t', flush=True)
        with open(self.raw_fpath, 'rb') as rbf:
            graphs = torch.load(rbf)
        print('Done')
        raw_graph_list: list[Graph] = []
        label_graph_list: list[Graph] = []
        for graph in tqdm(graphs, desc=f'Processing {self._split} dataset'):
            """
            PCQM-Contact
            Each `graph` is a tuple (x, edge_attr, edge_index,
                                    edge_label_index, edge_label)
                Shape of x : [num_nodes, 9]
                Shape of edge_attr : [num_edges, 3]
                Shape of edge_index : [2, num_edges]
                Shape of edge_label_index: [2, num_labeled_edges]
                Shape of edge_label : [num_labeled_edges]

                where,
                num_labeled_edges are negative edges and link pred labels,
                https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/loader/dataset/pcqm4mv2_contact.py#L192
            """
            nodel_label = graph[0]
            edge_label = graph[1]
            edge_index = graph[2]

            num_nodes = nodel_label.shape[0]
            hash = edge_index[0, :] * num_nodes + edge_index[1, :]
            sorted, indices = torch.sort(hash)
            edge_index = torch.stack((sorted // num_nodes, sorted % num_nodes))
            edge_label = edge_label[indices, :]

            raw_graph = Graph(num_nodes=num_nodes,
                              edge_index=edge_index,
                              node_attr=None, node_label=nodel_label,
                              edge_attr=None, edge_label=edge_label,
                              graph_attr=None, graph_label=None)
            raw_graph_list.append(raw_graph)

            nodepair_index = graph[3]
            nodepair_label = graph[4]
            label_graph = Graph(num_nodes, nodepair_index,
                                None, None, None, nodepair_label,
                                graph_attr=None, graph_label=None)
            label_graph_list.append(label_graph)

        return raw_graph_list, label_graph_list

    def transform(self, raw_graph_list: List[Graph]):
        print('Transforming graphs ...')
        tr_graph_list: List[Graph] = [
            self._transform_fn(graph, **self._transform_fn_kwargs)
            for graph in tqdm(raw_graph_list, dynamic_ncols=True)
        ]
        return tr_graph_list

    def __getitem__(self, idx: int) -> Graph:
        return self.input_graph_list[idx], self.label_graph_list[idx]

    def __len__(self):
        return len(self.input_graph_list)

    def __repr__(self) -> str:
        return 'PCQM-contact'


if __name__ == "__main__":
    dataset = PCQMContactDataset('train')
    print(dataset)
