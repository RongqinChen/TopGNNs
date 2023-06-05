import os
import pickle
import pickle as pkl
import random
from typing import Any, Callable, List, Optional, Union

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import download_url, extract_zip
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

from datautils.graph import Graph


class CSLDataset(Dataset):
    r"""A variety of artificially and semi-artificially generated graph
    datasets from the `"Benchmarking Graph Neural Networks"
    <https://arxiv.org/abs/2003.00982>`_ paper.

    .. note::
        The ZINC dataset is provided via
        :class:`torch_geometric.datasets.ZINC`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"PATTERN"`,
            :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
            :obj:`"TSP"`, :obj:`"CSL"`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
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
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 20 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #graphs
              - #nodes
              - #edges
              - #features
              - #classes
            * - PATTERN
              - 10,000
              - ~118.9
              - ~6,098.9
              - 3
              - 2
            * - CLUSTER
              - 10,000
              - ~117.2
              - ~4,303.9
              - 7
              - 6
            * - MNIST
              - 55,000
              - ~70.6
              - ~564.5
              - 3
              - 10
            * - CIFAR10
              - 45,000
              - ~117.6
              - ~941.2
              - 5
              - 10
            * - TSP
              - 10,000
              - ~275.4
              - ~6,885.0
              - 2
              - 2
            * - CSL
              - 150
              - ~41.0
              - ~164.0
              - 0
              - 10
    """

    root_url = 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1'

    def __init__(self, fold_idx: int,
                 transform_fn: Optional[Callable[[Graph, Any], Graph]] = None,
                 transform_fn_kwargs: Optional[dict] = None,
                 reload: bool = False):

        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        super().__init__()
        self.config_path()
        self.download()
        self.get_split(fold_idx)
        self.graph_list, self.data_sizes, self.labels = self.process()
        self.collate = self.graph_list[0].collate

    def config_path(self):
        self.raw_dir = "datasets/CSL/raw"
        self.split_path = "datasets/CSL/raw/split.pkl"
        proc_dir = "datasets/CSL/processed"
        os.makedirs(proc_dir, exist_ok=True)
        self.vanilla_path = f"{proc_dir}/vanilla.pt"

        if self._transform_fn is not None:
            transed_fname = f"{self._transform_fn.__name__}"
            if len(self._transform_fn_kwargs) > 0:
                t_arg_list = [f"{key}={val}" for key, val in
                              self._transform_fn_kwargs.items()]
                t_args = ".".join(t_arg_list)
                transed_fname = f"{transed_fname}.{t_args}"
            self.transed_path = f"{proc_dir}/{transed_fname}.pt"
        else:
            self.transed_path = None

    def get_split(self, fold_idx):
        if os.path.exists(self.split_path) and False:
            with open(self.split_path, 'rb') as rbf:
                class_idx_list = pkl.load(rbf)
        else:
            class_idx_list = [
                list(range(idx * 15, (idx + 1) * 15))
                for idx in range(10)
            ]
            for cidx in range(10):
                random.shuffle(class_idx_list[cidx])
            with open(self.split_path, 'wb') as wbf:
                pkl.dump(class_idx_list, wbf)

        ridx_list = [list(range(idx * 3, (idx + 1) * 3)) for idx in range(5)]

        test_idx = fold_idx
        valid_idx = (fold_idx+1) % 5
        train_idxs = set(range(5)).difference({test_idx, valid_idx})

        test_ridx = ridx_list[test_idx]
        valid_ridx = ridx_list[valid_idx]
        train_ridx = sum([ridx_list[idx] for idx in train_idxs], list())

        train_idxs = [class_idx_list[idx][ridx]
                      for idx in range(10) for ridx in train_ridx]
        valid_idxs = [class_idx_list[idx][ridx]
                      for idx in range(10) for ridx in valid_ridx]
        test_idxs = [class_idx_list[idx][ridx]
                     for idx in range(10) for ridx in test_ridx]
        return train_idxs, valid_idxs, test_idxs

    @property
    def raw_paths(self) -> List[str]:
        return [
            f"{self.raw_dir}/graphs_Kary_Deterministic_Graphs.pkl",
            f"{self.raw_dir}/y_Kary_Deterministic_Graphs.pt"
        ]

    def download(self):
        if os.path.exists(self.raw_paths[0]):
            return
        path = download_url(self.root_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        if os.path.exists(self.transed_path):
            print(f'Loading data from {self.transed_path}')
            transed_graph_batch, sizes = torch.load(self.transed_path)
            graph_label = transed_graph_batch.graph_label
            transed_graph_list = transed_graph_batch.uncollate()
            del transed_graph_batch
            return transed_graph_list, sizes, graph_label

        if os.path.exists(self.vanilla_path):
            print(f'Loading from {self.vanilla_path}')
            vanilla_graph_batch, sizes = torch.load(self.vanilla_path)
            graph_label = vanilla_graph_batch.graph_label
            vanilla_graph_list = vanilla_graph_batch.uncollate()
            del vanilla_graph_batch
        else:
            vanilla_graph_list = self.process_vanilla_graphs()
            vanilla_graph_batch = \
                vanilla_graph_list[0].collate(vanilla_graph_list)
            sizes = vanilla_graph_batch.get_sizes()
            torch.save((vanilla_graph_batch, sizes), self.vanilla_path)
            del vanilla_graph_batch

        if self._transform_fn is not None:
            transed_graph_list = self.transform(vanilla_graph_list)
            transed_graph_batch = \
                transed_graph_list[0].collate(transed_graph_list)
            sizes = transed_graph_batch.get_sizes()
            graph_label = transed_graph_batch.graph_label
            torch.save((transed_graph_batch, sizes), self.transed_path)
            del vanilla_graph_list
            del transed_graph_batch
            return transed_graph_list, sizes, graph_label
        else:
            return vanilla_graph_list, sizes, graph_label

    def process_vanilla_graphs(self) -> List[Graph]:
        with open(self.raw_paths[0], 'rb') as f:
            adjs = pickle.load(f)

        ys = torch.load(self.raw_paths[1]).tolist()

        graph_list = []
        for adj, y in zip(adjs, ys):
            row, col = torch.from_numpy(adj.row), torch.from_numpy(adj.col)
            edge_index = torch.stack([row, col], dim=0).to(torch.long)
            edge_index, _ = remove_self_loops(edge_index)
            hash = edge_index[0, :] * adj.shape[0] + edge_index[1, :]
            torch.all(torch.diff(hash) > 0)

            graph = Graph(
                adj.shape[0],
                edge_index,
                torch.ones((adj.shape[0], 1)),
                None, None, None, None,
                torch.LongTensor([y]))
            graph_list.append(graph)

        return graph_list

    def transform(self, vanilla_graph_list: List[Graph]):
        print('Transforming graphs ...')
        tran_graph_list: List[Graph] = [
            self._transform_fn(graph, **self._transform_fn_kwargs)
            for graph in tqdm(vanilla_graph_list, dynamic_ncols=True)
        ]
        return tran_graph_list

    def __getitem__(self, idx: Union[int, list]) \
            -> Union[Graph, List[Graph]]:
        if isinstance(idx, list):
            return Subset(self, idx)

        return self.graph_list[idx]

    def __len__(self):
        return len(self.graph_list)

    def __repr__(self) -> str:
        return 'CSL'
