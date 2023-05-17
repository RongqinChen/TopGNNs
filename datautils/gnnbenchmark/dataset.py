import os
import pickle
from typing import Any, Callable, List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import download_url, extract_zip
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

from datautils.graph import Graph


class MyGNNBenchmarkDataset(Dataset):
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

    names = ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']

    root_url = 'https://data.pyg.org/datasets/benchmarking-gnns'
    urls = {
        'PATTERN': f'{root_url}/PATTERN_v2.zip',
        'CLUSTER': f'{root_url}/CLUSTER_v2.zip',
        'MNIST': f'{root_url}/MNIST_v2.zip',
        'CIFAR10': f'{root_url}/CIFAR10_v2.zip',
        'TSP': f'{root_url}/TSP_v2.zip',
        'CSL': 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1',
    }

    def __init__(self, name: str, split: str,
                 transform_fn: Optional[Callable[[Graph, Any], Graph]] = None,
                 transform_fn_kwargs: Optional[dict] = None,
                 reload: bool = False):

        self.name = name
        assert self.name in self.names

        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        self._split = split
        super().__init__()
        self.config_path(name)
        self.download()
        self.graph_list, self.data_sizes = self.process()
        self.collate = self.graph_list[0].collate

    def config_path(self, name):
        self.raw_dir = f"datasets/GNNBenchmark/{name}/raw"
        proc_dir = f"datasets/GNNBenchmark/{name}/processed"
        os.makedirs(proc_dir, exist_ok=True)
        self.vanilla_path = f"{proc_dir}/{self._split}_vanilla.pt"

        if self._transform_fn is not None:
            transed_fname = f"{self._transform_fn.__name__}"
            if len(self._transform_fn_kwargs) > 0:
                t_arg_list = [f"{key}={val}" for key, val in
                              self._transform_fn_kwargs.items()]
                t_args = ".".join(t_arg_list)
                transed_fname = f"{transed_fname}.{t_args}"
            self.transed_path = f"{proc_dir}/{self._split}_{transed_fname}.pt"
        else:
            self.transed_path = None

    @property
    def raw_paths(self) -> List[str]:
        if self.name == 'CSL':
            return [
                f"{self.raw_dir}/graphs_Kary_Deterministic_Graphs.pkl",
                f"{self.raw_dir}/y_Kary_Deterministic_Graphs.pt"
            ]
        else:
            name = self.urls[self.name].split('/')[-1][:-4]
            return [f"{self.raw_dir}/{name}.pt"]

    def download(self):
        if os.path.exists(self.raw_paths[0]):
            return
        path = download_url(self.urls[self.name], self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        if os.path.exists(self.transed_path):
            transed_graph_batch, sizes = torch.load(self.transed_path)
            transed_graph_list = transed_graph_batch.uncollate()
            del transed_graph_batch
            return transed_graph_list, sizes

        if os.path.exists(self.vanilla_path):
            vanilla_graph_batch, sizes = torch.load(self.vanilla_path)
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
            torch.save((transed_graph_batch, sizes), self.transed_path)
            del transed_graph_batch
            return transed_graph_list, sizes
        else:
            return vanilla_graph_list, sizes

    def __repr__(self) -> str:
        return f'{self.name}'

    def process_vanilla_graphs(self) -> List[Graph]:
        if self.name == 'CSL':
            graph_list = self.process_CSL()
        else:
            idx = ['train', 'val', 'test'].index(self._split)
            inputs = torch.load(self.raw_paths[0])[idx]
            graph_list = []
            for datadict in tqdm(
                    inputs, 'Processing vanilla graphs ...'):
                num_nodes = datadict['x'].size(0)
                edge_index = datadict['edge_index']
                edge_index, _ = remove_self_loops(edge_index)
                hash = edge_index[0, :] * num_nodes + edge_index[1, :]
                torch.all(torch.diff(hash) > 0)
                graph = Graph(num_nodes, edge_index, datadict['x'],
                              None, None, None, None, None)
                graph_list.append(graph)

        return graph_list

    def process_CSL(self) -> List[Graph]:
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
        tran_graph_list: List[Graph] = [
            self._transform_fn(graph, **self._transform_fn_kwargs)
            for graph in tqdm(vanilla_graph_list, dynamic_ncols=True,
                              desc='Transforming graphs ...')
        ]
        return tran_graph_list
