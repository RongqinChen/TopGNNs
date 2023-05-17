import os
import os.path as osp
from typing import Any, Callable, List, Optional, Union

import torch
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from datautils.graph import Graph
from datautils.tu.reader import read_graph


class MyTUDataset(TUDataset):
    def __init__(
        self, name: str,
        transform_fn: Optional[Callable[[Graph, Any], Graph]] = None,
        transform_fn_kwargs: Optional[dict] = None,
        reload: bool = False
    ):
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._reload = reload
        self.config_path(name)
        # out = torch.load(self.processed_paths[0])
        # self.data, self.slices, self.sizes = out
        super().__init__('datasets/TUD', name)
        self.graph_label = self.data.graph_label
        self.collate = self.data.collate
        self._data_list = self.data.uncollate()
        del self.data

    def config_path(self, name):
        processed_dir = f"datasets/TUD/{name}/processed"
        self.vanilla_path = f"{processed_dir}/vanilla.pt"
        if self._transform_fn is not None:
            transed_fname = f"{self._transform_fn.__name__}"
            if len(self._transform_fn_kwargs) > 0:
                t_arg_list = [f"{key}={val}" for key, val in
                              self._transform_fn_kwargs.items()]
                t_args = ".".join(t_arg_list)
                transed_fname = f"{transed_fname}.{t_args}"
            self.transed_fpath = f"{processed_dir}/{transed_fname}.pt"
            self._processed_paths = self.transed_fpath
        else:
            self.transed_fpath = None
            self._processed_paths = self.vanilla_path

    @property
    def processed_paths(self) -> List[str]:
        return [self._processed_paths]

    def process(self):
        """
        dataset are loaded from `self.processed_paths[0]`
        via `super().__init__()`,
        so `self._processed_paths` should be assigned as the cache path
        """
        if not self._reload and self.transed_fpath is not None \
                and osp.exists(self.transed_fpath):
            # self._processed_paths = self.transed_fpath
            print(f"Loading cached dataset from `{self.transed_fpath}` ...")
        else:
            # self._processed_paths = self.vanilla_path
            if os.path.exists(self.vanilla_path):
                pass
            else:
                # generate vanilla graphs
                data_tuple = read_graph(self.raw_dir, self.name)
                print(f"Saving cache to `{self.vanilla_path}`")
                torch.save(data_tuple, self.vanilla_path)

            if self._transform_fn is None:
                print(f"Loading cached dataset from `{self.vanilla_path}` ...")
            else:
                dataset, _, _ = torch.load(self.vanilla_path)
                vanilla_graph_list = dataset.uncollate()
                tran_graph_list: List[Graph] = [
                    self._transform_fn(graph, **self._transform_fn_kwargs)
                    for graph in tqdm(vanilla_graph_list, dynamic_ncols=True,
                                      desc='Transforming graphs ...')
                ]
                tran_graphbatch = tran_graph_list[0].collate(
                    tran_graph_list, validate=True) # noqa
                sizes = tran_graphbatch.get_sizes()
                data_tuple = (tran_graphbatch, None, sizes)
                torch.save(data_tuple, self.transed_fpath)

    def __getitem__(self, idx: Union[int, list]) \
            -> Union[Graph, List[Graph]]:
        if isinstance(idx, list):
            return Subset(self, idx)

        return self._data_list[idx]
